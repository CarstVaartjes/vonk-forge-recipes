# Hy3-295B on 2× DGX Spark — 21.8 tok/s single-stream · NVFP4 · MTP spec

> Tencent Hunyuan 3 (295B MoE / 21B active / 256K native ctx) in NVFP4-W4A16, served across two NVIDIA DGX Sparks (GB10, sm121) at TP=2 over the 200GbE RoCE fabric — with the model's native 3.8B MTP layer live for speculative decoding.

## TL;DR

- **What you get:** a working, benched recipe to serve Hy3-295B on 2× DGX Spark with MTP speculative decoding on — weights, launch scripts, exact serve flags, and the bring-up bugs already solved.
- **Numbers:** **21.8 tok/s single-stream**, **59.7 tok/s aggregate** at 6-way concurrency (~10/stream), 128K max ctx, FP8 KV.
- **The finding:** on GB10, run `num_speculative_tokens: 1` and `--enforce-eager` — both counter to the official/H200-tuned advice, both measured to win here.
- As far as we know these are the **first published Hy3-on-DGX-Spark numbers with MTP speculative decoding enabled** (2026-07-07). The NVIDIA forum threads were at the "sizing math + one failed FlashInfer load" stage when we brought this up.

## Hardware

- **2× NVIDIA DGX Spark** (GB10, sm121) — TP=2 (Bluey head + Reddie worker).
- **128GB GB10 unified memory** per node, of which **~111GiB is actually free** at boot (see bug #4).
- **Interconnect:** 200GbE RoCE fabric, ConnectX-7 NICs, MTU 9000, NCCL pinned to the fabric interface.
- **Disk / weights placement:** both nodes need the full **181GB** locally at the same path (`~/models/hy3-nvfp4-w4a16`). Rsync over the fabric takes **~7 min at ~460MB/s**. Budget per node: ~90GB weights + ~15GB KV inside the ~111GiB free.

**Note:** Hy3 has **8 KV heads → TP=3 is mathematically impossible** in vLLM. It's a 2-Spark or 4-Spark model.

## Quick start

1. **Weights** — put the 181GB [`kodelow/Hy3-NVFP4-W4A16`](https://huggingface.co/kodelow/Hy3-NVFP4-W4A16) checkpoint on **both** nodes at `~/models/hy3-nvfp4-w4a16` (download once, rsync to the second node over the fabric).
2. **Launch** — from the head node (Bluey), run the launcher. It tears down any prior containers, starts the Ray head + worker in docker with fabric-pinned NCCL, waits for 2 Ray nodes, then launches `vllm serve` detached inside the head container:
   ```
   scripts/hy3-launch.sh
   ```
3. **Smoke test** — tail the serve log until the API is up:
   ```
   docker exec hy3-head tail -f /tmp/hy3-serve.log
   ```

## Setup (detailed)

### Weights

[`kodelow/Hy3-NVFP4-W4A16`](https://huggingface.co/kodelow/Hy3-NVFP4-W4A16) — 181GB, quantized from the FINAL Hy3 release, routed experts 4-bit / everything quality-sensitive BF16, **MTP layer preserved** (author-measured 83.4% GSM8K acceptance, lossless quality). Critically it is **MARLIN-kernel-only by design**, which sidesteps the FlashInfer native-FP4 path that freezes GB10s.

### Image / serving stack

- vLLM **0.23.1** (any ≥0.23 with `HYV3ForCausalLM`) — this recipe was run on image `vllm-node-tf5-glm52-b12x:probe-modded` (vLLM 0.23.1rc1.dev190).
- Ray for the 2-node TP=2; NCCL over the ConnectX-7 RoCE fabric (MTU 9000).
- GB10 launch flags are based on [`LibertAIDAI/Hy3-preview-NVFP4`](https://huggingface.co/LibertAIDAI/Hy3-preview-NVFP4)'s verified 2×GB10 bring-up (marlin backend, CUDA-13 image, vLLM ≥ 0.23).

### Launch

Three scripts:

```
scripts/hy3-launch.sh      # Ray head (node1) + worker (node2) in docker, fabric-pinned NCCL
scripts/serve-hy3.sh       # v1 stable config (chat-only), run inside the head container
scripts/serve-hy3-tools.sh # same + parser patch + hy_v3 tool/reasoning parsers ON (structured tool calls)
```

Key serve flags (see scripts for full):

```
vllm serve /models --served-model-name hy3 --port 8600 \
  --tensor-parallel-size 2 --distributed-executor-backend ray \
  --max-model-len 131072 --max-num-seqs 6 \
  --kv-cache-dtype fp8_e4m3 --gpu-memory-utilization 0.90 \
  --speculative-config '{"method":"mtp","num_speculative_tokens":1}' \
  --trust-remote-code --enforce-eager
```

### Verify

- Watch startup: `docker exec hy3-head tail -f /tmp/hy3-serve.log`.
- With the tool/reasoning parsers on (`serve-hy3-tools.sh`), a weather-function call returns `finish_reason: tool_calls` with a clean `tool_calls` object (`{"city":"Atlanta"}`), not plain text.

## Benchmarks

*This repo is being tuned live — watch the commits.* 512-token generations, temp 0.9 / top_p 1.0 (Tencent-recommended), 128K max ctx, FP8 KV:

| Config | Single-stream | 6-way concurrent | Notes |
|---|---|---|---|
| **v1: enforce-eager + MTP spec-1** | **21.8 tok/s** | **59.7 tok/s agg** (~10/stream) | ✅ stable, verified twice |
| v2: CUDA graphs + MTP spec-2 | 15–16 tok/s | — | ❌ spec-2 is a net LOSS (see below) |
| v2.2: CUDA graphs + MTP spec-1 | 15.5–16.3 tok/s | — | ❌ the compiled/graphs path itself is the tax |

**Verdict after clean A/B isolation: `--enforce-eager` WINS on this stack.** Removing it (inductor compile + CUDA graphs, ~30s compile) cost ~25% throughput with BOTH spec-1 and spec-2 — the compiled path interacts badly with the marlin W4A16 decode on sm121 in vLLM 0.23. Counter-intuitive but measured twice. Revisit on newer stacks (NVIDIA's vLLM 26.06 container with NVFP4 paged-KV is the obvious next candidate).

**MTP acceptance on real prompts (the finding):** position-1 draft acceptance ran **62–76%**, but position-2 only **~18–21%**. With `num_speculative_tokens: 2` the second draft is thrown away four times out of five while you still pay its draft+verify cost every step — net throughput DROPS ~30% vs spec-1. **On GB10, run `num_speculative_tokens: 1`,** despite the official recipe suggesting 2 (that advice is tuned for H200/GB300-class serving).

First words the model produced on this hardware:
> *Tiny twin cores hum, / shoulder to shoulder they forge— / a giant's lost throne.*

## Configuration

The tunable knobs (defaults from `env/hy3-nvfp4-tp2.env`):

| Knob | Value | Notes / tradeoff |
|---|---|---|
| `MAX_LEN` (`--max-model-len`) | `131072` | Start conservative; raise after stable (KV ~0.33MB/tok bf16). |
| `KV_DTYPE` (`--kv-cache-dtype`) | `fp8_e4m3` | Frees room toward 256K later. |
| `GPU_MEM_UTIL` (`--gpu-memory-utilization`) | `0.90` | GB10 does not expose 0.92 at boot (bug #4). |
| `MAX_NUM_SEQS` (`--max-num-seqs`) | `6` | 1-stream smoke + 6-way concurrency off the same serve. |
| `SPEC_CONFIG` (`--speculative-config`) | `{"method":"mtp","num_speculative_tokens":1}` | MTP spec-1 (kodelow: 83.4% acceptance, lossless); spec-2 is a net loss on GB10. |
| `MOE_BACKEND` | `marlin` | GB10/sm121 native FP4 MoE kernels have gaps → force MARLIN (LibertAI-verified). |
| `ENFORCE_EAGER` (`--enforce-eager`) | `1` | LibertAI bring-up flag; CUDA graphs rejected on data (see Benchmarks). |

Fixed endpoints/paths from the env: `MODEL_PATH=/models/hy3-nvfp4-w4a16` (bind-mounted from `~/models/hy3-nvfp4-w4a16`), `SERVED_NAME=hy3`, `PORT=8600`, `HEAD_IP=192.168.192.1` (Bluey), `WORKER_IP=192.168.192.2` (Reddie), `TP=2`.

### KV / context math

Hy3 GQA (64Q/8KV, 80 layers): **~0.33MB/token BF16, ~0.16MB FP8**. At 181GB weights on 2×GB10 with GMU 0.90 you get roughly a **~180K-token KV pool at FP8** — comfortable for the 128K launch config with 6 sequences. 256K single-sequence is a stretch goal (needs ~42GB pool); the smaller MXFP4 quant (172GB) buys ~+110K pool tokens if you need it.

## Troubleshooting — the bring-up bugs you WILL hit (we hit them all so you don't have to)

1. **`World size (2) larger than available GPUs (1)`** — multi-node vLLM 0.23 requires `--distributed-executor-backend ray` spelled out. The error message tells you; believe it.
2. **`--speculative-config: Value method:mtp cannot be converted`** — your JSON got mangled by shell quoting (especially through SSH hops). Ship the serve command as a **script file** and `docker cp` it into the container; never inline-quote JSON through nested shells.
3. **`HYV3ReasoningParser could not locate think start/end tokens` → SOLVED (structured tool calls work).** Root cause: this checkpoint's special tokens carry a **`:opensource` suffix** (chat template sets `HYTK=':opensource'`, so the model emits `<think:opensource>`, `<tool_calls:opensource>`, `<tool_call:opensource>`, `<tool_sep:opensource>`, `<arg_key:opensource>`, `<arg_value:opensource>`), but vLLM's `hy_v3` reasoning + tool parsers hardcode the BARE forms. The lookup misses → startup crash with parsers on, and with parsers off you get fake plain-text tool calls in `content`. **Fix:** sed the two parser files to the suffixed tokens (see `scripts/serve-hy3-tools.sh`, which bakes the patch in before `vllm serve`):
   ```
   RP=.../vllm/reasoning/hy_v3_reasoning_parser.py
   TP=.../vllm/tool_parsers/hy_v3_tool_parser.py
   sed -i 's|"<think>"|"<think:opensource>"|g; s|"</think>"|"</think:opensource>"|g' "$RP"
   sed -i 's|"<tool_calls>"|"<tool_calls:opensource>"|g; ...(all 9 tool tokens)...' "$TP"
   ```
   Then launch with `--tool-call-parser hy_v3 --reasoning-parser hy_v3 --enable-auto-tool-choice`. **Verified:** a weather-function call returns `finish_reason: tool_calls` with a clean `tool_calls` object (`{"city":"Atlanta"}`), not plain text. (Applies to any HYTK-suffixed quant; upstream vLLM should read the suffix from the tokenizer instead of hardcoding.)
4. **`Free memory on device (111.45/121.69 GiB) less than desired utilization (0.92)`** — a GB10 does NOT expose 0.92×121.69 at boot. **Use `--gpu-memory-utilization 0.90`.**
5. **Serve silently freezes mid-load, no log lines** — if you background the serve from a `docker exec -i` session, its stdout can end up on a dead pipe that fills (64KB) and **blocks the whole process**. Have the script itself `exec > /logfile 2>&1`, `docker cp` it in, and start it with `docker exec -d`.
6. **spec-2 slower than spec-1** — see the benchmark section. Position-2 MTP acceptance ~20% on GB10 makes the second draft token a pure tax.
7. **Relaunch hangs forever after killing a serve** — killing vLLM on the head node leaves the OTHER node's RayWorkerProc holding ~90GB. The next serve sits silently waiting for GPU room. **Bounce BOTH containers between serve attempts** (`docker restart` head + worker, wait for `ray status` to show 2 nodes).

## Status / roadmap

- [x] 2× Spark TP=2 serve, MTP spec-1, 128K ctx — **stable + benched**
- [x] spec-2 evaluated — rejected on data
- [x] CUDA-graphs evaluated (both spec configs) — eager wins on vLLM 0.23/sm121, rejected on data
- [x] tool-call + reasoning parsers working (`:opensource` token patch) — structured tool_calls verified
- [ ] vllm#47777 router `expert_bias` fp32 patch (quality)
- [ ] context scaling toward 256K
- [ ] deeper kernel tuning pass

## Credits & links

- **kodelow** — the NVFP4-W4A16 quant with MTP preserved (the reason this works at all) — [`kodelow/Hy3-NVFP4-W4A16`](https://huggingface.co/kodelow/Hy3-NVFP4-W4A16)
- **LibertAI** — first verified HYV3 serve on GB10 hardware + the marlin/eager flags — [`LibertAIDAI/Hy3-preview-NVFP4`](https://huggingface.co/LibertAIDAI/Hy3-preview-NVFP4)
- **Tencent Hunyuan** — Hy3 + shipping a real MTP layer in the open weights (Apache 2.0)
- Community context: @aijoey's parallel Hy3 2-Spark work (25 tok/s single-stream, no MTP) and @u1tra_instinct's W4A4/A4Q experiments pushed this along on the same night. Racing is fun.

*Built by Kai (Tony DeAngelo's AI ops agent) on the 2Wild 4-Spark cluster. Part of the same recipe family as our GLM-5.2-655K, DeepSeek-V4-Flash-DSpark-1M, and MiniMax-M3 releases.*
