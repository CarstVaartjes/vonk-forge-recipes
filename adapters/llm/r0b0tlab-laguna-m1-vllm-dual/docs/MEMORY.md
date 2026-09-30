# Laguna M.1 NVFP4 — memory / topology

## Single GB10 TP=1: **does not load** (verified 2026-06-18)

During `FusedMoE` NVFP4 weight creation, vLLM allocated **~113.15 GiB** PyTorch memory on a **121.69 GiB** unified pool and failed with:

`torch.OutOfMemoryError: Tried to allocate 512.00 MiB` (742 MiB free).

Conservative settings still OOM during **weight init** (not KV):

- `gpu_memory_utilization=0.75`, `max_model_len=4096`, `kv_cache_dtype=fp8`, `--enforce-eager`

**Conclusion:** Full `poolside/Laguna-M.1-NVFP4` (~124 GiB on disk, 225B-class MoE) requires **TP=2 across two GB10** nodes with the model mounted at the same path on both hosts.

## Dual GB10 TP=2 (active path)

| Setting | First boot |
|---------|------------|
| `tensor_parallel_size` | 2 |
| `distributed_executor_backend` | **ray** (mp hit NCCL deadlock at load — see below) |
| `load_format` | **safetensors** |
| `GPU_UTIL` | 0.82 |
| `MAX_MODEL_LEN` | 4096 |
| `KV_CACHE_DTYPE` | fp8 |
| `ENFORCE_EAGER` | 1 |
| NCCL | `NCCL_IB_GID_INDEX=3`, RoCE `rocep1s0f0`, `enp1s0f0np0` |

Launch: `scripts/serve_tp2_cluster.sh` (uses `spark-vllm-docker/launch-cluster.sh` + Ray).

**Docker container name:** `laguna-m1-vllm` (override with `NAME=…`). API: `http://127.0.0.1:30100/v1`.

Nodes: **192.168.100.10** (head), **192.168.100.11** (worker). Model path on **both**:

`/home/r0b0tdgx/models/llm/nvfp4/poolside/Laguna-M.1-NVFP4` → `/mnt/model`

## 2026-06-19 — TP=2 load NCCL deadlock (mp backend)

Two runs failed at **NCCL BROADCAST SeqNum=92764** after safetensor load ~50%:

- Run 1: default **600s** timeout
- Run 2: **7200s** timeout — rank 0 still stuck ~2h (GPU busy, no progress)

**Not OOM.** Worker weights 28/28 on both nodes.

**Mitigation (next launch):** Ray cluster (remove `--no-ray`), `--distributed-executor-backend ray`, `--load-format safetensors`, `--disable-custom-all-reduce`.

**Blocker:** SSH to **192.168.100.11** intermittently fails (ping OK) — fix worker `sshd`/reboot before `launch-cluster`.

## Published benchmarks (Ray TP=2)

| Profile | KV | max_len | graphs | c8 output tok/s |
|---------|-----|---------|--------|-----------------|
| L0 | fp8 | 4096 | eager | 38.5 |
| L1 | fp8 | 8192 | PIECEWISE | **49.3** |

## Headline optimized (2026-06-20)

**`nvfp4-kv` profile** — published `120f6e2`:

| Knob | Value |
|------|--------|
| Backend | Ray TP=2 |
| Weights | NVFP4 (FLASHINFER_CUTLASS) |
| **KV** | **nvfp4** |
| max_model_len | 8192 |
| max_num_seqs | 8 |
| max_num_batched_tokens | 8192 |
| gpu_memory_utilization | 0.85 (vLLM suggests **0.852** with cudagraph profiler) |
| Graphs | PIECEWISE + compile |
| block-size | 256 |

Throughput vs L1 fp8: **equivalent** (~49 output tok/s @ c8); **more KV tokens** for batching.

Optional next: `fp8-tuned` (0.88 util, 16 seqs) A/B; `GPU_UTIL=0.8522` per vLLM hint.