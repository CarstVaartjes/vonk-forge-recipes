# DeepSeek-V4-Flash on Dual DGX Spark (GB10 / SM121) — Native Blackwell FP8 Benchmark

Reproducible benchmark of **DeepSeek-V4-Flash** (official FP8) served across
**2× NVIDIA DGX Spark (GB10, SM121 Blackwell)** with tensor-parallel TP=2 over
RoCE, using the **fully native Blackwell tensor-core path** — DeepGEMM block-scaled
FP8 + MXFP4 MoE, sparse MLA, Lightning Indexer, MTP speculative decode. No Marlin,
no emulation, no CPU fallback.

## Headline Results

### Baseline (May 28, 2026 — Driver 580.142)
| Concurrency | Per-req (t/s) | Aggregate (t/s) | Reference | % of ref |
|---|---|---|---|---|
| 1 | **38.4** | 38.4 | 44 | 87% |
| 2 | 28.0 | **53.6** | ~45 | **119%** |
| 4 | 20.7 | 54.0 | — | — |
| 8 | 11.7 | 54.7 | — | — |

Throughput profile (65K context, 16 seqs):

| Concurrency | Per-req (t/s) | Aggregate (t/s) | Reference | % of ref |
|---|---|---|---|---|
| 1 | 36.1 | 36.1 | 44 | 82% |
| 2 | 28.8 | 57.0 | ~45 | 127% |
| 4 | 15.7 | 62.8 | — | — |
| 8 | 14.5 | **101.5** | ~96 | **106%** |
| 16 | 9.5 | **144.6** | — | — |

### Post MoE+IB Optimization (June 14, 2026 — Driver 580.159.03 ⚠️)

> **Note:** NVIDIA driver regression (580.142→580.159.03) causes ~3.5× throughput
> loss on GB10. The results below demonstrate the optimization improvements (2.2×
> at c=1) but absolute throughput is reduced by the environmental regression.
> Driver rollback to 580.142 restores full performance — see
> [Known Issues](#known-issue-driver-regression).

| Concurrency | Per-req (t/s) | Aggregate (t/s) | MTP Accept | SR |
|---|---|---|---|---|
| 1 | **11.32** | 11.32 | 68% | 100% |
| 2 | 8.60 | **17.20** | 60% | 100% |
| 4 | 4.87 | **19.47** | declining | 100% |
| 8 | 1.99 | 15.88 | 0% (GPU saturated) | 100% |
| 16 | 1.66 | **26.55** | 0% | 100% |

**Quality: 5/5 PASS** (math, code, factual, reasoning, instruction — via chat/completions)

## Two Optimizations Applied

### 1. MoE Padding Elimination (21× waste reduction at c=1)

vLLM's `compute_aligned_M()` in `deep_gemm_utils.py` over-allocates grouped GEMM
workspace by using worst-case `local_experts` as the M dimension. This patches it
to use `min(M*topk, local_experts)` — a tighter upper bound that is still safe.

| Concurrency | M_sum before | M_sum after | Reduction |
|---|---|---|---|
| c=1 decode | 16,384 | 768 | **21×** |
| c=8 decode | 16,384 | 6,144 | 2.7× |

**Measured improvement:** 5.0 → 11.0 t/s (2.2×) at c=1 with driver regression.
CUDA-graph safe — pure arithmetic, zero CPU synchronization.

### 2. NCCL RoCE Fix (19× faster all-reduce)

Driver 580.159.03 changed the RoCE GID table ordering, causing NCCL to fall back
from IB/RoCE to TCP socket transport. Explicit `NCCL_IB_GID_INDEX=3` restores
the correct RoCE transport.

| Transport | All-reduce latency |
|---|---|
| TCP fallback (broken) | 424 μs |
| RoCE/IB (fixed) | **22 μs** |

Also requires `--device=/dev/infiniband` in Docker container passthrough.

## Reproduce

**Pull the prebuilt image (no build needed):**
```bash
docker pull ghcr.io/r0b0tlab/vllm-dsv4-flash-gb10:cu130-sm121-arm64-dda4668b
```

**Launch with both optimizations:**
```bash
# CRITICAL: pass env vars via -e in VLLM_SPARK_EXTRA_DOCKER_ARGS.
# Bare shell exports do NOT propagate through launch-cluster.sh.
VLLM_SPARK_EXTRA_DOCKER_ARGS="\
  -e MAX_NUM_SEQS=16 -e MAX_MODEL_LEN=65536 -e MAX_BATCHED=8192 \
  -e GPU_UTIL=0.85 \
  -v /path/to/DeepGEMM-nvdev:/mnt/deep_gemm \
  -v /path/to/DeepSeek-V4-Flash:/mnt/model" \
./launch-cluster.sh \
  -n <node1_ip>,<node2_ip> \
  -t ghcr.io/r0b0tlab/vllm-dsv4-flash-gb10:cu130-sm121-arm64-dda4668b \
  --name vllm_ds4 --eth-if enp1s0f0np0 --ib-if rocep1s0f0 --no-ray \
  --apply-mod mods/deepgemm-sm121 --apply-mod mods/moe-padding-fix \
  --launch-script examples/dsv4-launch.sh -d
```

See [`docs/REPRODUCE.md`](docs/REPRODUCE.md) for full build/launch instructions.
AI agents: read [`AGENTS.md`](AGENTS.md) first.

## Known Issue: Driver Regression

NVIDIA driver **580.142 → 580.159.03** (May 2026) causes ~3.5× throughput
regression on GB10 (SM121). Raw GEMM performance is unaffected (85.9 TFLOPS),
but decode throughput drops from 38→11 t/s. Likely a kernel launch overhead or
CUDA graph capture efficiency regression.

**Fix:** Downgrade to 580.142 (available via apt snapshot PPA):
```bash
sudo apt install --allow-downgrades \
  nvidia-driver-580-open=580.142-0ubuntu0.24.04.1
sudo reboot
```

After rollback, the MoE + IB optimizations are expected to push:
- c=1: ~50+ t/s (baseline 38.4 × MoE factor)
- c=8: ~130+ t/s (baseline 101.5 × MoE factor)
- c=16: ~190+ t/s (baseline 144.6 × MoE factor)

## Reference Baseline

NVIDIA Developer Forums — "DeepSeek-V4-Flash (official FP8) running across 2x DGX
Spark — TP=2, MTP, 200K ctx": ~44 tok/s c=1, ~45 tok/s c=2 aggregate, ~96 tok/s
c=8 aggregate, on identical hardware.

---
Built and benchmarked by [@mr-r0b0t](https://x.com/mr_r0b0t) · r0b0tlab
