# Laguna M.1 TP=2 — end-to-end optimization ladder

## Critical: do not relaunch during load

`auto_tp2_when_node2_up.sh` must skip when **`laguna-m1-vllm`** (or legacy `laguna_tp2`) is **running**. Cron `laguna-tp2-auto-launch` paused during bring-up.

## Bring-up (once)

| Env | Value |
|-----|--------|
| `DISTRIBUTED_EXECUTOR_BACKEND` | `ray` |
| `LOAD_FORMAT` | `safetensors` (then try `fastsafetensors` on L1 if stable) |
| `ENFORCE_EAGER` | `1` |
| `KV_CACHE_DTYPE` | `fp8` |
| `MAX_MODEL_LEN` | `4096` |
| `GPU_UTIL` | `0.82` |
| `DISTRIBUTED_TIMEOUT_SECONDS` | `7200` |

## L0 → L1 (throughput, after API stable)

| Step | Change | Gate |
|------|--------|------|
| L0 | Current first boot | `verify_native_kernels` + smoke + c1–c8 |
| L1 | `ENFORCE_EAGER=0`, `CUDAGRAPH_MODE=PIECEWISE`, len **8192**, seqs **8**, util **0.85**, `block-size 256` | c1 not >10% below L0 |
| L2 | `KV_CACHE_DTYPE=nvfp4` @ 8192 | see `NVFP4_KV_SM121.md` |
| L3 | `MAX_MODEL_LEN=16384` if mem allows | smoke OK |

Run: `OPT_RELAUNCH=1 bash scripts/run_tp2_e2e_pipeline.sh` after first API ready.

## NCCL / Spark

- `NCCL_IB_GID_INDEX=3`, `enp1s0f0np0`, `rocep1s0f0`
- **Never** use `mp` no-ray for this model (SeqNum 92764 deadlock)
- Reboot worker if SSH banner hangs after failed load

## Publish gate

Native NVFP4 weights + honest KV label + concurrency benchmarks + HTML visual QA.