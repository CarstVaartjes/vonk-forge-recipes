# NVFP4 KV on SM121 (Laguna M.1)

## Checkpoint default

`poolside/Laguna-M.1-NVFP4` declares `kv_cache_scheme.num_bits: 8` (FP8-calibrated KV). Serving may use **`--kv-cache-dtype nvfp4`** on SM121 when FlashInfer path accepts it (unlike Step3.7 TllmGen-only failure).

## Ladder result (2026-06-20) — **PASS**

| Item | Result |
|------|--------|
| Config | Ray TP=2, max_len **8192**, PIECEWISE graphs, util **0.85**, seqs **8** |
| `KV_CACHE_DTYPE` | **nvfp4** |
| API / smoke | OK |
| Native MoE | FLASHINFER_CUTLASS, no MARLIN headline |
| KV capacity (log) | **~508k tokens** vs ~458k fp8 L1 (more headroom) |
| vs L1 fp8 throughput | c1 **+0.4%**, c8 **−0.3%** (within noise) |

**Headline publish:** dual GB10 TP=2, container **`laguna-m1-vllm`**, **NVFP4 weights + NVFP4 KV**, fp8-equivalent throughput at c8 ~**49 tok/s**.

## If blocked on future images

Log excerpt → `evidence/kv-opt-ladder.log`. Fall back to **fp8 KV** and label HTML honestly.

## Step 3.7 precedent

Step 3.7 failed nvfp4 KV via `TllmGenFmhaRunner` on SM12.1. **Laguna M.1 + FLASHINFER attention** accepts nvfp4 KV on this stack (empirically verified).