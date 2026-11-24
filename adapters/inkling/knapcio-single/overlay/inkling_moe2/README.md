# inkling_moe2: expert grouped-GEMM tiles for DSpark verify batches (c2-c24)

Switch: `INKLING_MOE2=tiles` (unset = nothing is imported). Exact: only *which* existing FlashInfer 0.7.4 kernel
instance runs changes; every chosen configuration was measured bit-identical to the shipped choice.

| GEMM | calls (tokens = verify rows) | shipped | inkling_moe2 |
|---|---|---|---|
| NVFP4 w13 (layers 0-19, K 2304) | 9-64 tokens | moe_tiles `swap_ab=0, tile_n=64, tile_k=256` | `swap_ab=1, tile_n=32, tile_k=256` |
| NVFP4 w13 + w2 | 65-128 tokens | stock `swap_ab=1, tile_n=32, tile_k=128` | `swap_ab=1, tile_n=32, tile_k=256` |
| MXFP8 w2 (layers 20-23, K 1152) | 512-1536 routed rows (c8-c24) | FlashInfer heuristic: CuTe `<128, 8>` swap-AB | tactic `tile_m=32, tile_n=128` |
| MXFP8 w13 (K 2304) | 1024-1536 routed rows (c16-c24) | same heuristic | tactic `tile_m=32, tile_n=128` |

Everything else (c1 = 8 tokens, > 128 tokens NVFP4, other row counts MXFP8, prefill) keeps the previous choice.

## How it hooks SGLang

- NVFP4: wraps `sglang.kernels.ops.gemm.inkling2_grouped_sm120._tiles` (module global read on every
  `nvfp4_expert_matmul` call). The wrapper answers from its table for the token ranges it covers and otherwise calls
  the function it wrapped: `inkling_kernels.moe_tiles`' wrapper when `INKLING_KERNELS` has `moe_tiles`, else stock.
  It extends moe_tiles (its 9-64 rule refines moe_tiles' `<=64@2304` rule; `<=8` stays moe_tiles').
- MXFP8: wraps `inkling2_grouped_sm120.mxfp8_expert_matmul`; for table rows it calls the same FlashInfer module's
  `moe_gemm_mxfp8_nt_groupwise_tuned` (the entry FlashInfer's own autotuner launches) with the tactic, otherwise the
  stock function. The first eager call per (rows, weight shape) also runs the stock function and compares the BF16
  outputs bit for bit (one host sync during warm-up; never during graph capture); a difference ends the process
  (`SystemExit(78)`). A call that reaches graph capture for a (rows, shape) not yet checked runs the stock function,
  so a graph never records an unchecked tactic (SGLang warms every shape eagerly before capture, so in practice the
  tactic is checked and captured). Serving's MXFP8 calls ran FlashInfer's heuristic kernel (c24 profile trace:
  all 320 calls `SM120BlockScaledBuilder<128, 8, ...>`), the same baseline the bench and the self-check compare to.
- Both choices are host-side functions of static shapes, so each captured CUDA graph keeps the kernel it was
  captured with. Patch targets are checked as source strings at `install()` (no sglang/torch import); a changed
  target raises `RuntimeError` and the combined loader exits 78; a target found changed at import time raises
  `tiles.PatchError` (a `BaseException`, so SGLang's model registry cannot swallow it).

Tables can be replaced: `INKLING_MOE2_NVFP4`, `INKLING_MOE2_MXFP8` (grammar in `tiles.py`; `none` = that half off).
`INKLING_MOE2_VERBOSE=1` logs the "applied" line.

### sitecustomize snippet (after the `INKLING_KERNELS` block, so the wrapper sits outside moe_tiles')

```python
    if _on("INKLING_MOE2"):  # after inkling_kernels: per-token tile tables over moe_tiles' _tiles wrapper
        import inkling_moe2
        inkling_moe2.install()
```

and in `scripts/launch.sh`: `-e INKLING_MOE2="${MOE2:-tiles}"`.

## Evidence (GB10, Spark_06, 2026-11-21)

Bench: `results/kbench/round2/moe/bench_moe2.py` with real routing: the CPU oracle's top-8 ids (engine/ref/inkling2_ref.py,
teacher-forced on 16 SGLang greedy outputs, 420 positions, 48 visits) cut into verify steps of B requests x 8
consecutive positions; modes `full` (all rows live), `capc` (shipped verify-cap dead-row remap, requests drawn from
B/3 texts: unions 52 / 74 / 103 at c4 / c8 / c16, close to the serving steplog's 68.8 at c8), `fullc`. Real layer-0
NVFP4 and layer-20 MXFP8 weights, inputs from SGLang's own route_metadata + routed quantize, cold L2 (128 MiB read
before every call), median of 30, 6 steps per (bs, mode). Every one of the 12 NVFP4 and 6 MXFP8 configurations was
bit-identical to the shipped one on every step (`results/run1.log`).

Per verify step (44 NVFP4 + 4 MXFP8 visits, w13 + w2), shipped -> inkling_moe2, ms:

| bs (tokens) | mode | shipped | moe2 | saved |
|---|---|---|---|---|
| 4 (32) | capc / full | 55.6 / 77.3 | 54.1 / 75.7 | 1.5 / 1.6 (2.7 / 2.1 %) |
| 8 (64) | capc / full | 77.8 / 103.3 | 75.0 / 101.2 | 2.8 / 2.1 (3.6 / 2.0 %) |
| 16 (128) | capc / full | 112.9 / 125.1 | 105.8 / 117.2 | 7.1 / 7.9 (6.3 %) |
| 24 (192) | full / fullc | 125.0 / 115.4 | 123.4 / 113.6 | 1.6 / 1.8 (1.3 / 1.6 %) |

Per GEMM: NVFP4 w13 +3.4..5.6 % at 32-64 tokens, +8 % at 128; NVFP4 w2 +3 % at 128; MXFP8 w2 +2..6 % at c8,
+8..10 % at c16, +16..19 % at c24; MXFP8 w13 +1..2 % at c16, +3..4.5 % at c24.

GPU check of the installed overlay (`results/kbench/round2/moe/gpu_check_moe2.py`, with the shipped moe_tiles): hooks applied
in order (`nvfp4 over moe_tiles, mxfp8`), w13 / swiglu / w2 / MXFP8 outputs bit-identical to stock eagerly and from
a captured CUDA graph at 32/64/128/192 tokens, MXFP8 w13 and w2 included (`results/gpucheck2.log`). CPU tests:
`results/kbench/round2/moe/tests/test_moe2.py`.

Not taken (measured, no gain): `inkling_prefill`'s `quant_once` at decode sizes is 1.4-3.7x SLOWER than the stock
routed quantize (13-45 us vs 50-64 us); a multi-row swiglu-quantize kernel was bit-identical but at best 9 % faster
at c24 (32.6 -> 29.7 us/call, ~0.13 ms/step) and slower below.
