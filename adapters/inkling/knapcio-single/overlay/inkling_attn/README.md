# inkling_attn: prefill attention with the relative bias computed in-kernel

Switch: `INKLING_ATTN=prefill_swa[,prefill_global]` (or `all`); unset = nothing is imported.

## What it replaces

Stock prefill, per layer visit (`Inkling2Attention._forward`, non-compact branch):

1. `RelLogitsProj`: cuBLAS einsum `thd,de->the`, writes BF16 `[t, 24, 1024]` (403 MB for an 8K chunk);
2. FA4 SM120 (`flash_attn_with_kvcache` -> `_flash_attn_varlen_func`): a shear kernel copies it into
   `[round_up(t, 64), 24, 1280]`, then the attention mainloop TMA-loads 2 bytes of bias per score.

(`inkling_prefill.bias_layout` fuses 1 and the shear copy; the mainloop still reads the sheared tensor.)

With this package, one CUDA kernel (`csrc/attn_kernel.cuh`) computes the bias inside the attention mainloop: each
warp holds its 16 query rows' `r` (`[16, 16]`, the rel-clip output) as an MMA A fragment, multiplies it by the
`proj` columns of the tile's 79 distances (10 extra m16n8k16 MMAs per 16 x 64 tile, about +8 % MMA work), stages
the BF16-rounded `[16, 80]` result in shared memory and reads it back sheared. No `[t, 24, 1024]` tensor exists.
FA2-style kernel: 4 warps x 16 rows (BM 64), 64-key tiles, Q in registers, cp.async K/V from the paged cache
(page 128), ldmatrix on XOR-swizzled rows, FP32 online softmax, BF16 P, FP32 output + LSE in FA4's layout.
232-236 registers, no spills, 43 KB smem, 2 CTAs per SM. `proj` is repacked per call by a tiny kernel so each lane loads its
B fragment with one 8-byte load.

| lever | layers | taken calls |
|---|---|---|
| `prefill_swa` | 40 sliding-window visits per token (window 1024) | prefill / extend, BF16 KV, no CP, no breakable graph |
| `prefill_global` | 8 global visits per token | same; bias work only on tiles within 1024 keys of the diagonal |

Decode, target verify and draft extend never enter a visit (stock paths untouched).

## Long prefixes: 8 warps from `INKLING_ATTN_W8_FROM` keys (default 131072)

The 4-warp kernel (BM 64, 2 CTAs / SM) keeps ~71 TF/s up to ~256-320K prefix keys in isolation (~150K in serving)
and falls to the stock rate (~47 TF/s) at 512K-1M as its K/V traffic per FLOP stops being served from L2. The 8-warp
instantiation (BM 128, half the bytes per FLOP) holds ~69 TF/s at every prefix and computes every row
bit-identically. On global layers, for the BF16 loader and the packed-prefix loader alike, the binding launches two
grids per call (`launch_global`): requests whose prefix (seqused_k - new rows) is >= `INKLING_ATTN_W8_FROM` run on
8 warps, the others on 4; each CTA reads its request's lengths on the device and returns before any shared-memory
access when the other grid owns the request (no host sync, CUDA-graph capturable). `INKLING_ATTN_W8_FROM=0`, SWA
layers and `INKLING_ATTN_WARPS=8` keep a single launch. Checks: `tests/test_w8split_cpu.py`,
`r2/tracks/longpf/tools/gpu_check_longpf_kv12s.py` (GPU, both loaders, 256K-1M, graph replay).

## Math and exactness

Same math as stock: bias = BF16(FP32 dot of r and proj) for distances 0..1023 (0 beyond on global layers),
s = qk * scale + bias in FP32, causal (+ window 1023 on SWA layers), P rounded to BF16 before PV, FP32 output,
natural-log LSE (-inf for a row with no key). Differences are FP32 accumulation order (QK, bias dot, PV and
online-softmax tile boundaries: 64 keys here, 128 in FA4). Measured against an FP32 reference (no TF32) on real
`proj` weights: see Results; the kernel's error is at or below stock FA4's at every shape.

## Patch points (patch.py)

- `Inkling2Attention._forward`: marks an eligible visit (context variable); nothing else changes.
- `RelLogitsProj.__call__`: inside the visit returns `r` itself (no einsum); SGLang then narrows it like any
  `rel_bias` (row prefix views keep the data pointer).
- `flash_attention_v4_sm120._flash_attn_varlen_func`: a `rel_bias` that is (a row prefix of) the visit's `r` runs
  `ops.prefill_attention` with the layer's `proj` and returns FA4's `(out, lse)`; a call the kernel does not cover
  falls back to stock FA4 with the einsum bias; an `r` marker outside its visit raises.

Install-time checks: `check_sources` looks for the exact code the patches rely on in the installed sglang
(`_forward`, `RelLogitsProj`, the SM120 wrapper, the backend's `rel_bias` / LSE handling, FA4's bias application
and shear pads, the SM120 module selection, CP and breakable-graph helpers); a mismatch raises (exit 78 through
sitecustomize). `RelLogitsProj.forward`'s signature is checked at patch time.

## Arming (sitecustomize.py)

Add after the `INKLING_PREFILL` block (order does not matter for correctness; see coexistence):

```python
    if _on("INKLING_ATTN"):
        import inkling_attn
        inkling_attn.install()
```

and forward `INKLING_ATTN` into the container in `scripts/launch.sh` like the other switches. The extension is
built when the model imports `inkling2_attn` (before any CUDA-graph capture), about 1 min on the first start, then
reused from `INKLING_ATTN_CACHE` (default `/root/.cache/inkling-attn`). Pre-build once with
`python3 -m inkling_attn.build`. Start-up log: `[inkling_attn pid=...] enabled prefill_swa ...` and
`extension built|reused ...`. `inkling_attn.patch.STATS` counts kernel / fallback / shadowed visits (eager mode).

## Coexistence with inkling_prefill

- `bias_layout` becomes unnecessary for the layers this package takes: in their visits `RelLogitsProj.__call__`
  returns `r` before `bias_layout`'s `forward` wrapper runs, and `bias_layout`'s varlen wrapper passes the width-16
  marker through. Layers not taken (e.g. global layers with only `prefill_swa`) still get `bias_layout`. Keeping
  `bias_layout` armed is therefore harmless and useful for `prefill_swa`-only.
- Works in either install order (dispatch is at `__call__`, one level before any `forward` wrapper).
- `qkvr_split`, `quant_once`, `partial_bf16`: independent.

## CUDA graphs

SGLang captures prefill graphs for this model (Backend.FULL, buckets up to `chunked_prefill_size` 8192, chunks
with a prefix replay too; docs/prefill/PREFILL.md 2.3). The kernel is captured inside them: static grid
`(cdiv(q rows, 64) + requests) x heads`, every request split read from `cu_seqlens_q` / `seqused_k` on the device,
no host sync, no allocation except the LSE `torch.empty` (graph pool). Rows past `cu_seqlens_q[-1]` are not
written, as FA4. Breakable prefill graphs (attention replayed outside the visit) are excluded.

## Results (Spark_06, GB10, clocks untouched, 2026-11-21)

`results/kbench/round2/attn/check_attn.py`, run5 (`runs/run5.log`, `runs/results_run5.json`). Inputs: real layer-0 / layer-5
`proj`, q, k ~ N(0, 1.5^2), v ~ N(0, 1), r ~ N(0, 2^2), shuffled pages of 128.

**Correctness**: max abs error vs an FP32 reference (no TF32, the stock BF16-rounded bias), sampled rows incl.
query-block edges, all 24 heads:

| case | kind | out err ours / stock | LSE err ours / stock |
|---|---|---|---|
| chunk 8192, prefix 0 | SWA / global | 5.149e-3 / 5.149e-3, 3.437e-3 / 3.437e-3 | 1.647e-3 / 1.646e-3, 5.7e-6 / 1.5e-5 |
| chunk 8192, prefix 8K | SWA / global | 3.678e-3 / 3.682e-3, 2.463e-3 / 2.452e-3 | 2.375e-4 / 2.384e-4, 7.06e-5 / 7.15e-5 |
| chunk 8192, prefix 32K | SWA / global | 2.609e-3 / 2.606e-3, 5.686e-3 / 5.735e-3 | 4.0e-5 / 4.0e-5, 1.698e-3 / 1.698e-3 |
| chunk 2048, prefix 0 | SWA / global | 3.633e-3 / 3.634e-3, 3.870e-3 / 3.869e-3 | 3.8e-6 / 3.8e-6, 9.3e-5 / 9.3e-5 |
| 5 requests + empty slots | SWA / global | 3.732e-3 / 3.732e-3, 3.548e-3 / 3.548e-3 | 2.77e-5 / 2.67e-5, 4.1e-5 / 4.1e-5 |
| tiny rows (1, 3, 65, 17) | SWA / global | 3.653e-3 / 3.653e-3, 3.141e-3 / 3.141e-3 | 1.9e-6 / 2.9e-6, 1.9e-6 / 2.9e-6 |

Ours and stock are equally far from the reference (both dominated by the BF16 rounding of P); the worst relative
excess is +0.4 % on out and +3.6 % on one LSE at the 1e-5 absolute level. No NaN, every covered row written.
CUDA-graph capture replayed with 4 different request splits: bit-equal to eager, rows past `cu_seqlens_q[-1]`
untouched. Patch routing: wrapper == direct call, fallback == stock bit for bit, escaped marker raises,
RelLogitsProj dispatch correct in both install orders with bias_layout. CPU emulation of the lane-level index
math (`emulate_kernel.py`) matches a dense FP64 reference.

**Speed** (cold L2: 128 MiB read before each rep; median of 20; useful FLOPs = 4 * 24 * 128 * visible keys;
stock = einsum + FA4 with shear; bias_layout = Triton sheared bias + FA4 presheared; ours = repack + kernel):

| kind | chunk | prefix | stock us (TF/s) | bias_layout us (TF/s) | ours us (TF/s) | vs stock | vs bias_layout |
|---|---|---|---|---|---|---|---|
| SWA | 2048 | 0 | 2336 (8.3) | 1561 (12.4) | 427 (45.2) | 5.47x | 3.65x |
| SWA | 2048 | 8K | 2453 (10.5) | 1680 (15.3) | 524 (49.2) | 4.68x | 3.20x |
| SWA | 2048 | 32K | 2465 (10.5) | 1672 (15.4) | 520 (49.5) | 4.74x | 3.21x |
| SWA | 8192 | 0 | 9659 (10.0) | 6245 (15.5) | 1846 (52.3) | 5.23x | 3.38x |
| SWA | 8192 | 8K | 9830 (10.5) | 6420 (16.1) | 1933 (53.3) | 5.08x | 3.32x |
| SWA | 8192 | 32K | 9816 (10.5) | 6419 (16.1) | 1966 (52.4) | 4.99x | 3.26x |
| global | 2048 | 0 | 2460 (10.5) | 1684 (15.3) | 520 (49.6) | 4.73x | 3.24x |
| global | 2048 | 8K | 6735 (34.4) | 5881 (39.4) | 3298 (70.3) | 2.04x | 1.78x |
| global | 2048 | 32K | 19753 (43.1) | 18934 (44.9) | 12063 (70.5) | 1.64x | 1.57x |
| global | 8192 | 0 | 16780 (24.6) | 13336 (30.9) | 6259 (65.9) | 2.68x | 2.13x |
| global | 8192 | 8K | 33887 (36.5) | 30264 (40.9) | 17916 (69.0) | 1.89x | 1.69x |
| global | 8192 | 32K | 86358 (43.0) | 82526 (45.0) | 53135 (69.8) | 1.63x | 1.55x |
| global | 8192 | 120K | 279140 (45.8) | 275093 (46.5) | 179440 (71.2) | 1.56x | 1.53x |

(The einsum alone is 2.0 ms per 8K chunk.) Global layers reach the bias-free FA4 rate (67-70 TF/s in
docs/prefill/PREFILL.md); SWA reaches 52-53 TF/s against FA4's bias-free 55-56.

Variants measured (run4): `INKLING_ATTN_WARPS=8` (BM 128, 1 CTA / SM) is 4-15 % slower; the unpacked proj loads
(4 x LDG.16 per fragment instead of one 8-byte load) are 10-12 % slower on SWA. Defaults: 4 warps, packed.

**End-to-end estimate** (`estimate_e2e.py`: per 8K chunk, 40 SWA + 8 global visits at the measured per-visit
times, global interpolated over the prefix):

| prompt | recipe today (all inkling_prefill levers) | + inkling_attn (SWA only) | + inkling_attn (SWA + global) |
|---|---|---|---|
| 8K | 1.93 s, 4,253 tok/s | 1.75 s | 1.69 s, ~4,840 tok/s (-12 %) |
| 32K | 8.28 s, 3,958 tok/s | 7.57 s | 7.08 s, ~4,630 tok/s (-15 %) |
| 128K | 47.7 s, 2,749 tok/s | 44.8 s | 38.3 s, ~3,420 tok/s (-20 %) |

Against the stock engine (no inkling_prefill): 2.40 -> 2.00 s, 10.1 -> 8.2 s, 54.0 -> 42.0 s. Microbench-based;
to be confirmed by the serving benchmark.

## Files

- `csrc/attn_kernel.cuh` kernel, `csrc/inkling_attn_ext.cu` binding, `build.py`, `ops.py`, `patch.py`, `__init__.py`.
- Tests: `tests/test_install.py` (CPU: env parsing, source checks). GPU tests and bench (scratch): `results/kbench/round2/attn/check_attn.py` (GPU: correctness vs FP32 reference and stock, graph
  replay, patch routing, microbench), `emulate_kernel.py` (CPU emulation of the kernel's lane-level index math),
  `estimate_e2e.py` (end-to-end estimate from the bench).
