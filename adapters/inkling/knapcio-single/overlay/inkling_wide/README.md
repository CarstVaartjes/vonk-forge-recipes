# inkling_wide: weight-streaming BF16 GEMM for 17..256 rows (DSpark verify batches)

Reads pico's packed 12-bit BF16 format (`bf16x`, pico by Andy Shu, MIT) as coded by `inkling_pico12`.

At c3..c24 the target's verify step sends 24..192 rows (batch x 8 draft rows) through the dense projections; the
looped layers' stacked q/k/v call sees three sources. Stock SGLang sends every call above 16 rows to cuBLAS, which
streams the weights at 151..206 GB/s on GB10 (wo_ud at M=64: `cutlass_80_wmma`, stacked qkvr at M=192 and the LM
head: `nvjet`). `inkling_pico12` covers 1..16 rows; its multi-row use re-reads the weight per 16 rows.

This package adds one kernel for 17..256 rows (`csrc/wide_ext.cu`) that reads every weight byte from DRAM once:

- a CTA owns 128 weight rows (16 tiles of 8 rows) and a K slice; its warps stream their tiles straight into
  registers (16 bytes per lane and tile step, one 64-value batch ahead), decode `bf16x` tiles in registers (pico's
  reader, `csrc/bf16x_format.cuh`), and multiply them with ALL M rows (`mma.sync` m16n8k16, BF16 in, FP32
  accumulate): the weight tile is the MMA's B operand in pico's lane order, the activations are the A operand,
  staged per 64-value chunk in shared memory by `cp.async` (3 stages, swizzled, conflict-free);
- 2 tiles per warp (8 warps) up to 64 rows (2 CTAs per SM), 2 tiles and two warp groups along M (16 warps) at
  65..128, 1 tile per warp (16 warps) above; row buckets of 16..256; bf16x tile headers held in registers
  (shuffled), weights prefetched into L2 one batch ahead of the register loads;
- split-K over CTAs when the column blocks alone do not fill the GPU (wo_ud), with a deterministic fixed-order
  reduction by the last CTA of a column block (counter per block, reset by that CTA);
- stacked q/k/v: the CTA's column block picks its source (k columns -> k rows, v columns -> v rows, else q rows)
  and its output row offset, so the three sources share ONE read of the weight (2/3 fewer MMAs than stock);
- the same kernel on the raw BF16 weight (`fmt=raw`): the layer's own storage, no copy; same output bits.

Status (2026-11-21): CPU-tested, reviewed, GPU-checked on GB10: all gates PASS on real
layer-0 / unembed weights (decode == weight, packed == raw, PDL on == off, graph == eager, variants and prefetch
bit-identical, FP64 bound, error <= cuBLAS's). Microbench below (L2-cold, medians, real weights, us):

| shape | M | cuBLAS | pico (16-row calls) | wide packed | wide raw | packed GB/s | cuBLAS / wide packed |
|---|---:|---:|---:|---:|---:|---:|---:|
| qkvr stacked (3 sources) | 17x3 | 186.4 | - | 143.4 | 185.8 | 213 | 1.30 |
| qkvr stacked | 32x3 | 190.5 | - | 143.5 | 187.4 | 213 | 1.33 |
| qkvr stacked | 64x3 | 200.3 | - | 149.6 | 191.6 | 205 | 1.34 |
| qkvr stacked | 85x3 | 207.2 | - | 181.2 | 198.7 | 169 | 1.14 |
| qkvr 8600x2304 | 9 | 179.6 | 144.4 | 139.3 | 183.3 | 220 | 1.29 |
| qkvr | 17 | 185.3 | 276.0 | 145.0 | 185.3 | 211 | 1.28 |
| qkvr | 32 | 186.2 | 275.0 | 141.4 | 184.8 | 217 | 1.32 |
| qkvr | 64 | 182.3 | 533.5 | 147.5 | 187.4 | 208 | 1.24 |
| qkvr | 96 | 186.5 | 794.3 | 189.4 | 197.5 | 162 | 0.98 |
| qkvr | 128 | 192.5 | 1053.7 | 204.8 | 205.4 | 149 | 0.94 |
| qkvr | 256 | 207.0 | 2094.0 | 268.4 | 270.8 | 114 | 0.77 |
| wo_ud 2304x3072 | 9 | 65.6 | 60.5 | 57.4 | 71.2 | 190 | 1.14 |
| wo_ud | 17 | 67.7 | 112.1 | 59.4 | 73.7 | 184 | 1.14 |
| wo_ud | 32 | 68.6 | 116.8 | 61.5 | 75.9 | 178 | 1.12 |
| wo_ud | 48 | 69.0 | 170.0 | 65.2 | 82.0 | 168 | 1.06 |
| wo_ud | 64 | 69.7 | 223.9 | 71.7 | 88.1 | 152 | 0.97 |
| wo_ud | 128 | 73.7 | 443.7 | 98.3 | 108.6 | 111 | 0.75 |
| head 201024x2304 (FP32) | 9 | 4127.6 | 3058.6 | 3103.6 | 4189.2 | 230 | 1.33 |
| head | 17 | 5245.3 | 5894.1 | 3140.4 | 4230.1 | 228 | 1.67 |
| head | 64 | 4577.3 | 11958.0 | 3415.1 | 4463.6 | 209 | 1.34 |
| head | 128 | 4789.3 | 23936.1 | 3608.7 | 4820.0 | 198 | 1.33 |
| head | 192 | 5279.7 | 35895.4 | 4051.7 | 4996.9 | 176 | 1.30 |
| head | 256 | 5325.9 | 47964.2 | 4662.5 | 5376.8 | 153 | 1.14 |

(Final run with the shipped defaults: prefetch distance 1, two warp groups at 65..128 rows, split-K 3 for wo_ud up
to 64 rows. Logs: `results/kbench/round2/wide/run4-final.log`, sweeps in `run2-full.log` / `run3-full.log`.) At 9..16 rows
wide packed matches or beats pico's GEMV (qkvr 139 vs 144 us, wo 57-58 vs 60-61 us, head 3.10-3.13 vs 3.01-3.06
ms). Up to ~64 rows the kernel streams the coded weight at 205..230 GB/s; above, it is issue-bound (packed and raw
take the same time; qkvr's 68 column blocks give 1.4 waves at 1 CTA per SM, and split-K measured slower), so the
default per-kind caps route each kind to wide only where it wins: qkvr (plain, and stacked per source) <= 96
rows, wo_ud <= 32, head <= 256. Summed dense time at M = 64 (plain qkvr + wo): cuBLAS 252.0 us vs wide 219.2 us
= 1.15x (at the kill line); with wo_ud routed to cuBLAS at 64 rows (the default cap) 217.2 us = 1.16x. Accuracy:
wide's max error vs FP64 equals cuBLAS's at most M; at 40..64 rows cuBLAS's wo errs 2.2x more (3.4e-2 vs 1.5e-2),
apparently from its split-K reduction.

Estimated effect at c8 (verify 64 rows a source; microbench deltas): 28 stacked qkvr calls x 51 us + 20 plain qkvr
x 35 us + head 1.16 ms ~= 3.3 ms per step (in the c8 profile cuBLAS ran slower in-graph, 231 / 200 / 4486 us, so the
in-serving gain may be up to ~4 ms). wo_ud at 64 rows stays on cuBLAS.

## Switches

| env | default | meaning |
|---|---|---|
| `INKLING_WIDE` | empty (off) | `dense` (qkvr plain + stacked, wo_ud), `lmhead` (target FP32 head), `all` |
| `INKLING_WIDE_MIN_M` | 17 | rows of a call taken from (per source for stacked calls) |
| `INKLING_WIDE_MAX_M` | 256 | rows of a call taken up to (<= 256) |
| `INKLING_WIDE_FORMAT` | `auto` | `auto`: pico's coded copy when `inkling_pico12` made one, else the raw weight; `raw`: always raw |
| `INKLING_WIDE_SCOPE` | `verify` | `verify`: only inside target decode / target-verify forwards (prefill, extend and anything outside the target forward keep the stock path); `all`: every eligible call |
| `INKLING_WIDE_MAX_M_KIND` | `qkvr=96,wo=32,head=256` | per-kind row caps (where wide beats cuBLAS on GB10) |
| `INKLING_WIDE_SPLITK` | by shape | force split-K slices per kind, e.g. `wo=4,qkvr=1,head=1` |
| `INKLING_WIDE_MG` | 2 | warp groups along M at 65..128 rows (2: 16 warps, half the rows each; 1: 8 warps) |
| `INKLING_WIDE_MT` | 0 | tiles per warp above 128 rows (0/1: 16 warps; 2: 8 warps, measured slower) |
| `INKLING_WIDE_PFD` | 1 | L2 prefetch distance in 64-value batches (0 = off; 2..16 measured flat or slower) |
| `INKLING_WIDE_PDL` | 1 | programmatic dependent launch (weights of the first batch load before `griddepcontrol.wait`) |
| `INKLING_WIDE_CACHE` | `~/.cache/inkling-wide` | extension build cache (`/root/.cache/inkling-wide` in the container) |
| `INKLING_WIDE_VERBOSE` | off | nvcc output (`-Xptxas -v`) and the armed line |

Suggested: `INKLING_PICO12=dense,lmhead INKLING_WIDE=dense,lmhead` (pico <= 16 rows, wide 17..256, both on the
same coded buffers: no extra memory beyond pico's and a split-K workspace of ~12 MB).

## Arming snippet for `sitecustomize.py`

Install AFTER `inkling_pico12` (wide's `post_load_weights` wrapper must run after pico coded the weights; a wrong
order exits 78 with a clear message) and preferably after `inkling_prefill` (so that stacked calls of 128..256
rows per source reach wide before prefill's cuBLAS split; prefill keeps every chunk above `INKLING_WIDE_MAX_M`):

```python
    if _on("INKLING_PREFILL"):  # (existing)
        import inkling_prefill
        inkling_prefill.install()
    if _on("INKLING_WIDE"):  # after pico12 (reads its packs) and prefill: outermost _project_sources for 17..256 rows
        import inkling_wide
        inkling_wide.install()
```

and in `scripts/launch.sh` pass `WIDE` through as `INKLING_WIDE` like `PICO12`.

## Patch points

| target (sglang) | what wide takes | everything else |
|---|---|---|
| `UnquantizedLinearMethod.apply` / `apply_into` of the registered layers only (each gets a copy of its method object with a subclass `Wide<its class>`, e.g. `WidePico12UnquantizedLinearMethod`) | BF16 x with MIN_M..MAX_M rows, no bias, weight storage unchanged since registration, not compiling, not batch-invariant; `apply_into` also needs an `[M, N]` output with contiguous rows | the class it subclassed (pico for <= 16 rows, then stock) |
| `Inkling2Attention._project_sources` | three distinct [M, 2304] sources with MIN_M <= M <= MAX_M, k/v column bounds multiples of 128 (3072 / 4096 / 5120 at TP1) | the method it wrapped (prefill / pico / qkvr_gemv / stock); markers `inkling_cfg`, `inkling_pico12`, `inkling_prefill_qkvr_split` are forwarded |
| `unquant.sm12x_fp32_lm_head` / `sm12x_fp32_lm_head_applies` (module attributes read at call time by `logits_processor._compute_lm_head`) | the target head, MIN_M..MAX_M rows, `out` absent or `[M, N]` FP32 with contiguous rows | the wrapped functions (pico <= 16 rows, stock); above 16 rows stock `applies` is False and SGLang would run `torch.mm(..., out_dtype=float32)` |
| `Inkling2ForConditionalGeneration.post_load_weights` | registers qkvr, wo_ud, the head after every (re)load; allocates the split-K workspace | - |
| `Inkling2ForConditionalGeneration.forward` | marks the forward's scope from `forward_batch.forward_mode` (decode / target verify) for `INKLING_WIDE_SCOPE=verify`, restored on return; nothing else | - |

Patch targets are checked as source strings in `install()` (RuntimeError -> sitecustomize exits 78) and as
attributes when each module is imported (exit 78 with a `FATAL` line). Draft layers are not covered (the default
`DRAFT_BYTES` already reads 8-bit copies there).

## Exactness

- **Same weights:** `bf16x` decodes to the BF16 weight bit for bit (gated by `tools/gpu_check_wide.py` through this
  package's own reader, `ext.decode`), and packed and raw runs of the kernel give identical bits.
- **Same math:** BF16 x BF16 products with FP32 accumulation and one rounding (BF16 for qkvr / wo, none for the FP32
  head). Against cuBLAS only the FP32 summation order differs: per 32-value step two MMAs, steps in K order, split-K
  slices added in slice order. The GPU check gates the FP64-reference bound and reports max abs / rel errors next
  to cuBLAS's.
- **Row independence:** an output element does not depend on M, the bucket or the other rows (stacked == per-source
  plain call bit for bit, gated). It does depend on the split-K count, which is a function of (N, K, M) only.
- Not bit-identical to stock serving (cuBLAS) or to pico (<= 16 rows); greedy text identity is not a valid gate
  for summation-order changes on this model (see `inkling_pico12/README.md`).

## Scope

A first serving screen (GB10, 2026-11-21, `scope` not yet present) gained +8 % on thinking c8 and +3..5 % on
greedy c4..c16, but changed the c1 text: with 8 verify rows wide never engages at c1, so prefill / extend chunks of
17..256 tokens (short prompts) were going through wide instead of cuBLAS / inkling_prefill. `INKLING_WIDE_SCOPE=verify`
(the default) limits wide to decode and target-verify forwards. The marker is set in Python before the model's layers
run, from the batch's forward mode, which is fixed when SGLang captures a graph, so captured graphs hold the
choice made for their mode.

## CUDA graphs

Per call: shape checks in Python, `torch.empty` output (graph pool under capture), at most a `.contiguous()` for
strided sources, one launch on the current stream. No host sync, no host-device copy, no data-dependent shapes.
The split-K workspace and counters are allocated in `post_load_weights` (before warm-up and capture); the reducing
CTA resets its counter, so every replay starts from zero (the GPU check replays a captured split-K call twice).
`cudaFuncSetAttribute` for every kernel runs at load (`ext.prepare()`).

PDL: the kernel loads its first weight batch, triggers `griddepcontrol.launch_dependents`, then
`griddepcontrol.wait` before reading activations or touching outputs, the workspace or counters.

## Memory

None beyond pico12's packs (shared) and the split-K workspace (max over registered weights of slices x MAX_M x N
FP32; ~12 MB at the defaults). With `INKLING_WIDE_FORMAT=raw` or without pico12 it reads the BF16 originals.

## Build

`build.py` compiles `csrc/wide_ext.cu` with `-gencode=arch=compute_121a,code=sm_121a -O3 -std=c++20
--expt-relaxed-constexpr -lineinfo` into `$INKLING_WIDE_CACHE/<tag>` (hash of sources, flags, torch and CUDA).
24 kernels (2 formats x 12 bucket variants); ~45 s first build. One tolerated 16-byte header spill in the bf16x
48..64-row kernel (launch bounds keep 2 CTAs per SM). Prebuild: `python3 -m inkling_wide.build`.

## Checks

- CPU: `python3 -m unittest discover -s inkling_wide/tests -v` from `overlay/` (needs torch; uses the stand-in
  sglang and pico extension of `tests/pico12`).
- GPU (server stopped): `PYTHONPATH=<overlay> python3 inkling_wide/tools/gpu_check_wide.py [--sweep]` inside the
  image with `/model` mounted. Gated: decode == weight, packed == raw, PDL on == off, graph == eager (split-K
  replayed twice), FP64 bound for split 1 and the default split, error no worse than cuBLAS (x1.25), stacked kept
  elements == per-source calls with the stock view layout. Reported: L2-cold times of cuBLAS, pico (16-row calls),
  wide packed / raw, the stacked call, and the summed dense time at M = 64 (kill line 1.15x).

## Files and licences

| path | origin | licence |
|---|---|---|
| `csrc/bf16x_format.cuh` | layout description, `bf16x_raw_off`, `mad_imm`, `bf16x_dec`, `ldg_bf16x` copied from pico (dsgx10 `ds41f-4xgx10-switchless/engine/kernels/gemv16.cu`, commit `f2117916`, via `inkling_pico12/third_party`) | MIT, Copyright (c) 2026 Andy Shu, `third_party/pico/LICENSE` |
| `csrc/wide_ext.cu`, `*.py`, `tests/`, `tools/` | ours | the repository's licence |

`_hook.py` is a copy of `inkling_pico12/_hook.py`.
