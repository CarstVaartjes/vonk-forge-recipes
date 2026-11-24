# inkling_pico12: small-M BF16 GEMVs on exactly coded 12-bit weights

Packed 12-bit BF16 GEMV: pico by Andy Shu (MIT).

At c1 decode the target reads its BF16 dense attention projections (stacked `qkvr` 8600 x 2304 and `wo_ud`
2304 x 3072, 24 layers, 48 layer visits per forward with the loop) and the LM head (201024 x 2304, FP32 logits)
every step, about 3.7 GB. pico's `bf16x` format stores the same BF16 values exactly in ~0.76 of the bytes: each
tile of 8 rows x 32 values keeps every sign and mantissa byte and stores the exponents as 4-bit offsets from the
tile's least (a tile whose exponents span more than 16 values stays raw). pico's GEMV decodes a tile in registers
and runs the same tensor-core MMAs as its raw BF16 GEMV, so its outputs are the raw kernel's bit for bit.

This package builds pico's kernel as a PyTorch extension at start-up, codes the eligible weights after they
load, and sends eligible calls of 1-16 rows (decode, verify, the draft's block) to it. The BF16 originals stay:
prefill, more than 16 rows and every other reader use them through the stock path.

Status (2026-11-21): written and CPU-tested (35 tests, stand-in extension); built and checked on a GB10.
`tools/gpu_check_pico12.py` checks it on a Spark (2026-11-21: PASS, packed == raw bit for bit on real weights).

## Switches

| env | default | meaning |
|---|---|---|
| `INKLING_PICO12` | empty (off) | comma list: `dense` (qkvr + wo_ud), `lmhead` (target FP32 head), `drafthead` (the draft's BF16 projection through the target head), `draft` (the draft's own BF16 linears), or `all`; plus `stack1` (with `dense`, not part of `all`): stacked q/k/v calls read the coded qkvr once (below) |
| `INKLING_PICO12_STACK1_KV_MT` | 1 | `stack1`: n-tiles per warp of the k/v range call (1 or 2; same bits, same time on GB10) |
| `INKLING_PICO12_STACK1_FUSE` | 0 | `stack1`: 1 = the q and k/v parts in one launch at <= 8 rows a source (same bits; within +-2 us of two launches on GB10) |
| `INKLING_PICO12_MAX_M` | 16 | rows of a plain call taken by pico (1-16) |
| `INKLING_PICO12_STACKED_MAX_M` | 16 | rows per source of a stacked q/k/v call taken by pico (0-16); 0 leaves stacked calls to the stock / `qkvr_gemv` path; <= 5 needs no kv pack (see below) |
| `INKLING_PICO12_WK` | 16 | warps along K per CTA (pico's `GemvCfg.wk`; the probe's setting) |
| `INKLING_PICO12_PDL` | 1 | launch with programmatic dependent launch, as the probe did; 0 = plain launches (same bits) |
| `INKLING_PICO12_CACHE` | `~/.cache/inkling-pico12` | extension build cache (`/root/.cache/inkling-pico12` in the container) |
| `INKLING_PICO12_STD` | `c++20` | nvcc `-std` (pico's own); `c++17` is the fallback if the torch headers object |
| `INKLING_PICO12_VERBOSE` | off | print the armed line in every process and the nvcc build output |

Suggested start: `INKLING_PICO12=dense,lmhead`. With the default `DRAFT_BYTES` (`gemv,fold8,head8,draftmx8`) the
draft options do nothing: `head8` and `draftmx8` read 8-bit copies of the same weights, which is fewer bytes, and
they keep precedence (below).

## Patch points

| target (sglang) | what pico takes | everything else |
|---|---|---|
| `UnquantizedLinearMethod.apply` / `apply_into` of the packed layers only (each such layer gets a copy of its method object with a subclass `Pico12<base>`; other layers and the shared class are untouched) | BF16 x with 1 to `MAX_M` rows, no bias, the layer's weight storage unchanged since coding, not under `torch.compile`, not in batch-invariant mode; `apply_into` also needs the caller's output to be `[M, N]` with contiguous rows | the base class's `apply` / `apply_into` |
| `Inkling2Attention._project_sources` (`srt/models/inkling_common/inkling2_attn.py`) | three distinct [M, 2304] sources with M <= `STACKED_MAX_M` (global layers, and the looped core layers in passes 2-3) | the method it wrapped (stock or `inkling_kernels.qkvr_gemv`); shared sources go to `_project_qkvr` -> `qkvr` -> the `apply` above |
| `unquant.sm12x_fp32_lm_head` and `sm12x_fp32_lm_head_applies` (module attributes; `logits_processor._compute_lm_head` imports both at call time) | the coded target head, 1 to `MAX_M` rows, `out` absent or `[M, N]` FP32 with contiguous rows | the stock functions (`applies` is stock's answer OR pico's) |
| `Inkling2ForConditionalGeneration.post_load_weights` (`srt/models/inkling2.py`) | codes qkvr (+ its kv rows), wo_ud of every `Inkling2Attention`, and the head, after every (re)load | - |
| `inkling2_mtp.project_through_lm_head` (`drafthead`) | the draft's base logits through the coded target head, BF16 hidden, 1 to `MAX_M` rows | the wrapped function when `--speculative-dspark-lm-head-dtype` is set or `inkling_draftbytes` `head8` packed the head (`lm_head._idb_head`) |
| `Inkling2ForConditionalGenerationMTP.load_weights` (`draft`) | codes every BF16 unquantized linear under the draft's `.layers` (no bias, N % 8 == 0, K % 32 == 0) | - |

Patch targets are checked twice: the source strings before anything is imported (`install()`, RuntimeError ->
`sitecustomize` exits 78), and the attributes when each module is imported (exit 78 with a `FATAL` line). A
failed build, a weight that does not code exactly, or any other error while coding also ends the process with 78:
a half-armed overlay never serves.

### The stacked qkvr call and `inkling_kernels.qkvr_gemv`

With three distinct sources the stock code stacks them into [3M, 2304], projects all 3M rows through the whole
8600-row weight (cuBLAS once 3M > 16) and keeps, per source, only its columns (q source: q, r, gate, sinks; k
source: k; v source: v). The pico path:

- 3M <= 16 (M <= 5): one call over all 3M rows and all columns: the same elements as the stock projection.
- 3M > 16: the q rows over all 8600 columns (writing rows [0, M) of the [3M, N] output directly), then the k and
  v rows (2M, in calls of at most 16 rows) over a second coded copy of qkvr's adjacent k and v rows
  [3072, 5120), copied into place with one `copy_`. Elements outside the kept columns of the k and v rows are
  left unwritten; `_project_sources` never returns them.
- The returned six views are made from the [3M, N] buffer with the stock `chunk(3)` / `split(split_sizes)`, so
  shape, stride and storage offset are the stock ones (checked in the CPU tests and the GPU check).

Bytes per layer at M = 6-8: 8600 + 2048 coded rows (~38 MB) against `qkvr_gemv`'s 8600 BF16 rows (39.6 MB) or
stock cuBLAS's 8600 BF16 rows at a lower rate; at M = 9-16 the kv pack is read twice (~45 MB).
`STACKED_MAX_M=8` keeps 9-16 on `qkvr_gemv` if the GPU check shows it faster there.

#### `stack1`: each coded byte read once

The path above reads 2048 rows twice per stacked visit at M = 6-8: the q rows' call goes over all 8600 rows of the
coded qkvr, k and v rows included (their outputs are discarded), and the k/v rows' call reads the kv pack, a second
coded copy of the same rows. At c1 (8 rows a source, 28 stacked visits a step) that is 28 x 7.3 MB = 204 MB per
step, plus a `copy_` each. With `stack1` the stacked call reads the one coded copy of the whole matrix once:

- the q rows over the columns outside the k/v rows [3072, 5120), the k and v rows (calls of at most 16 rows) over
  those only, both written in place into the [3M, N] output: no kv pack (~175 MB less memory), no copy;
- the kernels (`csrc/pico12_range.cu`, a second small extension built only when `stack1` is on) are pico's own
  kernel body (`gemv16_body`, reached by including pico's `gemv16.cu` verbatim) with the template arguments,
  launch bounds, block, shared memory and PDL launch of `gemv_bf16x`'s kernels, plus one test in front: a CTA
  whose output rows lie outside the asked range returns before reading or writing anything. So every element it
  writes has `gemv_bf16x`'s bits (gated: kept elements == the kv-pack path == the per-source calls, bit for bit,
  layers 0 and 6, M = 6, 8, 12, 16, both mt, fused and not; nothing outside the kept columns written; graph replay);
- M <= 5 (one call) is unchanged; the views returned are the stock ones as before.

GB10, real layer weights, L2-cold (`results/kbench/round2/c1leak/bench2.py`, 2026-11-22), us per stacked call:

| M per source | layer 0: today -> stack1 | layer 6: today -> stack1 | one full 8600-row call (shared visits) |
|---|---|---|---|
| 6 | 178.2 -> 143.4 | 157.7 -> 122.0 | 136.4 / 114.6 |
| 8 | 178.7 -> 143.9 | 158.2 -> 122.9 | 135.2 / 114.7 |
| 12 | 221.2 -> 182.8 | 201.7 -> 159.9 | 143.4 / 123.9 |
| 16 | 223.2 -> 187.5 | 203.8 -> 162.9 | 144.0 / 124.9 |

At c1 (M = 8, 28 stacked visits per step): ~35 us a visit, ~1.0 ms per step (~2.5 % of the 39.9 ms step).

Both packages patch `_project_sources`. `sitecustomize` installs `inkling_kernels` first, so its post-import hook
runs first and pico wraps it. If the hooks ever run the other way round, the pico hook applies `qkvr_gemv`'s patch
itself before wrapping, and forwards `qkvr_gemv`'s `inkling_cfg` marker so that `qkvr_gemv`'s own hook then finds
the method already patched. Either way the order is: pico for stacked calls it takes, then `qkvr_gemv` for the
rest of its range (M <= `INKLING_QKVR_MAX_M`), then stock. The start-up line says
`Inkling2Attention._project_sources (outside inkling_kernels.qkvr_gemv)` when both are on.

### With `inkling_draftbytes`

`draftmx8` swaps the draft linears' method class for a subclass at draft attach, after `draft` coded them at
draft load: its `apply` runs first and falls through to pico's for rows beyond its own bound. `head8` marks the
head (`_idb_head`) and `drafthead` then stands down. `gemv` (draft logits through the Triton GEMV) is inside the
function `drafthead` wraps, so with `drafthead` on, pico takes those calls first.

## Exactness

- **Coding is lossless and checked.** After `bf16x_encode`, `bf16x_check` decodes every 16-byte piece and
  compares it with the BF16 weight; any difference stops the process. The GPU check repeats it.
- **Packed == raw.** pico's coded GEMV gives its raw BF16 GEMV's bits (pico's claim; an earlier standalone probe found it on the
  real layer-0 qkvr / wo and the head at 1-16 rows; the GPU check gates it again, also with PDL on vs off and
  CUDA-graph replay vs eager).
- **Row independence.** A (row, column) output has the same bits whatever M (1-16), the CTA tiling (mt) or the
  other rows: the stacked path relies on this and the GPU check gates it (kept elements == the per-source call).
- **Against SGLang's stock path** (Triton `sm12x_bf16_gemv`, b12x for a one-row head, cuBLAS) only the FP32
  summation order differs: both compute BF16 x BF16 products with FP32 accumulation and one rounding (BF16 for
  qkvr / wo, none for the FP32 head). The outputs are therefore not bit-identical to stock serving; the GPU check
  reports the max difference and gates pico against an FP64 reference (`|err| <= 2^-23 sum|terms| (K/16 + 32)`
  plus half a BF16 ulp for BF16 outputs). As measured on this model, greedy-text identity is not a valid gate for a change
  of summation order on this model (rounding chaos through the W4A4 quantisers); use per-visit tensor
  comparisons and statistical text gates.

## CUDA graphs

Per call: shape checks in Python, `torch.empty` outputs (the graph pool under capture), a `.contiguous()` /
`.clone()` only for strided or misaligned rows, pico's launch on PyTorch's current stream, and for stacked
calls one `copy_`. No host sync, no host-device copy, no data-dependent shapes: the path chosen depends only on
shapes, so capture and replay take the same one. Load-time work (build, `tile`, `encode`, `check`, which
allocate and synchronise) happens in `post_load_weights` / the draft's `load_weights`, before SGLang's warm-up
and capture.

pico's kernels launch with programmatic stream serialization (`INKLING_PICO12_PDL=1`), load their first weight
tiles, call `griddepcontrol.launch_dependents`, then `griddepcontrol.wait` before reading the activations or
writing anything. A following kernel that was itself launched with PDL may therefore start early; every such
kernel must call `griddepcontrol.wait` before reading its inputs, which is the PDL contract SGLang's own PDL
kernels already follow (they also trigger early).

A weight reload re-codes into the existing buffers, so captured graphs keep pointing at valid, refreshed data;
if a reloaded weight needs more raw tiles than its buffer holds, the process exits (78) rather than serve stale
weights. A weight whose storage was replaced without `post_load_weights` falls back to stock (one warning).

## Memory

Coded copies are extra; the BF16 originals stay. From an earlier standalone probe's real-weight sizes (TP1):

| per | coded MB | count | MB |
|---|---|---|---|
| qkvr 8600 x 2304 | 30.6 | 24 | 735 |
| qkvr kv rows 2048 x 2304 (only when `STACKED_MAX_M` >= 6) | ~7.3 | 24 | ~175 |
| wo_ud 2304 x 3072 | 10.9 | 24 | 263 |
| head 201024 x 2304 (`lmhead` or `drafthead`) | 715.1 | 1 | 715 |
| total `dense,lmhead` | | | ~1.89 GB |

`draft` adds ~0.76 x the draft's BF16 linears (3 layers, ~0.4 GB). The memory comes out of the KV-cache budget at
the same `--mem-fraction-static`. The start-up line prints the exact figure; with the probe's sizes it would read

```
[inkling_pico12 pid=123] active (target): dense: qkvr 24 (+kv 24) wo_ud 24, head [201024x2304] for lmhead;
packed 1887.5 MB extra beside 2217.3 MB of BF16 originals (kept); patched unquant.sm12x_fp32_lm_head,
Inkling2Attention._project_sources (outside inkling_kernels.qkvr_gemv),
Inkling2ForConditionalGeneration.post_load_weights; ext reused in 0.4 s (/root/.cache/inkling-pico12/<tag>);
encode 1.9 s; max_m=16 stacked_max_m=16 wk=16 pdl=1
```

## Build

`build.py` calls `torch.utils.cpp_extension.load` on `csrc/pico12_ext.cu` (ours: bindings, a tiling kernel into
pico's documented layout, argument checks) and `third_party/pico/kernels/gemv16.cu`, with pico's flags
`-gencode=arch=compute_121a,code=sm_121a -O3 -std=c++20 --expt-relaxed-constexpr -lineinfo` plus
`-U__CUDA_NO_{HALF,BFLOAT16}_*` (torch defines them; pico was built without them). The build directory is
`$INKLING_PICO12_CACHE/<tag>`, `<tag>` a hash of the sources, flags, torch and CUDA versions: a later start
with the same inputs imports the built `.so` (well under a second), a change rebuilds in a new directory. It
needs nvcc and ninja (both in the image), honours `MAX_JOBS`, and does not need a GPU. The build runs in the
process that loads the weights; a first build should take on the order of a minute (not measured yet).

Prebuild once, e.g. right after pulling the overlay:

```sh
docker run --rm -v $PWD/overlay:/opt/inkling-overlay:ro -v $HOME/.cache/inkling-2-mini-sglang:/root/.cache \
  -e PYTHONPATH=/opt/inkling-overlay --entrypoint python3 lmsysorg/sglang:inkling-2-mini -m inkling_pico12.build
```

## Docker additions

`scripts/launch.sh` already mounts the overlay read-only, sets `PYTHONPATH`, and mounts `$CACHE_DIR` on
`/root/.cache`, so the build cache persists across restarts. It passes `PICO12` (default empty = off) as
`INKLING_PICO12`:

```sh
PICO12=dense,lmhead scripts/launch.sh up
```

Other `INKLING_PICO12_*` settings go through `SGLANG_ENV`, e.g. `SGLANG_ENV="-e INKLING_PICO12_STACKED_MAX_M=8"`.

## Checks

- GPU (server stopped; builds the extension, ~3.5 GB peak, a few minutes):

  ```sh
  docker run --rm --gpus all --ipc=host -v $PWD/overlay:/opt/inkling-overlay:ro \
    -v $HOME/models/Inkling-2-Mini:/model:ro -v $HOME/.cache/inkling-2-mini-sglang:/root/.cache \
    -e PYTHONPATH=/opt/inkling-overlay --entrypoint python3 lmsysorg/sglang:inkling-2-mini \
    /opt/inkling-overlay/tools/gpu_check_pico12.py 2>&1 | tee pico12-check-$(date +%Y%m%d-%H%M).log
  ```

  Gated: coding exact; packed == raw, PDL on == off, graph == eager, within the FP64 bound at M = 1, 8, 16 for
  qkvr, wo and the head (random and real layer-0 / unembed weights); the stacked 3x4, 3x8 and 3x16 cases equal
  the per-source calls, the patched `_project_sources` returns the stock view layout, and their graph replays
  are exact. Reported: max difference against stock and against `qkvr_gemv`, and L2-cold times (packed, pico
  raw, stock, cuBLAS; stacked: pico vs `qkvr_gemv` vs stock). Last line `pico12 gpu_check result PASS|FAIL`.
- CPU: `python3 -m unittest discover -s tests/pico12 -v` from `overlay/` (needs torch; no GPU, nvcc or sglang).

Expected effect (kernel times at M = 8 from an earlier standalone probe, not yet a serving measurement): qkvr 112.4 us vs ~171 us
raw and ~240 us for SGLang's Triton GEMV, wo 49 us vs 117.5 us, head 2.94 ms vs ~3.8 ms; that probe estimated
about -2.5 ms per c1 step beyond the Triton pack12 estimate.

## Files and licences

| path | origin | licence |
|---|---|---|
| `third_party/pico/kernels/gemv16.cu`, `common.cuh`, `dsk.h`, `third_party/pico/common/check.h` | pico (dsgx10 `ds41f-4xgx10-switchless/engine/`, commit `f2117916`), copied from an earlier standalone probe's vendored copies | MIT, Copyright (c) 2026 Andy Shu, `third_party/pico/LICENSE`. Each file carries a 2-line header naming its origin; the rest is unmodified (sha256 of the bodies equal the originals': gemv16.cu `3699c898...`, common.cuh `8b3a3c8d...`, dsk.h `16d78ecc...`, check.h `c607dcc1...`) |
| `csrc/pico12_ext.cu`, `csrc/pico12_range.cu`, `*.py` | ours | the repository's licence |

The default extension calls only pico's public API (`dsk.h`): `bf16x_encode`, `bf16x_check`, `gemv_bf16x`,
`gemv_bf16` (the last for the GPU check's raw comparison). The `stack1` extension (`csrc/pico12_range.cu`, built
only with `stack1`) includes pico's `gemv16.cu` unmodified to launch its internal `gemv16_body` over a range of
CTAs.

## Limits

- TP1 is the tested design (one Spark); nothing here is TP-specific, but TP > 1 shards were not checked.
- Rows above 16 always take the stock path (pico's 32/64-row kernels re-read weights and are slower than raw).
- Layers with a bias are not coded (Inkling-2-Mini has none: `q_bias` / `o_bias` are off, which the MTP draft
  requires).
- Under `torch.compile` or batch-invariant mode every call takes the stock path.
