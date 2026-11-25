# inkling_rsfix: exact final-token sampling for speculative verification

Overlay package for the Inkling-2-Mini single-Spark SGLang recipe (image
`lmsysorg/sglang@sha256:efac853b46811ccb760545b7b29e7119bdf907537ccf6e863fac3011fe8b98de`). It ships alone: it
imports no other overlay and patches only the SGLang modules of the four parts below.

| switch | value | effect |
|---|---|---|
| `INKLING_RSFIX` | unset / `0` | nothing is imported |
| | `1` | all parts: `reject`, `markov`, `blockwise`, `selector` |
| | e.g. `reject,markov` | only the listed parts |

Start-up: one line `[inkling_rsfix] enabled: parts=...`. A target whose source differs from the pinned image text
(sha256 per function, `inkling_rsfix/_sources.py`) stops the server at start-up (exit 78 through `sitecustomize.py`)
instead of running the stock path.

## Loading

`overlay/sitecustomize.py` installs it last (`if _on("INKLING_RSFIX")`); `scripts/launch.sh` passes `RSFIX=1` by
default. Order does not matter: it touches no function another overlay wraps.

## Parts

### `reject`: `sglang.kernels.ops.speculative.reject_sampling` (DSpark and DFLASH sampled verification)

`chain_speculative_sampling_triton` draws the bonus / residual token in a CDF pass:
`speculative_sampling_classic_kernel` pass 2 (chain and block verification), or `_split_cdf_kernel` (the split path,
taken for `batch_size < SM count` without block verification, i.e. sampled chain verifies at c1..c47).

Stock bug: the pass picks the first lane with `total_cumsum > target_u`. There is no `lane < VOCAB_SIZE` condition
and no positive-mass condition. `target_u = coin * norm_sum`, where `norm_sum` is a sum of per-tile `tl.sum` values
(or, after block verification, the residual mass summed over 32768-lane tiles). The pass scans with `tl.cumsum`. The
scan can end below `target_u`, and a later zero-mass lane is then the first lane above it: a padded vocabulary row
(Inkling: ids 200,058..201,023 of the 201,024-wide logits have p = 0) or a masked lane >= VOCAB_SIZE (seen on GPU:
bonus == 24 at V = 24). When nothing matches, the stock token is VOCAB_SIZE - 1. In the split path this is the
initial value of `Result`. It can again be a padded row.

Fixed rule, on both paths:
1. Select the first lane with `(total_cumsum > target_u) & (lane < VOCAB_SIZE) & (val > 0)`.
2. If no lane matches, select the last lane with positive residual mass over the whole row.
3. If the residual has no positive lane (`norm_sum == 0`, the degenerate residual), select the last lane with positive
   target mass. Only a target row with no positive entry, which is not a distribution, keeps VOCAB_SIZE - 1.

Edits are listed in `tests/rsfix/data/reject_edits.diff`, and a test pins them. Classic kernel: the lane condition, and a
per-tile `max` of a lane key (VOCAB_SIZE + v for a positive-residual lane, v for a positive-target lane, else -1). The
scan reaches the end of the row exactly when nothing matched, so the key covers the whole row. Split kernels:
- `_split_residual_tile` also returns the target tile.
- `_split_sum_kernel` (tile 0) writes `Result = VOCAB_SIZE` ("no match") and `Fallback = -1`.
- `_split_cdf_kernel` applies the lane condition and posts each tile's key with `atomic_max`.
- The last arrival reads both and decodes the fallback.
- `_split_scratch` makes one extra int32 per row. Like `Result`, it is written by `_split_sum_kernel` one launch
  before use, so it resets in stream order. The scratch set is made once per (device, stream, tile count) and is not
  created under capture: CUDA-graph safety is the same as stock. As with stock, calls that share one scratch set
  (same device, stream and tile count) must not run concurrently. A replay runs on the caller's current stream:
  eager calls and replays issued on the capture stream are serialized by it, and replays issued on other streams
  need explicit ordering against them. A graph captured on a stream with no eager warm-up has no scratch and captures
  the classic kernel, which carries the same fix.

Exactness: the accept walk, block verification, normalizers, RNG and the split gate are the stock code. In exact
arithmetic the first lane above `target_u` always has `val > 0`, so the law is unchanged. When the stock rule
emits a lane with positive mass, the fixed rule emits the same lane. Only rounding events change, and they no longer
emit a zero-mass or out-of-range token.

### `markov`: `dspark.markov_sample` (DSpark Markov draft sampler), SGLang PR #51681

A 32-bit Philox word is turned into `u = (w + 0.5) * 2**-32` in FP32. Words >= 0xFFFFFF80 round to u == 1, so the
exponential noise `-log1p(-u)` is +inf and the token's key is -inf. This leaves a ~3e-8 hole per token and draw in
the draft law, which the verify's q does not account for. If that token is the only finite-logit lane, it ties with
the masked lanes and a masked token wins. The PR's fix folds the word before the conversion:
`folded = (min(w, ~w) + 0.5) * 2**-32`. The lower half then uses `-log1p(-folded)` and the upper half `-log(folded)`.
This package uses the PR's `markov_sample.py` hunks verbatim (not its `sharded_accept_hip.py` hunk), a test pins
them, and they cover both the Triton kernel and the CPU torch
reference. Rebound: `_markov_sample_partial_kernel` (a copy of the stock text that calls the fixed `_select_block`),
`_select_block`, `_exponential_from_philox`, `MarkovSampleStep.torch`. `inkling_draftbytes` draws through
`MarkovSampleStep.execute`, so it is covered too.

### `blockwise`: `dspark.blockwise_rejection`, SGLang PR #51692

This path runs only with `SGLANG_OPT_DSV41_BLOCKWISE_REJECTION=1`. That variable defaults to
`SGLANG_ENABLE_NV_MEGA_RUNTIME_OPTIMIZATION`, which is off, and the recipe sets neither. The PR's fixes are patched
in, but the path is still not exact (see the cancellation case below). Stock bugs fixed:
- `max + log(sum)` loses the log-sum against a large row maximum.
- `log_p > log(u) + log_q` rounds a coin just below 1 back to `log_q`, so equal laws reject. The rejected row then
  resamples a residual that may hold only zero-target-mass tokens.
- An uncentred Gumbel race can round distinct draws to a tie.

The PR's code is used verbatim, and a test pins it. It is not fully exact. A review found a residual
cancellation case: unit temperatures, target logits `[-inf, 0, -inf]`, draft logits `[-inf, 0, -32]`, proposal 2.
FP32 normalization cancels the only positive residual lane, and the all `-inf` race then returns lane 0, which has
zero target mass. A stable residual and a positive-target fallback are needed before anyone enables this route. The
recipe does not use it.

### `selector`: the DFLASH / DFlash2 selector walk (`dflash._selector_walk_kernel`, `CandidateSelector.sample_path`, `lilicorr._selector_walk_torch`)

Stock bug: a sampled slot picks `index = #{lanes: u >= cumsum(q)}` clamped to `top_k - 1`. When the coin reaches or
passes the scan's last value (FP32: the softmax sum is just below 1), the count is top_k and the clamp picks the last
candidate, which can have q = 0. The draft then proposes a token outside the support of the q it reports.

Fixed rule (sampled rows only; greedy rows are untouched): the first lane with `(cumsum(q) > u) & (q > 0)`; no match
-> the last lane with q > 0 (lane 0 only for a row with no positive q, which is not a distribution). In exact
arithmetic the first lane above u always has q > 0, so the law is unchanged. The CUDA kernel is rebound as a module
global (every launcher caller gets it); `srt/models/dflash.py` is pinned per function because the DFlash2
compatibility patch rewrites other parts of that file. Details: `selector.py`.

## Not patched (findings)

- `inkling_vcap/fused_accept.py` (`VCAP_FUSED_ACCEPT`, our kernel): fixed in the package itself since v2.7.3 (the
  rules below); `inkling_dfaccept`'s sparse-q race (DFLASH) uses the same rules. The finding, as first recorded: the
  bonus race had the PR #51681 bug.
  - At lines 283-287, `hi = r0 * 2**-32` rounds to 1.0 for r0 >= 0xFFFFFF80, giving `e = -log(1 - 1) = +inf` and a key
    of -inf.
  - Lines 271, 292 and 365 then fall back to `V - 1` (201,023, a padded row) when no key is finite. With a one-hot
    residual this happens with probability ~3e-8 per draw.
  - A separate change applies the fix:
    - Fold the Philox word before the FP32 conversion: `upper = w0 >= 0x80000000`,
      `f0/f1 = where(upper, ~w0/~w1, w0/w1)`, `small = f0 * 2**-32 + (f1 + 0.5) * 2**-64` (V, or 1 - V in the upper
      half), `e = where(upper, -log(small), -log1p(-small))`.
    - Clamp positive-mass keys to a finite floor: `key = where(m & (w > 0), max(log w - log e, -3e38), -inf)`.
    - When a row has no finite key, emit the last lane with `pv > 0` (a per-split max written by `_race_kernel` and
      combined in `TritonPrims.sample`; `TorchPrims.sample` line 168 the same) instead of `V - 1`.
- `inkling_vcap/fastprobs.py`: softmax only, no selection. Not affected.
- The `inkling_vcap` BV mask feeds the stock classic kernel and is covered by `reject`.
- `inkling_draftbytes`: has no sampler of its own. It calls the stock `MarkovSampleStep.execute`
  (`patches.py:275`), which is covered by `markov`.
- Image, off the Inkling path, not patched:
  - `dspark/sharded_accept_hip.py:185` (HIP, vocab-sharded bonus, same #51681 pattern; the PR's second hunk).
  - `dspark/dspark_accept_split.py:201` (ROCm logits-split accept, `SGLANG_ROCM_DSPARK_SPLIT_ACCEPT`): the same
    unguarded CDF and `vocab - 1` fallback.
  - `dspark/simulated_bonus.py:35-36` (simulated-acceptance mode, `-log(-log(tl.rand))`): an endpoint `u` gives
    infinite noise, and a lone finite lane can then tie the masked ones.

## Files

- `inkling_rsfix/__init__.py`: switch, `install()`, `apply(part, module)` / `use()`, start-up log.
- `_hook.py`: the post-import hook, a copy of `inkling_noexp`'s.
- `_sources.py`: the source pins.
- `reject.py`, `markov.py`, `blockwise.py`, `selector.py`: the fixed objects.
- `reference.py`: a pure-torch model of the CDF pass (stock and fixed rules).
- `tests/rsfix/test_rsfix_cpu.py` (under `overlay/`): CPU tests.
- `tools/gpu_check_rsfix.py` (under `overlay/`): the GPU check.

## Checks

CPU (any machine with torch and pytest, or the image). `RSFIX_SGLANG_SRC` points at a copy of the image's `sglang` package directory; inside the image the
installed package is used:

```
cd overlay && RSFIX_SGLANG_SRC=/path/to/sglang python -m pytest -q tests/rsfix/test_rsfix_cpu.py
```

GPU (inside the image, no server, about 10 minutes; `--quick` for a smoke run):

```
cd overlay && PYTHONPATH=. python3 tools/gpu_check_rsfix.py
```
