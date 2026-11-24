# inkling_cert: certified greedy argmax of the target LM head

On greedy verify steps only the argmax of each row's 201,024 logits matters. This package reads a 6-bit copy of
the head (358 MB) instead of pico's coded head (715 MB), bounds every logit rigorously, and runs pico's exact head
GEMV only on the 16-row tiles that can hold the argmax. Every other logit is written as -inf. The stock softcap,
padding mask and argmax then run unchanged and return the same token as the full head, bit for bit.

Status (2026-11-21): written, CPU-tested (21 tests), and GPU-checked on Spark_06 with real weights:
`tools/gpu_check_cert.py` PASS, `tools/smoke_hooks.py` PASS (all hooks land on the image's sglang, cert outside
pico). Not yet run in a serving boot: the next step is `INKLING_CERT_HEAD=check` for a few thousand steps, then an
A/B against `=0`.

## GPU check (2026-11-21, GB10, real head, 5,983 real LM-head rows from the CPU oracle)

| gate | result |
|---|---|
| 6-bit copy decodes to the quantised head (by the kernel's rule); tables finite | ok |
| screen accumulation error / model (C_ACC ||D|| ||x||), 512 rows x 201,024 ids | max 0.0002 |
| pico accumulation error / model (C_ACC ||W|| ||x||) | max 0.0004 |
| s - B <= pico <= s + B for every logit | 0 outside; max abs(pico - s) / B = 0.13 |
| certified token == full-head token (stock softcap, mask, argmax), M = 8 | 5,976 rows, 0 mismatches, 747/747 steps certified |
| same, M = 16 | 5,968 rows, 0 mismatches, 373/373 certified |
| computed tiles bit-identical to pico's full head, others -inf | 0 bad steps |
| `on` = 0 equals pico's full head bit for bit | ok |
| near-tie rows (x along W_a + W_b, norms 150-800) | 2,048 rows, 0 mismatches |
| duplicate head rows (1 pair in the vocab): exact tie resolves to the lower id | ok |
| CUDA graph replay == eager, 40 steps with `on` toggled | ok |

Candidate tiles per step: M = 8 mean 57.6, p50 29, p90 96, p99 238, max 9,639 (one flat row; still exact, just
slower). M = 16: mean 105, p50 52, p99 433.

Microbench (L2 flushed with a 256 MiB read before each rep, CUDA events, median of 30):

| | M = 8 | M = 16 |
|---|---:|---:|
| pico full head (current `lmhead`) | 2.898 ms (247 GB/s) | 3.036 ms (236 GB/s) |
| certified path, median-candidate batch (18 / 30 tiles) | **1.671 ms** | **1.730 ms** |
| certified path, p90 batch (40 / 61 tiles) | 1.680 ms | 1.730 ms |
| screen alone (screen + rowmax + select) | 1.569 ms = 228 GB/s of 358 MB | 1.618 ms = 221 GB/s |
| `on` = 0 step (full head through the masked kernel) | 2.971 ms | 3.027 ms |

Final kernel (`gpu4.log`). Saving per certified step: -1.23 ms at c1 and -1.31 ms at c2. A persistent grid-stride screen (96-384 CTAs, x
loaded once per CTA) was tried and rejected: 140 GB/s at M = 8 and 202 at M = 16, because the x loads then go
ahead of the weight stream. A step that is not eligible pays about +90 us over
pico.

## Switch

| env | default | meaning |
|---|---|---|
| `INKLING_CERT_HEAD` | unset (off) | `1`: certified head on eligible steps; `check`: serve pico's full head, build the certified result beside it and count token mismatches (must stay 0); `0`: off |
| `INKLING_CERT_HEAD_LOG_EVERY` | 2000 | counter line every N verify steps (0 = never) |
| `INKLING_CERT_HEAD_CACHE` | `~/.cache/inkling-cert` | extension build cache |

The package needs `INKLING_PICO12` with `lmhead`, installed first. pico's GEMV is both the exact pass and the
fallback, and the certified token must equal pico's.

Arming in `sitecustomize.py`, after the `INKLING_PICO12` block:

```python
    if _on("INKLING_CERT_HEAD"):  # after inkling_pico12: wraps its lm-head function and reads its coded head
        import inkling_cert
        inkling_cert.install()
```

`_on()` already treats `check` as on, and `install()` reads the mode itself. `scripts/launch.sh` passes the
variable through `SGLANG_ENV` (`-e INKLING_CERT_HEAD=1`); `scripts/launch.sh` sets it from `CERT_HEAD`.

## Measured before building (kill-check)

Real LM-head inputs: 5,983 rows from the CPU oracle (`engine/ref/inkling2_ref.py`, FP32). They come from
teacher-forcing the 16 speedbench greedy transcripts, rounded to BF16. The real unembedding was used. Script:
`results/kbench/round2/cert/cand_count.py`. The head rows have norm 1.00 and ||x|| is about 223. The top logit has a median of
50.7, and the gap to the second logit has a median of 13.6.

| screen | MB read | bound B (median) | candidate tiles / step c1 (mean, p99, max) | c2 (mean, p99) | est. ms |
|---|---:|---:|---|---|---:|
| INT8 per row | 471 | 2.0 | 13, 25, 26 | | 2.06 |
| **INT6, FP16 scale per 192** | **358** | **6.2** | **43, 236, 606** | **78, 328** | **1.59** |
| INT6, FP16 scale per 32 | 382 | 5.0 | 23, 92, 213 | 43, 138 | 1.68 |
| INT7, scale per 192 | 416 | 3.1 | 14, 37, 52 | 26, 57 | 1.82 |
| INT5, scale per 32 | 324 | 10.4 | 305, 1638, 2279 | | 1.52 (heavy tail) |
| INT4, scale per 32 | 290 | 21 | ~10k (everything) | | dead |

The estimate charges screen bytes at 235 GB/s, candidate tiles at 57 KB each at 200 GB/s, plus 50 us of small
kernels. pico's full head is 2.90 ms at 8 rows. The measured screen error is about 1.4 % of the bound, because the
Cauchy-Schwarz bound is far from tight. Only the very peaked logits make it work.

## Per step (M <= 16 verify rows, all in the verify CUDA graph)

| kernel | what |
|---|---|
| `k_screen` (12564 CTAs x 12 warps) | Computes s = x . D^T from the 6-bit copy with BF16 MMAs. Each warp covers one 192-value group and scales its partial by the group's FP16 scale. ||x_g|| is computed from the same registers. Epilogue: B, then lo/hi, then the max of lo and of hi per (row, 16-row tile) over the real vocab. A non-finite value sets a flag. |
| `k_rowmax` | T[m] = max lo, H[m] = max hi |
| `k_select` | Per tile: a mask byte, set when some row has hi >= T - delta in that tile. All tiles are set when the step is not eligible (`on` = 0), a value was non-finite, or a row fails the softcap range test. Also updates the counters. |
| `k_exact` | pico's `gemv16_body<kBF16X, mt 2>` on the masked CTAs. This is the code of pico's coded-head GEMV (the same file, compiled into this extension) and gives its bits. Every other CTA writes -inf. |

Then the stock code continues: `fused_softcap` in place, ids >= `vocab_size` set to -inf, and the folded greedy
accept's argmax.

## Why the token is the same (cert_math.py has the full statement)

1. **Bound.** |pico - <W_v, x>| <= C_ACC ||W_v|| ||x|| and |s - <D_v, x>| <= C_ACC ||D_v|| ||x||, where
   C_ACC = (K/16 * 17 + 32) 2^-23 is the tensor-core accumulation model (an assumption, measured by the GPU check;
   pico12's own check already gates pico at (K/16 + 32) 2^-23). Also |<W_v - D_v, x>| <= sum_g ||R_vg|| ||x_g||
   (Cauchy-Schwarz per group). Every table is rounded up, and every FP32 step is inflated, so lo <= l <= hi.
2. **The winner is computed.** Let w be the stock token and u the id with lo_u = T. If l_w >= l_u, then
   hi_w >= T. Otherwise f(l_w) >= f(l_u) with l_w < l_u, where f is the stock FP32 softcap, which is monotone only
   up to E = 2^-14. Then l_u - l_w <= delta = 2 * 2E / exp(-2L/30) on the range L checked on the device, so
   hi_w >= T - delta.
3. **No -inf beats it.** f(-inf) = -30, and f(l_w) >= 30 tanh(T/30) - E > -30 under the range test (|T|, |H| up
   to about 160; real logits are 50-70).
4. **Ties.** Computed ids hold the stock values. The set of maximisers among them therefore contains w, the lowest
   stock maximiser, and nothing below it. The stock argmax (lowest index) returns w.

x is finite, so every logit is finite and no NaN reaches the argmax. A non-finite screen value forces the full
head anyway.

## When the certified path runs (`on` = 1)

The device flag is written before each verify replay by the `TargetVerifyExecutor.run_non_compact` wrapper. It is
1 only when all of these hold:
- `begin_static_step(bs, armed)` was called with `armed = fold_eligible`. That means a static-mode step whose
  in-graph folded greedy accept is used: all requests greedy, `verify_logits_adjustments_are_noop` (no penalties,
  logit bias, grammar mask or custom processor), and no grammar.
- The batch has no `return_logprob`, no watermark candidates and no sampling-mask capture.
- None of the DSpark debug observers that read the verify logits is set (checked at load; the package disables
  itself if one is).

`begin_step` (compact mode, idle participation) sets it to 0. Eager forwards (no graph) always compute the full
head. The first eager call per row bucket goes through these kernels in full-fallback mode, and later eager calls
go straight to pico. Rows > 16 (c3 and up) never reach here: pico itself takes only 1-16.

## Patch points

| target | what |
|---|---|
| `models/inkling2.py: Inkling2LogitsProcessor._get_logits` | marks target-verify calls (`forward_mode.is_target_verify()`) |
| `models/inkling2.py: Inkling2ForConditionalGeneration.post_load_weights` | after pico coded the head: builds the 6-bit copy and the tables on the GPU (chunked, float64), runs the self-test (full fallback == pico bit for bit, certified argmax == full argmax on random rows at M = 8 and 16; this also loads every kernel before capture), and logs one line |
| `layers/quantization/unquant.py: sm12x_fp32_lm_head` | wraps pico's wrapper. It takes target-verify calls that pico would take, as long as the processor has softcap 30, no `logit_scale`, no TP gather, and `vocab_size` <= rows. Everything else goes to pico or stock. |
| `speculative/dspark_components/dspark_verify.py: DsparkVerifyEpilogue.begin_static_step / begin_step, TargetVerifyExecutor.run_non_compact` | the per-step device flag (above) and the periodic counter line |

Source strings are checked at `install()`, and attributes when each module is imported. A mismatch exits with 78.
TP > 1 disables the package with a log line.

Coexistence:
- `inkling_pico12`: required, and its wrapper sits inside ours.
- `inkling_vcap`: independent. Dead rows are still verify rows with real hidden states, and their logits are
  certified like the others.
- `inkling_draftbytes`: the draft head is not touched.

## Cost and memory

- Extra memory is 358 MB (Q 347.4, scales 4.8, residual norms 4.8, row norms 0.8) plus about 1.7 MB of buffers.
- A step that is not eligible (`on` = 0: sampled, logprobs, ...) costs pico's head plus about 20-30 us. The screen
  and row-max kernels return at once, and the masked kernel runs every tile (pico's full head). An eligible step
  that fails the range test pays the screen and the full head; this is never seen on real rows.

## Reviews and logs

- GPU logs (Spark_06): `results/kbench/round2/cert/gpu2.log` (first PASS), `gpu3.log` (persistent variant, rejected),
  `gpu4.log` (final kernel).

## Files

| path | origin |
|---|---|
| `csrc/cert_ext.cu`, `*.py`, `tests/`, `tools/` | ours (ported in spirit from our GLM-5.3-Flash `glm_cert_head.py` / `cert_math.py`) |
| pico's `gemv16.cu`, `common.cuh`, `dsk.h`, `check.h` | not copied: `#include`d from `../inkling_pico12/third_party/pico` (MIT, Copyright (c) 2026 Andy Shu) |

## Checks

- CPU: `python3 -m unittest discover -s inkling_cert/tests -v` from `overlay/` (needs numpy and torch). It covers
  pack/unpack, the lane/MMA k-mapping of the packed layout, tables (torch vs numpy), bound coverage, adversarial
  screen errors, exact ties, softcap ties, padded ids, range fallback and the fill.
- GPU: `python3 inkling_cert/tools/gpu_check_cert.py --model /model --hidden DIR`, where DIR holds real LM-head
  input rows as `.npz`. It gates the packing, the accumulation model (screen and pico vs FP64), coverage
  (lo <= pico <= hi for every logit), and token identity on every real row at M = 8 and 16. It also gates near-tie
  rows, duplicate head rows, computed tiles bit-identical to pico, `on` = 0 equal to pico, and graph replay equal
  to eager. It then benches.
