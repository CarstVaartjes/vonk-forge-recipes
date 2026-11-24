# Speed overlays for SGLang (Inkling-2-Mini on GB10)

Import-time patches for the stock `lmsysorg/sglang:inkling-2-mini` image. Nothing in the image is edited: the
directory is mounted read-only into the container and put on `PYTHONPATH`; `sitecustomize.py` arms each package
only when its environment switch is set. With no switch set, nothing is imported and the engine runs exactly as
shipped. `scripts/launch.sh` sets the switches (`OVERLAY=0` turns all of them off).

A package whose patch target is missing or whose signature changed stops the server at start-up (exit 78) instead
of silently running the stock path.

| package | switch | what it changes | effect on the output |
|---|---|---|---|
| `inkling_vcap/` | `INKLING_VERIFY_CAP`, `INKLING_VERIFY_CAP_BV`, `INKLING_VERIFY_CAP_FASTPROBS`, `INKLING_VERIFY_CAP_FUSED_ACCEPT` | Verify-width cap for DSpark: per request and step, verify only the first L of the 8 draft rows, L from the draft's own confidence and the batch size (`policy:` survival tables and one price, or `cost:` thresholds); the dead rows reuse the anchor row's experts, so the MoE reads fewer distinct experts; acceptance is capped at row L-1. With `--speculative-use-block-verification`, sampled rows use block verification with the dead drafts handed to the kernel as a stopped draft (`BV=mask`). `FASTPROBS`: the max draft probabilities and the sampled accept's draft probabilities from one or two reads of the BF16 draft logits. `FUSED_ACCEPT`: the sampled accept (chain or block verification) computed from the logits for >= 3 requests. | Target distribution unchanged (greedy: every emitted token is still the target argmax; sampled: the width is a left-to-right stopping rule on draft-side information, and block verification over the stopped draft is Sun et al.'s algorithm; both Monte Carlo tested). FASTPROBS / FUSED_ACCEPT: same function, FP32 summation order only. |
| `inkling_draftbytes/` | `INKLING_DRAFT_BYTES`, `INKLING_DRAFT_HOTVOCAB_FILE` | Draft-side byte cuts: draft logits through the bandwidth-efficient GEMV (`gemv`), FP8 copies of the Markov fold matrix (`fold8`), of the LM head for the draft only (`head8`) and of the three draft layers (`draftmx8`), a ranked draft vocabulary (`hotvocab:N`), single-read FP8 draft GEMM above 64 rows (`wide8`), narrow-N tiles for the MXFP8 draft layers (`w8tile`), and an attention sink + last-W-token window for the draft's global layer (`dwin:W[:S]`). | Draft proposals only; verification still uses the BF16 target. Can change acceptance, never the target distribution. |
| `inkling_kernels/` | `INKLING_KERNELS` | `moe_tiles`: GB10 tile choice for the NVFP4 expert GEMMs at decode (`tile_k` 256), bit-exact vs stock (bench + end-to-end text). `qkvr_gemv`: keeps the stacked Q/K/V projection of the looped layers on one GEMV instead of cuBLAS at <= 16 rows per source (bit-identical in the GPU check). | none (same math; any difference is FP summation order) |
| `inkling_prefill/` | `INKLING_PREFILL` | Prefill: per-source qkvr projection, relative bias written in FA4's layout, MoE input quantised once per token, BF16 MoE combine. | none (bit-identical in the GPU check) |
| `inkling_pico12/` | `INKLING_PICO12` | pico's packed 12-bit BF16 GEMV (exactly coded weights, ~0.76 of the bytes) for qkvr / wo_ud (`dense`), the target FP32 LM head (`lmhead`) and optionally the draft (`drafthead`, `draft`) at 1-16 rows; BF16 originals kept for prefill and > 16 rows. Builds a CUDA extension at start-up (cached under `/root/.cache`). GPU check PASS (packed == raw bit for bit on real weights). | FP32 summation order only (packed == pico's raw BF16 GEMV bit for bit; vs stock not bit-identical) |
| `inkling_cert/` | `INKLING_CERT_HEAD` | Certified greedy LM head: a 6-bit screen with a rigorous per-logit bound picks the vocabulary tiles that can hold the argmax; pico's exact GEMV computes only those, the rest is -inf before the stock softcap and argmax. Greedy target-verify steps of <= 16 rows without logprobs / grammar; everything else runs the full head. Needs `INKLING_PICO12` with `lmhead`. | Identical argmax by construction (serving check mode: 45,584 rows, 0 mismatches). |
| `inkling_wide/` | `INKLING_WIDE`, `INKLING_WIDE_SCOPE` | Packed 12-bit GEMM (pico's coded weights read once per call, mma.sync) for 17-256-row calls of qkvr / wo_ud / LM head on decode / verify forwards. | FP32 summation order only (weights decode to the same BF16 bits). |
| `inkling_vattn/` | `INKLING_VATTN` | Target-verify attention on the split-KV relative-bias kernel for up to 48 requests; above 32 rows the bias is computed per key tile in the kernel. | Same math; bias bit-identical, output within 1 BF16 ulp of the stock compact path. |
| `inkling_attn/` | `INKLING_ATTN` | Prefill / extend attention with the relative bias computed in the kernel (FA2-style, paged KV, sliding window and global layers); falls back to the stock FA4 path per call. | Same math; error vs an FP32 reference equal to stock's. |
| `inkling_pfuse/` | `INKLING_PFUSE` | Prefill glue fusion (loop-message mixes, attention-residual mix + RMSNorm, short conv + residual update) for extend batches of >= 512 rows. | FP32 row-sum order only; short conv bit-identical. |
| `inkling_moe2/` | `INKLING_MOE2` | GB10 tile tables for the routed-expert grouped GEMMs at verify batch sizes (NVFP4 9-128 tokens, MXFP8 tactic at 512-1536 routed rows). | none (every chosen configuration bit-identical to the shipped one) |
| `inkling_scodes/` | `INKLING_SCODES` | NVFP4 expert scales stored as a row base + 5-bit codes and expanded inside a modified FlashInfer SM120 grouped-GEMM mainloop (escapes read the resident stock scales). | none (bit-identical to the stock kernel) |
| `inkling_kv12/` | `INKLING_KV12`, `INKLING_KV12_MAX_PAGES` | Full, committed KV pages of the global attention layers also kept in a packed 12-bit side pool (sign + mantissa byte, 4-bit exponent code, escapes; slots follow the pages being decoded); decode / verify attention decodes packed tiles in registers. Install after `inkling_vattn`. | none (decoded tiles equal the BF16 cache; attention partials bit-identical to stock) |
| `inkling_kv12s/` | `INKLING_KV12S`, `INKLING_KV12S_POOL`, `INKLING_KV12S_OVERFLOW_HEADS`, `INKLING_KV12S_FALLBACK_PAGES` | Packed global KV store (default): the target's global-layer KV lives only in the 12-bit form of `inkling_kv12`; open pages stay BF16 until committed. `POOL=static` (default since v2.7.2, `static.py`) builds it inside SGLang's stock hybrid sliding-window pools: the draft keeps its own BF16 pool, prefill keeps its CUDA graphs and reads packed prefixes through a fixed BF16 scratch gathered on the device (the packed loader when a prefix does not fit). `POOL=unified` is the v2.7.1 binding on `--enable-unified-memory` (page-major window band, `swa_layout.py`). A host guard (`pressure.py`) keeps the exact overflow slots from running out. Refuses unsupported server settings at startup. Checks: `tools/gpu_check_kv12s.py`. | none (every store and read bit-identical; greedy texts identical to the BF16 store) |
| `inkling_noexp/` | `INKLING_NOEXP` | Verify rows the verify cap has cut (`inkling_vcap` remap) route to a phantom expert E inside captured verify graphs, so the routed-row quantisation, both grouped GEMMs, SwiGLU and the combine skip them; eager calls keep stock routing. Inactive when the verify cap or its remap is off. | none for every row that can be accepted (live rows bit-identical; cut rows are never read) |

Measured effect of each package and of the stack: `docs/experiments.md`.

## Layout

- `sitecustomize.py`: combined loader.
- `inkling_*/`: the packages; each has its own README with patch points, exactness argument, CUDA-graph safety
  and limits.
- `tools/`: GPU checks and microbenchmarks to run inside the image on a Spark with the server stopped
  (`gpu_check_qkvr.py`, `gpu_check_pico12.py`, `gpu_check_prefill.py`, `bench_moe_tiles.py`, `gpu_check.py` and `bench_draft.py` for the draft package,
  `build_hotvocab.py` for the hot-vocabulary draft, `gpu_check_bv.py`, `gpu_check_fastprobs.py`,
  `gpu_check_fused_accept.py` and `bench_fused_accept.py` for the sampled accept).
- `tests/`: CPU unit tests (`tests/vcap`, `tests/draftbytes`, `tests/pico12`, `tests/prefill`), run them inside the
  image: `docker run --rm --entrypoint bash -v $PWD/overlay:/ov lmsysorg/sglang:inkling-2-mini -c "cd /ov &&
  python3 -m unittest discover -s tests/vcap"`.

## Writing a new overlay

Patch at import time through a post-import hook, check every target's signature, raise a `BaseException`
subclass on mismatch (SGLang's model registry swallows `Exception` during model imports), keep every per-step
action graph-safe (device buffers written before replay, no host sync), and log one line at start-up.
