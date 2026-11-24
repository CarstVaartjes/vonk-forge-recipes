# inkling_draftbytes: fewer bytes per step on the DSpark draft (Inkling-2-Mini, 1x GB10)

An import-time overlay for `lmsysorg/sglang:inkling-2-mini` (sglang
0.0.0.dev1+g03fde9cc3.d20261120). It changes **only what the Inkling2 MTP draft
proposes**. The target forward, the verify and the accept rule are not
touched. Nothing here has run on a GPU yet. The CPU tests pass (23/23).
`gpu_check.py` and `bench_draft.py` run inside the image on a Spark with the server stopped (copies in overlay/tools/).

## Options (`INKLING_DRAFT_BYTES`, comma list; unset or empty = no-op)

| option | what it does | bytes/step at c1 (stock -> new) | expected | extra memory |
|---|---|---|---|---|
| `gemv` | Draft base logits (926 MB BF16 LM head, M = 7*bs rows) go through the image's `sm12x_bf16_gemv` Triton GEMV when M <= 16, instead of cuBLAS `torch.matmul`. `INKLING_DRAFT_GEMV_MAX_M` (default 16) also sends 16 < M <= that value to this package's unscaled kernel. | same bytes, better streaming | -0 to -1.4 ms (the image's docstring: ~175 GB/s for cuBLAS vs ~240 for the GEMV at 1 row; M=7 is unmeasured) | 0 |
| `foldk` | The Markov fold `addmm(fold_bias, latent, fold_weight.T)` through this package's kernel on the **BF16** fold matrix. This is a pure kernel A/B against cuBLAS at M = bs. | same | unknown (cuBLAS at M=1, K=256, N=201024) | 0 |
| `fold8[:row\|gN\|mx]` | FP8 e4m3 copy of the 201024x256 fold matrix, with a per-row fp32 scale by default. It feeds the same kernel (dequant on the fly, FP32 accumulate). | 7 x 103 MB -> 7 x 52 MB (-354 MB) | -1.6 ms | +52 MB |
| `head8[:row\|gN\|mx]` | FP8 copy of the LM head for the draft base logits, own kernel. It is the fallback if `--speculative-dspark-lm-head-dtype mxfp8` (FlashInfer cute-dsl) does not run on sm_121. It stands down when that flag is set. | 926 -> 464 MB | -2.1 ms | +464 MB |
| `draftmx8[:mx\|row]` | MXFP8 (default) copies of every BF16 linear in the 3 draft layers (qkv 5120x2304, o 2304x3072, gate_up 18432x2304, down 2304x9216, kv_proj_ctx 2048x2304), made in memory after load. Rows <= `INKLING_DRAFT_W8_MAX_M` (64) use the FP8 kernel. Larger batches use the stock BF16 path. | ~524 -> ~270 MB | -1.1 ms | +270 MB (BF16 kept) |
| `hotvocab:N` | The draft proposes only from the first N ids of `INKLING_DRAFT_HOTVOCAB_FILE`, which must be frequency-ranked (see `build_hotvocab.py`). The base logits read N LM-head rows, and the fold reads N fold rows. Stacks with `fold8`, `head8` and `gemv`. | N=32768: 1647 -> ~210 MB with fold8 (~135 MB with head8 too) | -6.3 to -6.9 ms, minus the acceptance lost to misses | +151 MB (N=32K, BF16 head rows) + 8-17 MB |
| `w8tile` | With `draftmx8`: the draft linears' Triton launches use a narrow tile (BN 16, BK 256 for K = 2304 / 128 for K = 3072, 9216; `row`/`gN` keep their BK) instead of BN 32 / BK 256: twice the CTAs on the narrow-N shapes (o, down: 2304 rows). Bit-identical (the MX path is one MMA chain per output whatever the tile; gated on the GPU at 8, 16, 24, 32, 64 rows). Recommended with `draftmx8`. | same bytes, better streaming | GB10 L2-cold, 8 rows, the 4 linears of one layer: 549 -> 430 us (155 -> 198 GB/s): -0.36 ms per c1 step (below) | 0 |
| `w8gemv` | With `draftmx8` (mx): the draft linears at 1-16 rows through `csrc/w8gemv.cu`, a CUDA weight-streaming GEMV on the same MXFP8 pack (no new copy): one warp per 8 weight rows over the whole K, weights 4-8 steps ahead in registers, activations 2-4 steps ahead; Triton's per-output arithmetic (one ascending-k MMA chain, its kWidth-4 operand order, exact dequant, FP32 accumulation). Rows above 16 keep Triton (`w8tile` applies there). `INKLING_DRAFT_W8GEMV_CFG` forces a variant. Built at draft prepare (`INKLING_DRAFT_W8GEMV_CACHE`). **Experimental, not recommended:** bit-identical to the Triton kernel (gated), but slower than `w8tile` on GB10 (8 rows, one layer's 4 linears: 478 vs 430 us; o 49 vs 41 us, down 135 vs 113 us). | same bytes | none over `w8tile` | 0 |
| `wide8` | For calls of more than 64 rows (batched verify / draft blocks, e.g. the head at c24: 192 rows), this package's kernels run one CTA per output tile over ALL 64-row blocks (row blocks outer, `_acc_tile` unchanged; the CTA's W tile is re-read from L2) with a wider tile (BN 64, 8 warps, 3 stages), instead of one CTA per 64-row block re-reading W from DRAM. Calls of <= 64 rows launch exactly as before. Output bit-identical to the default kernel (gated on the GPU for head8 row/mx, fold8, the BF16 head, the hot step). | head at 192 rows: 3 x 464 MB -> 464 MB from DRAM | head8:row 192 rows 6.16 -> 3.52 ms, 384 rows 12.3 -> 6.9 ms; fold8 192 rows 1.03 -> 0.60 ms | 0 |
| `tau:X` (or `INKLING_DRAFT_TAU=X`) | Draft temperature. Sampled rows draft at X x T instead of T. `resolve_draft_params` scales both the sampling temperature and the temperature the accept uses to build q, by the same FP32 product. The target keeps T. Greedy rows and all-greedy batches are unchanged. X in [0.05, 4]; 1 = off. Refuses the sampling-accept CUDA graph, which reads the draft temperatures as the target's. Credit: the idea comes from our DeepSeek recipe's draft-temperature finding (0.7-0.8 best there). | same | acceptance only: expect +1-2 % on thinking (T = 0.8) at tau 0.7-0.8, 0 at T = 0 | 0 |
| `dwin:W[:S]` | The drafter's global attention layer (layer 1; `local_layer_ids = [0, 2]`) reads the first S tokens (sink, default 128 = one page; 0 = none) plus every page holding one of the last W tokens of its own K/V instead of the whole context (`dwin.py`). Whole pages: `ceil(W/128)+1` window columns, W <= kept tail < W+128. Identity below about S + W. The FA4 call gets the window's static `max_seqlen_k` (SplitKV planning; `INKLING_DRAFT_DWIN_MAXK=0` = stock bound). A wrapper on `FlashAttentionBackend.forward_extend` swaps in a gathered page table / lengths for that layer only (TARGET_VERIFY, topk 1); built by device tensor ops, CUDA-graph safe. Draft-only: outputs exact, cost is acceptance only. | 4,096 B per context token -> (S + W) x 4,096 B: 128K 537 MB -> 67 MB (W 16K) | GB10, FA4 graph replay, bs 1: 128K 2.30 -> 0.32 ms, 256K 4.47 -> 0.32 ms (W 16K; W 8K 0.17 ms); +~22 us of metadata ops at any context | 0 |

Byte math at 220 GB/s: LM head 201024x2304x2 = 926 MB; fold 201024x256x2 = 103 MB,
read 7 times = 720 MB; draft linears 165 MB per layer x 3. The ms figures are the
bytes saved / 220 GB/s. They are upper bounds that hold only if the new kernels
stream at the stock rate. `bench_draft.py` measures the actual rate.

Stacked best case on paper: `hotvocab:32768,fold8,head8,draftmx8` saves ~8 ms of
~60 ms (~13 %) before acceptance changes. `gemv,fold8,draftmx8` gives -3 to -4 ms
with no change to the proposal vocabulary. Acceptance must be measured.

Tunables (env): `INKLING_DRAFT_FOLD_BN` (64), `INKLING_DRAFT_FOLD_BK` (128),
`INKLING_DRAFT_LINEAR_BN` (32), `INKLING_DRAFT_LINEAR_BK` (256),
`INKLING_DRAFT_NUM_WARPS` (4). Set them from the `bench_draft.py` sweeps.
`wide8` tunables: `INKLING_DRAFT_WIDE8_MODE` (1; 2 = K outer with up to 6 row-block accumulators in registers,
also bit-identical but spills and is slower), `INKLING_DRAFT_WIDE8_BN` (64), `INKLING_DRAFT_WIDE8_WARPS` (8),
`INKLING_DRAFT_WIDE8_STAGES` (3), from `tools/gpu_check_wide8.py --tile-sweep` (2026-11-21, GB10: every
configuration bit-identical to the default kernel). Check: `tools/gpu_check_wide8.py --ref <pre-wide8 _triton.py>`
(modes, graph replay, hot step) and `--ops` (the package's dispatch, wide8 off vs on).

### `w8tile` / `w8gemv` measurements (GB10, L2-cold, random MXFP8 packs of the draft shapes, `results/kbench/round2/c1leak/bench2.py`, 2026-11-22)

us per call; GB/s of FP8 + scale bytes. Triton default = BN 32 / BK 256 / 4 warps (today); `w8tile` = BN 16 / BK 256
(K 2304) or 128 (K 3072, 9216); `w8gemv` = its best variant per cell.

| rows | shape [N x K] (MB) | default | `w8tile` | `w8gemv` |
|---|---|---|---|---|
| 8 | qkv 5120 x 2304 (12.2) | 81.4 (149) | 68.7 (177) | 70.7 (172) |
| 8 | o 2304 x 3072 (7.3) | 61.6 (119) | 41.0 (178) | 49.3 (148) |
| 8 | gate_up 18432 x 2304 (43.8) | 228.4 (192) | 208.0 (211) | 222.4 (197) |
| 8 | down 2304 x 9216 (21.9) | 177.2 (124) | 112.8 (194) | 135.2 (162) |
| 8 | kv_proj_ctx 2048 x 2304 (4.9) | 42.0 (116) | 30.9 (158) | 35.0 (139) |
| 8 | one layer's 4 linears | 548.6 (155) | 430.4 (198) | 477.5 (178) |
| 1 / 16 | one layer's 4 linears | 531.6 / 531.0 | 434.3 / 436.3 | 471.2 / 490.1 |
| 24 / 32 / 64 | o | 75.9 / 71.8 / 90.2 | 41.0 / 43.0 / 55.3 | - |
| 24 / 32 / 64 | down | 215.0 / 200.7 / 260.2 | 114.8 / 116.8 / 157.7 | - |
| 24 / 32 / 64 | qkv, gate_up | 113 / 98 / 137, 336 / 289 / 413 | 78 / 76 / 125, 246 / 238 / 414 | - |

Bits: `w8tile` == default at 8, 16, 24, 32, 64 rows on every shape (the tile sweep at 8 and 16 rows: all 48
configurations identical); `w8gemv` == default at 1-16 rows, every variant, with and without bias, strided
activations, MX scales 2^-54 .. 2^8 and an all-zero block (it reproduces Triton's kWidth-4 MMAv2 operand order).
At c1 (8-row draft block, 3 layers) `w8tile` takes the 12 draft-layer GEMMs from 1.65 to 1.29 ms (-0.36 ms per
step, the profile's 1.64 ms at 156 GB/s), plus ~-0.03 ms on the three kv_proj_ctx calls. Neither reaches 220 GB/s
on the narrow shapes (o, down: 2304 output rows): `w8tile` is the recommended setting.

## Patch points

All patches are installed by post-import hooks when the target module is first
imported. Weights are prepared once, right after
`Inkling2ForConditionalGenerationMTP.attach_shared_modules` (draft worker init),
which is before any CUDA graph capture.

| target (file:function) | option | change |
|---|---|---|
| `srt/models/inkling2_mtp.py: Inkling2ForConditionalGenerationMTP.attach_shared_modules` | any | runs the stock attach first, then `prepare()`: FP8 packs, hot subsets, draft-linear packs, and one log line |
| `srt/models/inkling2_mtp.py: project_through_lm_head` (the name this module imported from `models/dspark.py`, so only the Inkling2 draft is affected) | gemv, head8 | GEMV / FP8 kernel. It stands down if the LM head is quantized or `--speculative-dspark-lm-head-dtype` is set |
| `srt/models/inkling2_mtp.py: Inkling2MTPHead.step_bias` | fold8, foldk | same `_rnn_step`, then the fold kernel instead of `torch.addmm`, then the same `_gather_logits` crop |
| `srt/models/inkling2_mtp.py: Inkling2MTPHead.refresh_fold` | any | after a weight refresh, re-prepares the packs in place, so captured graphs keep valid pointers |
| `srt/models/inkling2_mtp.py: Inkling2ForConditionalGenerationMTP.compute_base_logits` | hotvocab | base logits on the N hot rows, scattered into a full-width [M, V] buffer that is -inf elsewhere |
| `srt/speculative/dspark_components/dspark_draft.py: _draft_step` | hotvocab | if the head has a hot state: `_rnn_step`, then one fused kernel (fold on the hot rows + base + the stock roundings + softcap + vocab limit, scattered into a -inf row), then the **stock** `MarkovSampleStep` with a zero bias and no softcap. It writes the kept step logits and picks the token. Otherwise it calls the stock step |
| draft linears' `quant_method` (instances under `model.layers`) | draftmx8 | swapped for a subclass of the same class, so `isinstance` checks still pass, whose `apply` uses the FP8 kernel at <= 64 rows |

The fail-loud rules:
- A missing patch target raises `DraftBytesPatchError` at import. It derives from
  `BaseException` because sglang's model registry swallows `Exception`.
- A bad option string exits the process with code 78 from `sitecustomize.py`.
  Python ignores sitecustomize exceptions, so raising would not stop it.
- `prepare()` raises if TP > 1, if the LM head is not BF16, or if the
  `_draft_step` patch did not apply.

There is one startup line: `[inkling_draftbytes] active: options=...; patched ...; <packs and sizes>`.

Not compatible with `--enable-nvidia-megakernel`: its specialized draft sampler
would bypass the patched step. It was screened flat anyway.

## Exactness

Every option changes only the draft's proposal q. Nothing touches the target
distribution p or how a token is accepted.

- **Greedy (T=0) verify** accepts a drafted token iff it equals the target argmax.
  It then emits the target's own token at the first mismatch. The emitted
  sequence is therefore the target's greedy sequence for **any** proposal. All
  options are exact by construction: FP8 noise, a different kernel, a restricted
  vocabulary.
- **Sampled verify** (speculative sampling: accept x ~ q with min(1, p(x)/q(x)),
  otherwise resample from norm(max(0, p - q))) returns exactly p for **any** q, as
  long as the q the accept rule reads is the q the token was drawn from. In this
  code, `MarkovSampleStep` writes the step logits it selects from (Gumbel-max at
  the row temperature) into `logits_out`. The accept path then turns exactly those
  logits into q (`SoftmaxTemp`, or block-wise rejection). The patches keep that
  pairing:
  - gemv / foldk / fold8 / head8 / draftmx8 only change the numbers that flow
    into `MarkovSampleStep`. It stores and samples from the same values.
  - tau: q is SoftmaxTemp(kept logits, tau x T), and the token was drawn by Gumbel-max at
    the same tau x T (one FP32 tensor feeds both), so the pairing holds. The target's p stays
    at T (sampling_info). `tests/draftbytes/test_tau.py` has a Monte Carlo law test: max
    abs error 5e-4 to 8e-4 at tau 0.5 / 0.7 / 1.3; a mismatched q is off by 0.21.
  - hotvocab: the fused kernel writes the final hot step logits into a full-width
    row that is -inf off the hot set. The stock `MarkovSampleStep` then runs with a
    zero bias (`x + 0`, and the BF16 round of a BF16 value is the identity) and no
    softcap, which was already applied in the hot kernel with the stock rounding
    order. So the kept logits are exactly the sampled distribution, with q = 0 off
    the set. Off-set tokens are never proposed, and the residual max(0, p - q) = p
    there gives them their full target mass. This is checked in closed form in
    `tests/test_draftbytes.py::RestrictedProposalExactness`. -inf already occurs
    in these logits (the stock vocab limit), so the accept code handles it.
  - Why not put -inf into the base logits and keep the stock step? The stock step
    applies softcap after the add, and tanh(-inf) * cap = -cap. That is a finite
    logit, so off-set tokens would keep a small probability. It would still be
    exact (q is still what is sampled from), but the proposal would be worse.
    The patched step avoids that.
- Cost is acceptance only. FP8 copies perturb q (fold8 rel. error ~2.3 %, see
  below). hotvocab rejects every position whose target token is outside the set.
  About 1 % of misses costs about 1 % of accepted drafts.

## Why the 7 fold reads cannot be fused

Step s needs `latent_s = RNN(state_{s-1}, embed(token_{s-1}), h_s, f)`, and
`token_{s-1}` is sampled from the full-vocab logits of step s-1, which include
that step's fold. The chain is strictly sequential, so no step's fold can be
computed before the previous token exists. Algebra does not help either:
`fold = lm_head @ hidden_out / mult`, so `base + bias = lm_head @ (h_s +
hidden_out(latent_s)) / mult`. Folding the bias into one pass would read the
926 MB head 7 times instead of the 103 MB fold 7 times. The rank-256 fold is
already the cheapest form. The remaining levers are fewer bytes per read (FP8)
and fewer rows (hotvocab). With hotvocab plus fold8 at N = 32768, the fold is
8.4 MB, and it may stay in L2 across the 7 steps (GB10 L2 size not confirmed
here). The rnn_proj weight (7.9 MB, an `nn.Linear`) is also read 7 times per
step (~55 MB). It is left alone.

## Numerics (CPU reference, fold-shaped data: 8192 x 256, 8 latent rows)

Relative L2 error of the FP8 fold vs the BF16 fold:

| mode | rel. error | test bound |
|---|---|---|
| row | 0.0233 | 0.035 |
| g32 | 0.0215 | 0.035 |
| g64 | 0.0224 | - |
| mx | 0.0239 | 0.040 |

e4m3 keeps 3 mantissa bits, so per-element rounding is at most 2^-4 relative.
Finer scale groups barely help because the error comes from the mantissa, not
from range. Per-row is the default (fewest scale bytes).

Kernel math: FP8 to BF16 is exact, and e8m0 scales are powers of two, so the MX
dequant in BF16 is exact. fp32 scales multiply the FP32 partial of each K tile.
The kernel therefore equals `quant.linear_ref` up to FP32 summation order, and
`gpu_check.py` gates within 1 BF16 ulp on >= 99.9 % of outputs.

## Docker additions

```
-v /path/to/repo/overlay:/opt/inkling-overlay:ro
-e PYTHONPATH=/opt/inkling-overlay          # append to any PYTHONPATH the image already sets
-e INKLING_DRAFT_BYTES=gemv,fold8          # for example
-e INKLING_DRAFT_HOTVOCAB_FILE=/opt/inkling-overlay/hot.json   # only with hotvocab:N
```

Check `docker inspect lmsysorg/sglang:inkling-2-mini | grep PYTHONPATH` first.
`-e PYTHONPATH=` replaces the image's value. `sitecustomize.py` arms the overlay
in every interpreter, including the spawned scheduler. It is silent without the
env, and it chains to any other `sitecustomize.py` it shadows. A combined loader
only needs `import inkling_draftbytes; inkling_draftbytes.install()`. It is
idempotent and does not import sglang.

## Test plan

1. **CPU (done here):** `python -m unittest discover -s tests -v`. It covers
   options, the no-op contract, FP8 numerics vs BF16, the patch plumbing end to
   end against CPU stand-ins of the patched modules (`tests/stubs`), and the loud
   failures. With the hot set equal to the whole valid vocab, it checks that the
   hot path is bit-identical to stock (tokens and kept logits).
2. **GPU check (server stopped, ~3-5 GB GPU, nothing pinned):**
   `python3 /opt/inkling-overlay/gpu_check.py` (`--vocab 65536` for a smaller run).
   - Part 1: kernels vs reference (gated).
   - Part 2: the real `Inkling2MTPHead`, real `draft_markov_block` and
     `MarkovSampleStep`, run for every variant against stock. Hot-step structure
     is gated. Each variant is captured in a CUDA graph, and its replay must
     equal eager bit for bit (gated).
   - Part 3: ms of the draft vocab work per step for each variant.
   - Exit 0 = pass.
3. **Microbench:** `python3 /opt/inkling-overlay/bench_draft.py --json /tmp/b.json`.
   It times stock vs ours for the LM head (M 1..56, plus FlashInfer `mm_mxfp8`
   if it runs on sm_121), the fold (M 1..8, with BN/BK/warps sweeps), hot vocab
   per step, and the draft linears (with BN/BK sweeps). Set the tunables from it.
   If `head8`/`draftmx8` stream below ~200 GB/s at M=7/8, the tiles need work
   before a serving arm.
4. **Hot list:** collect output traces on the Spark (SGLang `/generate` with
   `output_ids` is best, since it includes special tokens exactly as drafted, or
   OpenAI responses including `reasoning_content`). Then run
   `python3 build_hotvocab.py --tokenizer /model --out hot.json traces/*.jsonl`
   and read the held-out coverage at 8K/16K/32K/64K. Pick N where held-out
   coverage is >= ~99 %.
5. **Serving arms** (one boot each, same workload, record speedbench c1/c4 tok/s
   and mean accept length from the server log):
   - `stock`, `gemv`, `fold8`, `gemv,fold8`, `draftmx8`, `gemv,fold8,draftmx8`
   - `head8` only if the mxfp8 flag fails
   - `hotvocab:N,fold8` (+`gemv`, +`head8`)

   Gate: T=0 outputs on the eval set equal stock (target path untouched). If any
   differ, bracket them with a stock-vs-stock rerun before blaming the overlay.
   Speed must beat stock beyond the noise band, and the accept-length drop must
   not eat the step-time gain.

## Files

- `inkling_draftbytes/`
  - `__init__.py`: `install()`
  - `_opts.py`: env parsing
  - `_hooks.py`: post-import hooks
  - `patches.py`: patches and `prepare()`
  - `ops.py`: CUDA/Triton vs torch-reference dispatch
  - `_triton.py`: the `w8_mm` and `hot_step` kernels
  - `quant.py`: FP8 packs and reference math
  - `timing.py`: graph-replay timing with L2 flush
- `sitecustomize.py`: single-track loader
- `gpu_check.py`, `bench_draft.py`: GPU checks and microbench (run in the container)
- `build_hotvocab.py`: frequency-ranked hot list and coverage from traces
- `tests/`: CPU unit tests and CPU stand-ins of the patched sglang modules

## Known unknowns

- Triton kernel speed on GB10 at these shapes (fp8 to bf16 convert plus `tl.dot`
  at MP=16) has not been measured. `bench_draft.py` decides.
- cuBLAS rate for the stock draft LM head at M=7 and the stock fold at M=1 is
  unknown, so the `gemv`/`foldk` gains are unknown until benched.
- GB10 L2 size, which decides whether the hot fold stays resident across the 7
  steps.
- The acceptance cost of fold8/head8/draftmx8 noise and of hot-set misses. Only a
  serving run measures it.
