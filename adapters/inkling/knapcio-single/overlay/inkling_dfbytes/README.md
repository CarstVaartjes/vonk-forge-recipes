# inkling_dfbytes: draft-side byte cuts for the DFlash2 drafter

Import-time overlay for the DFLASH worker (`SPEC=dflash2`). It changes only what the drafter proposes.

| switch | value | effect |
|---|---|---|
| `INKLING_DFLASH_BYTES` | unset / `0` | nothing is patched |
| | `mx8` | MXFP8 copies of the drafter's decoder linears, served at <= 16 rows (c1 / c2) |
| | `hot8` / `hot16` | the selector's top-16 candidates from a hot-vocabulary plane of the target head |
| `INKLING_DFLASH_HOT_N` | default 65536 | size of the hot plane (first N ids of `INKLING_DRAFT_HOTVOCAB_FILE`) |
| `INKLING_DFLASH_CTX8` | `0` / `1` | `1`: the context projections on b12x MXFP8 (`ctx8.py`, below); independent of `INKLING_DFLASH_BYTES` |
| `INKLING_DFLASH_CTX8_MAX_M` | default 256 | largest row count `ctx8` serves; larger calls (long prefills) keep BF16 |

`scripts/launch.sh` sets `DFLASH_BYTES=mx8,mx8w,hot8` and `DFLASH_CTX8=1` by default with `SPEC=dflash2` and the overlay on.

## Parts

- `mx8`: after `DFlashDraftModel.load_weights`, every BF16 unquantized linear under `model.layers` gets an MXFP8 copy
  (one E8M0 scale per 32 inputs) and `inkling_draftbytes`' W8 linear method. At <= 16 rows it runs the CUDA
  weight-streaming GEMV; larger batches keep BF16 (the Triton W8 kernel does not beat cuBLAS there).
  `inkling_draftbytes` itself is not armed: only its kernels are configured. The context K/V projections keep their
  BF16 weights.
- `hot8` / `hot16`: `DFlash2DraftModel.compute_candidates` takes the top-16 over a plane holding the target head's
  rows for the hot ids (FP8 row-scaled for `hot8`, 151 MB; BF16 for `hot16`, 302 MB) instead of the full BF16 head
  (922 MB). The plane is built from the target `lm_head` in `DFlashWorkerV2._maybe_build_draft_sampler`, before the
  draft graph is captured; it is skipped under TP > 1, the fused candidate merge and the megakernel head, and a moved
  head (online weight update) is refused.

- `ctx8`: after every verify the worker appends the committed rows' target features to the draft window, eagerly:
  the context `fc` (9216 -> 2304, 42.5 MB BF16) then `hidden_norm`, and the five layers' stacked K/V projection
  (2304 -> 10240, 47.2 MB BF16) then the fused K RMSNorm + RoPE and one prefix-valid ring write per layer. With
  `INKLING_DFLASH_CTX8=1` the two GEMMs read MXFP8 copies (the `mx8` format, scales swizzled for b12x; 21.9 + 24.3 MB)
  through `flashinfer.mxfp8_quantize` + `mm_mxfp8(backend="b12x")` for 1-256 rows. `DFlashDraftModel.project_target_hidden`
  and `FusedKVMaterializeHelper.materialize` are wrapped; on the MXFP8 path the latter runs the stock body line by line
  with only its `torch.mm` replaced. The three stock functions it relies on are pinned by source sha256 (a different
  image refuses at start-up). One row count per b12x tile plan (1-256) is run at load, so no b12x kernel compiles while serving.
  Any other set-up or kernel error leaves that projection on BF16 (logged). The copies are derived weights: the
  drafter and the target declare `_derived_weight_cache_error`, so SGLang refuses online weight updates (before any
  runner is written) while ctx8 is on. With SGLang's aux CUDA graphs (`SGLANG_OPT_DFLASH_AUX_GRAPH`) requested, ctx8 is not armed. The server log shows
  `ctx8: first fc call on b12x` / `first K/V call` once the path is in use and `ctx8: 1000 fc calls on b12x` /
  `1000 K/V calls` once decode appends run on it; `stock BF16 from now on` (ctx8 fell back) and SGLang's
  `DFLASH fused KV ... failed` / `fused KV materialization disabled` (the stock helper fell back) must not appear.
  Only ValueError / NotImplementedError / TypeError / ImportError / AttributeError / KeyError / AssertionError whose text
  names no CUDA fault fall back; anything else stops the server (`DFLASH_CTX8=0` to run without it). b12x runs its
  default tactic (`autotune(False, skip_ops={"mxfp8_gemm"})`), so the load-time warm-up covers every kernel it uses. `tools/gpu_check_ctx8.py` checks the copy bit for bit (GEMM swapped back), the error
  against BF16, CUDA graph replay and the timing; `../tools/ctx8_agree.py` measures the drafter's token agreement on two
  steplog captures.

## Exactness

Draft-only. The selector's q is computed from the candidates the path is drawn from, so the sampled verification
keeps the target law; greedy output is the target argmax chain. Proposals and acceptance change and were measured.

## Measured (GB10, 2026-11-24)

- `tools/bench_dfbytes.py`, cold L2 in a CUDA graph: the 20 drafter linears at 8 rows 3.17 -> 1.99 ms; the candidate
  projection at 7 / 56 / 168 rows 4.02 / 4.39 / 5.15 ms (stock) -> 0.68 / 0.77 / 1.40 ms (64K `hot8`). On 5,983
  captured target rows, 64K `hot8` holds 96.5 % of the stock top-16 and the stock top-1 in its top-16 on 99.6 %.
- Serving A/B against the stock drafter: speedbench quick +8.4 %, prose c1 +6.5 %, code c1 +4.4 %, prose c8 +6 %;
  greedy hashes equal except one known near-tie prompt.

## ctx8 measured (box B, 2026-11-25, `tools/gpu_check_ctx8.py`, real drafter weights, no server)

- Exactness of the copy: with the GEMM swapped back to `torch.mm`, the ctx8 `materialize` writes K/V bytes identical
  to the stock one at 1 / 8 / 32 / 64 / 192 / 256 rows; 257 rows take the stock path bit for bit; the b12x GEMM writes
  the stock workspace rows in place (1 / 3 / 127 / 129 / 255 rows); CUDA graph replay equals eager bitwise.
- Error against BF16 (relative Frobenius / lowest row cosine, every row count): fc output 3.6-3.8 % / 0.9992; K and V
  from the K/V GEMM alone 3.5-3.9 %; end to end from the target features 5.1-5.5 % / 0.998.
- Copies: 47.7 MB (fc 22.6 + K/V 25.1) next to the 89.7 MB of BF16 weights. Four b12x tile plans; no kernel compiles
  after the load-time warm-up (rows 1-256, both call shapes).
- Eager fc + K/V append with ring writes, L2 flushed: 0.447 -> 0.269 ms at 8 rows (-0.179 ms), -0.191 / -0.183 /
  -0.181 ms at 32 / 64 / 192 rows. In a CUDA graph: 0.403 -> 0.225 ms at 8 rows (-0.178 ms), -0.193 / -0.190 / -0.183
  ms at 32 / 64 / 192 rows.

ctx8 in serving, box A, 2026-11-25: one steplog capture per arm (full-width verify), speedbench quick, greedy,
`tools/ctx8_agree.py`.
- Tokens per step: 3.250 with ctx8 against 3.247 without (1,613 and 1,621 greedy steps).
- Matched steps: 1,349 with the same prompt and context and the same token stream so far, out of 88.7 % of each
  arm's steps. On these, the drafted tokens agree on 97.1 / 93.1 / 90.6 / 87.5 / 85.4 / 82.4 / 79.8 % of slots 1-7
  and on 70.6 % of whole blocks. Accepted drafts per step: 2.137 against 2.136.
- Greedy outputs: identical on 28 / 30 prompts. The other two (longctx-09, agentic-02) first diverge at output tokens
  78 and 140. A change in acceptance moves the verify block boundaries, which changes the floating-point order of a
  verified row: the verify-schedule class the cap and the draft-tail snapshots also show.
- Gate for ctx8: greedy differences limited to this verify-schedule floating-point class, and no drop in the quality
  bracket.
