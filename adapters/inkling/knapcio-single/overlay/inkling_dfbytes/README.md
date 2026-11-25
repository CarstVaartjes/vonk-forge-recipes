# inkling_dfbytes: draft-side byte cuts for the DFlash2 drafter

Import-time overlay for the DFLASH worker (`SPEC=dflash2`). It changes only what the drafter proposes.

| switch | value | effect |
|---|---|---|
| `INKLING_DFLASH_BYTES` | unset / `0` | nothing is patched |
| | `mx8` | MXFP8 copies of the drafter's decoder linears, served at <= 16 rows (c1 / c2) |
| | `hot8` / `hot16` | the selector's top-16 candidates from a hot-vocabulary plane of the target head |
| `INKLING_DFLASH_HOT_N` | default 65536 | size of the hot plane (first N ids of `INKLING_DRAFT_HOTVOCAB_FILE`) |

`scripts/launch.sh` sets `DFLASH_BYTES=mx8,hot8` by default with `SPEC=dflash2` and the overlay on.

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

## Exactness

Draft-only. The selector's q is computed from the candidates the path is drawn from, so the sampled verification
keeps the target law; greedy output is the target argmax chain. Proposals and acceptance change and were measured.

## Measured (GB10, 2026-11-24)

- `tools/bench_dfbytes.py`, cold L2 in a CUDA graph: the 20 drafter linears at 8 rows 3.17 -> 1.99 ms; the candidate
  projection at 7 / 56 / 168 rows 4.02 / 4.39 / 5.15 ms (stock) -> 0.68 / 0.77 / 1.40 ms (64K `hot8`). On 5,983
  captured target rows, 64K `hot8` holds 96.5 % of the stock top-16 and the stock top-1 in its top-16 on 99.6 %.
- Serving A/B against the stock drafter: speedbench quick +8.4 %, prose c1 +6.5 %, code c1 +4.4 %, prose c8 +6 %;
  greedy hashes equal except one known near-tie prompt.
