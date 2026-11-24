# inkling_vcap: verify-width cap for Inkling-2 DSpark (dead-row expert remap)

This overlay is for Inkling-2-Mini on one DGX Spark, with SGLang DSpark static verify
(8 rows per request). Before each target verify it picks a live length `L` for each
request, using only information the draft has already produced. Verify rows `>= L` are
"dead". It overwrites their routed expert ids and weights with those of the request's
anchor row, so the grouped GEMMs read only the experts the live rows need. It also
caps the accept at row `L-1`.

The static `[bs, 8]` shape and every CUDA graph stay as they are. When
`INKLING_VERIFY_CAP` is unset or `off`, the overlay does nothing.

All code was written from scratch against the image's sglang sources (read-only
reference). The overlay patches at import time and edits no engine file.

## Does the engine already have this?

It does not. Read this section before choosing modes.

- `SGLANG_RAGGED_VERIFY_MODE=cap-accept` already caps the accept, through the same
  `cutoff_verify_lens` plumbing used here. It has four problems:
  - It needs a draft confidence head or the online acceptance estimator. The bundled
    Inkling draft has no confidence head.
  - Its budget is planned on the host from confidences that are about two steps old,
    and it is a batch-wide top-k.
  - Leaving `static` mode switches off the static verify epilogue. That is the folded
    in-graph greedy accept/commit (`static_mode` in `DSparkWorkerV2.__init__`), so the
    step glue gets slower.
  - It never touches the MoE routing. The verify still reads the expert union of all
    8 rows, so cap-accept alone only throws away acceptance and saves no bandwidth.
- `SGLANG_RAGGED_VERIFY_MODE=compact` packs fewer tokens. At bs=1 it cannot shrink
  anything: `plans_verify_budget` is false because the floor tier and the full tier
  are the same graph.

So the overlay keeps `static` mode and adds two pieces:

1. A same-step live length that it computes itself.
2. The cutoff in the static epilogue's in-graph greedy accept. Today that accept never
   caps; `_accept` is called without `cutoff_verify_lens`, and the fused tail writes a
   zero trim.

It also adds the MoE dead-row remap, which no engine mode has.

## Modes (environment)

| Variable | Values | Default |
|---|---|---|
| `INKLING_VERIFY_CAP` | `off` (or unset); `fixed:K` (K in 1..8); `conf:T` (alias `confmax:T`); `cost:B1=T1,B2=T2,...`; `policy:/path/policy.json` (see "The `policy:` mode" below). Any mode can be followed by `,min=A`, `,max=B`, `,bsmax=N` | off |
| `INKLING_VERIFY_CAP_REMAP` | `1` = remap dead rows; `0` = cap only (for the exactness A/B) | 1 |
| `INKLING_VERIFY_CAP_FUSED` | `1` = remap is one Triton launch per MoE visit, bit-checked against `index_select` on the device at startup, with fallback; `0` = two `index_select` launches | 1 |
| `INKLING_VERIFY_CAP_LOG_EVERY` | prints a stats line every N decode steps; `0` = no stats kernels at all | 500 |
| `INKLING_VERIFY_CAP_UNION` | `1` = count the distinct experts at each MoE visit in target verify (graph-safe device counters, per layer, in the stats line). When unset or `0`, nothing is recorded. | 0 |
| `INKLING_VERIFY_CAP_FASTPROBS` | `1` = the draft max probabilities (`conf`/`cost`/`policy`) and the sampled accept's draft probabilities come from one or two reads of the BF16 step logits, using the image's split softmax kernels (`fastprobs.py`). It is self-checked against the torch reference at startup and falls back to it on a mismatch. `0` = torch / FlashInfer as before. | 0 |
| `INKLING_VERIFY_CAP_FUSED_ACCEPT` | `1` = plain-temperature sampled accepts (chain or block verification) are computed from the logits, with no materialised probabilities (`fused_accept.py`). It is self-checked at startup and falls back to the stock accept otherwise. | 0 |
| `INKLING_VERIFY_CAP_BV` | Only with `--speculative-use-block-verification` (section "Block verification under the cap"). `mask` = hand the stock kernel the dead drafts as a stopped draft; `stock` = stock kernel over every row, then the usual cap. Both exact. | mask |

The modes work like this:

- **`fixed:K`**: every request verifies rows `0..K-1`, which is `K-1` drafts plus the
  bonus.
- **`conf:T`** works per request, using the draft's kept step logits. It takes the
  per-position **max** draft probability `m_j`; sampled rows use their temperature and
  greedy rows use temperature 1. It computes `surv_j = prod_{i<=j} m_i` and sets
  `L = 1 + #{j : surv_j >= T}` (this count is a leading run because `surv` never
  increases). `L` is then clamped to `[min, max]`.
  - For greedy rows the draft token is the argmax, so the max probability is also the
    proposed token's probability.
  - A cut on the proposed token's own probability is deliberately not offered, because
    it is not exact for sampled requests (see below).
- **`cost:B1=T1,...`** is `conf` with a survival threshold chosen by the decode batch
  size.
  - A step of `bs` requests uses the smallest key `>= bs`. Above the largest key, it
    uses the largest key's threshold.
  - `T = 0` means every row is live. Such a step does no draft-logit work, and once its
    buffers are set it launches nothing.
  - Example: `cost:1=0.30,2=0.25,4=0.15,8=0.0`.

  The idea is to verify the next row only while its expected accepted tokens pay for
  the row's cost. That cost is high at c1, where the expert union is far from
  saturation. At c8 it is near zero, because the union is already near saturation, so
  `T` there should be about 0.
- **`bsmax=N`** (any mode): when `bs > N`, every request verifies all 8 rows. The
  decision is made on the host from the batch size, with no sync.

The intended runtime is the recipe launch: `--tp 1 --ep 1 --attention-backend fa4
... --speculative-algorithm DSPARK --speculative-num-draft-tokens 8` plus the four
`SGLANG_OPT_*` flags, with `SGLANG_RAGGED_VERIFY_MODE` unset (it defaults to `static`).

## Patch points (all in the image's `sglang` package)

| Target | What the overlay does |
|---|---|
| `srt/models/inkling_common/inkling2_moe.py: Inkling2MoE.forward` | Sets a flag only when `forward_batch.forward_mode.is_target_verify()`, so prefill, decode and draft are never touched. |
| `... : Inkling2Gate.forward` | Takes the `(weights, ids)` from `route_topk` and returns them gathered by `src_index[:M]`: `index_select` on rows, graph-safe. With UNION=1 it also adds the distinct-expert count to a device counter. |
| `srt/speculative/dspark_components/dspark_verify.py: TargetVerifyExecutor.run_non_compact` | Runs right before the static verify replay, on the same stream. It writes the step's live lengths and remap index (no host sync), then calls the original. `Inkling2MTPVerifyExecutor` inherits it. |
| `... : DsparkVerifyEpilogue._accept_static_greedy` | Records `self._accept(..., cutoff_verify_lens=verify_lens)` into the verify graph, so the folded greedy accept is capped by the device buffer `verify_lens_buf`. The fused accept tail never caps, so it is bypassed while the overlay is on. |
| `... : accept_draft_tokens` (module function) | When the static path passes `cutoff_layout=None`, injects a cutoff whose `verify_lens` is the live-length buffer. This covers the eager accept (sampled steps, grammar, non-graph). With `block_verification=True` and `INKLING_VERIFY_CAP_BV=mask` it also arms the draft mask for the call. |
| `... : _draft_probs` (module function) | With the mask armed: `logic.mask_dead_draft_rows` on the draft probabilities (see "Block verification under the cap"). Otherwise a pass-through. Required only when block verification is on. |
| `.../dspark_sampling_accept_graph.py: accept_draft_tokens` (imported name) | Same wrapper, for the sampling-accept CUDA-graph region (`SGLANG_OPT_DSV41_SAMPLING_ACCEPT_GRAPH`). |
| `.../dspark_draft.py: DraftBlockProposer.propose` | Stashes the proposal (draft tokens, kept step logits, temperatures, greedy mask) for the verify step that follows. |
| `.../dspark_worker_v2.py: DSparkWorkerV2.__init__` | After the original: validates the setup, allocates buffers, enables the overlay, and prints one startup line. |
| `... : DSparkWorkerV2._forward_decode` | After the original: accumulates the stats (mean L, commit, uncapped commit) on the device. |

The live-length buffer **is** the static epilogue's own `verify_lens_buf` (int64,
`[max_bs]`), so the in-graph accept reads it at replay. That buffer exists for Inkling
because `Inkling2ForConditionalGenerationMTP.folds_static_verify_commit = True`.
Without an epilogue, the overlay uses its own buffer and the accept runs eagerly.

### Fail-loud guards

- Every hook checks that its targets exist, with exact parameter lists, plus a few
  source invariants. Examples: `run_static` slices `verify_lens_buf` and passes it to
  `accept`; `_forward_decode` calls propose, then `begin_static_step`, then
  `run_non_compact`, then `accept_and_finalize`; `Inkling2MoE.forward` calls
  `self.gate(x)` before `self.experts(`. A mismatch raises inside the sglang import.
- At worker init the overlay refuses to start in these cases:
  - Ragged mode is not `static`.
  - `SGLANG_SIMULATE_ACC_LEN` is set.
  - `SGLANG_DSPARK_SIMULATE_FOLD` is set.
  - PP > 1.
  - Vocab-sharded accept.
  - No `Inkling2MoE` in the target.
  - `conf` mode with tp > 1 or attention DP. The live lengths would not be
    rank-synced; `fixed` mode works there.
  - `conf` mode with `SGLANG_DSPARK_USE_GREEDY_DRAFT=1`, which keeps no step logits.
- At the first decode step, `conf` mode refuses a megakernel or int8-markov draft
  sampler, because those may leave greedy rows' step logits unwritten.
- At the first decode step the overlay also refuses to run if the remap was recorded
  into the verify graphs but the capped accept was not.
- A malformed `INKLING_VERIFY_CAP` stops the process at interpreter start (SystemExit
  from sitecustomize).

## Why the output stays exact

1. **Live rows are bit-identical to the uncapped verify.** Row `r` reads only rows
   `<= r`:
   - Attention is causal within the verify chain.
   - The short convolutions (`attn_sconv`, `mlp_sconv`) are causal.
   - Attn-res and loop messages are per token.
   - The MoE is per token. Each token routes on its own gate logits. The NVFP4/MXFP8
     activation scales are per row (`nvfp4_routed_quantize` returns a per-row amax).
     The grouped-GEMM tile is chosen from the static total token count (`_tiles(tokens)`).
     The combine sums each token's own top-k rows.

   Dead rows are later than every live row of their request, so overwriting their
   routing changes nothing a live row reads. Dead rows still compute finite values:
   real experts, real weights, real inputs. So no 0*NaN can leak through masked
   attention tiles.
2. **Acceptance never uses a dead row.** The cap sets `correct = min(correct, L-1)`,
   so the bonus comes from row `<= L-1`. Commit lengths are `<= L`, and every piece
   of bookkeeping is keyed on them: target KV, SWA ring, conv/mamba state commit
   (`commit_lens - 1`), draft KV/hidden commit (`gated = min(commit, verify_lens) * gate`),
   radix cache, FutureMap relay. A capped step therefore looks to the engine exactly
   like a natural rejection at row `L-1`.
3. **Greedy:** every committed token is the target argmax given an exact prefix, which
   is the same chain the uncapped engine produces.
4. **Sampled:** the decision to verify draft `d_j` depends only on `m_0..m_j`. Each
   `m_i` is a function of the draft distribution at position `i` (the context plus
   `d_{<i}`), never of `d_j` itself. So given the accepted prefix, `d_j` is still
   drawn from the draft law `q_j`, and rejection sampling still outputs `p`.
   Truncating is a stopping rule on past information; a truncated step emits the row
   `L-1` output, which is distributed as `p`.

   A cut on `q_j(d_j)` (the proposed token's probability) conditions `d_j` on itself
   and is **biased**. The Monte Carlo test shows both facts: the max-probability rule
   is within 4e-3 of the exact joint, and the token-probability rule is off by
   about 0.037.

   **What the stock sampled kernels do under the cap** (checked against
   `speculative_sampling_classic_kernel`, its split variant, `blockwise_rejection`
   (`_rejection_kernel` + `_insert_kernel`), and the ROCm split accept):
   - The rejection walk runs over every row, dead rows included. `correct` is then
     `min(raw, L-1)`.
   - If the cap trims (`raw > L-1`), the bonus is `candidates[L]`, which is `d_{L-1}`.
     That token sits in the first dead row, but it was *accepted at row L-1* using row
     `L-1`'s live target distribution and coin. In the classic kernel this is
     `predicts[accept_index[L-1]]`. Blockwise uses the `capped < raw` branch.
   - Otherwise (`raw <= L-1`, including the boundary `raw == L-1`), the bonus is the
     residual sample at row `raw`, which is live.
   - So the token emitted at the boundary is the rejection-sampling output of row
     `L-1`, which is distributed as `p_{L-1}`. Dead rows only influence `raw` beyond
     `L-1` and the final sample at a dead row, and both are discarded.
   - This is the standard argument (the output at each verified row is `p`-distributed
     given the prefix) plus a stopping time that is fixed before the target runs.
   - The alternative "stop before `d_{L-1}` and draw a fresh `p_{L-1}` bonus" is equally
     exact. It is not needed.

   `StockCappedSamplingLawTest` checks this with a multi-step Monte Carlo that mirrors
   the stock kernels:
   - Order-1 Markov target and draft, gamma = 3, L from the max-probability rule.
   - Dead rows are given a deliberately wrong target distribution.
   - Boundary cases `raw == L-1` (about 18 % of steps) and trims (about 42 %) are
     asserted to occur.
   - The first three emitted tokens match the exact joint law to within 3.5e-3
     (measured 8e-4). The fresh-bonus variant also passes.
   - A negative control that leaks the dead-row residual into the bonus is off by
     about 0.11, and the test asserts that it is detected.

   Block verification (`--speculative-use-block-verification`, Sun et al.) is exact
   under the cap too. Earlier versions of this README said otherwise and refused it.
   The proof is in the next section.

5. **Full width is inert.** If no batch size can ever get `L < 8` (`fixed:8`, `conf:0`,
   or `cost:` with every `T = 0`), the overlay sets `capping=off (inert)` at startup:
   - no remap is recorded;
   - no cutoff is injected;
   - the stock accept, including the fused tail, is kept.

   Only the stats and the optional union counter remain. The counter is observational:
   it gathers the ids into a separate tensor, and graph-padding rows read as row 0, so
   padding does not inflate the count.

   In capping modes, graph-padding requests (`L = 0`) are deliberately remapped to
   row 0 and capped to 0. Their outputs are discarded by the engine, and this only
   shrinks the union.

## Block verification under the cap

Block verification (Sun et al., "Block Verification Accelerates Speculative Decoding",
ICLR 2025, Algorithm 2) is the engine's `BLOCK_VERIFICATION` branch of
`speculative_sampling_classic_kernel`, which is used for sampled rows when the server
runs with `--speculative-use-block-verification`. With drafts `X_1..X_g`, target rows
`P_0..P_g` and draft rows `Q_0..Q_{g-1}` (row `i` is the law after `i` drafts), the
kernel computes:

```
p_0 = 1,  p_i = min(p_{i-1} P_{i-1}(X_i) / Q_{i-1}(X_i), 1)     (0 if Q_{i-1}(X_i) is not in (0, 1])
r_i = sum_x max(p_i P_i(x) - Q_i(x), 0),  h_i = r_i / (r_i + 1 - p_i)  (i < g),  h_g = p_g
tau = the largest i with coin_i < h_i (0 if none); emit X_1..X_tau and a bonus Y
Y ~ P_g if tau = g, else Y ~ max(p_tau P_tau - Q_tau, 0) normalized
```

Under the cap, request `r` verifies `L` rows (`g = L - 1` live drafts). `L` is a stopping
rule: whether `X_k` is live depends on `X_1..X_{k-1}` only (point 4 of the previous
section). Dead target rows (`>= L`) are wrong because of the expert remap.

**Lemma.** For any target rows (wrong ones included), `P(tau >= i | X_1..X_i) = p_i`.
Proof by backward induction. At `i = g` this is `h_g = p_g`. For `i < g`, the coins are
independent, so

`P(tau >= i | X_1..X_i) = 1 - (1 - h_i)(1 - E[P(tau >= i+1 | X_1..X_{i+1}) | X_1..X_i])`.

By the induction hypothesis the inner term is
`E[p_{i+1} | X_1..X_i] = sum_x Q_i(x) min(p_i P_i(x)/Q_i(x), 1) = sum_x min(p_i P_i(x), Q_i(x)) = p_i - r_i`.
Since `1 - h_i = (1 - p_i) / (r_i + 1 - p_i)`, the whole expression equals `p_i`. The edge
cases `p_i` in {0, 1} also hold. The induction uses only the fact that `P_i` is a
probability vector and that the same `P_i` enters both `p_{i+1}` and `r_i`. The kernel
reads one row for both.

So `P(tau >= i | X_1..X_i) = p_i` and `P(tau = i | X_1..X_i) = p_i - (p_i - r_i) = r_i`.
Both are functions of rows `< i+1` only. For `i <= L-1` that means live rows only.

**Two exact ways to run it under the cap.**

- `INKLING_VERIFY_CAP_BV=stock`: the stock kernel runs over all 8 rows, then the
  existing cap applies (`correct = min(tau, L-1)`). When the cap trims, the bonus is
  `predicts[accept_index[L-1]] = X_L`. What is emitted:
  - `X_1..X_L` with probability `P(X_1..X_L) p_L`, where `p_L` uses live row `L-1`;
  - `X_1..X_k, Y` (`k <= L-1`) with probability `P(X_1..X_k) r_k residual_k(Y)`.

  This is the same law as a block verification that stops after `L-1` drafts, so the
  dead rows cannot change it. Neither can the stopping rule, because `{L = l}` is
  decided by `X_1..X_{l-1}` (the same argument as point 4). The earlier "not exact"
  claim was wrong.
- `INKLING_VERIFY_CAP_BV=mask` (default): before the kernel, `logic.mask_dead_draft_rows`
  zeroes the draft row `L-1` and sets `q(X_j) = 0` for every dead draft. This is exactly
  the draft of a chain that stops: after the stop it emits a token outside the
  vocabulary (target probability 0) with probability 1. Over that augmented draft the
  kernel is the stock fixed-length algorithm, so Sun et al.'s theorem applies directly:
  - `p_i = 0` past the cut, so a dead level is never accepted and never scanned;
  - `h_{L-1} = p_{L-1}`, because its residual mass is `p_{L-1} sum P_{L-1} - 0`;
  - the bonus at `tau = L-1` comes from `P_{L-1}` alone.

  The expected acceptance is the same as with `stock`: `sum_{k<L} E[p_k]` in both. `mask`
  saves the kernel's full-vocabulary scans of dead levels. It also never reads a dead
  target row, so it would stay exact even if dead rows were not finite. Its cost is one
  row per request read and written (`index_select`/`index_copy_`) plus a `[bs, 7]`
  gather/scatter. Graph-safe: fixed shapes, no host sync, reads the live-length buffer.
  The overlay skips the mask on eager steps where every row is live (constant
  full-width steps, e.g. above `bsmax`).

**Tests.** `tests/vcap/test_block_verify.py` mirrors the kernel line by line in float64
(`block_verify_ref`). It uses a 3-token Markov target and draft, garbage dead rows, and
`L` from the max-probability rule, and it checks the joint law of the first three
tokens (300k sequences). All errors are max abs over the 27 cells, with tolerance
3.5e-3:

| arm | max abs error |
|---|---|
| full width | 9.6e-4 |
| `mask`, gamma 3 | 9.3e-4 |
| `mask`, gamma 5 | 1.2e-3 |
| `stock` + cap, gamma 3 | 7.4e-4 |
| `stock` + cap, gamma 5 | 8.7e-4 |

- With `mask` at gamma 3, the steps cover every case: accepted up to the cut, stopped
  inside the live block, `L = 1`, and full width.
- Accepted drafts per step: 0.978 with `mask`, 0.976 with `stock`.
- Negative controls, both detected: a dead-row leak (on a trim, emit the dead row's
  final sample) is off by 0.12; a look-ahead cut (on the proposed token's own
  probability) is off by 0.17.

`MaskTest` and `WrapperTest` cover the mask pattern and the accept/`_draft_probs`
wrappers. `tools/gpu_check_bv.py` repeats the Monte Carlo on the GPU
through the real `accept_draft_tokens` and the stock Triton kernel.

## Draft statistics in fewer bytes (`INKLING_VERIFY_CAP_FASTPROBS=1`)

The R7S profile (thinking, T = 0.8; 2026-11-22) found two vocabulary-wide eager costs
between the graphs. Per-kernel times are from `tools/profile_eager.py`.

- The policy rule's `m_j` (`logic.draft_token_probs`) makes about 8 FP32 passes over
  `[bs*7, 201024]`: a copy, a division, amax twice, sub, exp and sum. That costs
  2.0 ms per step at c8. It runs at every batch size the cap covers (bs <= 16), on
  greedy rows too.
- The accept's draft probabilities go through FlashInfer's softmax after a BF16-to-FP32
  copy of the step logits: 0.28 ms for the copy alone at c8, 0.93 ms at c24.

`fastprobs.py` replaces both with the image's own split kernels (`dspark_accept`'s
`_softmax_temp_stats_kernel` and `_softmax_temp_write_kernel`, the ROCm path of
`softmax_temp_triton`), run on CUDA:

- `m_j` = 1 / sum exp(x/T - max), from one read of the BF16 logits;
- q = softmax(x/T), from two reads of the BF16 logits and one FP32 write.

Both compute the same function as before. They differ only in FP32 summation order:
`m_j` only sets the live length, and q is the draft law up to FP32 rounding, exactly as
FlashInfer's q is. They work with draft tau and with the block-verification mask
(applied after).

## Sampled accept from the logits (`INKLING_VERIFY_CAP_FUSED_ACCEPT=1`)

The stock plain sampled accept materialises `[bs*8, V]` target probabilities and
`[bs*7, V]` draft probabilities: FlashInfer softmax, two reads and one write each,
plus an FP32 copy of the BF16 draft logits. It then reads a few values and one row
pair per request. At c24 that is about 1.07 GB and 5 ms per step (R7S).

`fused_accept.accept_core` reads every logit row about once:

1. the lse of each target row (at T) and each draft row (at `T_d`), from the image's
   split stats kernel;
2. `p(d_j)` and `q(d_j)` as `exp(z/T - lse)`, `[bs, 7]` gathers;
3. the walk:
   - chain: the leading run of `u_j q_j < p_j`;
   - block verification: the prefix probabilities, `r_i` only for the levels that
     can accept (`_mass_kernel` skips every other row), `h_i`, and the backward walk;
4. the bonus, drawn from `max(s P_tau - Q_tau, 0)` by an exponential race
   (`_race_kernel`: argmax of `log w - log E`, `E ~ Exp(1)`, Philox). This selects `x`
   with probability `w(x)/sum w`, in one pass with no normalisation.

Under the cap the walk stops after the `g = L - 1` live drafts, and the bonus at
`tau = g` comes from the target row `g` alone. That is the stopped-chain form, exact
by the stopping-rule argument above, and nothing past row `g` is read. The law is the
stock one: the same formulas, with FP32 rounding of p and q from lse, and a different
exact sampler for the bonus. `cap_trim_lens` is 0, because the walk never verifies a
dead draft, so the stats line's `mean_uncapped` equals `mean_commit` for these steps.

Eligible calls: plain temperature (no top-k / top-p), a non-greedy draft, no vocab
shard, no block-wise rejection noise, CUDA, not inside a graph capture. Everything
else takes the stock path. Mixed batches combine with the stock greedy accept
through `SelectMixedAccept`, as the engine does. Tests:

- `tests/vcap/test_fused_accept.py`: law Monte Carlo through the torch primitives.
  Max abs error 6e-4 to 1.4e-3 for chain and BV, capped and uncapped. A "no stop"
  negative control is off by 0.15.
- `tools/gpu_check_fused_accept.py` (GB10, 2026-11-22, all pass):
  - the self-check passes;
  - race law at V = 201024 across split boundaries: max abs error 0.0036 over 32k
    samples, no off-support picks;
  - `L = 1` with all-`-inf` draft rows: bonus law 0.0037 vs `P_0`;
  - law Monte Carlo through the real `accept_draft_tokens`: chain and BV, capped and
    uncapped, errors 4.8e-4 to 1.1e-3.
  - an independent code review before the first GPU run: five findings, all fixed.

Cold-L2 cost of one accept call, stock (FlashInfer probabilities + the classic
kernel) vs fused, in us:

| bs | 1 | 2 | 3 | 4 | 6 | 8 | 24 |
|---|---|---|---|---|---|---|---|
| chain | 163 / 302 | 317 / 338 | 532 / 382 | 748 / 429 | 1198 / 520 | 1649 / 612 | 5125 / 1372 |
| BV | 272 / 516 | 500 / 534 | 732 / 586 | 950 / 632 | 1432 / 728 | 1893 / 817 | 5628 / 1584 |

The fused path is launch-bound at bs 1-2, so it runs only from
`INKLING_VERIFY_CAP_FUSED_ACCEPT_MIN_BS` (default 3).

## How dead rows shrink the expert union

Each MoE application reads the union of top-8 experts over the step's rows. Dead rows
get the anchor row's ids, so per request the union becomes the union over rows `0..L-1`.
Experts with no tokens get no grouped-GEMM work and their weights are not read.

That saving happens at each of the 48 layer applications:

- 24 stored layers, with 6-17 looped 3 times.
- About 4.48 MB per NVFP4 expert and 8.21 MB per MXFP8 expert, so roughly 230 MB per
  unit of union.

### Rough c1 model

Assumptions:

- About 16.9 ms of non-MoE reads per step: 3.72 GB at 220 GB/s.
- The rest of the measured 60 ms is MoE, which means a union of about 41 at 8 rows.
- The union grows like independent top-8 draws, scaled down to match that 41.
- Per-position acceptance of 0.6.

| L | union | step (ms) | tokens/step | tok/s |
|---|---|---|---|---|
| 8 (today) | 41 | 60 | 2.46 | 41 |
| 5 | 28 | 46 | 2.31 | 50 |
| 4 | 23 | 41 | 2.18 | 53 |
| 3 | 18 | 36 | 1.96 | 55 |
| 2 | 12 | 30 | 1.60 | 54 |

This is only a model. Correlated routing inside one request makes the union grow
*slower* than independent draws, which shrinks the gain. Draft time and launch costs
are ignored. `conf:T` should beat the best fixed K, because it spends rows only where
the draft is confident. Measure the real union with `INKLING_VERIFY_CAP_UNION=1`.

## CUDA-graph safety

- `live_lens` (the epilogue's `verify_lens_buf`) and `src_index` (`[max_bs*8]` int64)
  are static device buffers. They are written by plain tensor ops right before the
  replay, on the same stream, with no `.item()` or `.tolist()`.
- During capture `src_index` is the identity and `verify_lens_buf` holds zeros, so
  capture and warmup outputs are unaffected.
- The graphs record `index_select` (remap), `CapCorrectLen` plus the bonus gather (the
  same in-graph kernels ragged `compact` already captures), and the gated commit.
- Graph padding requests get `L = 0`: they are remapped to row 0 and capped to 0.
- If a step's batch is larger than the captured maximum, the remap (by row count) and
  the cutoff (by request count) both skip, so they stay consistent.
- Interaction with the user's folding flags:
  - `MID_FOLD`: verify ids from the draft graph tail. No effect.
  - `PREPARE_INPUTS_ONCE`: the window and anchor are built in one launch. No effect.
  - `SKIP_PREFILLED_INPUTS`: input_ids and positions are not re-copied. No effect.
  - `INKLING2_VERIFY_REL_BIAS_KERNEL`: an attention kernel. No effect.

  None of them touch these two buffers. Under `TARGET_METADATA_EARLY` the side stream
  stages metadata only, and the replay still runs on the main stream after these
  writes.
- Per-step eager cost:
  - Constant-L steps (`fixed`, above `bsmax`, `T=0`) launch nothing once the buffers
    hold the right values. The overlay only refills them when the batch size or the
    epilogue's `(bs, armed)` state changes.
  - `conf`/`cost` steps with `T>0` take about 15 small launches: softmax stats over
    `[bs*7, V]`, cumprod, and the index rebuild. That is roughly 0.1-0.2 ms at c1.
  - Stats take 3 launches per step for constant L and 5 for dynamic L, plus one host
    sync every `LOG_EVERY` steps. They are off with `LOG_EVERY=0`.
- Inside the graph:
  - Remap: one fused launch per MoE visit, 48 per verify graph. Before this change it
    was two `index_select` launches per visit, 96 in all.
  - Capped greedy accept: about 2 more small kernels than the uncapped one.
  - With `UNION=1`, the counter adds about 6 kernels per visit, about 290 per graph.
    Measure speed with it off.

## Docker additions

```
-v /path/to/repo/overlay:/opt/inkling-overlay:ro
-e PYTHONPATH=/opt/inkling-overlay            # append ":$PYTHONPATH" if the image sets one
-e INKLING_VERIFY_CAP=conf:0.3             # or fixed:4, or off
-e INKLING_VERIFY_CAP_LOG_EVERY=500        # optional
-e INKLING_VERIFY_CAP_UNION=1              # optional, measurement runs only
```

- Do not set `SGLANG_RAGGED_VERIFY_MODE`; it must stay `static`.
- The server must be launched by a normal `python` (not `-I` or `-S`), or
  sitecustomize is ignored.
- The overlay's `sitecustomize.py` runs any `sitecustomize` it shadows.
- Check it is active: the scheduler log should show
  `[inkling-vcap] enabled mode=... patched: ...`. The first decode step then prints
  `[inkling-vcap] first decode step: remap recorded in N MoE calls ...`. N > 0 means
  the remap is inside the verify graphs; expect 48 × (warmups + captures) per captured
  batch size. Stats lines follow, `steps=... mean_live=... mean_commit=...
  mean_uncapped=... [union_mean=...]`. `mean_uncapped` is commit plus the drafts the
  cap trimmed, so `mean_uncapped - mean_commit` is the acceptance the cap gave up.

## Test plan

### CPU (done here)

Run from this directory:

```
python -m unittest discover -s tests -v
```

22 tests, all passing on torch 2.15 CPU. They cover:

- Config parsing.
- Live lengths.
- The remap index (identity, dead rows to anchor, padding rows to 0, partial rebuild).
- Union shrink.
- The greedy cap.
- The sampled stopping rule: a two-token Monte Carlo, plus the multi-step
  stock-kernel Monte Carlo with garbage dead rows, boundary cases and a negative
  control.
- Import-hook and signature guards.
- An end-to-end toy DSpark loop through the real `sitecustomize`. It runs on stand-in
  modules with the engine's names and signatures and a float64 causal toy with
  `Inkling2MoE` layers (one of them looped) and per-position caches. It checks:
  - Greedy output is byte-identical for off, `fixed:{8,3,1}`, `conf`, and `confmax`.
  - It holds with remap on and off.
  - It holds on both the eager accept path and the epilogue-style accept path.
  - The union drops with `fixed:3`.

### On the Spark (c1, greedy, temperature 0)

Use N = 50 prompts of mixed prose and code, 512 output tokens each.

**A. Hooks are inert.** Compare `off` with `fixed:8`. The output must be
byte-identical, and the tok/s gap should be within noise (only the
`index_select`/accept-path overhead differs).

**B. The remap is exact (the key test).** Compare `fixed:3` with `REMAP=0` against
`fixed:3` with `REMAP=1`, then do the same for `fixed:5` and `conf:0.3`.

- The schedule is identical in both runs, so the outputs **must** be byte-identical.
- Any difference means a live row read something a dead row changed. To find it, dump
  the first divergent step's logits and diff by layer.

**C. End-to-end vs baseline.** Compare `off` with `conf:T` for T in
{0.15, 0.25, 0.35, 0.5}, and with `fixed:K` for K in {3, 4, 5}.

- Expect identical text on almost all prompts.
- Changing the step schedule can, in principle, move a position to another row of
  another block. Attention split or reduction order may depend on the block's KV
  length, so a rare near-tie argmax flip is possible. The same is true of the baseline
  against a different acceptance pattern.
- For every divergence, check the top-2 logit gap at the first differing token. Tiny
  means a near-tie (benign); large means a bug, so go back to B.

**D. Speed.** Run your speedbench or RigMark c1 prose and code for off, the best
`fixed:K`, and a few values of `conf:T`. Do this after the warm-up boot (the first run
after a boot is invalid). Record tok/s, `mean_live`, `mean_commit` and `mean_uncapped`.

**E. Union measurement.** Run `fixed:8`, `fixed:4` and `fixed:2` with `UNION=1`, and
read `union_mean` and the per-layer values. This gives the real union growth u(L) that
the model above assumed.

**F. Sampled sanity.** Run `conf:0.3` at temperature 0.7 on a fixed seed. Check there
are no errors and the acceptance stats look sane. If the harness has a quality eval
(qeval), run it against off; distribution exactness cannot be shown by byte identity.

**G. bs > 1.** Run c4 with `conf:0.3`: per-request L, padded graph rows. Repeat B at
c4.

## Measured on the Spark (2026-11-21, before cost/bsmax/fused)

- The overlay loads. The first-step line showed the remap recorded in 3312 MoE calls
  and the capped accept recorded 69 times.
- `fixed:8` with `UNION=1`:
  - Greedy output was byte-identical on 16/16 prompts.
  - Union was 28.3 experts per visit at c1, and about 55-60 at c8 thinking.
  - Step time went from 61.4 to 63.0 ms. Most of that is likely the union counter.
- `fixed:4`:
  - Union 19.2; commit 2.04 against uncapped 2.19.
  - Prose c1 went from 32.3 to 35.4 tok/s (+9.7 %).
  - At c8 thinking it fell from 187.8 to 174.9 tok/s (-7 %), because the union is
    saturated there. `bsmax` and `cost` address this.

## Risks and limits

- **Gain depends on the real union curve.** If one request's 8 rows share most
  experts, dead rows saved little to begin with. Measure E first; it is cheap.
- **Acceptance loss.** Every cap gives up the drafts it trims (`mean_uncapped -
  mean_commit`). Small T or K can lose more than it saves.
- **`conf` cost.** It reads the `[bs, 7, V≈200K]` draft step logits each step: 5.6 MB
  of fp32 temporaries at c1, chunked at 64 rows for larger batches. A fused Triton
  kernel could replace about 10 launches if this ever shows up in a profile.
- **Remap overhead.** It adds 48 tiny kernels per verify graph (fused). Folding the
  remap into `route_topk` itself would remove them, but that means changing an engine
  kernel. The fused Triton kernel was not compiled here (there is no GPU locally). It
  is bit-checked at startup and falls back to `index_select` on any mismatch.
- **Graph capture ordering.** If a future engine captures the target verify graphs
  before `DSparkWorkerV2.__init__` returns, the remap is absent from the graphs. That
  is still exact: the cap is applied eagerly, but nothing is saved. The first-step log
  warns about it.
- **Engine changes.** The patches depend on private methods (`_accept_static_greedy`,
  `verify_lens_buf`, `run_non_compact`). The guards make a mismatch fail at startup
  rather than misbehave.
- **The fused accept tail is bypassed** while the overlay is on. It is only enabled
  under `SGLANG_ENABLE_NV_MEGA_RUNTIME_OPTIMIZATION`, which is off by default.
- **Scope.** TP or attention DP with `conf` mode is refused. EP>1 and the MegaMoE path
  take the same gate output and should work with `fixed:K`, but were not reviewed.

## The `policy:` mode (round 2: calibrated, priced per batch size)

`INKLING_VERIFY_CAP=policy:/opt/inkling-overlay/inkling_vcap/policies/r2-theta004.json`

### Rule

Per request, from the same draft max probabilities `m_j` as `conf`:

```
a_j = calib[greedy|sampled][j][bin(m_j)]      measured P(d_j accepted | d_<j accepted, bin)
S_j = a_0 * ... * a_j                          (estimated P(d_0..d_j all accepted))
d_j is live while S_j >= theta * c_j           c_j = u1(j+2) - u1(j+1)
L   = 1 + leading run, clamped to [min, max]
```

- `calib` is fitted on full-width steplogs, separately for greedy rows (m at T = 1)
  and sampled rows (m at the request temperature). Bins: `m` edges 0.3, 0.5, 0.7,
  0.8, 0.9, 0.95, 0.98, 0.995.
- `c_j` is the c1 marginal expert union of the row that verifies `d_j`
  (`u1` = the measured c1 prefix union 8.0 12.5 16.0 19.0 21.7 24.0 26.0 27.9): late
  rows are cheaper than early ones, which a flat `conf` threshold ignores.
- `theta` is a price in "expected accepted tokens per expert of union". It is looked
  up by batch size like `cost:`; `theta = 0` means full width with no draft-logit
  work. The replay found ONE theta (0.04) best at every batch size (per-bs tuning
  gave 0.033-0.053 and no extra gain), so the shipped file has a single key.
- Optional batch-aware rounds (`"pool"`, `"rounds"`): request r's price is scaled by
  `P_r = prod_{s != r} (1 - u1(L_s) / E)`, the chance that its extra experts are
  not already read for another request, iterated K times from `L = 1`. The replay
  found no gain over the per-request rule (within +-0.3 %), so the shipped file
  uses `rounds = 0`. The option stays (tested) for later cost models.
- Stateless: no per-request history (prev_full added nothing in the replay).
  Graph-safe as `conf`: eager tensor ops before the verify replay, no host sync;
  about 8 more small launches than `conf` per step.

### Exactness

Same argument as `conf` (section "Why the output stays exact", point 4): `S_j` is a
function of `m_0..m_j`, and `m_j` depends on `d_<j` only, so `{L = l}` is decided by
`d_0..d_{l-2}`: a stopping rule. With the batch-aware rounds the other requests'
widths enter too; they are independent of this request's drafts, and the rounds are
monotone from `L = 1`, so each round only reads this request's `S` up to its final
cut. `tests/vcap/test_policy.py` checks this three ways:

- perturbing a request's `m_j` for every `j >= L_r` never changes any width of the
  batch (with rounds on);
- a Monte Carlo of the stock capped sampled accept with two interacting requests
  per batch (gamma 3, dead rows given a garbage target, about 13 % trims and 4 %
  boundary cases; shuffling the partner request changes about 40 % of the widths)
  matches the exact joint
  law of the first three tokens within 3.5e-3; a look-ahead rule is detected (> 1e-2);
- the device rule equals the replay's numpy reference decision for decision.

### Replay (round 2 data, stock engine, full width + union counter)

Data: `R2SL-s06` (Spark_06, 2026-11-21): 36,201 request-steps; thinking c1/c4/c8
(sampled T = 0.8), decode table prose+code c1-c16 (greedy), speedbench quick c1.
Replay tool: `bsreplay.py` (development tool, not shipped). Step time `T = F(bs) + 0.97 ms x U`
(`U` = expert union per MoE visit; `F(1) = 33.3`, `F(8) = 59.6` ms from the c1 and
c8 profiles, linear in between), plus 0.05 + 0.13 x bs ms for the draft-logit
statistics of a dynamic rule. Mixed widths: the step's own measured fixed-width
union curve, read at the batch's mean c1 union (exact for uniform widths).
2-fold cross-fit (calibration and theta fitted on the other fold), 95 % CIs from a
paired cluster bootstrap (request at bs 1, 32-step blocks at bs > 1).

Gain over the shipped `cost:1=0.30,2=0.25,4=0.15,8=0.0`, all workloads mixed:

| bs | steps | `policy:` theta 0.04 | `cost:1=0.30,24=0.25` | oracle |
|---|---|---|---|---|
| 1 | 8054 | +0.5 % [+0.3, +0.7] | 0 | +15.7 % |
| 2 | 64 | +0.7 % [-0.5, +2.6] | 0 | +17.6 % |
| 4 | 1840 | +2.2 % [+1.8, +2.6] | +1.0 % | +18.6 % |
| 6 | 123 | +3.9 % [+2.1, +8.2] | +2.9 % | +18.7 % |
| 7 | 524 | +2.5 % [+1.8, +3.4] | +1.7 % | +17.5 % |
| 8 | 1101 | +5.0 % [+3.5, +6.3] | +4.2 % | +19.4 % |
| 16 | 365 (greedy only) | +5.9 % [+4.5, +7.3] | +6.0 % | +20.5 % |

By workload, `policy:` theta 0.04: thinking (sampled) c4 +1.5 %, c8 +1.8 %; greedy
decode c4 +3.8 %, c8 +10.4 %, c16 +5.9 %. The greedy decode load accepts only
1.2-1.3 drafts per step at c4-c16, so its late rows rarely pay. Sensitivity (union
model "mean row count", F +15 %, b = 1.19 at bs >= 4, overhead x2): the gain stays
positive at every bs with enough data (bs 8: +3.5 to +7.2 %). bs 9-15 and 24 are
not in the data; theta 0.04 is extrapolated.

## Tests (round 2)

`python -m unittest discover -s tests/vcap` from the overlay root: 33 tests (the 22
above plus 11 in `test_policy.py`: JSON format and validation, env grammar, the
shipped policy file, device rule vs replay reference, round monotonicity, the
width-ignores-confidences-past-the-cut property, the two-request sampled Monte
Carlo and its negative control, `plan_step`, and greedy byte identity end to end
for a per-request policy, a batch-aware policy and an inert policy on both accept
paths with remap on and off). `test_policy.py` imports `bsreplay.py` from the
overlay root for the reference rule.

## Tests (round 3, sampled path)

Running `python -m unittest discover -s tests/vcap` from the overlay root gives 51 tests.
The round-3 files:

- `test_block_verify.py`: kernel mirror, law Monte Carlo for `mask` and `stock`, negative
  controls, the mask pattern, and the wrappers.
- `test_fastprobs.py`: the combine math against the torch reference.
- `test_fused_accept.py`: law Monte Carlo for chain and BV, capped and uncapped, plus the
  no-stop negative control.

GPU checks, run in the image: `tools/gpu_check_bv.py`, `tools/gpu_check_fastprobs.py`,
`tools/gpu_check_fused_accept.py`. The last one includes the combined stacks: FASTPROBS q
with the BV mask, and FASTPROBS with fused accept and BV, with the draft at tau x T. All
passed on GB10 on 2026-11-22 (errors at most 1.5e-3).

`tools/bench_fused_accept.py` times stock vs fused per batch size.
`tools/profile_eager.py` lists a trace's eager kernels together with their CPU stacks.
