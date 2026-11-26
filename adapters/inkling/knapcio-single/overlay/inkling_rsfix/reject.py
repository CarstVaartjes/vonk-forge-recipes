"""Fixed copies of the final-token (residual / bonus) sampling kernels of
``sglang.kernels.ops.speculative.reject_sampling`` (pinned image text, see _sources).

Stock bug: the CDF pass selects the first lane with ``total_cumsum > target_u``, with no
lane-range condition and no positive-mass condition. ``target_u = coin * norm_sum`` uses a
normalizer accumulated with ``tl.sum`` per tile (or, after block verification, the
residual mass summed over BLOCK_REDUCE tiles), while the pass scans with ``tl.cumsum``.
The scan can end below target_u. A later zero-mass lane then matches: a padded
vocabulary row (Inkling: rows 200058..201023 of the 201024-wide logits have p = 0), or a
masked lane >= VOCAB_SIZE (observed on GPU: bonus == VOCAB_SIZE at V = 24). When nothing
matches, the stock token is VOCAB_SIZE - 1 (classic kernel) or the Result initial value
VOCAB_SIZE - 1 (split kernels), again possibly a padded lane.

Fixed rule (both paths):
- select the first lane with ``(total_cumsum > target_u) & (lane < VOCAB_SIZE) & (val > 0)``;
- no match: the last lane with positive residual mass over the whole row;
- no positive residual lane at all (norm_sum == 0, the degenerate residual): the last lane
  with positive target mass;
- a target row with no positive entry (not a distribution) keeps VOCAB_SIZE - 1.
Nothing else changes: the accept walk, block verification, normalizers, RNG and the
split gate are the stock code. In exact arithmetic the first lane above target_u always
has val > 0, so the law is unchanged; only rounding events select differently.

The split path keeps its CUDA-graph safety: one extra int32 per row (the fallback lane key,
an atomic maximum), written by _split_sum_kernel one launch before _split_cdf_kernel uses
it, like the stock Result; the scratch set is made once per (device, stream, tile count).

Module globals rebound by inkling_rsfix (the stock launcher looks them up at call time):
speculative_sampling_classic_kernel, _split_sum_kernel, _split_cdf_kernel, _split_scratch,
_chain_speculative_sampling_split.
"""

import torch
import triton
import triton.language as tl

# Bound from the stock module by inkling_rsfix.apply() before any kernel here is used
# (no import of the stock module from this file: no import cycle with the post-import hook).
_STOCK_DEPS = ("_split_chain_walk",)

# rsfix's own scratch sets (4 tensors each); the stock _SPLIT_SCRATCH is left unused.
_SPLIT_SCRATCH: dict = {}


@triton.jit
def speculative_sampling_classic_kernel(
    # Pointers
    Predicts,
    AcceptIndex,
    AcceptTokenNum,
    Candidates,
    RetriveIndex,
    UniformSamples,
    UniformSamplesFinal,
    TargetProbs,
    DraftProbs,
    # Strides
    stride_cand_b,
    stride_cand_s,
    stride_idx_b,
    stride_idx_s,
    stride_uni_b,
    stride_uni_s,
    stride_tp_b,
    stride_tp_s,
    stride_tp_v,
    stride_dp_b,
    stride_dp_s,
    stride_dp_v,
    # Constants
    NUM_SLOTS: tl.constexpr,
    VOCAB_SIZE: tl.constexpr,
    BLOCK_V: tl.constexpr,
    BLOCK_VERIFICATION: tl.constexpr = False,
    BLOCK_REDUCE: tl.constexpr = 4096,
):
    pid = tl.program_id(0)
    cur_prob_row = 0

    cand_ptr_base = Candidates + pid * stride_cand_b
    idx_ptr_base = RetriveIndex + pid * stride_idx_b
    uni_ptr_base = UniformSamples + pid * stride_uni_b

    root_global_idx = tl.load(idx_ptr_base + 0 * stride_idx_s)
    tl.store(AcceptIndex + pid * stride_idx_b + 0 * stride_idx_s, root_global_idx)
    last_accepted_global_idx = root_global_idx

    num_accept = 0
    residual_scale = 1.0
    norm_sum = -1.0

    if BLOCK_VERIFICATION:
        # Algorithm 2: https://arxiv.org/abs/2403.10444
        # Keep the short recurrence in registers; only the longest accepted prefix matters.
        prefix_prob = 1.0
        prefix_probs = ()
        for step in tl.static_range(1, NUM_SLOTS):
            draft_token = tl.load(cand_ptr_base + step * stride_cand_s)
            p = tl.load(
                TargetProbs
                + pid * stride_tp_b
                + (step - 1) * stride_tp_s
                + draft_token * stride_tp_v
            )
            q = tl.load(
                DraftProbs
                + pid * stride_dp_b
                + (step - 1) * stride_dp_s
                + draft_token * stride_dp_v
            )
            q_is_prob = (q > 0.0) & (q <= 1.0)
            prefix_prob = tl.where(q_is_prob, tl.minimum(prefix_prob * p / q, 1.0), 0.0)
            prefix_probs += (prefix_prob,)

        for step in tl.static_range(NUM_SLOTS - 1, 0, -1):
            if num_accept == 0:
                prefix_prob = prefix_probs[step - 1]
                coin = tl.load(uni_ptr_base + (step - 1) * stride_uni_s)
                # For normalized distributions h_i <= r_i; reject before scanning.
                if coin < prefix_prob:
                    accept_prob = prefix_prob
                    residual_mass = -1.0
                    if (
                        (step < NUM_SLOTS - 1)
                        & (prefix_prob > 0.0)
                        & (prefix_prob < 1.0)
                    ):
                        residual_mass = 0.0
                        for v_start in range(0, VOCAB_SIZE, BLOCK_REDUCE):
                            v_offsets = v_start + tl.arange(0, BLOCK_REDUCE)
                            mask = v_offsets < VOCAB_SIZE
                            p_next = tl.load(
                                TargetProbs
                                + pid * stride_tp_b
                                + step * stride_tp_s
                                + v_offsets * stride_tp_v,
                                mask=mask,
                                other=0.0,
                            )
                            q_next = tl.load(
                                DraftProbs
                                + pid * stride_dp_b
                                + step * stride_dp_s
                                + v_offsets * stride_dp_v,
                                mask=mask,
                                other=0.0,
                            )
                            residual_mass += tl.sum(
                                tl.maximum(prefix_prob * p_next - q_next, 0.0)
                            )
                        accept_prob = residual_mass / (
                            residual_mass + 1.0 - prefix_prob
                        )
                    if coin < accept_prob:
                        num_accept = step
                        residual_scale = prefix_prob
                        norm_sum = residual_mass

        cur_prob_row = num_accept
        for step in range(1, num_accept + 1):
            draft_token = tl.load(cand_ptr_base + step * stride_cand_s)
            tl.store(Predicts + last_accepted_global_idx, draft_token)
            last_accepted_global_idx = tl.load(idx_ptr_base + step * stride_idx_s)
            tl.store(
                AcceptIndex + pid * stride_idx_b + step * stride_idx_s,
                last_accepted_global_idx,
            )
    else:
        step = 1
        continue_verifying = 1
        # At each token-verification loop entry, cur_prob_row == num_accept == step - 1.
        while (step < NUM_SLOTS) and (continue_verifying == 1):
            draft_token = tl.load(cand_ptr_base + step * stride_cand_s)
            p = tl.load(
                TargetProbs
                + pid * stride_tp_b
                + cur_prob_row * stride_tp_s
                + draft_token * stride_tp_v
            )
            q = tl.load(
                DraftProbs
                + pid * stride_dp_b
                + cur_prob_row * stride_dp_s
                + draft_token * stride_dp_v
            )
            coin = tl.load(uni_ptr_base + (step - 1) * stride_uni_s)
            # A proposal token must have positive probability under its draft distribution.
            q_is_prob = (q > 0.0) & (q <= 1.0)
            if q_is_prob & (coin * q < p):
                num_accept += 1
                cur_prob_row = step
                tl.store(Predicts + last_accepted_global_idx, draft_token)
                last_accepted_global_idx = tl.load(idx_ptr_base + step * stride_idx_s)
                tl.store(
                    AcceptIndex + pid * stride_idx_b + num_accept * stride_idx_s,
                    last_accepted_global_idx,
                )
                step += 1
            else:
                continue_verifying = 0

    tl.store(AcceptTokenNum + pid, num_accept)

    # Final Sampling
    all_drafts_accepted = num_accept == NUM_SLOTS - 1
    coin_final = tl.load(UniformSamplesFinal + pid)

    tp_base_ptr = TargetProbs + (pid * stride_tp_b) + (cur_prob_row * stride_tp_s)
    # DraftProbs has only num_steps rows (TargetProbs has num_steps + 1). When
    # all drafts are accepted cur_prob_row == num_steps is out of bounds for
    # DraftProbs, but the all-accepted branch samples pure target p and never
    # dereferences this pointer; on rejection cur_prob_row <= num_steps - 1.
    dp_base_ptr_safe = DraftProbs + (pid * stride_dp_b) + (cur_prob_row * stride_dp_s)

    # Reuse the selected prefix mass as the correction normalizer.
    if norm_sum < 0.0:
        norm_sum = 0.0
        # Pass 1: Sum
        for v_start in range(0, VOCAB_SIZE, BLOCK_V):
            v_offsets = v_start + tl.arange(0, BLOCK_V)
            mask = v_offsets < VOCAB_SIZE

            p_ptr = tp_base_ptr + v_offsets * stride_tp_v
            p_val = tl.load(p_ptr, mask=mask, other=0.0)

            if all_drafts_accepted:
                val = p_val
            else:
                q_ptr = dp_base_ptr_safe + v_offsets * stride_dp_v
                q_val = tl.load(q_ptr, mask=mask, other=0.0)
                # Treat any non-probability q (NaN, +-inf, negative) as 0: the
                # residual falls back to p. A comparison against NaN is false, so
                # the range test rejects it along with the infinities.
                q_val = tl.where((q_val >= 0.0) & (q_val <= 1.0), q_val, 0.0)
                diff = residual_scale * p_val - q_val
                val = tl.where(diff > 0.0, diff, 0.0)

            norm_sum += tl.sum(val)

    # Pass 2: CDF. rsfix: only in-range lanes with positive residual mass can be
    # selected. target_u comes from norm_sum, a per-tile tl.sum (or the block-verification
    # residual mass, a sum over BLOCK_REDUCE tiles), while the scan below is a tl.cumsum:
    # the scan can end below target_u, and without the lane condition a zero-mass lane
    # (padded vocabulary rows, lanes >= VOCAB_SIZE) could be the first lane above it.
    # No match (target_u at or past the scan's end) emits the last lane with positive
    # residual mass; a residual with no positive lane at all (norm_sum == 0) emits the
    # last lane with positive target mass. Only a target row with no positive entry
    # (not a distribution) keeps the stock VOCAB_SIZE - 1.
    target_u = coin_final * norm_sum
    cum_sum = 0.0
    final_token = VOCAB_SIZE - 1
    found = 0
    # Largest lane key seen: VOCAB_SIZE + v for a positive-residual lane v, v for a
    # positive-target lane, -1 for neither. The scan reaches the end of the row exactly
    # when no lane matches, so the key then covers the whole row.
    fallback_key = -1

    for v_start in range(0, VOCAB_SIZE, BLOCK_V):
        if found == 0:
            v_offsets = v_start + tl.arange(0, BLOCK_V)
            mask = v_offsets < VOCAB_SIZE

            p_ptr = tp_base_ptr + v_offsets * stride_tp_v
            p_val = tl.load(p_ptr, mask=mask, other=0.0)

            if all_drafts_accepted:
                val = p_val
            else:
                q_ptr = dp_base_ptr_safe + v_offsets * stride_dp_v
                q_val = tl.load(q_ptr, mask=mask, other=0.0)
                # Same guard as pass 1.
                q_val = tl.where((q_val >= 0.0) & (q_val <= 1.0), q_val, 0.0)
                diff = residual_scale * p_val - q_val
                val = tl.where(diff > 0.0, diff, 0.0)

            block_cumsum = tl.cumsum(val, axis=0)
            total_cumsum = cum_sum + block_cumsum

            positive = mask & (val > 0.0)
            candidates_mask = (total_cumsum > target_u) & positive
            has_match = tl.max(candidates_mask, axis=0)

            if has_match:
                match_idx = tl.argmax(candidates_mask.to(tl.int32), axis=0)
                final_token = v_start + match_idx
                found = 1

            lane_key = tl.where(
                positive,
                VOCAB_SIZE + v_offsets,
                tl.where(mask & (p_val > 0.0), v_offsets, -1),
            )
            fallback_key = tl.maximum(fallback_key, tl.max(lane_key, axis=0))

            cum_sum += tl.sum(val)

    if found == 0:
        if fallback_key >= VOCAB_SIZE:
            final_token = fallback_key - VOCAB_SIZE
        elif fallback_key >= 0:
            final_token = fallback_key

    tl.store(Predicts + last_accepted_global_idx, final_token)


@triton.jit
def _split_residual_tile(
    TargetProbs,
    DraftProbs,
    pid,
    cur_prob_row,
    all_drafts_accepted,
    v_start,
    stride_tp_b,
    stride_tp_s,
    stride_tp_v,
    stride_dp_b,
    stride_dp_s,
    stride_dp_v,
    VOCAB_SIZE: tl.constexpr,
    BLOCK_V: tl.constexpr,
):
    """One tile of the residual distribution, identical to the tile body of both
    full-vocabulary passes above, including the non-probability guard on q.
    rsfix: also returns the target tile (the degenerate-residual fallback reads it)."""
    residual_scale = 1.0
    tp_base_ptr = TargetProbs + (pid * stride_tp_b) + (cur_prob_row * stride_tp_s)
    dp_base_ptr_safe = DraftProbs + (pid * stride_dp_b) + (cur_prob_row * stride_dp_s)
    v_offsets = v_start + tl.arange(0, BLOCK_V)
    mask = v_offsets < VOCAB_SIZE
    p_val = tl.load(tp_base_ptr + v_offsets * stride_tp_v, mask=mask, other=0.0)
    if all_drafts_accepted:
        val = p_val
    else:
        q_val = tl.load(
            dp_base_ptr_safe + v_offsets * stride_dp_v, mask=mask, other=0.0
        )
        q_val = tl.where((q_val >= 0.0) & (q_val <= 1.0), q_val, 0.0)
        diff = residual_scale * p_val - q_val
        val = tl.where(diff > 0.0, diff, 0.0)
    return val, p_val


@triton.jit
def _split_sum_kernel(
    Predicts,
    AcceptIndex,
    AcceptTokenNum,
    Candidates,
    RetriveIndex,
    UniformSamples,
    TargetProbs,
    DraftProbs,
    TileSums,
    Result,
    Fallback,
    stride_cand_b,
    stride_cand_s,
    stride_idx_b,
    stride_idx_s,
    stride_uni_b,
    stride_uni_s,
    stride_tp_b,
    stride_tp_s,
    stride_tp_v,
    stride_dp_b,
    stride_dp_s,
    stride_dp_v,
    NUM_SLOTS: tl.constexpr,
    VOCAB_SIZE: tl.constexpr,
    BLOCK_V: tl.constexpr,
    NUM_TILES: tl.constexpr,
):
    """Pass 1, split: every program repeats the chain walk and reduces one tile.
    Tile 0 additionally performs the stores the stock kernel makes along the walk."""
    pid = tl.program_id(0)
    tile = tl.program_id(1)
    num_accept, cur_prob_row, _ = _split_chain_walk(
        Predicts,
        AcceptIndex,
        Candidates,
        RetriveIndex,
        UniformSamples,
        TargetProbs,
        DraftProbs,
        pid,
        tile == 0,
        stride_cand_b,
        stride_cand_s,
        stride_idx_b,
        stride_idx_s,
        stride_uni_b,
        stride_uni_s,
        stride_tp_b,
        stride_tp_s,
        stride_tp_v,
        stride_dp_b,
        stride_dp_s,
        stride_dp_v,
        NUM_SLOTS,
    )
    if tile == 0:
        tl.store(AcceptTokenNum + pid, num_accept)
        # rsfix: VOCAB_SIZE = "no tile matched" (every match index is below it); the
        # fallback key starts at -1 = "no positive lane". Both are written here, one
        # launch before _split_cdf_kernel updates them, so they reset in stream order.
        tl.store(Result + pid, VOCAB_SIZE)
        tl.store(Fallback + pid, -1)
    all_drafts_accepted = num_accept == NUM_SLOTS - 1
    val, _ = _split_residual_tile(
        TargetProbs,
        DraftProbs,
        pid,
        cur_prob_row,
        all_drafts_accepted,
        tile * BLOCK_V,
        stride_tp_b,
        stride_tp_s,
        stride_tp_v,
        stride_dp_b,
        stride_dp_s,
        stride_dp_v,
        VOCAB_SIZE,
        BLOCK_V,
    )
    tl.store(TileSums + pid * NUM_TILES + tile, tl.sum(val))


@triton.jit
def _split_cdf_kernel(
    Predicts,
    Candidates,
    RetriveIndex,
    UniformSamples,
    UniformSamplesFinal,
    TargetProbs,
    DraftProbs,
    TileSums,
    Result,
    Fallback,
    Arrived,
    stride_cand_b,
    stride_cand_s,
    stride_idx_b,
    stride_idx_s,
    stride_uni_b,
    stride_uni_s,
    stride_tp_b,
    stride_tp_s,
    stride_tp_v,
    stride_dp_b,
    stride_dp_s,
    stride_dp_v,
    NUM_SLOTS: tl.constexpr,
    VOCAB_SIZE: tl.constexpr,
    BLOCK_V: tl.constexpr,
    NUM_TILES: tl.constexpr,
):
    """Pass 2, split. Each program adds the tile sums in tile order, which gives both
    the normalizer and the running sum in front of its own tile, then applies the stock
    tile step. The stock loop takes the first tile with a match and its first matching
    index, which is the smallest matching index of the row, so keeping the minimum over
    all tiles with an atomic reproduces the stock token.

    rsfix: a lane matches only when it is in range and has positive residual mass, and
    each tile also posts its largest lane key (VOCAB_SIZE + v for the last positive-
    residual lane v, else v for the last positive-target lane, else -1) with an atomic
    maximum. When no tile matched (the scan ended at or below target_u), the last
    arrival emits the row's last positive-residual lane, or, for a residual with no
    positive lane, the last positive-target lane."""
    pid = tl.program_id(0)
    tile = tl.program_id(1)
    num_accept, cur_prob_row, last_idx = _split_chain_walk(
        Predicts,
        Predicts,  # AcceptIndex is unused here: do_store is False
        Candidates,
        RetriveIndex,
        UniformSamples,
        TargetProbs,
        DraftProbs,
        pid,
        False,
        stride_cand_b,
        stride_cand_s,
        stride_idx_b,
        stride_idx_s,
        stride_uni_b,
        stride_uni_s,
        stride_tp_b,
        stride_tp_s,
        stride_tp_v,
        stride_dp_b,
        stride_dp_s,
        stride_dp_v,
        NUM_SLOTS,
    )
    all_drafts_accepted = num_accept == NUM_SLOTS - 1

    # Accumulated in tile order from 0.0, as the stock pass-1 loop accumulates them.
    norm_sum = 0.0
    cum_sum = 0.0
    for t in tl.static_range(NUM_TILES):
        if t == tile:
            cum_sum = norm_sum
        norm_sum += tl.load(TileSums + pid * NUM_TILES + t)

    target_u = tl.load(UniformSamplesFinal + pid) * norm_sum
    v_start = tile * BLOCK_V
    val, p_val = _split_residual_tile(
        TargetProbs,
        DraftProbs,
        pid,
        cur_prob_row,
        all_drafts_accepted,
        v_start,
        stride_tp_b,
        stride_tp_s,
        stride_tp_v,
        stride_dp_b,
        stride_dp_s,
        stride_dp_v,
        VOCAB_SIZE,
        BLOCK_V,
    )
    total_cumsum = cum_sum + tl.cumsum(val, axis=0)
    v_offsets = v_start + tl.arange(0, BLOCK_V)
    mask = v_offsets < VOCAB_SIZE
    positive = mask & (val > 0.0)
    candidates_mask = (total_cumsum > target_u) & positive
    if tl.max(candidates_mask, axis=0):
        match_idx = tl.argmax(candidates_mask.to(tl.int32), axis=0)
        tl.atomic_min(Result + pid, v_start + match_idx, sem="relaxed", scope="gpu")
    lane_key = tl.where(
        positive,
        VOCAB_SIZE + v_offsets,
        tl.where(mask & (p_val > 0.0), v_offsets, -1),
    )
    tile_key = tl.max(lane_key, axis=0)
    if tile_key >= 0:
        tl.atomic_max(Fallback + pid, tile_key, sem="relaxed", scope="gpu")

    # Arrival counter (acq_rel RMW, gpu scope): each program's atomics above are
    # ordered before its arrival, and the last arrival acquires every earlier one
    # through the RMW chain on the counter, so its read below sees all of them. The
    # counter is reset here, so the buffer is reusable in stream order and the launch
    # is safe to capture in a CUDA graph.
    tl.debug_barrier()
    arrived = tl.atomic_add(Arrived + pid, 1, sem="acq_rel", scope="gpu")
    if arrived == NUM_TILES - 1:
        final_token = tl.atomic_add(Result + pid, 0, sem="relaxed", scope="gpu")
        if final_token >= VOCAB_SIZE:
            # No tile matched: the last positive-residual lane, else the last
            # positive-target lane; a row with neither keeps the stock VOCAB_SIZE - 1.
            key = tl.atomic_add(Fallback + pid, 0, sem="relaxed", scope="gpu")
            final_token = tl.where(
                key >= VOCAB_SIZE,
                key - VOCAB_SIZE,
                tl.where(key >= 0, key, VOCAB_SIZE - 1),
            )
        tl.store(Predicts + last_idx, final_token)
        tl.atomic_xchg(Arrived + pid, 0, sem="relaxed", scope="gpu")


def _split_scratch(device, num_tiles):
    """Tile sums, sampled token, arrival counters and (rsfix) the fallback lane key, one
    set per (device, stream, tile count), made once and never replaced, so pointers
    captured in a CUDA graph stay valid. Returns ``None`` while capturing until an eager launch has made the set,
    because a zero fill recorded under capture runs only when the graph replays."""
    stream = torch._C._cuda_getCurrentRawStream(device.index)
    key = (device.index, stream, num_tiles)
    scratch = _SPLIT_SCRATCH.get(key)
    if scratch is None and not torch.cuda.is_current_stream_capturing():
        # One row per multiprocessor; the gate splits only smaller batches.
        rows = torch.cuda.get_device_properties(device).multi_processor_count
        scratch = _SPLIT_SCRATCH.setdefault(
            key,
            (
                torch.empty((rows, num_tiles), dtype=torch.float32, device=device),
                torch.empty((rows,), dtype=torch.int32, device=device),
                # Left at zero; _split_cdf_kernel resets each counter it completes.
                torch.zeros((rows,), dtype=torch.int32, device=device),
                # rsfix: written by _split_sum_kernel before every use.
                torch.empty((rows,), dtype=torch.int32, device=device),
            ),
        )
    return scratch


def _chain_speculative_sampling_split(
    predicts,
    accept_index,
    accept_token_num,
    candidates,
    retrive_index,
    uniform_samples,
    uniform_samples_for_final_sampling,
    target_probs,
    draft_probs,
    *,
    batch_size,
    num_slots,
    vocab_size,
    num_tiles,
    block_v,
    num_warps,
):
    """Same outputs as ``speculative_sampling_classic_kernel`` on the
    non-block-verification path. Returns False if the split path cannot run now."""
    scratch = _split_scratch(device=target_probs.device, num_tiles=num_tiles)
    if scratch is None:
        return False
    tile_sums, result, arrived, fallback = scratch
    strides = (
        candidates.stride(0),
        candidates.stride(1),
        retrive_index.stride(0),
        retrive_index.stride(1),
        uniform_samples.stride(0),
        uniform_samples.stride(1),
        target_probs.stride(0),
        target_probs.stride(1),
        target_probs.stride(2),
        draft_probs.stride(0),
        draft_probs.stride(1),
        draft_probs.stride(2),
    )
    consts = dict(
        NUM_SLOTS=num_slots,
        VOCAB_SIZE=vocab_size,
        BLOCK_V=block_v,
        NUM_TILES=num_tiles,
    )
    grid = (batch_size, num_tiles)
    _split_sum_kernel[grid](
        predicts,
        accept_index,
        accept_token_num,
        candidates,
        retrive_index,
        uniform_samples,
        target_probs,
        draft_probs,
        tile_sums,
        result,
        fallback,
        *strides,
        **consts,
        num_warps=num_warps,
    )
    _split_cdf_kernel[grid](
        predicts,
        candidates,
        retrive_index,
        uniform_samples,
        uniform_samples_for_final_sampling,
        target_probs,
        draft_probs,
        tile_sums,
        result,
        fallback,
        arrived,
        *strides,
        **consts,
        num_warps=num_warps,
    )
    return True
