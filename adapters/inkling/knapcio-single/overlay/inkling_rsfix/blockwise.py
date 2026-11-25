"""SGLang PR #51692 ("Stabilize blockwise rejection sampling under logit shifts") for
``sglang.kernels.ops.speculative.dspark.blockwise_rejection`` (pinned image text, see
_sources). The path runs only with SGLANG_OPT_DSV41_BLOCKWISE_REJECTION=1 (default: the
value of SGLANG_ENABLE_NV_MEGA_RUNTIME_OPTIMIZATION, off), for sampled chain verifies
without block verification, top-k or top-p; the Inkling recipe does not set it.

Stock bugs (the PR's description):
- ``_global_lse`` returns ``max + log(sum)``; adding the small log-sum to a large row
  maximum loses its low bits before the caller subtracts the sum from a token logit.
- accept tests ``log_p > log(u) + log_q``; with a coin just below 1, log(u) is tiny and
  ``log(u) + log_q`` rounds back to log_q, so equal laws (p == q) reject. The rejected row
  then resamples a residual that may hold only zero-target-mass tokens.
- the bonus / one-hot residual Gumbel race adds noise to large uncentred scores, which can
  round distinct draws to a tie.

Fix (the PR's code): keep (max, log-sum) apart through accept and resample, accept iff
``log(u) < log_p - log_q``, centre the bonus race by the target row max (the stats pass now
also reads the bonus row's max), reuse the last verified row's statistics on rejection.

Not fully exact: FP32 normalization can cancel the only positive residual lane
(e.g. target logits [-inf, 0, -inf], draft [-inf, 0, -32], proposal 2), and the all -inf
race then returns lane 0. Needs a stable residual and a positive-target fallback before
this route is enabled; the Inkling recipe does not use it.

Rebound by inkling_rsfix: _log_normalizer_parts (new), _stats_kernel, _rejection_kernel,
_resample_kernel, blockwise_rejection_triton (BlockwiseRejection.execute looks it up at
call time), blockwise_rejection_torch.
"""

from __future__ import annotations

from typing import Optional

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

# Bound from the stock module by inkling_rsfix.apply() before any kernel here is used
# (no import of the stock module from this file: no import cycle with the post-import hook).
_STOCK_DEPS = (
    "_RESAMPLE_BLOCK",
    "_STATS_BLOCK",
    "_TL_RAND_MIN",
    "RejectionNoise",
    "_insert_kernel",
    "_log1p_neg_stable",
    "_murmur3_uniform32",
    "_step_seed",
)


@triton.jit
def _log_normalizer_parts(
    max_ptr, sumexp_ptr, row, NUM_BLOCKS: tl.constexpr, PADDED: tl.constexpr
):
    blocks = tl.arange(0, PADDED)
    mask = blocks < NUM_BLOCKS
    maxes = tl.load(max_ptr + row * NUM_BLOCKS + blocks, mask=mask, other=float("-inf"))
    sumexps = tl.load(sumexp_ptr + row * NUM_BLOCKS + blocks, mask=mask, other=0.0)
    global_max = tl.max(maxes, axis=0)
    # Keep the shift separate: adding a small log sum to a large maximum
    # can round it away before the caller subtracts it from a token logit.
    return global_max, tl.log(tl.sum(sumexps * tl.exp(maxes - global_max)))



@triton.jit
def _stats_kernel(
    t_max_ptr,
    t_sumexp_ptr,
    d_max_ptr,
    d_sumexp_ptr,
    target_logits_ptr,
    target_stride,
    draft_logits_ptr,
    draft_stride_0,
    draft_stride_1,
    target_temp_ptr,
    draft_temp_ptr,
    target_vocab,
    draft_vocab,
    STRIDE: tl.constexpr,
    NUM_BLOCKS: tl.constexpr,
    BLOCK: tl.constexpr,
    HAS_DRAFT_LOGITS: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    req = row // STRIDE
    step = row % STRIDE
    block_idx = tl.program_id(1)
    offs = block_idx * BLOCK + tl.arange(0, BLOCK)
    t_temp = tl.load(target_temp_ptr + req).to(tl.float32)
    t = (
        tl.load(
            target_logits_ptr + row * target_stride + offs,
            mask=offs < target_vocab,
            other=float("-inf"),
        ).to(tl.float32)
        / t_temp
    )
    t_max = tl.max(t, axis=0)
    tl.store(t_max_ptr + row * NUM_BLOCKS + block_idx, t_max)
    if step == STRIDE - 1:
        # The bonus Gumbel race only needs the target row maximum.
        return
    tl.store(
        t_sumexp_ptr + row * NUM_BLOCKS + block_idx,
        tl.where(t_max > float("-inf"), tl.sum(tl.exp(t - t_max)), 0.0),
    )
    if HAS_DRAFT_LOGITS:
        d_temp = tl.load(draft_temp_ptr + req).to(tl.float32)
        d = (
            tl.load(
                draft_logits_ptr + req * draft_stride_0 + step * draft_stride_1 + offs,
                mask=offs < draft_vocab,
                other=float("-inf"),
            ).to(tl.float32)
            / d_temp
        )
        d_max = tl.max(d, axis=0)
        tl.store(d_max_ptr + row * NUM_BLOCKS + block_idx, d_max)
        tl.store(
            d_sumexp_ptr + row * NUM_BLOCKS + block_idx,
            tl.where(d_max > float("-inf"), tl.sum(tl.exp(d - d_max)), 0.0),
        )



@triton.jit
def _rejection_kernel(
    correct_len_ptr,
    raw_len_ptr,
    trim_ptr,
    row_stats_ptr,
    candidates_ptr,
    candidates_stride,
    verify_lens_ptr,
    t_max_ptr,
    t_sumexp_ptr,
    d_max_ptr,
    d_sumexp_ptr,
    target_logits_ptr,
    target_stride,
    draft_logits_ptr,
    draft_stride_0,
    draft_stride_1,
    target_temp_ptr,
    draft_temp_ptr,
    target_vocab,
    draft_vocab,
    bs,
    key,
    counter_ptr,
    STRIDE: tl.constexpr,
    NUM_BLOCKS: tl.constexpr,
    PADDED_BLOCKS: tl.constexpr,
    HAS_DRAFT_LOGITS: tl.constexpr,
    HAS_CUTOFF: tl.constexpr,
):
    req = tl.program_id(0).to(tl.int64)
    seed = _step_seed(key, counter_ptr)
    t_temp = tl.load(target_temp_ptr + req).to(tl.float32)
    d_temp = 1.0
    if HAS_DRAFT_LOGITS:
        d_temp = tl.load(draft_temp_ptr + req).to(tl.float32)
    accepted = tl.zeros((), tl.int32)
    verifying = True
    t_max = 0.0
    t_log_sum = 0.0
    d_max = 0.0
    d_log_sum = 0.0
    for i in range(STRIDE - 1):
        if verifying:
            row = req * STRIDE + i
            token = tl.load(candidates_ptr + req * candidates_stride + i + 1).to(
                tl.int64
            )
            valid = (token >= 0) & (token < target_vocab)
            token = tl.minimum(tl.maximum(token, 0), target_vocab - 1)
            t_max, t_log_sum = _log_normalizer_parts(
                t_max_ptr, t_sumexp_ptr, row, NUM_BLOCKS, PADDED_BLOCKS
            )
            log_p = (
                tl.load(target_logits_ptr + row * target_stride + token).to(tl.float32)
                / t_temp
                - t_max
                - t_log_sum
            )
            log_q = 0.0
            if HAS_DRAFT_LOGITS:
                d_max, d_log_sum = _log_normalizer_parts(
                    d_max_ptr, d_sumexp_ptr, row, NUM_BLOCKS, PADDED_BLOCKS
                )
                log_q = (
                    tl.load(
                        draft_logits_ptr
                        + req * draft_stride_0
                        + i * draft_stride_1
                        + token,
                        mask=token < draft_vocab,
                        other=float("-inf"),
                    ).to(tl.float32)
                    / d_temp
                    - d_max
                    - d_log_sum
                )
            u = tl.maximum(tl.rand(seed, (req * STRIDE + i).to(tl.int32)), _TL_RAND_MIN)
            # A proposal must have positive draft probability (classic kernel's guard).
            # Form the log ratio first so equal laws always accept a coin < 1.
            ok = valid & (log_q > float("-inf")) & (tl.log(u) < log_p - log_q)
            verifying = ok
            accepted += ok.to(tl.int32)
    tl.store(raw_len_ptr + req, accepted)
    capped = accepted
    if HAS_CUTOFF:
        cap = tl.load(verify_lens_ptr + req).to(tl.int32) - 1
        capped = tl.minimum(accepted, cap)
    tl.store(correct_len_ptr + req, capped)
    tl.store(trim_ptr + req, accepted - capped)
    # On rejection, the loop already computed both parts for the selected row.
    if accepted == STRIDE - 1:
        blocks = tl.arange(0, PADDED_BLOCKS)
        row = req * STRIDE + accepted
        t_max = tl.max(
            tl.load(
                t_max_ptr + row * NUM_BLOCKS + blocks,
                mask=blocks < NUM_BLOCKS,
                other=float("-inf"),
            ),
            axis=0,
        )
    tl.store(row_stats_ptr + req, t_max)
    tl.store(row_stats_ptr + bs + req, t_log_sum)
    if HAS_DRAFT_LOGITS:
        tl.store(row_stats_ptr + 2 * bs + req, d_max)
        tl.store(row_stats_ptr + 3 * bs + req, d_log_sum)



@triton.jit
def _resample_kernel(
    res_max_ptr,
    res_arg_ptr,
    raw_len_ptr,
    row_stats_ptr,
    candidates_ptr,
    candidates_stride,
    target_logits_ptr,
    target_stride,
    draft_logits_ptr,
    draft_stride_0,
    draft_stride_1,
    target_temp_ptr,
    draft_temp_ptr,
    target_vocab,
    draft_vocab,
    bs,
    key,
    counter_ptr,
    STRIDE: tl.constexpr,
    NUM_BLOCKS: tl.constexpr,
    BLOCK: tl.constexpr,
    HAS_DRAFT_LOGITS: tl.constexpr,
):
    req = tl.program_id(0).to(tl.int64)
    block_idx = tl.program_id(1)
    seed = _step_seed(key, counter_ptr)
    r = tl.load(raw_len_ptr + req).to(tl.int64)
    row = req * STRIDE + r
    is_bonus = r == STRIDE - 1
    offs = block_idx * BLOCK + tl.arange(0, BLOCK)
    mask = offs < target_vocab
    t_temp = tl.load(target_temp_ptr + req).to(tl.float32)
    t = (
        tl.load(
            target_logits_ptr + row * target_stride + offs,
            mask=mask,
            other=float("-inf"),
        ).to(tl.float32)
        / t_temp
    )
    # Use the same shift in every vocab block before adding Gumbel noise.
    t = t - tl.load(row_stats_ptr + req)
    if is_bonus:
        residual = t
    else:
        rejected = tl.load(candidates_ptr + req * candidates_stride + r + 1)
        if HAS_DRAFT_LOGITS:
            d_temp = tl.load(draft_temp_ptr + req).to(tl.float32)
            log_p = t - tl.load(row_stats_ptr + bs + req)
            d = (
                tl.load(
                    draft_logits_ptr + req * draft_stride_0 + r * draft_stride_1 + offs,
                    mask=offs < draft_vocab,
                    other=float("-inf"),
                ).to(tl.float32)
                / d_temp
            )
            log_q = (d - tl.load(row_stats_ptr + 2 * bs + req)) - tl.load(
                row_stats_ptr + 3 * bs + req
            )
            # log max(p - q, 0) = log p + log1p(-q / p), -inf where q >= p.
            ratio = tl.exp(log_q - log_p)
            residual = tl.where(
                ratio < 1.0, log_p + libdevice.log1p(-ratio), float("-inf")
            ).to(tl.float32)
        else:
            # One-hot draft: the residual is p without the rejected token.
            residual = tl.where(offs != rejected, t, float("-inf"))
    u = _murmur3_uniform32(seed, req, offs)
    noised = tl.where(mask, residual - tl.log(-_log1p_neg_stable(u)), float("-inf"))
    value, idx = tl.max(noised, axis=0, return_indices=True)
    tl.store(res_max_ptr + req * NUM_BLOCKS + block_idx, value)
    tl.store(res_arg_ptr + req * NUM_BLOCKS + block_idx, block_idx * BLOCK + idx)



def blockwise_rejection_triton(
    *,
    candidates: torch.Tensor,
    target_logits: torch.Tensor,
    draft_logits: Optional[torch.Tensor],
    target_temperatures: torch.Tensor,
    draft_temperatures: torch.Tensor,
    noise: RejectionNoise,
    cutoff_verify_lens: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """``draft_logits`` is pre-temperature, or None for a one-hot draft. Returns
    (correct_len int32, bonus int64, cap_trim_lens int32), each [bs]."""
    bs, stride = candidates.shape
    target_vocab = target_logits.shape[-1]
    assert target_logits.stride(-1) == 1 and target_logits.shape[0] == bs * stride
    has_draft = draft_logits is not None
    draft_vocab = draft_logits.shape[-1] if has_draft else target_vocab
    if has_draft:
        assert draft_logits.stride(-1) == 1 and draft_logits.shape[1] >= stride - 1
    device = candidates.device
    target_temperatures = target_temperatures.reshape(-1).contiguous()
    draft_temperatures = draft_temperatures.reshape(-1).contiguous()
    num_blocks = triton.cdiv(max(target_vocab, draft_vocab), _STATS_BLOCK)
    padded_blocks = triton.next_power_of_2(num_blocks)
    stats = torch.empty(
        (4, bs * stride, num_blocks), dtype=torch.float32, device=device
    )
    lens = torch.empty((3, bs), dtype=torch.int32, device=device)
    row_stats = torch.empty(
        (4 if has_draft else 2, bs), dtype=torch.float32, device=device
    )
    res_blocks = triton.cdiv(target_vocab, _RESAMPLE_BLOCK)
    res_max = torch.empty((bs, res_blocks), dtype=torch.float32, device=device)
    res_arg = torch.empty((bs, res_blocks), dtype=torch.int64, device=device)
    bonus = torch.empty((bs,), dtype=torch.int64, device=device)
    correct_len, raw_len, trim = lens[0], lens[1], lens[2]
    t_max, t_sumexp, d_max, d_sumexp = stats[0], stats[1], stats[2], stats[3]
    d_ptr = draft_logits if has_draft else target_logits
    d_s0 = draft_logits.stride(0) if has_draft else 0
    d_s1 = draft_logits.stride(1) if has_draft else 0
    _stats_kernel[(bs * stride, num_blocks)](
        t_max, t_sumexp, d_max, d_sumexp,
        target_logits, target_logits.stride(0),
        d_ptr, d_s0, d_s1,
        target_temperatures, draft_temperatures,
        target_vocab, draft_vocab,
        STRIDE=stride, NUM_BLOCKS=num_blocks, BLOCK=_STATS_BLOCK,
        HAS_DRAFT_LOGITS=has_draft,
    )  # fmt: skip
    has_cutoff = cutoff_verify_lens is not None
    verify_lens = (
        cutoff_verify_lens.to(device=device).contiguous() if has_cutoff else correct_len
    )
    _rejection_kernel[(bs,)](
        correct_len, raw_len, trim, row_stats,
        candidates, candidates.stride(0), verify_lens,
        t_max, t_sumexp, d_max, d_sumexp,
        target_logits, target_logits.stride(0),
        d_ptr, d_s0, d_s1,
        target_temperatures, draft_temperatures,
        target_vocab, draft_vocab, bs,
        noise.key, noise.counter,
        STRIDE=stride, NUM_BLOCKS=num_blocks, PADDED_BLOCKS=padded_blocks,
        HAS_DRAFT_LOGITS=has_draft, HAS_CUTOFF=has_cutoff,
        num_warps=1,
    )  # fmt: skip
    _resample_kernel[(bs, res_blocks)](
        res_max, res_arg, raw_len, row_stats,
        candidates, candidates.stride(0),
        target_logits, target_logits.stride(0),
        d_ptr, d_s0, d_s1,
        target_temperatures, draft_temperatures,
        target_vocab, draft_vocab, bs,
        noise.key, noise.counter,
        STRIDE=stride, NUM_BLOCKS=res_blocks, BLOCK=_RESAMPLE_BLOCK,
        HAS_DRAFT_LOGITS=has_draft,
    )  # fmt: skip
    _insert_kernel[(bs,)](
        bonus, correct_len, raw_len, res_max, res_arg,
        candidates, candidates.stride(0), noise.counter,
        NUM_BLOCKS=res_blocks, PADDED_BLOCKS=triton.next_power_of_2(res_blocks),
        num_warps=1,
    )  # fmt: skip
    return correct_len, bonus, trim



def blockwise_rejection_torch(
    *,
    candidates: torch.Tensor,
    target_logits: torch.Tensor,
    draft_logits: Optional[torch.Tensor],
    target_temperatures: torch.Tensor,
    draft_temperatures: torch.Tensor,
    noise: RejectionNoise,
    cutoff_verify_lens: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Reference with the same math; draws from a generator seeded by (key, counter)."""
    bs, stride = candidates.shape
    gamma = stride - 1
    vocab = target_logits.shape[-1]
    gen = torch.Generator(device=candidates.device)
    gen.manual_seed((noise.key + int(noise.counter.item()) * 1_000_003) % (2**63))
    t_temp = target_temperatures.reshape(bs, 1, 1).float()
    log_p = torch.log_softmax(
        target_logits.float().view(bs, stride, vocab) / t_temp, -1
    )
    if draft_logits is not None:
        d_temp = draft_temperatures.reshape(bs, 1, 1).float()
        d = draft_logits[:bs, :gamma].float() / d_temp
        log_q = torch.full((bs, gamma, vocab), float("-inf"), device=d.device)
        log_q[..., : d.shape[-1]] = torch.log_softmax(d, -1)[..., :vocab]
    else:
        log_q = torch.full((bs, gamma, vocab), float("-inf"), device=log_p.device)
        log_q.scatter_(-1, candidates[:, 1:].unsqueeze(-1), 0.0)
    coins = torch.rand((bs, gamma), generator=gen, device=candidates.device)
    gumbel = -torch.log(
        -torch.log(torch.rand((bs, vocab), generator=gen, device=candidates.device))
    )
    tokens = candidates[:, 1:].long()
    lp = log_p[:, :gamma].gather(-1, tokens.unsqueeze(-1)).squeeze(-1)
    lq = log_q.gather(-1, tokens.unsqueeze(-1)).squeeze(-1)
    ok = (lq > float("-inf")) & (torch.log(coins.clamp_min(4.6566127342e-10)) < lp - lq)
    raw = torch.cumprod(ok.int(), dim=-1).sum(-1).int()
    rows = torch.arange(bs, device=candidates.device)
    lp_rej = log_p[rows, raw.long()]
    lq_rej = torch.cat([log_q, log_q[:, :1]], 1)[rows, raw.long()]
    residual = torch.where(
        raw.unsqueeze(-1) == gamma,
        lp_rej,
        torch.where(
            lq_rej < lp_rej,
            lp_rej + torch.log1p(-torch.exp(lq_rej - lp_rej)),
            float("-inf"),
        ),
    )
    bonus = (residual + gumbel).argmax(-1)
    if cutoff_verify_lens is not None:
        capped = torch.minimum(raw, cutoff_verify_lens.to(raw.device).int() - 1)
    else:
        capped = raw
    capped_bonus = candidates.gather(
        1, (capped.long() + 1).clamp_max(gamma).unsqueeze(-1)
    ).squeeze(-1)
    bonus = torch.where(capped < raw, capped_bonus, bonus)
    noise.counter.add_(1)
    return capped, bonus.long(), raw - capped
