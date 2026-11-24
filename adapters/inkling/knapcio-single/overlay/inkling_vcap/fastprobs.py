"""Vocabulary-wide draft statistics in one or two reads of the draft's kept step logits,
instead of the many full-width FP32 temporaries the torch reference allocates
(INKLING_VERIFY_CAP_FASTPROBS=1; default off).

Two uses:

1. ``max_prob``: the per-position max draft probability ``m_j`` that the conf, cost and
   policy rules read before each target verify. The torch reference
   (``logic.draft_token_probs(use_max=True)``) makes about 8 full-width passes over
   [bs*7, V] in FP32: a dtype copy, the division by T, amax, sub, exp, sum and amax
   again. The R7S profile (thinking T=0.8, c8) shows 2.0 ms per step for it. Here one
   read of the BF16 logits gives each row's online (max, sum-exp) of x/T, from the
   image's stock split kernel ``_softmax_temp_stats_kernel``. The row's max probability
   is then 1 / sum_i exp(x_i/T - max/T), and a [rows, splits, 2] combine runs in torch.
2. ``softmax_temp``: the draft probabilities q that the sampled accept reads
   (``dspark_verify._draft_probs``). The stock path is FlashInfer's softmax, which
   first copies the BF16 logits to FP32 (0.93 ms per step at c24, 0.28 ms at c8) and
   then reads that copy twice. Here the image's stock split pair
   (``_softmax_temp_stats_kernel`` + ``_softmax_temp_write_kernel``, written for ROCm)
   reads the BF16 logits twice and writes FP32 probabilities once.

Both are the same mathematical function as the reference: the softmax of x/T, with the
max taken first. They differ only in FP32 summation order, a rounding-level difference.
``m_j`` only decides the live length, and any function of the draft distribution at
position j keeps the cap exact. q is the draft's law up to FP32 rounding, as the
FlashInfer q is. No new kernel is written here; both kernels are the image's.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch

_BLOCK_V = 2048
_TARGET_PROGRAMS = 1024  # the image's choice for its split path


def _stock_kernels():
    try:
        from sglang.kernels.ops.speculative.dspark import dspark_accept as da
    except Exception:  # noqa: BLE001
        return None
    stats = getattr(da, "_softmax_temp_stats_kernel", None)
    write = getattr(da, "_softmax_temp_write_kernel", None)
    if stats is None or write is None:
        return None
    return stats, write


def available(device) -> bool:
    return torch.device(device).type == "cuda" and _stock_kernels() is not None


def _split(num_rows: int, vocab: int) -> Tuple[int, int]:
    import triton

    num_splits = max(1, min(64, triton.cdiv(_TARGET_PROGRAMS, max(1, num_rows))))
    chunk = triton.cdiv(triton.cdiv(vocab, num_splits), _BLOCK_V) * _BLOCK_V
    return triton.cdiv(vocab, chunk), chunk


def _check_rows(logits2d: torch.Tensor) -> None:
    if logits2d.dim() != 2 or logits2d.stride(1) != 1:
        raise ValueError("fastprobs needs [rows, V] logits with a unit last stride")


def row_stats(
    logits2d: torch.Tensor, temps: torch.Tensor, rows_per_request: int
) -> Tuple[torch.Tensor, int, int]:
    """[rows, splits, 2] (max, sum-exp relative to that max) of logits / T per row
    chunk; temps [rows // rows_per_request] FP32. Returns (stats, num_splits, chunk)."""
    _check_rows(logits2d)
    kernels = _stock_kernels()
    if kernels is None:
        raise RuntimeError("the image's split softmax kernels are missing")
    stats_kernel, _ = kernels
    rows, vocab = logits2d.shape
    num_splits, chunk = _split(rows, vocab)
    stats = torch.empty((rows, num_splits, 2), dtype=torch.float32, device=logits2d.device)
    if rows:
        stats_kernel[(rows, num_splits)](
            logits2d, temps, stats, vocab, chunk, rows_per_request, logits2d.stride(0),
            num_splits, BLOCK_V=_BLOCK_V,
        )
    return stats, num_splits, chunk


def combine_max_prob(stats: torch.Tensor) -> torch.Tensor:
    """[rows] max probability from [rows, splits, (max, sum-exp)]: 1 / sum_s
    sum_s * exp(max_s - max). Chunks that saw only -inf carry a zero sum. A row with
    no finite logit, or a NaN, gives 0 (the torch reference's nan_to_num(0))."""
    part_max, part_sum = stats[..., 0], stats[..., 1]
    row_max = part_max.amax(dim=-1, keepdim=True)
    live = part_max > float("-inf")
    scaled = torch.where(live, part_sum * torch.exp(part_max - row_max), torch.zeros_like(part_sum))
    total = scaled.sum(dim=-1)
    out = torch.where(total > 0, 1.0 / total, torch.zeros_like(total))
    return torch.nan_to_num(out, nan=0.0).clamp_(0.0, 1.0)


def request_temps(
    bs: int, temperatures: Optional[torch.Tensor], greedy_mask: Optional[torch.Tensor],
    device,
) -> torch.Tensor:
    """Per-request temperature with the reference's rule: the request's temperature
    clamped at 1e-5; greedy rows (and no temperatures at all) use 1."""
    if temperatures is None:
        return torch.ones(bs, dtype=torch.float32, device=device)
    temps = temperatures.reshape(-1)[:bs].to(device=device, dtype=torch.float32).clamp_min(1e-5)
    if greedy_mask is not None:
        temps = torch.where(greedy_mask.reshape(-1)[:bs].to(device=device, dtype=torch.bool),
                            torch.ones_like(temps), temps)
    return temps.contiguous()


def max_prob(
    step_logits: torch.Tensor, *, temperatures: Optional[torch.Tensor],
    greedy_mask: Optional[torch.Tensor],
) -> torch.Tensor:
    """[bs, gamma] max draft probability per position (drop-in for
    logic.draft_token_probs(..., use_max=True))."""
    bs, gamma, vocab = step_logits.shape
    flat = step_logits.reshape(bs * gamma, vocab)
    temps = request_temps(bs, temperatures, greedy_mask, step_logits.device)
    stats, _, _ = row_stats(flat, temps, gamma)
    return combine_max_prob(stats).view(bs, gamma)


def softmax_temp(
    logits2d: torch.Tensor, temperatures: torch.Tensor, rows_per_request: int
) -> torch.Tensor:
    """FP32 softmax(logits / T) per row, T = temperatures[row // rows_per_request]
    (the SoftmaxTemp contract), reading the logits in their own dtype."""
    _check_rows(logits2d)
    kernels = _stock_kernels()
    if kernels is None:
        raise RuntimeError("the image's split softmax kernels are missing")
    import triton

    _, write_kernel = kernels
    rows, vocab = logits2d.shape
    bs = rows // rows_per_request
    temps = temperatures.reshape(-1)[:bs].to(torch.float32).contiguous()
    stats, num_splits, chunk = row_stats(logits2d, temps, rows_per_request)
    out = torch.empty((rows, vocab), dtype=torch.float32, device=logits2d.device)
    if rows:
        write_kernel[(rows, num_splits)](
            logits2d, temps, stats, out, vocab, chunk, rows_per_request,
            logits2d.stride(0), num_splits, BLOCK_V=_BLOCK_V,
            BLOCK_S=triton.next_power_of_2(num_splits),
        )
    return out
