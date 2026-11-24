"""rnnhoist: the step-invariant part of the Markov walk's rnn_proj, computed once per walk.

Inkling2MTPHead._rnn_step runs rnn_proj = Linear(5120 -> 768) on
[state (256), markov_w1[prev] (256), hidden_s (2304), f_rnn (2304)] at each of the 7 draft steps. Only the first
512 inputs depend on the previous step. hidden_s for every step (draft_hidden[:, s]) and f_rnn are known when
the walk starts (init_state), so:

  init_state:  pre[b, s] = hidden[b, s] @ Wh^T + (f_rnn[b] @ Wf^T + bias)     FP32 [bs, gamma, 768]
  step s:      proj = bf16(x @ Ws^T + pre[b, s]),  x = [state, markov_w1[prev]]  (FP32 partial, one rounding)
  then the stock rnn_gates(proj, state).

The stock op is the same sum in one FP32 chain with one BF16 rounding (cuBLAS), so this differs only in FP32
summation order. Per step the walk reads Ws (768 x 512 BF16, 0.79 MB) instead of all of W (7.9 MB).

The step index of a _rnn_step call is recovered from the hidden slice it receives (draft_markov_block passes
draft_hidden[:, step], a view into the tensor init_state saw): (ptr - base) / stride(1). Any call that does not
match the walk init_state prepared (other tensor, other rows, other f_rnn) runs the stock step.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F

from . import ops
from .quant import W8Pack, bf16_pack


@dataclass
class RnnPacks:
    ws: W8Pack  # [3R, 2R] state + prev-embedding columns
    wh: W8Pack  # [3R, H] hidden columns
    wf: W8Pack  # [3R, H] f_rnn columns
    bias32: torch.Tensor  # [3R] fp32

    def nbytes(self) -> int:
        return sum(p.nbytes() for p in (self.ws, self.wh, self.wf)) + self.bias32.numel() * 4


@dataclass
class Walk:
    pre: torch.Tensor  # [bs, gamma, 3R] fp32
    base_ptr: int
    step_bytes: int
    row_stride: int
    gamma: int
    rows: int
    f_ptr: int


def build_packs(rnn_proj, rank: int, hidden: int) -> RnnPacks:
    w = rnn_proj.weight.data
    assert w.shape == (3 * rank, 2 * rank + 2 * hidden) and w.dtype == torch.bfloat16
    ws = w[:, : 2 * rank].contiguous()
    wh = w[:, 2 * rank: 2 * rank + hidden].contiguous()
    wf = w[:, 2 * rank + hidden:].contiguous()
    bias = rnn_proj.bias
    bias32 = (bias.data.float() if bias is not None else torch.zeros(3 * rank, device=w.device)).contiguous()
    return RnnPacks(bf16_pack(ws), bf16_pack(wh), bf16_pack(wf), bias32)


def begin_walk(packs: RnnPacks, draft_hidden: Optional[torch.Tensor], f_rnn: torch.Tensor,
               max_rows: int) -> Optional[Walk]:
    """Hoisted partials for the walk over draft_hidden [bs, gamma, H]; None -> the stock step runs."""
    if draft_hidden is None or draft_hidden.dim() != 3 or draft_hidden.dtype != torch.bfloat16:
        return None
    bs, gamma, hid = draft_hidden.shape
    if bs == 0 or bs > max_rows or hid != packs.wh.k or f_rnn.shape != (bs, hid) or draft_hidden.stride(2) != 1:
        return None
    flat = draft_hidden.reshape(bs * gamma, hid)
    # <= 64 rows per call: one row block per launch (the multi-block wide mode with a BF16 weight needs more shared
    # memory than GB10 has); rows are independent, so chunking changes nothing numerically
    pre = torch.cat([ops.w8_linear(flat[i:i + 64], packs.wh, None, torch.float32) for i in range(0, bs * gamma, 64)]
                    ).view(bs, gamma, -1)
    fpart = ops.w8_linear(f_rnn, packs.wf, packs.bias32, torch.float32)
    pre = pre + fpart[:, None, :]
    esize = draft_hidden.element_size()
    return Walk(pre=pre, base_ptr=draft_hidden.data_ptr(), step_bytes=draft_hidden.stride(1) * esize,
                row_stride=draft_hidden.stride(0), gamma=gamma, rows=bs, f_ptr=f_rnn.data_ptr())


def step_index(walk: Optional[Walk], hidden: torch.Tensor, f_rnn: torch.Tensor) -> int:
    """The walk step whose hidden slice this is, or -1."""
    if walk is None or hidden.dim() != 2 or hidden.shape[0] != walk.rows or f_rnn.data_ptr() != walk.f_ptr:
        return -1
    if walk.rows > 1 and hidden.stride(0) != walk.row_stride:
        return -1
    off = hidden.data_ptr() - walk.base_ptr
    if off < 0 or off % walk.step_bytes:
        return -1
    s = off // walk.step_bytes
    return s if s < walk.gamma else -1


def hoisted_proj(packs: RnnPacks, walk: Walk, s: int, state: torch.Tensor, prev_tokens: torch.Tensor,
                 w1: torch.Tensor) -> torch.Tensor:
    """bf16 rnn_proj output of step s."""
    x = torch.cat([state, F.embedding(prev_tokens.long(), w1).to(state.dtype)], -1)
    part = ops.w8_linear(x, packs.ws, None, torch.float32)
    return (part + walk.pre[:, s]).to(torch.bfloat16)
