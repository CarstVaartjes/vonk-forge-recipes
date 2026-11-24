"""FP8 (e4m3) weight packs and their pure-torch reference math.

A pack holds q[N, K] (float8_e4m3fn, or the BF16 weight itself for an unscaled
pack) plus scales: fp32 [N, K // group] for row/group modes, uint8 e8m0
[N, K // 32] for mx. Dequantized weight = q * scale, broadcast over each group.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch

FP8 = torch.float8_e4m3fn
FP8_MAX = 448.0
SCALE_NONE, SCALE_FP32, SCALE_E8M0 = 0, 1, 2


@dataclass
class W8Pack:
    q: torch.Tensor  # [N, K] fp8 or bf16
    scale: Optional[torch.Tensor]  # [N, K // group] fp32 | uint8 | None
    group: int  # inputs per scale (K for per-row)
    scale_kind: int  # SCALE_NONE | SCALE_FP32 | SCALE_E8M0
    label: str = "bf16"

    @property
    def n(self) -> int:
        return int(self.q.shape[0])

    @property
    def k(self) -> int:
        return int(self.q.shape[1])

    def nbytes(self) -> int:
        size = self.q.numel() * self.q.element_size()
        if self.scale is not None:
            size += self.scale.numel() * self.scale.element_size()
        return size


def bf16_pack(weight: torch.Tensor) -> W8Pack:
    """An unscaled pack that reads the BF16 weight in place (no copy)."""
    assert weight.dtype == torch.bfloat16 and weight.dim() == 2
    return W8Pack(weight, None, int(weight.shape[1]), SCALE_NONE, "bf16")


def _quant_rows(w: torch.Tensor, kind: str, group: int):
    """w: [R, K] any float dtype -> (q fp8 [R, K], scale [R, K // group])."""
    rows, k = w.shape
    g = w.float().view(rows, k // group, group)
    amax = g.abs().amax(dim=-1)
    if kind == "mx":
        # Smallest power of two that keeps the block amax within e4m3's range,
        # so nothing clips. Zero blocks get exponent 0.
        safe = torch.where(amax > 0, amax / FP8_MAX, torch.ones_like(amax))
        exp = torch.ceil(torch.log2(safe)).clamp_(-127, 127)
        scale = torch.exp2(exp)
        stored = (exp + 127).to(torch.uint8)
    else:
        scale = torch.where(amax > 0, amax / FP8_MAX, torch.ones_like(amax))
        stored = scale.to(torch.float32)
    q = (g / scale.unsqueeze(-1)).clamp_(-FP8_MAX, FP8_MAX).to(FP8).view(rows, k)
    return q, stored


def quantize(
    weight: torch.Tensor,
    kind: str,
    group: int = 0,
    *,
    out: Optional[W8Pack] = None,
    chunk_rows: int = 16384,
) -> W8Pack:
    """Quantize weight[N, K] row-chunk by row-chunk (bounded temp memory).
    kind: "row" | "group" | "mx". With ``out`` the existing tensors are
    refilled in place (their addresses stay valid for captured graphs)."""
    assert weight.dim() == 2
    n, k = weight.shape
    if kind == "row":
        group, scale_kind, label = k, SCALE_FP32, "row"
    elif kind == "group":
        scale_kind, label = SCALE_FP32, f"g{group}"
    elif kind == "mx":
        group, scale_kind, label = 32, SCALE_E8M0, "mx"
    else:
        raise ValueError(kind)
    if k % group:
        raise ValueError(f"K={k} is not a multiple of the scale group {group}")
    if out is None:
        q = torch.empty((n, k), dtype=FP8, device=weight.device)
        scale = torch.empty(
            (n, k // group),
            dtype=torch.uint8 if scale_kind == SCALE_E8M0 else torch.float32,
            device=weight.device,
        )
        out = W8Pack(q, scale, group, scale_kind, label)
    else:
        assert out.q.shape == (n, k) and out.group == group
    for start in range(0, n, chunk_rows):
        rows = slice(start, min(start + chunk_rows, n))
        q, s = _quant_rows(weight[rows], kind, group)
        out.q[rows].copy_(q)
        out.scale[rows].copy_(s)
    return out


def quantize_mode(weight: torch.Tensor, mode, out: Optional[W8Pack] = None) -> W8Pack:
    """mode: an _opts.QuantMode."""
    return quantize(weight, mode.kind, mode.group, out=out)


def dequant_scale(pack: W8Pack) -> Optional[torch.Tensor]:
    """fp32 scale per element group [N, K // group]."""
    if pack.scale_kind == SCALE_NONE:
        return None
    if pack.scale_kind == SCALE_E8M0:
        return torch.exp2(pack.scale.float() - 127.0)
    return pack.scale.float()


def dequantize(pack: W8Pack) -> torch.Tensor:
    """fp32 [N, K]."""
    w = pack.q.float()
    s = dequant_scale(pack)
    if s is None:
        return w
    return (w.view(pack.n, -1, pack.group) * s.unsqueeze(-1)).view(pack.n, pack.k)


def linear_ref(
    x: torch.Tensor,
    pack: W8Pack,
    bias: Optional[torch.Tensor] = None,
    out_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """x[M, K] @ dequant(pack)^T (+ bias) in FP32, rounded once to out_dtype.
    The spec the Triton kernel implements (up to FP32 summation order)."""
    y = x.float() @ dequantize(pack).t()
    if bias is not None:
        y = y + bias.float()
    return y.to(out_dtype)
