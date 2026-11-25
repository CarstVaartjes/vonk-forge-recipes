"""Device dispatch: CUDA tensors run the Triton kernels, anything else runs the
pure-torch reference (the spec, and what the CPU unit tests exercise).

Nothing here syncs with the host or reads tensor values on the host, so every
op is safe inside CUDA graph capture once the weights are prepared.
"""

from __future__ import annotations

from typing import Optional

import torch

from .quant import W8Pack, linear_ref

_cfg = {"fold_bn": 64, "fold_bk": 128, "linear_bn": 32, "linear_bk": 256, "num_warps": 4, "rows": 0, "stages": 0,
        "wide_bn": 64, "wide_warps": 8}


def configure(opts) -> None:
    _cfg.update(
        fold_bn=opts.fold_bn,
        fold_bk=opts.fold_bk,
        linear_bn=opts.linear_bn,
        linear_bk=opts.linear_bk,
        num_warps=opts.num_warps,
        rows=opts.wide8_mode if getattr(opts, "wide8", False) else 0,
        stages=getattr(opts, "wide8_stages", 0),
        wide_bn=getattr(opts, "wide8_bn", 64),
        wide_warps=getattr(opts, "wide8_warps", 8),
    )


def _wide_kw(m: int = 0):
    """wide8: the row mode, stage count and tile for a launch of m rows; {} when off or when m fits one 64-row
    block (then the launch is exactly the default one)."""
    if not _cfg["rows"] or m <= 64:
        return {}
    return {"rows": _cfg["rows"], "num_stages": _cfg["stages"] or None, "bn": _cfg["wide_bn"],
            "num_warps": _cfg["wide_warps"]}


def _triton():
    from . import _triton as kernels

    return kernels


def w8_linear(
    x: torch.Tensor,
    pack: W8Pack,
    bias: Optional[torch.Tensor] = None,
    out_dtype: torch.dtype = torch.bfloat16,
    *,
    narrow_k: bool = False,
    bn: Optional[int] = None,
    bk: Optional[int] = None,
    num_warps: Optional[int] = None,
) -> torch.Tensor:
    """x[..., K] @ dequant(pack)^T (+ bias). narrow_k picks the fold tile
    (K = 256: wide BN); otherwise the linear tile (BN 32, BK 256)."""
    lead = x.shape[:-1]
    x2 = x.reshape(-1, x.shape[-1])
    if not x2.is_cuda:
        return linear_ref(x2, pack, bias, out_dtype).view(*lead, pack.n)
    if x2.dtype != torch.bfloat16:
        x2 = x2.to(torch.bfloat16)
    if x2.stride(-1) != 1:
        x2 = x2.contiguous()
    out = torch.empty((x2.shape[0], pack.n), dtype=out_dtype, device=x2.device)
    if x2.shape[0]:
        kw = dict(
            bn=bn or (_cfg["fold_bn"] if narrow_k else _cfg["linear_bn"]),
            bk=bk or (_cfg["fold_bk"] if narrow_k else _cfg["linear_bk"]),
            num_warps=num_warps or _cfg["num_warps"],
        )
        if bn is None and num_warps is None:
            kw.update(_wide_kw(x2.shape[0]))
        _triton().w8_mm(x2, pack, bias, out, **kw)
    return out.view(*lead, pack.n)


def hot_step_ref(
    latent: torch.Tensor,
    pack: W8Pack,
    fold_bias: torch.Tensor,
    base: torch.Tensor,
    ids: torch.Tensor,
    out: torch.Tensor,
    *,
    softcap: Optional[float],
    vocab_limit: Optional[int],
    base_compact: bool = False,
) -> torch.Tensor:
    """Reference of the fused hot step: writes the hot columns of ``out``. base_compact: ``base`` holds only the hot
    columns, in hot order."""
    step_dtype = out.dtype
    bias = linear_ref(latent, pack, fold_bias, torch.bfloat16).float()
    ids = ids.long()
    b = (base[:, :ids.numel()] if base_compact else base[:, ids]).float()
    value = (b + bias).to(step_dtype).float()
    if softcap is not None:
        value = (torch.tanh(value / softcap) * softcap).to(step_dtype).float()
    if vocab_limit is not None:
        value = torch.where(ids[None, :] < vocab_limit, value, float("-inf"))
    out[:, ids] = value.to(out.dtype)
    return out


def hot_step(
    latent: torch.Tensor,
    pack: W8Pack,
    fold_bias: torch.Tensor,
    base: torch.Tensor,
    ids32: torch.Tensor,
    out: torch.Tensor,
    *,
    softcap: Optional[float],
    vocab_limit: Optional[int],
    bn: Optional[int] = None,
    bk: Optional[int] = None,
    num_warps: Optional[int] = None,
    base_compact: bool = False,
) -> torch.Tensor:
    """out[m, ids[j]] = limit(softcap(bf16(base[m, ids[j]] + bf16(fold_j(latent_m)))));
    other columns of ``out`` are left untouched (the caller fills them -inf). base_compact: base[m, j] stands for
    the full-width base[m, ids[j]] (the hot head's output before it was scattered)."""
    if base_compact and (base.shape[-1] != ids32.numel() or base.shape[0] != latent.shape[0]):
        raise ValueError("inkling_draftbytes: compact base logits do not match the hot set")
    if not latent.is_cuda:
        return hot_step_ref(
            latent, pack, fold_bias, base, ids32, out,
            softcap=softcap, vocab_limit=vocab_limit, base_compact=base_compact,
        )
    if latent.dtype != torch.bfloat16:
        latent = latent.to(torch.bfloat16)
    if latent.stride(-1) != 1:
        latent = latent.contiguous()
    if latent.shape[0]:
        kw = dict(bn=bn or _cfg["fold_bn"], bk=bk or _cfg["fold_bk"], num_warps=num_warps or _cfg["num_warps"])
        if bn is None and num_warps is None:
            kw.update(_wide_kw(latent.shape[0]))
        _triton().hot_step(
            latent, pack, fold_bias, base, ids32, out,
            softcap=softcap, vocab_limit=vocab_limit, base_compact=base_compact, **kw,
        )
    return out


def bf16_gemv(x: torch.Tensor, weight: torch.Tensor, max_m: int, gemv_mod) -> torch.Tensor:
    """Draft base logits x[M, K] @ weight[N, K]^T in BF16: the image's sm12x
    Triton GEMV for <= 16 rows, this package's unscaled kernel for
    16 < M <= max_m, cuBLAS (torch.matmul, the stock call) otherwise."""
    x = x.to(weight.dtype)
    if gemv_mod is not None and gemv_mod.use_sm12x_bf16_gemv(x, weight):
        return gemv_mod.sm12x_bf16_gemv(x, weight)
    if (
        x.is_cuda
        and x.dim() == 2
        and 16 < x.shape[0] <= max_m
        and weight.dtype == torch.bfloat16
        and weight.stride(1) == 1
        and weight.stride(0) == weight.shape[1]
    ):
        from .quant import bf16_pack

        return w8_linear(x, bf16_pack(weight), None, weight.dtype)
    return torch.matmul(x, weight.T)
