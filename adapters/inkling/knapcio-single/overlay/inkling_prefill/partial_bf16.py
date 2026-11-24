"""C3d partial_bf16: the MoE combine writes BF16 directly.

Stock (sglang/srt/models/inkling_common/inkling2_moe.py, Inkling2Experts._forward_sm120):
    out = routed_partial(down, offsets, src2dst, top_k, nvfp4_scales=scales).to(x.dtype)
routed_partial (kernels/ops/moe/inkling2_moe.py) stores the per-token FP32 sum of the
top-k expert rows, then .to(bfloat16) reads it back and writes BF16: at an 8K prefill
chunk that is 75 MB written and read again per MoE layer for nothing.

This lever runs a copy of the stock `_routed_partial` Triton kernel whose only change
is the final store: `total.to(tl.bfloat16)` (round-to-nearest-even, the same rounding
as torch's FP32 -> BF16 cast). The FP32 sum itself is computed by the identical code,
so the result is bit-identical. It is used only for the routed_partial call inside
_forward_sm120 when x is BF16 (a context flag set by a thin _forward_sm120 wrapper);
every other routed_partial caller keeps the stock FP32 function. The returned BF16
tensor makes the stock `.to(x.dtype)` a no-op.

CUDA graphs: same launch shape as stock, no host sync.
"""

import contextvars

import torch

MODEL_MODULE = "sglang.srt.models.inkling_common.inkling2_moe"
KERNEL_MODULE = "sglang.kernels.ops.moe.inkling2_moe"
MARK = "inkling_prefill_partial_bf16"

_OUT_DTYPE = contextvars.ContextVar("inkling_prefill_partial_dtype", default=None)

_kernel = None


def _build_kernel():
    from . import _triton_partial

    return _triton_partial._routed_partial_bf16, _triton_partial.PDL


def routed_partial_bf16(x, offsets, src2dst, top_k, *, nvfp4_scales=None):
    """BF16 twin of kernels.ops.moe.inkling2_moe.routed_partial."""
    import triton

    global _kernel
    if _kernel is None:
        _kernel = _build_kernel()
    kernel, pdl = _kernel
    experts = offsets.numel() - 1
    out = torch.empty(
        (src2dst.numel() // top_k, x.shape[1]), device=x.device, dtype=torch.bfloat16
    )
    if out.numel():
        x_amax, w_amax, w_denominator = nvfp4_scales or (None, None, None)
        kernel[(out.shape[0], triton.cdiv(out.shape[1], 1024))](
            x,
            x_amax,
            w_amax,
            w_denominator,
            offsets,
            src2dst,
            out,
            out.shape[1],
            experts,
            top_k,
            nvfp4_scales is not None,
            1024,
            experts.bit_length(),
            launch_pdl=pdl,
        )
    return out


def make_routed_partial(stock):
    def routed_partial(x, offsets, src2dst, top_k, *, nvfp4_scales=None):
        if _OUT_DTYPE.get() is torch.bfloat16 and x.is_cuda:
            return routed_partial_bf16(x, offsets, src2dst, top_k, nvfp4_scales=nvfp4_scales)
        return stock(x, offsets, src2dst, top_k, nvfp4_scales=nvfp4_scales)

    routed_partial.__wrapped__ = stock
    setattr(routed_partial, MARK, True)
    return routed_partial


def make_forward_sm120(prev):
    def _forward_sm120(self, x, weights, ids, combine):
        token = _OUT_DTYPE.set(x.dtype)
        try:
            return prev(self, x, weights, ids, combine)
        finally:
            _OUT_DTYPE.reset(token)

    _forward_sm120.__wrapped__ = prev
    setattr(_forward_sm120, MARK, True)
    return _forward_sm120


def patch(model_module):
    cls = getattr(model_module, "Inkling2Experts", None)
    prev = getattr(cls, "_forward_sm120", None) if cls is not None else None
    stock = getattr(model_module, "routed_partial", None)
    if prev is None or stock is None:
        raise RuntimeError(
            f"inkling_prefill.partial_bf16: {MODEL_MODULE}.Inkling2Experts._forward_sm120 "
            "or routed_partial is missing"
        )
    if getattr(stock, MARK, False):
        return False
    model_module.routed_partial = make_routed_partial(stock)
    cls._forward_sm120 = make_forward_sm120(prev)
    return True
