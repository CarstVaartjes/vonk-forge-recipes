"""C3a quant_once: quantize each MoE input token once, then gather the packed rows.

Stock (kernels/ops/moe/inkling2_experts_sm120.py, nvfp4_routed_quantize, called by
Inkling2Experts._sm120_nvfp4 for the NVFP4 layers 0-19): one NVFP4 quantization per
ROUTED row, each reading its token's BF16 row; with top-8 every token row is read
and quantized 8 times (8K chunk: 302 MB read, 1.87 ms per MoE visit).

The quantization of a row is a pure function of that row (routed_quant.cuh: the
row amax is a max-reduction, then per-16 block scales and e2m1 packing), so:
  1. run the stock kernel once over the T token rows (dst2src = arange, top_k 1,
     all rows in one expert) -> token values, token scales (swizzled 128x4 atoms of
     that single expert), token amax;
  2. a Triton gather copies, for every routed row r, the packed values and amax of
     token dst2src[r] // top_k, and moves its 36 scale words from the token layout to
     the routed layout of expert(r) (routed_quant.cuh scale_base / scale_offset);
  3. zero each expert's scale tail rows like zero_expert_tail.
Every byte the stock kernel writes is reproduced bit for bit (gpu_check_prefill.py
compares values, amax and all written scale bytes, and the grouped GEMM output).
Bytes the stock kernel never writes (torch.empty padding) stay unwritten.

Scope: top_k > 1, T >= INKLING_PREFILL_QUANT_MIN_T tokens (default 256), BF16
contiguous input whose width is a multiple of 64, single-rank routing
(dst2src.numel() == T * top_k), int32 metadata. Otherwise the stock function runs.
MXFP8 layers 20-23 are not touched (different scale layout; 4 of 48 visits).

CUDA graphs: static grids, buffers from torch.empty/arange/fill, no host sync.
"""

import torch

TARGET_MODULE = "sglang.kernels.ops.moe.inkling2_experts_sm120"
TARGET_ATTR = "nvfp4_routed_quantize"
MARK = "inkling_prefill_quant_once"


def _pow2(n):
    p = 1
    while p < n:
        p *= 2
    return p


def quant_once(stock, outputs, x, dst2src, top_k, offsets):
    """Same (values, scales, amax) as stock(x, dst2src, top_k, offsets)."""
    from . import _triton_quant

    t, width = x.shape
    rows = dst2src.numel()
    experts = offsets.numel() - 1
    device = x.device
    ident = torch.arange(t, dtype=torch.int32, device=device)
    one = torch.empty(2, dtype=torch.int32, device=device)
    one.fill_(t)
    one[:1].zero_()
    tok_vals, tok_scales, tok_amax = stock(x=x, dst2src=ident, top_k=1, offsets=one)
    values, scales, amax = outputs(rows=rows, width=width, experts=experts, device=device)
    if rows:
        c4 = width // 64  # scale_k / 4 words per row, scale_k = width / 16
        vwords = width // 8
        _triton_quant._gather_routed[(rows,)](
            tok_vals.view(torch.int32),
            tok_scales.view(torch.int32),
            tok_amax,
            dst2src,
            offsets,
            values.view(torch.int32),
            scales.view(torch.int32),
            amax,
            TOP_K=top_k,
            VWORDS=vwords,
            VBLOCK=_pow2(vwords),
            C4=c4,
            CBLOCK=_pow2(c4),
            E=experts,
            SEARCH=max(1, (experts - 1).bit_length()),
        )
        _triton_quant._zero_tails[(experts,)](
            offsets, scales.view(torch.int32), C4=c4, CBLOCK=_pow2(c4)
        )
    return values, scales, amax


def eligible(x, dst2src, top_k, offsets, min_t):
    return (
        top_k > 1
        and x.is_cuda
        and x.dim() == 2
        and x.dtype == torch.bfloat16
        and x.is_contiguous()
        and x.shape[0] >= min_t
        and x.shape[1] % 64 == 0
        and dst2src.dtype == torch.int32
        and offsets.dtype == torch.int32
        and dst2src.numel() == x.shape[0] * top_k
        and not torch.compiler.is_compiling()
    )


def make_quantize(stock, outputs, cfg):
    min_t = cfg["min_t"]

    def nvfp4_routed_quantize(x, dst2src, top_k, offsets):
        if eligible(x, dst2src, top_k, offsets, min_t):
            return quant_once(stock, outputs, x, dst2src, top_k, offsets)
        return stock(x=x, dst2src=dst2src, top_k=top_k, offsets=offsets)

    nvfp4_routed_quantize.__wrapped__ = stock
    setattr(nvfp4_routed_quantize, MARK, cfg)
    return nvfp4_routed_quantize


def patch(module, cfg):
    stock = getattr(module, TARGET_ATTR, None)
    outputs = getattr(module, "_nvfp4_outputs", None)
    if stock is None or outputs is None:
        raise RuntimeError(
            f"inkling_prefill.quant_once: {TARGET_MODULE}.{TARGET_ATTR} or _nvfp4_outputs is missing"
        )
    if getattr(stock, MARK, None) is not None:
        return False
    setattr(module, TARGET_ATTR, make_quantize(stock, outputs, cfg))
    return True
