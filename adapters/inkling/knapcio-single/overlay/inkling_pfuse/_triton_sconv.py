"""Triton kernel of the sconvres lever (imported only on the GPU path).

A copy of sglang.kernels.ops.mamba.inkling2_sconv._sconv_prefill_kernel (static weights, residual) whose store is
replaced by the layer's residual update:
    stock   y = bf16(conv)                                      (_sconv_prefill_kernel, BF16 output)
            partial' = bf16(((0 + partial) + attention) + y)   (inkling2_residual_add, FP32 sum in input order)
         or partial' = bf16(attention + y)                       (block boundary: torch BF16 add, FP32 internally)
    here    the same expressions on the conv accumulator rounded to BF16 in registers; one store.
Bit-identical by construction: identical FP32 conv code (copied, enable_fp_fusion=False like stock), identical
BF16 rounding of the conv output, identical order of the residual sum.
"""

import triton
import triton.language as tl


@triton.jit
def _sconv_residual_kernel(
    X,
    Weight,
    Cache,
    Indices,
    Mask,
    Cu,
    Seq,
    Attn,
    Partial,
    Y,
    n_rows,
    SX: tl.constexpr,
    SW0: tl.constexpr,
    SW1: tl.constexpr,
    SC0: tl.constexpr,
    SC1: tl.constexpr,
    SC2: tl.constexpr,
    D: tl.constexpr,
    W: tl.constexpr,
    SCALE: tl.constexpr,
    HAS_PARTIAL: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
):
    rows = tl.program_id(0) * BM + tl.arange(0, BM)
    col = tl.program_id(1) * BN + tl.arange(0, BN)
    valid = (rows < n_rows)[:, None] & (col < D)[None, :]
    rows = tl.minimum(rows, n_rows - 1)
    seq = tl.load(Seq + rows)
    bos = tl.load(Cu + seq)
    slot = tl.load(Indices + seq).to(tl.int64)
    has_cache = (tl.load(Mask + seq) != 0) & (slot > 0)
    current = tl.load(X + rows[:, None] * SX + col[None, :], valid, other=0).to(tl.float32)
    acc = tl.full((BM, BN), 0, tl.float32)
    for tap in tl.static_range(W):
        lag = W - 1 - tap
        src = rows - lag
        from_x = (src >= bos)[:, None]
        value = tl.load(X + src[:, None] * SX + col[None, :], valid & from_x, other=0).to(tl.float32)
        prefix = tl.load(
            Cache + slot[:, None] * SC0 + (src - bos + W - 1)[:, None] * SC1 + col[None, :] * SC2,
            valid & ~from_x & has_cache[:, None],
            other=0,
        ).to(tl.float32)
        value = tl.where(from_x, value, prefix)
        weight = (
            tl.load(Weight + col * SW0 + tap * SW1, col < D, other=0).to(X.dtype.element_ty).to(tl.float32)
        )[None, :]
        acc = tl.fma(value, weight, acc)
    acc = acc + current
    acc = acc * SCALE
    conv = acc.to(tl.bfloat16).to(tl.float32)
    offs = rows[:, None].to(tl.int64) * D + col[None, :]
    attn = tl.load(Attn + offs, valid, other=0).to(tl.float32)
    if HAS_PARTIAL:
        total = tl.zeros((BM, BN), tl.float32)
        total += tl.load(Partial + offs, valid, other=0).to(tl.float32)
        total += attn
        total += conv
    else:
        total = attn + conv
    tl.store(Y + offs, total.to(tl.bfloat16), valid)
