"""Fused short conv + post-op for the K / V prefill prologue of sliding-window layers (imported on the GPU path).

A copy of sglang.kernels.ops.mamba.inkling2_sconv._sconv_prefill_kernel (static weights, residual; the same FP32
conv code, enable_fp_fusion=False like stock) whose BF16 output feeds, in registers, the op stock runs next:
    POST = 1 (K): QKL2Clip without scale / output scale:  ss = sum(y^2); norm = ss * rsqrt(ss) (0 if ss == 0);
                  y * min(1, thr / (norm + eps))
    POST = 2 (V): RMSNorm without weight: y * rsqrt(ss / 128 + eps)
One program = 16 rows x one 128-wide head (BN = head_dim), so the head reduction stays in the program. With ORDER=1
the row sum of squares follows the CuTe kernels' reduction tree (_row_sumsq), so the result is bit-identical to the
stock pair; ORDER=0 is the earlier tl.sum version (FP32 order class, 1-ulp flips). Every rounding point (BF16 conv
output, BF16 result) is kept.
"""

import triton
import triton.language as tl

from sglang.kernels.numerics import rsqrt_approx_f32


# Row sum of squares in the CuTe kernels' order. CuTe's row_sum: each thread folds its own register elements
# (x.reduce, init 0), then cute.arch.warp_reduction combines threads_per_row lanes with butterfly shuffles.
#   QK L2 clip, N = 128: 8 threads x 2 vectors of 8 (thread t holds columns 8t..8t+7 and 64+8t..64+8t+7)
#   RMSNorm,    N = 128, no weight (widest dtype FP32 -> vecsize 4): 16 threads x 2 vectors of 4 (thread t holds
#               columns 4t..4t+3 and 64+4t..64+4t+3)
# PER (per-thread fold): 0 sequential, mul then add; 1 sequential FMA (acc = fma(v, v, acc)); 2 pairwise halves
#   (products first); 3 pairwise halves with the first level fused (fma(hi, hi, lo * lo)); 4 the same as
#   fma(lo, lo, hi * hi). XOR (lane combine): 0 butterfly offsets descending (n/2 first), 1 ascending (1 first).
# FP32 addition is commutative, so a butterfly gives every lane the same value; only the tree shape matters.


# Triton note: a `return` inside an `if` does not end code generation of the statements after it, so every helper
# below has one return at the end and if / else branches only.


@triton.jit
def _sq_add(acc, v, PER: tl.constexpr):
    if PER == 1:
        r = tl.fma(v, v, acc)
    else:
        r = acc + v * v
    return r


@triton.jit
def _fold2(acc, x, PER: tl.constexpr):  # x [..., 2] (the last dim is the next fold index)
    a, b = tl.split(x)
    return _sq_add(_sq_add(acc, a, PER), b, PER)


@triton.jit
def _fold4(acc, x, PER: tl.constexpr):  # x [..., 2(low bit), 2(high bit)]
    a, b = tl.split(x)
    return _fold2(_fold2(acc, a, PER), b, PER)


@triton.jit
def _fold8(acc, x, PER: tl.constexpr):  # x [..., 2, 2, 2] bits low .. high
    a, b = tl.split(x)
    return _fold4(_fold4(acc, a, PER), b, PER)


@triton.jit
def _fold16(acc, x, PER: tl.constexpr):  # x [..., 2, 2, 2, 2] bits low .. high
    a, b = tl.split(x)
    return _fold8(_fold8(acc, a, PER), b, PER)


@triton.jit
def _first_level(x, PER: tl.constexpr):  # pairwise halves, level 1 on squares: x [..., 2(high bit)] -> [...]
    lo, hi = tl.split(x)
    if PER == 3:
        r = tl.fma(hi, hi, lo * lo)
    elif PER == 4:
        r = tl.fma(lo, lo, hi * hi)
    else:
        r = lo * lo + hi * hi
    return r


@triton.jit
def _halves(x):  # pairwise halves of values already squared / summed: x [..., 2(high bit)] -> [...]
    lo, hi = tl.split(x)
    return lo + hi


@triton.jit
def _thread_sum16(x, BM: tl.constexpr, NT: tl.constexpr, PER: tl.constexpr):
    """x [BM, NT, 16] (k = element index in the thread's fold order) -> [BM, NT]."""
    if PER <= 1:
        b = tl.permute(tl.reshape(x, (BM, NT, 2, 2, 2, 2)), (0, 1, 5, 4, 3, 2))  # bits low .. high
        r = _fold16(tl.zeros((BM, NT), tl.float32), b, PER)
    else:
        b = tl.reshape(x, (BM, NT, 2, 2, 2, 2))  # dims: bit3, bit2, bit1, bit0
        b = tl.permute(b, (0, 1, 3, 4, 5, 2))  # high bit last
        b = _first_level(b, PER)  # [BM, NT, 2(bit2), 2(bit1), 2(bit0)]
        b = _halves(tl.permute(b, (0, 1, 3, 4, 2)))
        b = _halves(tl.permute(b, (0, 1, 3, 2)))
        r = _halves(b)
    return r


@triton.jit
def _thread_sum8(x, BM: tl.constexpr, NT: tl.constexpr, PER: tl.constexpr):
    """x [BM, NT, 8] -> [BM, NT]."""
    if PER <= 1:
        b = tl.permute(tl.reshape(x, (BM, NT, 2, 2, 2)), (0, 1, 4, 3, 2))
        r = _fold8(tl.zeros((BM, NT), tl.float32), b, PER)
    else:
        b = tl.permute(tl.reshape(x, (BM, NT, 2, 2, 2)), (0, 1, 3, 4, 2))
        b = _first_level(b, PER)
        b = _halves(tl.permute(b, (0, 1, 3, 2)))
        r = _halves(b)
    return r


@triton.jit
def _lane_sum(x, BM: tl.constexpr, NT: tl.constexpr, XOR: tl.constexpr):
    """Butterfly over NT (8 or 16) lanes: x [BM, NT] -> [BM]."""
    if NT == 16:
        b = tl.reshape(x, (BM, 2, 2, 2, 2))  # lane bits high .. low
        if XOR == 0:
            b = _halves(tl.permute(b, (0, 2, 3, 4, 1)))
            b = _halves(tl.permute(b, (0, 2, 3, 1)))
            b = _halves(tl.permute(b, (0, 2, 1)))
            r = _halves(b)
        else:
            b = _halves(b)
            b = _halves(b)
            b = _halves(b)
            r = _halves(b)
    else:
        b = tl.reshape(x, (BM, 2, 2, 2))
        if XOR == 0:
            b = _halves(tl.permute(b, (0, 2, 3, 1)))
            b = _halves(tl.permute(b, (0, 2, 1)))
            r = _halves(b)
        else:
            b = _halves(b)
            b = _halves(b)
            r = _halves(b)
    return r


@triton.jit
def _row_sumsq(y, BM: tl.constexpr, POST: tl.constexpr, PER: tl.constexpr, XOR: tl.constexpr):
    """Sum of squares of each 128-wide row of y [BM, 128] in the stock CuTe kernel's order (POST 1 clip, 2 RMSNorm)."""
    if POST == 1:
        x = tl.reshape(y, (BM, 2, 8, 8))  # column = 64 j + 8 t + e
        x = tl.reshape(tl.permute(x, (0, 2, 1, 3)), (BM, 8, 16))  # [row, t, 8 j + e]
        r = _lane_sum(_thread_sum16(x, BM, 8, PER), BM, 8, XOR)
    else:
        x = tl.reshape(y, (BM, 2, 16, 4))  # column = 64 j + 4 t + e
        x = tl.reshape(tl.permute(x, (0, 2, 1, 3)), (BM, 16, 8))  # [row, t, 4 j + e]
        r = _lane_sum(_thread_sum8(x, BM, 16, PER), BM, 16, XOR)
    return r


@triton.jit
def _sconv_post_kernel(
    X, Weight, Cache, Indices, Mask, Cu, Seq, Y, n_rows,
    SX: tl.constexpr, SW0: tl.constexpr, SW1: tl.constexpr, SC0: tl.constexpr, SC1: tl.constexpr,
    SC2: tl.constexpr, D: tl.constexpr, W: tl.constexpr, SCALE: tl.constexpr, POST: tl.constexpr, THR, EPS,
    BM: tl.constexpr, BN: tl.constexpr, PER: tl.constexpr = 1, XOR: tl.constexpr = 0, ORDER: tl.constexpr = 1,
    SS=None, WRITE_SS: tl.constexpr = False,
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
    y = acc.to(tl.bfloat16).to(tl.float32)  # the stock conv output (BF16)
    if ORDER:
        ss = _row_sumsq(y, BM, POST, PER, XOR)  # bit-exact: the CuTe kernels' reduction tree
    else:
        ss = tl.sum(y * y, 1)
    if WRITE_SS:  # check only: the FP32 row sum of squares, [n_rows, D // BN]
        tl.store(SS + rows * (D // BN) + tl.program_id(1), ss, (tl.program_id(0) * BM + tl.arange(0, BM)) < n_rows)
    if POST == 1:
        norm = tl.where(ss > 0.0, ss * rsqrt_approx_f32(ss), 0.0)
        coeff = tl.minimum(1.0, tl.div_rn(tl.full(norm.shape, THR, tl.float32), norm + EPS),
                           propagate_nan=tl.PropagateNan.ALL)  # stock minimumf propagates NaN
    else:
        coeff = rsqrt_approx_f32(tl.div_rn(ss, tl.full(ss.shape, BN, tl.float32)) + EPS)
    out = y * coeff[:, None]
    offs = rows[:, None].to(tl.int64) * D + col[None, :]
    tl.store(Y + offs, out.to(tl.bfloat16), valid)
