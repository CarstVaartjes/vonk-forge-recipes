"""Triton kernel of bias_layout (imported only on the GPU path).

out[t, h, :] is the sheared relative-bias row FA4's ShearingBias would write for
query token t, head h (inkling_prefill._layout.shear_params / simple_shear_row),
with pre[t, h, p] = sum_d r[t, h, d] * proj[d, p] computed here by one tl.dot
(K = 16, FP32 accumulate, BF16 round) instead of a materialised einsum output.
"""

import triton
import triton.language as tl


@triton.jit
def _sheared_rel_bias(
    R,
    P,
    Out,
    CuQ,
    SeqK,
    n_req,
    stride_rt,
    stride_rh,
    stride_ot,
    stride_oh,
    D: tl.constexpr,
    REL: tl.constexpr,
    PAD: tl.constexpr,
    LOCAL: tl.constexpr,
    WL: tl.constexpr,
    TM: tl.constexpr,
    BT: tl.constexpr,
    BC: tl.constexpr,
    SEARCH: tl.constexpr,
):
    h = tl.program_id(1)
    t = tl.program_id(0) * BT + tl.arange(0, BT)
    total = tl.load(CuQ + n_req)
    valid = t < total
    # request of each token: largest b with CuQ[b] <= t (zero-length slots skipped)
    lo = tl.zeros([BT], dtype=tl.int32)
    hi = tl.zeros([BT], dtype=tl.int32) + n_req
    for _ in tl.static_range(SEARCH):
        mid = (lo + hi) // 2
        start = tl.load(CuQ + mid, mask=valid, other=0)
        go = hi - lo > 1
        take = start <= t
        lo = tl.where(go & take, mid, lo)
        hi = tl.where(go & (~take), mid, hi)
    b = lo
    start = tl.load(CuQ + b, mask=valid, other=0)
    sq = tl.load(CuQ + b + 1, mask=valid, other=1) - start
    sk = tl.load(SeqK + b, mask=valid, other=1)
    m = t - start
    n_right = m + 1 + sk - sq
    if LOCAL:
        n_left = tl.maximum(m + sk - sq - WL, 0)
    else:
        n_left = tl.zeros([BT], dtype=tl.int32)
    nbv = n_right - tl.maximum(n_left, n_right - REL)
    m_block = m // TM
    nb_max = tl.minimum((sk + 127) // 128, ((m_block + 1) * TM + sk - sq + 127) // 128)
    nb_rel0 = (n_right - 1) // 128
    blk_right = 1 + tl.maximum(PAD // 128 - (nb_max - nb_rel0), 0)
    right = (blk_right - 1) * 128 + ((n_right - 1) % 128) + 1
    left = tl.maximum(right - nbv, 0)
    last_p = tl.minimum(right, nbv) - 1

    d = tl.arange(0, D)
    a = tl.load(
        R + t.to(tl.int64)[:, None] * stride_rt + h * stride_rh + d[None, :],
        mask=valid[:, None],
        other=0.0,
    )
    row = Out + t.to(tl.int64)[:, None] * stride_ot + h * stride_oh
    for c0 in tl.static_range(0, REL, BC):
        p = c0 + tl.arange(0, BC)
        w = tl.load(P + d[:, None] * REL + p[None, :])
        val = tl.dot(a, w).to(tl.bfloat16)
        col = right[:, None] - 1 - p[None, :]
        tl.store(row + col, val, mask=valid[:, None] & (p[None, :] <= last_p[:, None]))
    for c0 in tl.static_range(0, PAD, BC):
        i = c0 + tl.arange(0, BC)
        is_right = i[None, :] >= right[:, None]
        is_left = i[None, :] < left[:, None]
        if LOCAL:
            pad = tl.full([BT, BC], float("-inf"), tl.float32)
        else:
            pad = tl.where(is_right, float("-inf"), 0.0)
        tl.store(row + i[None, :], pad.to(tl.bfloat16), mask=valid[:, None] & (is_right | is_left))
