"""Triton kernels of the prefill bank hoist on top of inkling_pfuse's loop-mix path.

The loop message bank (2 BF16 sources) and the 48 message queries (12 core layers x attn 3 + mlp 1) are fixed for
a whole pass. pfuse's per-visit kernels redo the bank part every visit:
  _loop_stats_kernel   bank attn-res weights, message sum of squares, normed message planes (BF16, written)
  _controller_down_split  reads the normed message planes for K columns 2304-4607 (groups 2 part, 3, 4)
Hoisted (once per pass):
  _bank_stats_kernel   per row and (layer, side): the bank weights bp / bi (pfuse's _row_weights call, same
                       constexprs) and the message rstd (pfuse's _row_combine sum of squares, same CH / shapes)
  _bank_down_kernel    K groups 3 and 4 (wholly message columns) of every controller, the normed message tile
                       rebuilt on the fly from the bank (stats kernel's op order) and fed to the same tl.dot tile
                       chain as _controller_down_split; partials [NLS, NM, 2, T, R]
Per visit:
  _stats_dest_kernel   pfuse _loop_stats_kernel with the bank part removed (state weights, dest normed planes only)
  _down_visit_kernel   _controller_down_split restricted to groups 0-2; group 2's message columns (0-767 of the
                       message half) rebuilt from the bank like _bank_down_kernel
  _mix1_hoist_kernel   pfuse _loop_mix1_kernel whose hidden sums the visit partials and then the hoisted groups 3
                       and 4, in group order
Every hoisted or rebuilt value is computed by the same expressions (helpers imported from pfuse's module) in the
same order, so the pass output is meant to be bit-identical to pfuse's loop_mix (checked on the GPU).
"""

import triton
import triton.language as tl
from triton.language.extra import libdevice

from inkling_pfuse._triton_mix import _load_src, _pick, _row_combine, _row_weights  # noqa: F401
from sglang.kernels.numerics import rsqrt_approx_f32


@triton.jit
def _bank_stats_kernel(
    B, QM, BP, BI, MR, T, eps_res, eps_ctrl, LS_Q_STRIDE, Q_OFF,
    NM: tl.constexpr, QP: tl.constexpr, QPM: tl.constexpr, NBP: tl.constexpr, D: tl.constexpr, CH: tl.constexpr,
):
    """Program (row, ls): bank weights of the NM queries of layer-side ls (queries at QM + ls * LS_Q_STRIDE + Q_OFF)
    and the message rstd, exactly as _loop_stats_kernel computes them. BP [NLS, T, NBP, QP], BI [NLS, T, QP],
    MR [NLS, T, QPM]."""
    row = tl.program_id(0).to(tl.int64)
    ls = tl.program_id(1).to(tl.int64)
    qm = tl.arange(0, QPM)
    Q = QM + ls * LS_Q_STRIDE + Q_OFF
    bprob, binv = _row_weights(B, B[0], Q, row, eps_res, NM, QPM, NBP, False, True, D, CH)
    tl.store(BP + (ls * T + row) * (NBP * QP) + tl.arange(0, NBP)[:, None] * QP + qm[None, :], bprob)
    tl.store(BI + (ls * T + row) * QP + qm, binv)
    mss = tl.zeros([QPM, CH], tl.float32)
    for c in tl.static_range(D // CH):
        m = _row_combine(B, B[0], bprob, binv, row, c, QPM, NBP, False, D, CH)
        mss = mss + m * m
    mrstd = libdevice.rsqrt(tl.div_rn(tl.sum(mss, 1), tl.full([QPM], D, tl.float32)) + eps_ctrl)
    tl.store(MR + (ls * T + row) * QPM + qm, mrstd)


@triton.jit
def _normed_msg_tile(B, BP, BI, MR, rows, rmask, kk, kmask, i, T,
                     NBP: tl.constexpr, QP: tl.constexpr, QPM: tl.constexpr, D: tl.constexpr):
    """bf16 normed message i on rows x columns kk (of the message half): the value _loop_stats_kernel stores,
    bf16((sum_n fma(p_n, x_n, acc) * inv) * rstd) with acc from 0 in source order, 0 outside the masks
    (the masked load's other=0 of _controller_down_split). BP / BI / MR already offset to this layer-side."""
    mask = rmask[:, None] & kmask[None, :]
    acc = tl.zeros([rows.shape[0], kk.shape[0]], tl.float32)
    for n in tl.static_range(NBP):
        x = tl.load(B[n] + rows[:, None] * D + kk[None, :], mask, other=0.0).to(tl.float32)
        p = tl.load(BP + rows * (NBP * QP) + n * QP + i, rmask, other=0.0)
        acc = tl.fma(p[:, None], x, acc)
    m = acc * tl.load(BI + rows * QP + i, rmask, other=0.0)[:, None]
    m = m * tl.load(MR + rows * QPM + i, rmask, other=0.0)[:, None]
    if QPM > 1:
        # pfuse's _pick(m, i, QPM) sums the QPM query slots with the others set to +0.0: x + 0.0 (turns -0.0 into
        # +0.0, exact otherwise). QPM == 1 is a single-element sum: no addition.
        m = m + 0.0
    return tl.where(mask, m, 0.0).to(tl.bfloat16)


@triton.jit
def _bank_down_kernel(
    B, BP, BI, MR, Down, P34, T, LS_D_STRIDE, D_OFF,
    R: tl.constexpr, K: tl.constexpr, NM: tl.constexpr, NBP: tl.constexpr, QP: tl.constexpr,
    QPM: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, D: tl.constexpr,
):
    """Program (z = (ls * NM + m) * 2 + g, row block, rank block): K group 3 + g of controller m of layer-side ls.
    The group's tl.dot chain is the one _controller_down_split runs for that group (same BLOCK_K tiles in order)."""
    HALF: tl.constexpr = K // 2
    z = tl.program_id(0)
    g = z % 2
    lm = z // 2
    m = lm % NM
    ls = lm // NM
    group = 3 + g
    rows = tl.program_id(1) * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = tl.program_id(2) * BLOCK_N + tl.arange(0, BLOCK_N)
    rmask = rows < T
    rows64 = rows.to(tl.int64)
    lsT = ls.to(tl.int64) * T
    BPl = BP + lsT * (NBP * QP)
    BIl = BI + lsT * QP
    MRl = MR + lsT * QPM
    Wd = Down + ls.to(tl.int64) * LS_D_STRIDE + (D_OFF + m.to(tl.int64)) * R * K
    ks = tl.arange(0, BLOCK_K)
    partial = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
    for tile in range(1024 // BLOCK_K):
        k0 = group * 1024 + tile * BLOCK_K
        k = k0 + ks
        x = _normed_msg_tile(B, BPl, BIl, MRl, rows64, rmask, k0 - HALF + ks, k < K, m, T, NBP, QP, QPM, D)
        w = tl.load(Wd + cols[None, :] * K + k[:, None], (cols[None, :] < R) & (k[:, None] < K), other=0)
        partial = tl.dot(x, w, partial)
    out = P34 + (z.to(tl.int64) * T) * R
    tl.store(out + rows64[:, None] * R + cols[None, :], partial, rmask[:, None] & (cols[None, :] < R))


@triton.jit
def _stats_dest_kernel(
    S, A, QS, Normed, SP, SI, T, eps_res, eps_ctrl,
    NQS: tl.constexpr, QP: tl.constexpr, QPS: tl.constexpr, NSP: tl.constexpr,
    HAS_ADD: tl.constexpr, NORM_VALUES: tl.constexpr, D: tl.constexpr, CH: tl.constexpr, ND: tl.constexpr,
):
    """_loop_stats_kernel (SPLIT_NORMED layout) without the bank: sp / si and the ND normed dest planes
    [ND, T, D]."""
    row = tl.program_id(0).to(tl.int64)
    cols = tl.arange(0, CH)
    qs = tl.arange(0, QPS)
    sprob, sinv = _row_weights(S, A, QS, row, eps_res, NQS, QPS, NSP, HAS_ADD, NORM_VALUES, D, CH)
    tl.store(SP + row * (NSP * QP) + tl.arange(0, NSP)[:, None] * QP + qs[None, :], sprob)
    tl.store(SI + row * QP + qs, sinv)
    dss = tl.zeros([QPS, CH], tl.float32)
    for c in tl.static_range(D // CH):
        d = _row_combine(S, A, sprob, sinv, row, c, QPS, NSP, HAS_ADD, D, CH)
        dss = dss + d * d
    drstd = libdevice.rsqrt(tl.div_rn(tl.sum(dss, 1), tl.full([QPS], D, tl.float32)) + eps_ctrl)
    for c in tl.static_range(D // CH):
        d = _row_combine(S, A, sprob, sinv, row, c, QPS, NSP, HAS_ADD, D, CH) * drstd[:, None]
        for j in tl.static_range(ND):
            tl.store(Normed + (j * T + row) * D + c * CH + cols, _pick(d, j, QPS).to(tl.bfloat16))


@triton.jit
def _down_visit_kernel(
    Normed, B, BP, BI, MR, Down, Partials, T,
    R: tl.constexpr, K: tl.constexpr, SPLIT: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr, NUM: tl.constexpr, ND: tl.constexpr, NBP: tl.constexpr, QP: tl.constexpr,
    QPM: tl.constexpr, D: tl.constexpr, G0: tl.constexpr = 0, NG: tl.constexpr = 3,
):
    """_controller_down_split for K groups G0 .. G0 + NG - 1 (visit: 0-2; pass boundary: 3-4). Columns < K/2 read
    the dest planes Normed [ND, T, K/2]; columns >= K/2 rebuild the normed message from the bank (BP / BI / MR of
    this layer-side). One code path for both uses (the x operand reaches tl.dot the same way in every group).
    SPLIT: program z = j * NUM + m writes group G0 + j of controller m into Partials [NUM, NG, T, R];
    else z = m sums the NG groups in order into Partials [NUM, 1, T, R]."""
    HALF: tl.constexpr = K // 2
    z = tl.program_id(0)
    rows = tl.program_id(1) * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = tl.program_id(2) * BLOCK_N + tl.arange(0, BLOCK_N)
    rmask = rows < T
    rows64 = rows.to(tl.int64)
    if SPLIT:
        m = z % NUM
        first = G0 + z // NUM
        count = 1
        zout = m * NG + z // NUM
    else:
        m = z
        first = G0
        count = NG
        zout = z
    out = Partials + (zout.to(tl.int64) * T) * R
    if ND > 1:
        dsel = m
    else:
        dsel = m * 0
    Wd = Down + m.to(tl.int64) * R * K
    ks = tl.arange(0, BLOCK_K)
    total = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
    for group in range(first, first + count):
        partial = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
        for tile in range(1024 // BLOCK_K):
            k0 = group * 1024 + tile * BLOCK_K
            k = k0 + ks
            if k0 < HALF:
                x = tl.load(
                    Normed + (dsel.to(tl.int64) * T + rows64[:, None]) * HALF + k[None, :],
                    rmask[:, None] & (k[None, :] < K),
                    other=0,
                )
            else:
                x = _normed_msg_tile(B, BP, BI, MR, rows64, rmask, k0 - HALF + ks, k < K, m, T, NBP, QP, QPM, D)
            w = tl.load(Wd + cols[None, :] * K + k[:, None], (cols[None, :] < R) & (k[:, None] < K), other=0)
            partial = tl.dot(x, w, partial)
        if group == first:
            total = partial
        else:
            total = total + partial
    tl.store(out + rows64[:, None] * R + cols[None, :], total, rmask[:, None] & (cols[None, :] < R))


@triton.jit
def _mix1_hoist_kernel(
    S, A, B, SP, SI, BP, BI, Part, P34, Gain, Forget, Wn, Y, T, eps_norm,
    NM: tl.constexpr, QP: tl.constexpr, NSP: tl.constexpr, NBP: tl.constexpr,
    HAS_ADD: tl.constexpr, SHARED: tl.constexpr, NORM: tl.constexpr, GV: tl.constexpr, R: tl.constexpr,
    BR: tl.constexpr, D: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr, STAGES: tl.constexpr,
):
    """pfuse _loop_mix1_kernel; the hidden is the visit partials (GV = 3 groups, or 1 pre-summed) followed by the
    hoisted groups 3 and 4 (P34 [NM, 2, T, R] of this layer-side), added in group order."""
    i = tl.program_id(0)
    rows = tl.program_id(1) * BM + tl.arange(0, BM)
    rmask = rows < T
    rows = tl.minimum(rows, T - 1).to(tl.int64)
    rk = tl.arange(0, BR)
    ok = rmask[:, None] & (rk[None, :] < R)
    p = Part + ((i * GV) * T + rows[:, None]) * R + rk[None, :]
    h = tl.load(p, ok, other=0.0)
    for g in tl.static_range(1, GV):
        h = h + tl.load(p + g * T * R, ok, other=0.0)
    q = P34 + ((i * 2) * T + rows[:, None]) * R + rk[None, :]
    h = h + tl.load(q, ok, other=0.0)
    h = h + tl.load(q + T * R, ok, other=0.0)
    h = h.to(tl.bfloat16)
    if SHARED:
        j = i * 0
    else:
        j = i
    ss = tl.zeros([BM], tl.float32)
    rstd = ss
    PASSES: tl.constexpr = 2 if NORM else 1
    for ps in tl.static_range(PASSES):
        if ps == 1:
            rstd = rsqrt_approx_f32(tl.div_rn(ss, tl.full([BM], D, tl.float32)) + eps_norm)
        for start in tl.range(0, D, BN, num_stages=STAGES):
            cols = start + tl.arange(0, BN)
            mask = rmask[:, None] & (cols[None, :] < D)
            offs = rows[:, None] * D + cols[None, :]
            dacc = tl.zeros([BM, BN], tl.float32)
            for n in tl.static_range(len(S)):
                x = _load_src(S, A, n, offs, mask, HAS_ADD)
                pn = tl.load(SP + rows * (NSP * QP) + n * QP + j, rmask, other=0.0)
                dacc = tl.fma(pn[:, None], x, dacc)
            dest = dacc * tl.load(SI + rows * QP + j, rmask, other=0.0)[:, None]
            macc = tl.zeros([BM, BN], tl.float32)
            for n in tl.static_range(len(B)):
                x = tl.load(B[n] + offs, mask, other=0.0).to(tl.float32)
                pn = tl.load(BP + rows * (NBP * QP) + n * QP + i, rmask, other=0.0)
                macc = tl.fma(pn[:, None], x, macc)
            msg = macc * tl.load(BI + rows * QP + i, rmask, other=0.0)[:, None]
            wofs = i * D * R + cols[None, :] * R + rk[:, None]
            wok = (rk[:, None] < R) & (cols[None, :] < D)
            gain = tl.dot(h, tl.load(Gain + wofs, wok, other=0.0))
            forget = tl.dot(h, tl.load(Forget + wofs, wok, other=0.0))
            out = (1.0 - libdevice.tanh(forget)) * dest + libdevice.tanh(gain) * msg
            if ps == 0:
                if NORM:
                    ss = ss + tl.sum(tl.where(mask, out * out, 0.0), 1)
                else:
                    tl.store(Y + i * T * D + offs, out, mask)
            else:
                w = tl.load(Wn + i * D + cols, cols < D, other=0.0).to(tl.float32)
                tl.store(Y + i * T * D + offs, ((out * rstd[:, None]) * w[None, :]).to(tl.bfloat16), mask)
