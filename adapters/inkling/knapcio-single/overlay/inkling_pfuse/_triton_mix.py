"""Triton kernels of inkling_pfuse (imported only on the GPU path).

Notation: D = 2304 (hidden), a "source" is one [T, D] BF16 tensor of the attention-residual state (blocks and
partial) or of the loop message bank. attn_res over sources x_0..x_{N-1} with query q (already scaled by D**-0.5):

    rstd_n = rsqrt(mean(x_n^2) + 1e-6); logit_n = (x_n . q) * rstd_n; e = softmax-numerator(logit)
    out = (sum_n fma(e_n * rstd_n, x_n)) * (1 / sum_n e_n)          (normalize_values; else e_n instead of e_n*rstd_n)

exactly the per-element formula of the stock CuTe kernel (fma chain in source order starting from 0, then one
multiply by the reciprocal); only the order of the FP32 row reductions (sum of squares, dots, norm statistics)
differs. All kernels here are compiled with enable_fp_fusion=False like the stock Triton kernels, so the
elementwise parts are not contracted differently.

Kernels
  _resnorm_kernel     attn_res (1-3 queries, optional addend on the last source) + RMSNorm(weight) -> BF16, one row
                      per program, FP32 attn_res output never stored
  _loop_stats_kernel  state attn_res (dest) and bank attn_res (messages) of a loop message mix + the controller
                      RMS statistics; writes the controller's normed BF16 input [num, T, 2D] (what the stock
                      _controller_norm writes) and the softmax weights of both mixes (a few floats per row)
  _loop_mix_kernel    per block of rows: rebuilds dest and messages from the sources and the stored weights,
                      hidden = bf16(sum of the stock down-projection partials), out = (1 - tanh(h Wf)) dest +
                      tanh(h Wg) msg (the stock _controller_mix formula), then either stores out (FP32, NORM=False)
                      or RMS-normalizes it with the layer norm weight in a second pass and stores BF16 (NORM=True)
  _controller_down_split  the stock split-K down projection reading the normed input as [ND + NM, T, D] (dest
                      planes, then message planes): a shared dest is written and read once instead of three times
  _loop_mix1_kernel   default replacement of _loop_mix_kernel: one controller per program, pipelined column loop
"""

import triton
import triton.language as tl
from triton.language.extra import libdevice

from sglang.kernels.numerics import rsqrt_approx_f32


@triton.jit
def _load_src(S, A, n: tl.constexpr, offs, mask, HAS_ADD: tl.constexpr):
    """Row values of source n as FP32 (plus the addend for the last source, like the stock fadd_rn)."""
    x = tl.load(S[n] + offs, mask, other=0.0).to(tl.float32)
    if HAS_ADD:
        if n == len(S) - 1:
            x = x + tl.load(A + offs, mask, other=0.0).to(tl.float32)
    return x


@triton.jit
def _row_weights(
    S, A, Q, row, eps, NQ: tl.constexpr, QP: tl.constexpr, NSP: tl.constexpr, HAS_ADD: tl.constexpr,
    NORM_VALUES: tl.constexpr, D: tl.constexpr, CH: tl.constexpr,
):
    """Softmax weights [NSP, QP] (e_n, times rstd_n when NORM_VALUES) and 1 / sum_n e_n [QP] of one row."""
    NS: tl.constexpr = len(S)
    cols = tl.arange(0, CH)
    qi = tl.arange(0, QP)
    ni = tl.arange(0, NSP)
    qmask = qi < NQ
    ss_all = tl.zeros([NSP], tl.float32)
    dot_all = tl.zeros([NSP, QP], tl.float32)
    for n in tl.static_range(NS):
        ss = tl.zeros([CH], tl.float32)
        dq = tl.zeros([QP, CH], tl.float32)
        for c in tl.static_range(D // CH):
            x = _load_src(S, A, n, row * D + c * CH + cols, cols < CH, HAS_ADD)
            q = tl.load(Q + qi[:, None] * D + c * CH + cols[None, :], qmask[:, None], other=0.0)
            ss = tl.fma(x, x, ss)
            dq = tl.fma(x[None, :], q, dq)
        ss_all = tl.where(ni == n, tl.sum(ss, 0), ss_all)
        dot_all = tl.where(ni[:, None] == n, tl.sum(dq, 1)[None, :], dot_all)
    rstd = rsqrt_approx_f32(tl.fma(ss_all, 1.0 / D, eps))
    valid = (ni < NS)[:, None] & qmask[None, :]
    logit = tl.where(valid, dot_all * rstd[:, None], float("-inf"))
    top = tl.where(qmask, tl.max(logit, 0), 0.0)
    e = tl.where(valid, libdevice.exp(logit - top[None, :]), 0.0)
    total = tl.where(qmask, tl.sum(e, 0), 1.0)
    inv = tl.where(qmask, tl.div_rn(tl.full([QP], 1.0, tl.float32), total), 0.0)
    if NORM_VALUES:
        e = e * rstd[:, None]
    return e, inv


@triton.jit
def _row_combine(
    S, A, prob, inv, row, c: tl.constexpr, QP: tl.constexpr, NSP: tl.constexpr, HAS_ADD: tl.constexpr,
    D: tl.constexpr, CH: tl.constexpr,
):
    """attn_res outputs [QP, CH] of chunk c of one row."""
    cols = tl.arange(0, CH)
    ni = tl.arange(0, NSP)
    acc = tl.zeros([QP, CH], tl.float32)
    for n in tl.static_range(len(S)):
        x = _load_src(S, A, n, row * D + c * CH + cols, cols < CH, HAS_ADD)
        p = tl.sum(tl.where(ni[:, None] == n, prob, 0.0), 0)
        acc = tl.fma(p[:, None], x[None, :], acc)
    return acc * inv[:, None]


@triton.jit
def _pick(x, i: tl.constexpr, QP: tl.constexpr):
    """Row i of a [QP, CH] tensor (exact: the other rows contribute +0)."""
    qi = tl.arange(0, QP)
    return tl.sum(tl.where(qi[:, None] == i, x, 0.0), 0)


@triton.jit
def _resnorm_kernel(
    S, A, Q, W, Y, T, eps_res, eps_norm,
    NQ: tl.constexpr, QP: tl.constexpr, NSP: tl.constexpr, HAS_ADD: tl.constexpr, NORM_VALUES: tl.constexpr,
    D: tl.constexpr, CH: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    cols = tl.arange(0, CH)
    prob, inv = _row_weights(S, A, Q, row, eps_res, NQ, QP, NSP, HAS_ADD, NORM_VALUES, D, CH)
    ss = tl.zeros([QP, CH], tl.float32)
    for c in tl.static_range(D // CH):
        o = _row_combine(S, A, prob, inv, row, c, QP, NSP, HAS_ADD, D, CH)
        ss = ss + o * o  # stock RMSNorm: rounded square, then add (no contraction)
    rstd = rsqrt_approx_f32(tl.div_rn(tl.sum(ss, 1), tl.full([QP], D, tl.float32)) + eps_norm)
    for c in tl.static_range(D // CH):
        o = _row_combine(S, A, prob, inv, row, c, QP, NSP, HAS_ADD, D, CH) * rstd[:, None]
        for q in tl.static_range(NQ):
            w = tl.load(W[q] + c * CH + cols).to(tl.float32)
            y = _pick(o, q, QP) * w
            tl.store(Y + (q * T + row) * D + c * CH + cols, y.to(tl.bfloat16))


@triton.jit
def _loop_stats_kernel(
    S, A, QS, B, QM, Normed, SP, SI, BP, BI, T, eps_res, eps_ctrl,
    NQS: tl.constexpr, NM: tl.constexpr, QP: tl.constexpr, QPS: tl.constexpr, QPM: tl.constexpr,
    NSP: tl.constexpr, NBP: tl.constexpr,
    HAS_ADD: tl.constexpr, NORM_VALUES: tl.constexpr, SHARED: tl.constexpr, D: tl.constexpr, CH: tl.constexpr,
    SPLIT_NORMED: tl.constexpr, ND: tl.constexpr,
):
    """QP: query stride of the stored weights; QPS / QPM: padded query counts used for the state / bank math."""
    row = tl.program_id(0).to(tl.int64)
    cols = tl.arange(0, CH)
    qs = tl.arange(0, QPS)
    qm = tl.arange(0, QPM)
    sprob, sinv = _row_weights(S, A, QS, row, eps_res, NQS, QPS, NSP, HAS_ADD, NORM_VALUES, D, CH)
    bprob, binv = _row_weights(B, A, QM, row, eps_res, NM, QPM, NBP, False, True, D, CH)
    tl.store(SP + row * (NSP * QP) + tl.arange(0, NSP)[:, None] * QP + qs[None, :], sprob)
    tl.store(SI + row * QP + qs, sinv)
    tl.store(BP + row * (NBP * QP) + tl.arange(0, NBP)[:, None] * QP + qm[None, :], bprob)
    tl.store(BI + row * QP + qm, binv)
    dss = tl.zeros([QPS, CH], tl.float32)
    mss = tl.zeros([QPM, CH], tl.float32)
    for c in tl.static_range(D // CH):
        d = _row_combine(S, A, sprob, sinv, row, c, QPS, NSP, HAS_ADD, D, CH)
        m = _row_combine(B, A, bprob, binv, row, c, QPM, NBP, False, D, CH)
        dss = dss + d * d  # stock _controller_norm: tl.sum(dest * dest), fusion disabled
        mss = mss + m * m
    drstd = libdevice.rsqrt(tl.div_rn(tl.sum(dss, 1), tl.full([QPS], D, tl.float32)) + eps_ctrl)
    mrstd = libdevice.rsqrt(tl.div_rn(tl.sum(mss, 1), tl.full([QPM], D, tl.float32)) + eps_ctrl)
    for c in tl.static_range(D // CH):
        d = _row_combine(S, A, sprob, sinv, row, c, QPS, NSP, HAS_ADD, D, CH) * drstd[:, None]
        m = _row_combine(B, A, bprob, binv, row, c, QPM, NBP, False, D, CH) * mrstd[:, None]
        if SPLIT_NORMED:
            # [ND + NM, T, D]: normed dests (ND = 1 when shared), then the normed messages
            for j in tl.static_range(ND):
                tl.store(Normed + (j * T + row) * D + c * CH + cols, _pick(d, j, QPS).to(tl.bfloat16))
            for i in tl.static_range(NM):
                tl.store(Normed + ((ND + i) * T + row) * D + c * CH + cols, _pick(m, i, QPM).to(tl.bfloat16))
        else:
            for i in tl.static_range(NM):
                out = Normed + (i * T + row) * (2 * D) + c * CH + cols
                if SHARED:
                    tl.store(out, _pick(d, 0, QPS).to(tl.bfloat16))
                else:
                    tl.store(out, _pick(d, i, QPS).to(tl.bfloat16))
                tl.store(out + D, _pick(m, i, QPM).to(tl.bfloat16))


@triton.jit
def _tile_mix(S, A, P, I, rows, rmask, offs, j: tl.constexpr, NSP: tl.constexpr, QP: tl.constexpr,
              HAS_ADD: tl.constexpr, mask):
    """attn_res output j on a [BM, BN] tile from the stored row weights (same op order as _row_combine)."""
    acc = tl.zeros(offs.shape, tl.float32)
    for n in tl.static_range(len(S)):
        x = _load_src(S, A, n, offs, mask, HAS_ADD)
        p = tl.load(P + rows * (NSP * QP) + n * QP + j, rmask, other=0.0)
        acc = tl.fma(p[:, None], x, acc)
    return acc * tl.load(I + rows * QP + j, rmask, other=0.0)[:, None]


@triton.jit
def _hidden(Part, i: tl.constexpr, rows, rmask, T, GROUPS: tl.constexpr, R: tl.constexpr, BR: tl.constexpr):
    """bf16 controller hidden [BM, BR] of controller i: split-K partials summed in group order (stock mix)."""
    rk = tl.arange(0, BR)
    ok = rmask[:, None] & (rk[None, :] < R)
    p = Part + ((i * GROUPS) * T + rows[:, None]) * R + rk[None, :]
    h = tl.load(p, ok, other=0.0)
    for g in tl.static_range(1, GROUPS):
        h = h + tl.load(p + g * T * R, ok, other=0.0)
    return h.to(tl.bfloat16)


@triton.jit
def _ctrl_out(h, Gain, Forget, i: tl.constexpr, cols, dest, msg, D: tl.constexpr, R: tl.constexpr,
              BR: tl.constexpr):
    """(1 - tanh(h @ forget.T)) * dest + tanh(h @ gain.T) * msg on one tile (stock _controller_mix)."""
    rk = tl.arange(0, BR)
    wofs = i * D * R + cols[None, :] * R + rk[:, None]
    wok = (rk[:, None] < R) & (cols[None, :] < D)
    gain = tl.dot(h, tl.load(Gain + wofs, wok, other=0.0))
    forget = tl.dot(h, tl.load(Forget + wofs, wok, other=0.0))
    return (1.0 - libdevice.tanh(forget)) * dest + libdevice.tanh(gain) * msg


@triton.jit
def _loop_mix_kernel(
    S, A, B, SP, SI, BP, BI, Part, Gain, Forget, W, Y, T, eps_norm,
    NQS: tl.constexpr, NM: tl.constexpr, QP: tl.constexpr, NSP: tl.constexpr, NBP: tl.constexpr,
    HAS_ADD: tl.constexpr, SHARED: tl.constexpr, NORM: tl.constexpr, GROUPS: tl.constexpr, R: tl.constexpr,
    BR: tl.constexpr, D: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr,
):
    rows = tl.program_id(0) * BM + tl.arange(0, BM)
    rmask = rows < T
    rows = tl.minimum(rows, T - 1).to(tl.int64)
    h0 = _hidden(Part, 0, rows, rmask, T, GROUPS, R, BR)
    h1 = h0
    h2 = h0
    if NM > 1:
        h1 = _hidden(Part, 1, rows, rmask, T, GROUPS, R, BR)
        h2 = _hidden(Part, 2, rows, rmask, T, GROUPS, R, BR)
    ss0 = tl.zeros([BM], tl.float32)
    ss1 = tl.zeros([BM], tl.float32)
    ss2 = tl.zeros([BM], tl.float32)
    PASSES: tl.constexpr = 2 if NORM else 1
    rstd0 = ss0
    rstd1 = ss0
    rstd2 = ss0
    for p in tl.static_range(PASSES):
        if p == 1:
            rstd0 = rsqrt_approx_f32(tl.div_rn(ss0, tl.full([BM], D, tl.float32)) + eps_norm)
            rstd1 = rsqrt_approx_f32(tl.div_rn(ss1, tl.full([BM], D, tl.float32)) + eps_norm)
            rstd2 = rsqrt_approx_f32(tl.div_rn(ss2, tl.full([BM], D, tl.float32)) + eps_norm)
        for start in range(0, D, BN):
            cols = start + tl.arange(0, BN)
            mask = rmask[:, None] & (cols[None, :] < D)
            offs = rows[:, None] * D + cols[None, :]
            d0 = _tile_mix(S, A, SP, SI, rows, rmask, offs, 0, NSP, QP, HAS_ADD, mask)
            d1 = d0
            d2 = d0
            if not SHARED:
                if NQS > 1:
                    d1 = _tile_mix(S, A, SP, SI, rows, rmask, offs, 1, NSP, QP, HAS_ADD, mask)
                    d2 = _tile_mix(S, A, SP, SI, rows, rmask, offs, 2, NSP, QP, HAS_ADD, mask)
            o0 = _ctrl_out(h0, Gain, Forget, 0, cols, d0,
                           _tile_mix(B, A, BP, BI, rows, rmask, offs, 0, NBP, QP, False, mask), D, R, BR)
            o1 = o0
            o2 = o0
            if NM > 1:
                o1 = _ctrl_out(h1, Gain, Forget, 1, cols, d1,
                               _tile_mix(B, A, BP, BI, rows, rmask, offs, 1, NBP, QP, False, mask), D, R, BR)
                o2 = _ctrl_out(h2, Gain, Forget, 2, cols, d2,
                               _tile_mix(B, A, BP, BI, rows, rmask, offs, 2, NBP, QP, False, mask), D, R, BR)
            if p == 0:
                if NORM:
                    ss0 = ss0 + tl.sum(tl.where(mask, o0 * o0, 0.0), 1)
                    if NM > 1:
                        ss1 = ss1 + tl.sum(tl.where(mask, o1 * o1, 0.0), 1)
                        ss2 = ss2 + tl.sum(tl.where(mask, o2 * o2, 0.0), 1)
                else:
                    tl.store(Y + offs, o0, mask)
                    if NM > 1:
                        tl.store(Y + T * D + offs, o1, mask)
                        tl.store(Y + 2 * T * D + offs, o2, mask)
            else:
                w0 = tl.load(W[0] + cols, cols < D, other=0.0).to(tl.float32)
                tl.store(Y + offs, ((o0 * rstd0[:, None]) * w0[None, :]).to(tl.bfloat16), mask)
                if NM > 1:
                    w1 = tl.load(W[1] + cols, cols < D, other=0.0).to(tl.float32)
                    w2 = tl.load(W[2] + cols, cols < D, other=0.0).to(tl.float32)
                    tl.store(Y + T * D + offs, ((o1 * rstd1[:, None]) * w1[None, :]).to(tl.bfloat16), mask)
                    tl.store(Y + 2 * T * D + offs, ((o2 * rstd2[:, None]) * w2[None, :]).to(tl.bfloat16), mask)


@triton.jit
def _controller_down_split(
    Normed,
    Down,
    Partials,
    T,
    R: tl.constexpr,
    K: tl.constexpr,
    SPLIT: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    NUM: tl.constexpr,
    ND: tl.constexpr,
):
    """sglang _controller_down (split-K normed @ down.T, 1024-wide K groups) with the normed input stored as
    [ND + NUM, T, K / 2]: rows 0..ND-1 the normed dests (ND = 1: one dest shared by all controllers), rows ND + m
    the normed message of controller m. Controller m's K columns [0, K/2) read dest (m if ND > 1 else 0) and
    [K/2, K) read its message; a BLOCK_K tile never straddles K/2 (asserted on the host), so every tl.dot sees
    exactly the tile the stock kernel loads from its [NUM, T, K] buffer. Controllers vary fastest in the grid so
    the shared dest tiles are re-read from L2."""
    GROUPS: tl.constexpr = (K + 1023) // 1024
    HALF: tl.constexpr = K // 2
    z = tl.program_id(0)
    rows = tl.program_id(1) * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = tl.program_id(2) * BLOCK_N + tl.arange(0, BLOCK_N)
    if SPLIT:
        m = z % NUM
        first = z // NUM
        count = 1
        zout = m * GROUPS + first
    else:
        m = z
        first = 0
        count = GROUPS
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
            sel = tl.where(k0 < HALF, dsel, ND + m).to(tl.int64)
            kk = tl.where(k0 < HALF, k0, k0 - HALF) + ks
            k = k0 + ks
            x = tl.load(
                Normed + (sel * T + rows[:, None].to(tl.int64)) * HALF + kk[None, :],
                (rows[:, None] < T) & (k[None, :] < K),
                other=0,
            )
            w = tl.load(
                Wd + cols[None, :] * K + k[:, None],
                (cols[None, :] < R) & (k[:, None] < K),
                other=0,
            )
            partial = tl.dot(x, w, partial)
        if group == first:
            total = partial
        else:
            total = total + partial
    tl.store(
        out + rows[:, None].to(tl.int64) * R + cols[None, :],
        total,
        (rows[:, None] < T) & (cols[None, :] < R),
    )


@triton.jit
def _loop_mix1_kernel(
    S, A, B, SP, SI, BP, BI, Part, Gain, Forget, Wn, Y, T, eps_norm,
    NM: tl.constexpr, QP: tl.constexpr, NSP: tl.constexpr, NBP: tl.constexpr,
    HAS_ADD: tl.constexpr, SHARED: tl.constexpr, NORM: tl.constexpr, GROUPS: tl.constexpr, R: tl.constexpr,
    BR: tl.constexpr, D: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr, STAGES: tl.constexpr,
):
    """_loop_mix_kernel with one controller per program (controller = program_id(0), fastest, so the programs
    of one row block share their source tiles through L2) and a software-pipelined column loop. Wn: the stacked
    [NM, D] norm weights (NORM). Same per-element arithmetic as _loop_mix_kernel."""
    i = tl.program_id(0)
    rows = tl.program_id(1) * BM + tl.arange(0, BM)
    rmask = rows < T
    rows = tl.minimum(rows, T - 1).to(tl.int64)
    rk = tl.arange(0, BR)
    ok = rmask[:, None] & (rk[None, :] < R)
    p = Part + ((i * GROUPS) * T + rows[:, None]) * R + rk[None, :]
    h = tl.load(p, ok, other=0.0)
    for g in tl.static_range(1, GROUPS):
        h = h + tl.load(p + g * T * R, ok, other=0.0)
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
