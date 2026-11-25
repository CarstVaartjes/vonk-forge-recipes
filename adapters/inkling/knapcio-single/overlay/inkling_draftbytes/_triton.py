"""Triton kernels (CUDA only; imported lazily by ops.py).

w8_mm:    Y[M, N] = X[M, K] @ dequant(W)[N, K]^T (+ B), FP32 accumulate. W is FP8
          e4m3 with fp32 per-group scales or e8m0 MX scales, or plain BF16.
          FP8 -> BF16 is exact; e8m0 scales are powers of two, so the MX dequant
          in BF16 is exact too; fp32 scales multiply the FP32 partial of each
          K tile (tiles never straddle a scale group).
hot_step: one draft step over the hot vocabulary: the Markov fold on the hot
          rows, the same BF16 roundings as the stock MarkovSampleStep
          (bias -> bf16, base + bias -> bf16, softcap -> bf16, vocab limit),
          scattered to the hot columns of a full-width row buffer.

Weight-streaming layout: one CTA per BN output rows (W rows) and MP input rows;
the W tile is loaded [BK, BN] (K contiguous in memory). M <= MP is padded by
masking; larger M uses a second grid dimension (ROWS 0, the default), which reads
W once per MP-row block.

wide8 (ROWS 1 / 2): one CTA per BN output rows for ALL row blocks. ROWS 1 loops
over the row blocks around the very same _acc_tile: it issues the CTA's W loads
once per row block, the second and later ones hitting L2 while the tile stays
resident (no guarantee). ROWS 2 loops over K once and applies each loaded W tile
to up to WIDE_NRB row blocks (one FP32 accumulator each): one W load per K tile
per CTA up to 6 x 64 = 384 rows (more falls back to ROWS 1). ROWS 2 runs with
num_stages 1 unless told otherwise (each 64 x BK activation tile is staged in
shared memory; deeper pipelining would not fit). Both run, per row
block, the same tl.dot shapes, K tiles and scale multiplies as ROWS 0, so each
output element is computed by the same expression.
"""

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

from .quant import SCALE_FP32


@triton.jit
def _acc_tile(
    X,
    W,
    S,
    row0,
    M,
    n,
    nmask,
    SX,
    K: tl.constexpr,
    G: tl.constexpr,
    BK: tl.constexpr,
    MP: tl.constexpr,
    BN: tl.constexpr,
    SCALE: tl.constexpr,
    W_FP8: tl.constexpr,
):
    rows = row0 + tl.arange(0, MP)
    rmask = rows < M
    NG: tl.constexpr = K // G
    acc = tl.zeros((MP, BN), tl.float32)
    for k0 in range(0, K, BK):
        ks = k0 + tl.arange(0, BK)
        kmask = ks < K
        w = tl.load(
            W + n[None, :].to(tl.int64) * K + ks[:, None],
            nmask[None, :] & kmask[:, None],
            other=0,
        )
        if W_FP8:
            # FP8 planes arrive as uint8 (a masked load cannot fill fp8 with
            # an int constant); reinterpret the bits after the load.
            w = w.to(tl.float8e4nv, bitcast=True)
        x = tl.load(
            X + rows[:, None].to(tl.int64) * SX + ks[None, :],
            rmask[:, None] & kmask[None, :],
            other=0,
        ).to(tl.bfloat16)
        if SCALE == 2:
            e = tl.load(
                S + n[None, :].to(tl.int64) * NG + ks[:, None] // G,
                nmask[None, :] & kmask[:, None],
                other=127,
            )
            wf = w.to(tl.float32) * tl.exp2(e.to(tl.float32) - 127.0)
            acc += tl.dot(x, wf.to(tl.bfloat16))
        elif SCALE == 1:
            s = tl.load(S + n.to(tl.int64) * NG + k0 // G, nmask, other=0)
            acc += tl.dot(x, w.to(tl.bfloat16)) * s[None, :]
        else:
            acc += tl.dot(x, w.to(tl.bfloat16))
    return acc, rows, rmask


WIDE_NRB = 6  # row blocks ROWS 2 keeps in registers (M <= 6 * 64 = 384)


@triton.jit
def _wtile(W, S, n, nmask, k0, K: tl.constexpr, G: tl.constexpr, BK: tl.constexpr, SCALE: tl.constexpr,
           W_FP8: tl.constexpr):
    """_acc_tile's weight side of K tile k0: (BF16 W tile [BK, BN], per-column FP32 scale or None)."""
    ks = k0 + tl.arange(0, BK)
    kmask = ks < K
    NG: tl.constexpr = K // G
    w = tl.load(
        W + n[None, :].to(tl.int64) * K + ks[:, None],
        nmask[None, :] & kmask[:, None],
        other=0,
    )
    if W_FP8:
        w = w.to(tl.float8e4nv, bitcast=True)
    s = nmask.to(tl.float32)  # placeholder (used only when SCALE == 1)
    if SCALE == 2:
        e = tl.load(
            S + n[None, :].to(tl.int64) * NG + ks[:, None] // G,
            nmask[None, :] & kmask[:, None],
            other=127,
        )
        wb = (w.to(tl.float32) * tl.exp2(e.to(tl.float32) - 127.0)).to(tl.bfloat16)
    else:
        wb = w.to(tl.bfloat16)
        if SCALE == 1:
            s = tl.load(S + n.to(tl.int64) * NG + k0 // G, nmask, other=0)
    return wb, s


@triton.jit
def _blk(acc, X, row0, M, k0, wb, s, SX, K: tl.constexpr, BK: tl.constexpr, MP: tl.constexpr,
         SCALE: tl.constexpr):
    """_acc_tile's update of one MP-row block for K tile k0, with the W tile already loaded."""
    rows = row0 + tl.arange(0, MP)
    rmask = rows < M
    ks = k0 + tl.arange(0, BK)
    kmask = ks < K
    x = tl.load(
        X + rows[:, None].to(tl.int64) * SX + ks[None, :],
        rmask[:, None] & kmask[None, :],
        other=0,
    ).to(tl.bfloat16)
    if SCALE == 1:
        acc += tl.dot(x, wb) * s[None, :]
    else:
        acc += tl.dot(x, wb)
    return acc


@triton.jit
def _acc_rows(X, W, S, M, n, nmask, SX, K: tl.constexpr, G: tl.constexpr, BK: tl.constexpr,
              MP: tl.constexpr, BN: tl.constexpr, SCALE: tl.constexpr, W_FP8: tl.constexpr,
              NRB: tl.constexpr):
    """ROWS 2: up to 6 row blocks' accumulators over one pass of K (each W tile loaded once)."""
    a0 = tl.zeros((MP, BN), tl.float32)
    a1 = tl.zeros((MP, BN), tl.float32)
    a2 = tl.zeros((MP, BN), tl.float32)
    a3 = tl.zeros((MP, BN), tl.float32)
    a4 = tl.zeros((MP, BN), tl.float32)
    a5 = tl.zeros((MP, BN), tl.float32)
    for k0 in range(0, K, BK):
        wb, s = _wtile(W, S, n, nmask, k0, K, G, BK, SCALE, W_FP8)
        a0 = _blk(a0, X, 0 * MP, M, k0, wb, s, SX, K, BK, MP, SCALE)
        if NRB > 1:
            a1 = _blk(a1, X, 1 * MP, M, k0, wb, s, SX, K, BK, MP, SCALE)
        if NRB > 2:
            a2 = _blk(a2, X, 2 * MP, M, k0, wb, s, SX, K, BK, MP, SCALE)
        if NRB > 3:
            a3 = _blk(a3, X, 3 * MP, M, k0, wb, s, SX, K, BK, MP, SCALE)
        if NRB > 4:
            a4 = _blk(a4, X, 4 * MP, M, k0, wb, s, SX, K, BK, MP, SCALE)
        if NRB > 5:
            a5 = _blk(a5, X, 5 * MP, M, k0, wb, s, SX, K, BK, MP, SCALE)
    return a0, a1, a2, a3, a4, a5


@triton.jit
def _w8_store(acc, row0, M, n, nmask, B, Y, SY, MP: tl.constexpr, HAS_BIAS: tl.constexpr):
    rows = row0 + tl.arange(0, MP)
    rmask = rows < M
    if HAS_BIAS:
        acc += tl.load(B + n, nmask, other=0).to(tl.float32)[None, :]
    tl.store(
        Y + rows[:, None].to(tl.int64) * SY + n[None, :],
        acc.to(Y.dtype.element_ty),
        rmask[:, None] & nmask[None, :],
    )


@triton.jit
def _w8_mm_kernel(
    X,
    W,
    S,
    B,
    Y,
    M,
    N,
    SX,
    SY,
    K: tl.constexpr,
    G: tl.constexpr,
    BK: tl.constexpr,
    MP: tl.constexpr,
    BN: tl.constexpr,
    SCALE: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    W_FP8: tl.constexpr,
    ROWS: tl.constexpr = 0,
    NRB: tl.constexpr = 1,
):
    n = tl.program_id(0) * BN + tl.arange(0, BN)
    nmask = n < N
    if ROWS == 0:
        acc, rows, rmask = _acc_tile(
            X, W, S, tl.program_id(1) * MP, M, n, nmask, SX, K, G, BK, MP, BN, SCALE, W_FP8
        )
        _w8_store(acc, tl.program_id(1) * MP, M, n, nmask, B, Y, SY, MP, HAS_BIAS)
    elif ROWS == 1:
        for rb in range(0, tl.cdiv(M, MP)):
            acc, rows, rmask = _acc_tile(X, W, S, rb * MP, M, n, nmask, SX, K, G, BK, MP, BN, SCALE, W_FP8)
            _w8_store(acc, rb * MP, M, n, nmask, B, Y, SY, MP, HAS_BIAS)
    else:
        a0, a1, a2, a3, a4, a5 = _acc_rows(X, W, S, M, n, nmask, SX, K, G, BK, MP, BN, SCALE, W_FP8, NRB)
        _w8_store(a0, 0 * MP, M, n, nmask, B, Y, SY, MP, HAS_BIAS)
        if NRB > 1:
            _w8_store(a1, 1 * MP, M, n, nmask, B, Y, SY, MP, HAS_BIAS)
        if NRB > 2:
            _w8_store(a2, 2 * MP, M, n, nmask, B, Y, SY, MP, HAS_BIAS)
        if NRB > 3:
            _w8_store(a3, 3 * MP, M, n, nmask, B, Y, SY, MP, HAS_BIAS)
        if NRB > 4:
            _w8_store(a4, 4 * MP, M, n, nmask, B, Y, SY, MP, HAS_BIAS)
        if NRB > 5:
            _w8_store(a5, 5 * MP, M, n, nmask, B, Y, SY, MP, HAS_BIAS)


@triton.jit
def _hot_step_kernel(
    X,
    W,
    S,
    FB,
    BASE,
    IDS,
    OUT,
    M,
    N,
    SX,
    SBASE,
    SOUT,
    softcap,
    vocab_limit,
    K: tl.constexpr,
    G: tl.constexpr,
    BK: tl.constexpr,
    MP: tl.constexpr,
    BN: tl.constexpr,
    SCALE: tl.constexpr,
    SOFTCAP: tl.constexpr,
    LIMIT: tl.constexpr,
    W_FP8: tl.constexpr,
    ROWS: tl.constexpr = 0,
    NRB: tl.constexpr = 1,
    BASE_COMPACT: tl.constexpr = False,
):
    n = tl.program_id(0) * BN + tl.arange(0, BN)
    nmask = n < N
    if ROWS == 0:
        acc, rows, rmask = _acc_tile(
            X, W, S, tl.program_id(1) * MP, M, n, nmask, SX, K, G, BK, MP, BN, SCALE, W_FP8
        )
        _hot_store(acc, tl.program_id(1) * MP, M, n, nmask, FB, BASE, IDS, OUT, SBASE, SOUT, softcap,
                   vocab_limit, MP, SOFTCAP, LIMIT, BASE_COMPACT)
    elif ROWS == 1:
        for rb in range(0, tl.cdiv(M, MP)):
            acc, rows, rmask = _acc_tile(X, W, S, rb * MP, M, n, nmask, SX, K, G, BK, MP, BN, SCALE, W_FP8)
            _hot_store(acc, rb * MP, M, n, nmask, FB, BASE, IDS, OUT, SBASE, SOUT, softcap, vocab_limit, MP,
                       SOFTCAP, LIMIT, BASE_COMPACT)
    else:
        a0, a1, a2, a3, a4, a5 = _acc_rows(X, W, S, M, n, nmask, SX, K, G, BK, MP, BN, SCALE, W_FP8, NRB)
        _hot_store(a0, 0 * MP, M, n, nmask, FB, BASE, IDS, OUT, SBASE, SOUT, softcap, vocab_limit, MP,
                   SOFTCAP, LIMIT, BASE_COMPACT)
        if NRB > 1:
            _hot_store(a1, 1 * MP, M, n, nmask, FB, BASE, IDS, OUT, SBASE, SOUT, softcap, vocab_limit, MP,
                       SOFTCAP, LIMIT, BASE_COMPACT)
        if NRB > 2:
            _hot_store(a2, 2 * MP, M, n, nmask, FB, BASE, IDS, OUT, SBASE, SOUT, softcap, vocab_limit, MP,
                       SOFTCAP, LIMIT, BASE_COMPACT)
        if NRB > 3:
            _hot_store(a3, 3 * MP, M, n, nmask, FB, BASE, IDS, OUT, SBASE, SOUT, softcap, vocab_limit, MP,
                       SOFTCAP, LIMIT, BASE_COMPACT)
        if NRB > 4:
            _hot_store(a4, 4 * MP, M, n, nmask, FB, BASE, IDS, OUT, SBASE, SOUT, softcap, vocab_limit, MP,
                       SOFTCAP, LIMIT, BASE_COMPACT)
        if NRB > 5:
            _hot_store(a5, 5 * MP, M, n, nmask, FB, BASE, IDS, OUT, SBASE, SOUT, softcap, vocab_limit, MP,
                       SOFTCAP, LIMIT, BASE_COMPACT)


@triton.jit
def _hot_store(acc, row0, M, n, nmask, FB, BASE, IDS, OUT, SBASE, SOUT, softcap, vocab_limit,
               MP: tl.constexpr, SOFTCAP: tl.constexpr, LIMIT: tl.constexpr,
               BASE_COMPACT: tl.constexpr = False):
    rows = row0 + tl.arange(0, MP)
    rmask = rows < M
    acc += tl.load(FB + n, nmask, other=0).to(tl.float32)[None, :]
    # The stock fold (torch.addmm) returns a BF16 bias.
    bias = acc.to(tl.bfloat16).to(tl.float32)
    ids = tl.load(IDS + n, nmask, other=0)
    mask = rmask[:, None] & nmask[None, :]
    # BASE_COMPACT: base holds only the hot columns, in hot order (column j = vocab id ids[j]); the same values
    # the full-width base held at ids[j].
    if BASE_COMPACT:
        bcol = n
    else:
        bcol = ids
    base = tl.load(
        BASE + rows[:, None].to(tl.int64) * SBASE + bcol[None, :], mask, other=0
    ).to(tl.float32)
    value = (base + bias).to(tl.bfloat16).to(tl.float32)
    if SOFTCAP:
        value = (libdevice.tanh(value / softcap) * softcap).to(tl.bfloat16).to(
            tl.float32
        )
    if LIMIT:
        value = tl.where(ids[None, :] < vocab_limit, value, -float("inf"))
    tl.store(
        OUT + rows[:, None].to(tl.int64) * SOUT + ids[None, :],
        value.to(OUT.dtype.element_ty),
        mask,
    )


def _mp(m: int) -> int:
    if m <= 16:
        return 16
    if m <= 32:
        return 32
    return 64


def _bk(pack, cfg_bk: int) -> int:
    if pack.scale_kind != SCALE_FP32:
        return min(cfg_bk, triton.next_power_of_2(pack.k))
    bk = cfg_bk
    while bk >= 16 and pack.group % bk:
        bk //= 2
    if bk < 16:
        raise ValueError(f"no K tile >= 16 divides the scale group {pack.group}")
    return bk


def _scale_arg(pack, like):
    return pack.scale if pack.scale is not None else like


def _w_arg(pack):
    """(weight pointer, is_fp8): FP8 planes are passed as uint8 views."""
    if pack.q.dtype == torch.float8_e4m3fn:
        return pack.q.view(torch.uint8), True
    return pack.q, False


LAST = {}  # the last compiled kernel of each launcher (gpu_check reads registers / spills / shared memory)


def _stages(rmode: int, num_stages):
    """Triton launch options: the original launch (ROWS 0) keeps Triton's default stages; ROWS 2 defaults to 1."""
    if rmode == 0:
        return {}
    if num_stages is None and rmode == 2:
        num_stages = 1
    return {} if num_stages is None else {"num_stages": num_stages}


def _rows(m: int, mp: int, rows: int):
    """(ROWS, NRB, grid rows) for m rows: wide8 only when there is more than one row block; ROWS 2 holds at most
    WIDE_NRB blocks (more falls back to ROWS 1)."""
    nrb = triton.cdiv(m, mp)
    if rows == 0 or nrb == 1:
        return 0, 1, nrb
    if rows == 2 and nrb > WIDE_NRB:
        rows = 1
    return rows, nrb if rows == 2 else 1, 1


def w8_mm(x, pack, bias, out, *, bn: int, bk: int, num_warps: int, rows: int = 0, num_stages=None):
    m, k = x.shape
    assert k == pack.k and x.stride(1) == 1 and out.stride(1) == 1
    mp = _mp(m)
    rmode, nrb, gy = _rows(m, mp, rows)
    grid = (triton.cdiv(pack.n, bn), gy)
    extra = _stages(rmode, num_stages)
    w, w_fp8 = _w_arg(pack)
    LAST["w8_mm"] = _w8_mm_kernel[grid](
        x,
        w,
        _scale_arg(pack, w),
        bias if bias is not None else x,
        out,
        m,
        pack.n,
        x.stride(0),
        out.stride(0),
        K=pack.k,
        G=pack.group,
        BK=_bk(pack, bk),
        MP=mp,
        BN=bn,
        SCALE=pack.scale_kind,
        HAS_BIAS=bias is not None,
        W_FP8=w_fp8,
        ROWS=rmode,
        NRB=nrb,
        num_warps=num_warps,
        **extra,
    )
    return out


def hot_step(
    latent,
    pack,
    fold_bias,
    base,
    ids32,
    out,
    *,
    softcap,
    vocab_limit,
    bn: int,
    bk: int,
    num_warps: int,
    rows: int = 0,
    num_stages=None,
    base_compact: bool = False,
):
    m, k = latent.shape
    assert k == pack.k and latent.stride(1) == 1
    assert base.stride(1) == 1 and out.stride(1) == 1
    mp = _mp(m)
    rmode, nrb, gy = _rows(m, mp, rows)
    grid = (triton.cdiv(pack.n, bn), gy)
    extra = _stages(rmode, num_stages)
    w, w_fp8 = _w_arg(pack)
    LAST["hot_step"] = _hot_step_kernel[grid](
        latent,
        w,
        _scale_arg(pack, w),
        fold_bias,
        base,
        ids32,
        out,
        m,
        pack.n,
        latent.stride(0),
        base.stride(0),
        out.stride(0),
        float(softcap or 0.0),
        int(vocab_limit or 0),
        K=pack.k,
        G=pack.group,
        BK=_bk(pack, bk),
        MP=mp,
        BN=bn,
        SCALE=pack.scale_kind,
        SOFTCAP=softcap is not None,
        LIMIT=vocab_limit is not None,
        W_FP8=w_fp8,
        ROWS=rmode,
        NRB=nrb,
        BASE_COMPACT=base_compact,
        num_warps=num_warps,
        **extra,
    )
    return out

