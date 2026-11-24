"""Triton kernels of quant_once (imported only on the GPU path).

Index math: inkling_prefill._layout.scale_word / scale_base, i.e. routed_quant.cuh
scale_base and scale_offset in 32-bit words (four K-blocks of one row are four
consecutive bytes of one word).
"""

import triton
import triton.language as tl


@triton.jit
def _gather_routed(
    TokVals,
    TokScales,
    TokAmax,
    Dst2Src,
    Offsets,
    Vals,
    Scales,
    Amax,
    TOP_K: tl.constexpr,
    VWORDS: tl.constexpr,
    VBLOCK: tl.constexpr,
    C4: tl.constexpr,
    CBLOCK: tl.constexpr,
    E: tl.constexpr,
    SEARCH: tl.constexpr,
):
    lo = tl.load(Offsets)
    hi = tl.load(Offsets + E)
    r = lo + tl.program_id(0)
    if r < hi:
        t = tl.load(Dst2Src + r) // TOP_K
        w = tl.arange(0, VBLOCK)
        wm = w < VWORDS
        v = tl.load(TokVals + t.to(tl.int64) * VWORDS + w, wm)
        tl.store(Vals + r.to(tl.int64) * VWORDS + w, v, wm)
        tl.store(Amax + r, tl.load(TokAmax + t))
        ex = tl.arange(0, 1 << SEARCH)
        starts = tl.load(Offsets + 1 + ex, ex < E - 1, other=0x7FFFFFFF)
        e = tl.sum((starts <= r).to(tl.int32), 0)
        oe = tl.load(Offsets + e)
        local = r - oe
        base = ((oe - lo + 127 * e) // 128 * 128).to(tl.int64) * C4
        c = tl.arange(0, CBLOCK)
        cm = c < C4
        dst = base + ((local // 128) * C4 + c) * 128 + (local % 32) * 4 + (local // 32) % 4
        src = ((t // 128) * C4 + c).to(tl.int64) * 128 + (t % 32) * 4 + (t // 32) % 4
        tl.store(Scales + dst, tl.load(TokScales + src, cm), cm)


@triton.jit
def _zero_tails(
    Offsets,
    Scales,
    C4: tl.constexpr,
    CBLOCK: tl.constexpr,
):
    e = tl.program_id(0)
    lo = tl.load(Offsets)
    oe = tl.load(Offsets + e)
    count = tl.load(Offsets + e + 1) - oe
    used = count % 128
    if used != 0:
        base = ((oe - lo + 127 * e) // 128 * 128).to(tl.int64) * C4
        rows = tl.arange(0, 128)
        local = (count // 128) * 128 + rows
        c = tl.arange(0, CBLOCK)
        dst = (
            base
            + ((local[:, None] // 128) * C4 + c[None, :]) * 128
            + (local[:, None] % 32) * 4
            + (local[:, None] // 32) % 4
        )
        mask = (rows[:, None] >= used) & (c[None, :] < C4)
        tl.store(Scales + dst, tl.zeros([128, CBLOCK], dtype=tl.int32), mask)
