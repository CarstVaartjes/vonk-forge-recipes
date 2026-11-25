"""The prompt-copy matcher as one Triton kernel (rule.match_ref is its reference).

One program per batch row. Row b reads hist[slot_b, :L_b] (L = the anchor position + 1); a row that is not valid,
whose L is below KEY + 1, or above the history capacity, or whose slot is out of range, writes n = 0 and touches no
memory. The search walks tiles of candidate end positions e backwards from L - 1 and stops at the first tile holding a
full KEY-gram hit, taking its largest e (the most recent earlier occurrence); the KEY comparisons of a tile are masked
by the ones before them, so a tile without a last-token hit loads one value per lane. Then the backward match length
m (capped at MAXLEN) and the continuation c = hist[slot, e : min(e + G, L)].
"""

import torch

from . import rule

try:
    import triton
    import triton.language as tl
except ImportError:  # CPU tests
    triton = None
    tl = None

TILE = 256

if triton is not None:

    @triton.jit
    def _match_kernel(hist_ptr, hist_stride, rows, cap, slots_ptr, lens_ptr, valid_ptr,
                      m_ptr, n_ptr, cont_ptr, cont_stride,
                      KEY: tl.constexpr, MAXLEN: tl.constexpr, G: tl.constexpr, GP: tl.constexpr,
                      EP: tl.constexpr, TILE: tl.constexpr):
        b = tl.program_id(0)
        slot = tl.load(slots_ptr + b).to(tl.int64)
        L = tl.load(lens_ptr + b).to(tl.int64)
        ok = tl.load(valid_ptr + b) != 0
        ok = ok & (slot >= 0) & (slot < rows) & (L >= KEY + 1) & (L <= cap)
        base = hist_ptr + slot * hist_stride
        best = tl.full((), -1, tl.int64)
        if ok:
            lanes = tl.arange(0, TILE).to(tl.int64)
            t = L - 1
            while (t >= KEY) & (best < 0):
                e = t - lanes                                   # descending end positions
                hit = e >= KEY
                for i in tl.static_range(KEY):                  # compare from the last token backwards
                    qi = tl.load(base + (L - 1 - i))
                    v = tl.load(base + (e - 1 - i), mask=hit, other=-1)
                    hit = hit & (v == qi)
                best = tl.max(tl.where(hit, e, -1), axis=0)
                t = t - TILE
        found = best >= 0
        # backward extension, vectorized: m = KEY + the number of leading matches before the first mismatch
        i = tl.arange(0, EP).to(tl.int64)
        pa = best - KEY - 1 - i
        pb = L - KEY - 1 - i
        live = found & (i < MAXLEN - KEY) & (pa >= 0)
        va = tl.load(base + pa, mask=live, other=-1)
        vb = tl.load(base + pb, mask=live, other=-2)
        stop = ~(live & (va == vb))
        first = tl.min(tl.where(stop, i, MAXLEN - KEY), axis=0)
        m = tl.where(found, KEY + first, 0).to(tl.int64)
        n = tl.where(found, tl.minimum(L - best, G), 0).to(tl.int64)
        j = tl.arange(0, GP).to(tl.int64)
        cj = tl.load(base + best + j, mask=found & (j < n), other=-1)
        tl.store(cont_ptr + b * cont_stride + j, tl.where(j < n, cj.to(tl.int64), -1), mask=j < G)
        tl.store(m_ptr + b, m)
        tl.store(n_ptr + b, n)


def match(hist, slots, lens, valid, gamma, key=rule.KEY, maxlen=rule.MAXLEN, out=None):
    """(m [bs] int64, n [bs] int64, cont [bs, gamma] int64): the Triton kernel on CUDA tensors, the reference
    otherwise (CPU tests)."""
    bs = int(slots.shape[0])
    if not hist.is_cuda:
        return rule.match_ref(hist, slots, lens, valid, gamma, key, maxlen)
    if triton is None:
        raise RuntimeError("inkling_pcopy: Triton is not available for the CUDA matcher")
    if hist.dim() != 2 or hist.dtype != torch.int32 or hist.stride(1) != 1:
        raise ValueError(f"inkling_pcopy: history must be a row-contiguous int32 [rows, cap] tensor, got "
                         f"{hist.dtype} {tuple(hist.shape)} strides {hist.stride()}")
    if not (slots.shape == lens.shape == valid.shape == (bs,)):
        raise ValueError("inkling_pcopy: slots / lens / valid must be [bs]")
    if out is not None:
        mo, no, co = out
        if (mo.dtype != torch.int64 or no.dtype != torch.int64 or co.dtype != torch.int64 or mo.shape[0] < bs
                or no.shape[0] < bs or co.shape[0] < bs or co.shape[1] != gamma or co.stride(1) != 1):
            raise ValueError("inkling_pcopy: out buffers must be int64 [>= bs], [>= bs], [>= bs, gamma] row-contiguous")
    if out is None:
        m = torch.empty(bs, dtype=torch.int64, device=hist.device)
        n = torch.empty(bs, dtype=torch.int64, device=hist.device)
        cont = torch.empty((bs, gamma), dtype=torch.int64, device=hist.device)
    else:
        m, n, cont = (t[:bs] for t in out)
    if bs == 0:
        return m, n, cont
    _match_kernel[(bs,)](
        hist, hist.stride(0), hist.shape[0], hist.shape[1],
        slots.to(torch.int64).contiguous(), lens.to(torch.int64).contiguous(), valid.to(torch.int8).contiguous(),
        m, n, cont, cont.stride(0),
        KEY=key, MAXLEN=maxlen, G=gamma, GP=triton.next_power_of_2(max(gamma, 1)),
        EP=triton.next_power_of_2(max(maxlen - key, 1)), TILE=TILE,
    )
    return m, n, cont
