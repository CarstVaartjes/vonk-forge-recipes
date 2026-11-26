"""Host side of the prefill bank hoist (kernels/phoist_triton.py) for inkling_pfuse's loop_mix.

prepare(bank, scaled_queries, down_all, tokens)                 once per loop pass (passes 2 and 3)
loop_mix_hoisted(prep, layer, side, sources, addend, queries, normalize_values, bank, down, gain, forget,
                 norm_weights=None, eps_norm=1e-6)              == inkling_pfuse.mix.loop_mix(...) bit for bit

bank: the pass's 2 message-bank sources (BF16 [T, 2304], contiguous); scaled_queries: LoopMessageBank's
[12, 4, 2304] FP32 (rows attn, k, v, mlp); down_all: [12, 4, 64, 4608] BF16 (the core layers' stacked controller
down weights). Shapes are static per call and no host sync happens: CUDA-graph safe (prefill graph buckets).
"""

import os

import torch
import triton

D, R, K, NBP = 2304, 64, 4608, 2
SIDES = {"attn": (3, 4, 0), "mlp": (1, 1, 3)}  # NM, QPM, row offset in the 4 controller modules
VARIANT = {"name": "a"}  # "a" (Option A, default) | "exact"; set by inkling_phoist.install()


def _pf():
    from inkling_pfuse import mix

    return mix


def _k():
    from . import phoist_triton

    return phoist_triton


def _down_tiles(tokens, device):
    """pfuse loop_mix's down launch choice (stock message_controllers prefill branch)."""
    mix = _pf()
    k = mix._kernels()
    split = tokens <= 4096
    bk = 256 if tokens <= 1024 else 64
    if bk == 256 and k["max_smem"](device.index) < 1 << 17:
        bk = 128
    return split, 64, 64, bk


def _stages(bk):
    """pfuse's down runs 3 stages; with the message tile rebuilt in the kernel, BK 128 (<= 1024 rows on GB10) needs
    112 KB of shared memory at 3 stages (99 KB available): 2 stages there. Software pipelining does not change the
    order of the tl.dot accumulation (checked bit for bit by tools/phoist_probe.py)."""
    return 3 if bk <= 64 else 2


def prepare(bank, scaled_queries, down_all, tokens, variant=None):
    variant = variant or VARIANT["name"]
    assert variant in ("a", "exact")
    mix = _pf()
    kt = _k()
    assert len(bank) == NBP and all(b.shape == (tokens, D) and b.dtype == torch.bfloat16 and b.is_contiguous()
                                    for b in bank)
    assert scaled_queries.shape == (12, 4, D) and scaled_queries.dtype == torch.float32
    assert scaled_queries.is_contiguous()
    assert down_all.shape == (12, 4, R, K) and down_all.dtype == torch.bfloat16 and down_all.is_contiguous()
    dev = bank[0].device
    _, bm, bn, bk = _down_tiles(tokens, dev)
    QP = mix.QP
    prep = {"T": tokens, "bank": tuple(bank), "down_all": down_all, "variant": variant}
    for side, (nm, qpm, off) in SIDES.items():
        bp = torch.empty((12, tokens, NBP, QP), dtype=torch.float32, device=dev)
        bi = torch.empty((12, tokens, QP), dtype=torch.float32, device=dev)
        mr = torch.empty((12, tokens, qpm), dtype=torch.float32, device=dev)
        p34 = torch.empty((12, nm, 2, tokens, R), dtype=torch.float32, device=dev)  # groups 3-4
        if tokens:
            kt._bank_stats_kernel[(tokens, 12)](
                tuple(bank), scaled_queries, bp, bi, mr, tokens, mix.EPS_ATTN_RES, mix.EPS_CTRL, 4 * D, off * D,
                nm, QP, qpm, NBP, D, mix.CH, num_warps=4, enable_fp_fusion=False)
            if variant == "exact":
                # groups 3-4 with the visit's own down kernel (G0 = 3, one group per program), per layer-side: the
                # message operand reaches tl.dot through the same code path as group 2's message columns (the
                # all-rebuilt _bank_down_kernel compiled to different MMA arithmetic under the CUDA 13 ptxas SGLang
                # selects, fusion runs/w18-s06.log): bit-identical to pfuse, slower
                for ls in range(12):
                    kt._down_visit_kernel[(nm * 2, triton.cdiv(tokens, bm), triton.cdiv(R, bn))](
                        bank[0], tuple(bank), bp[ls], bi[ls], mr[ls], down_all[ls, off:off + nm], p34[ls], tokens,
                        R, K, True, bm, bn, bk, nm, 1, NBP, QP, qpm, D, 3, 2,
                        num_warps=4, num_stages=_stages(bk), enable_fp_fusion=False)
            else:
                # Option A (default): one launch per side for all 12 layers' groups 3-4. Same per-element
                # formulas, K tiles and BF16 rounding points; under the serving ptxas the MMA code differs from
                # pfuse's, so outputs match pfuse up to FP32 accumulation order (prefill rule since 2026-11-24)
                kt._bank_down_kernel[(12 * nm * 2, triton.cdiv(tokens, bm), triton.cdiv(R, bn))](
                    tuple(bank), bp, bi, mr, down_all, p34, tokens, 4 * R * K, off,
                    R, K, nm, NBP, QP, qpm, bm, bn, bk, D, num_warps=4, num_stages=_stages(bk),
                    enable_fp_fusion=False)
        prep[side] = (bp, bi, mr, p34)
    return prep


def loop_mix_hoisted(prep, layer, side, sources, addend, queries, normalize_values, bank, down, gain, forget,
                     norm_weights=None, eps_norm=1e-6):
    """pfuse loop_mix (split_down, per_ctrl, fused norm when norm_weights is given) with the bank work taken from
    `prep` (layer = core index 0..11, side "attn" | "mlp")."""
    mix = _pf()
    kt = _k()
    nqs = queries.shape[0]
    nm, qpm, off = SIDES[side]
    assert 0 <= layer < 12 and len(bank) == NBP and nqs in (1, nm)
    assert len(bank) == NBP and all(a.data_ptr() == b.data_ptr() and a.shape == b.shape
                                    for a, b in zip(bank, prep["bank"]))
    # groups 0-2 must use the same down weights the prepared groups 3-4 used
    down = prep["down_all"][layer, off:off + nm]
    num, rank, kdim = down.shape
    bank = prep["bank"]
    assert mix.MIX["split_down"] and mix.MIX["per_ctrl"]
    dev = sources[0].device
    tokens = sources[0].shape[0]
    assert tokens == prep["T"]
    shared = nqs == 1 and nm > 1
    nd = nqs
    QP, NSP = mix.QP, mix.NSP
    bp, bi, mr, p34 = (t[layer] for t in prep[side])
    if norm_weights is not None:
        assert len(norm_weights) == nm
        y = torch.empty((nm, tokens, D), dtype=torch.bfloat16, device=dev)
    else:
        y = torch.empty((nm, tokens, D), dtype=torch.float32, device=dev)
    if not tokens:
        return y
    sp = torch.empty((tokens, NSP, QP), dtype=torch.float32, device=dev)
    si = torch.empty((tokens, QP), dtype=torch.float32, device=dev)
    normed = torch.empty((nd, tokens, D), dtype=torch.bfloat16, device=dev)
    add = addend if addend is not None else sources[0]
    kt._stats_dest_kernel[(tokens,)](
        tuple(sources), add, queries, normed, sp, si, tokens, mix.EPS_ATTN_RES, mix.EPS_CTRL,
        nqs, QP, mix._qp(nqs), NSP, addend is not None, bool(normalize_values), D, mix.CH, nd,
        num_warps=4, enable_fp_fusion=False)
    split, bm, bn, bk = _down_tiles(tokens, dev)
    gv = 3 if split else 1
    partials = torch.empty((nm, gv, tokens, rank), dtype=torch.float32, device=dev)
    kt._down_visit_kernel[(nm * gv, triton.cdiv(tokens, bm), triton.cdiv(rank, bn))](
        normed, tuple(bank), bp, bi, mr, down, partials, tokens,
        rank, kdim, split, bm, bn, bk, nm, nd, NBP, QP, qpm, D, 0, 3,
        num_warps=4, num_stages=_stages(bk), enable_fp_fusion=False)
    wn = gain if norm_weights is None else torch.stack([w.reshape(D) for w in norm_weights])
    sfx = "" if norm_weights is not None else "f"
    bm1, bn1, warps1, stages1 = (mix.MIX[n + sfx] for n in ("bm1", "bn1", "warps1", "stages1"))
    kt._mix1_hoist_kernel[(nm, triton.cdiv(tokens, bm1))](
        tuple(sources), add, tuple(bank), sp, si, bp, bi, partials, p34, gain, forget, wn, y, tokens,
        float(eps_norm), nm, QP, NSP, NBP, addend is not None, shared, norm_weights is not None, gv, rank,
        max(16, triton.next_power_of_2(rank)), D, bm1, bn1, stages1, num_warps=warps1, enable_fp_fusion=False)
    return y

