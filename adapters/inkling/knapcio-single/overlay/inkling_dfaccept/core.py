"""Sparse-q sampled accept for the DFLASH selector draft, computed from the target logits.

The stock path (DFlashWorkerV2._selector_sampling_accept) scatters the selector's q (K = 16 probabilities per draft
slot, over the slot's candidate ids) into a persistent dense [bs, 7, V] FP32 buffer and calls DSpark's
accept_sampling, which materialises softmax(z / T) for every target row ([bs, 8, V] FP32) and runs the chain (or
block-verification) kernel over the dense tensors. At c24 that is 135 MB of q (almost all zeros) plus 154 MB of p,
written and read every step. Here, without either dense tensor (the structure of inkling_vcap.fused_accept, whose
target-row statistics kernel this reuses):

  1. per target row 0..g of each request, (max m, log-sum l) of z / T (inkling_vcap.fused_accept._stats_kernel);
  2. p(d_j) = exp(z_j(d_j) / T - m_j - l_j) by a [bs, 7] gather; q(d_j) = the selector's q at the candidate equal
     to d_j (the drafted token is always one of its slot's candidates);
  3. the accept walk on [bs, 7] tensors (chain: the leading run of {u_j q_j < p_j}; block verification: Sun et al.
     Algorithm 2, prefix probabilities and residual masses r_i = sum_x max(pp_i P_i(x) - Q_i(x), 0), Q_i sparse);
  4. the bonus: one sample from max(s P_tau - Q_tau, 0) (Q only while the draft has not stopped) by an exponential
     race over the target row with the slot's K (id, q) pairs subtracted in-kernel (the rsfix rules: Philox words
     folded before the FP32 conversion, a finite key floor for positive mass, the last positive-target lane when no
     lane has positive residual).

Same law as the stock path: the same formulas, p from a log-sum-exp instead of a materialised softmax (FP32 order),
q exactly the selector's values, and a different (exact up to the 64-bit uniform grid and FP32 math) sampler for the
bonus. Calls with a verify cutoff (cutoff_verify_lens) take the stock path: the stopped-chain form is the same law
only when the cutoff is a stopping rule, which a planner reading the current proposal's confidence does not
guarantee (a review found a counterexample), and the stock path also reports cap_trim_lens.
"""

from __future__ import annotations

import itertools
from typing import Optional

import torch

try:  # Triton is absent on the CPU test machine; the torch primitives cover the logic.
    import triton
    import triton.language as tl
    from triton.language.extra import libdevice
except Exception:  # noqa: BLE001
    triton = None
    tl = None

BLOCK_V = 2048
TARGET_PROGRAMS = 1024


def accept_core_sparse(prims, *, candidates, target_logits, cand_ids, q_rows, t_temps,
                       live_lens: Optional[torch.Tensor], block_verification: bool, coins: torch.Tensor):
    """(correct_len int32 [bs], bonus int64 [bs], cap_trim int32 [bs] (zeros)).
    candidates [bs, S] (anchor + G drafts), target_logits [bs*S, V], cand_ids [bs, G, K] int64, q_rows [bs, G, K]
    (the selector's q per slot over its candidates), t_temps [bs] (> 0), live_lens [>= bs] or None, coins [bs, G]."""
    bs, S = candidates.shape
    G = S - 1
    V = target_logits.shape[-1]
    dev = target_logits.device
    tl2 = target_logits.reshape(bs * S, V)
    cand = cand_ids.reshape(bs, G, -1).to(torch.int64).contiguous()
    qr = q_rows.reshape(bs, G, -1).to(torch.float32).contiguous()
    if live_lens is None:
        g = torch.full((bs,), G, dtype=torch.int64, device=dev)
    else:
        g = (live_lens[:bs].to(device=dev, dtype=torch.int64) - 1).clamp(0, G)
    t_need = torch.arange(S, device=dev)[None, :] <= g[:, None]
    pos = torch.arange(G, device=dev)
    live = pos[None, :] < g[:, None]
    mt, lt = prims.stats(tl2, t_temps, S, t_need.reshape(-1))
    mt, lt = mt.view(bs, S), lt.view(bs, S)
    tok = candidates[:, 1:].to(torch.int64).clamp(0, V - 1)
    zt = tl2.view(bs, S, V)[:, :G].gather(2, tok[..., None]).squeeze(2).float()
    p = torch.exp(zt / t_temps[:, None] - mt[:, :G] - lt[:, :G])
    hit = cand == tok[..., None]
    q = torch.where(hit, qr, torch.zeros_like(qr)).sum(-1)  # the drafted token is one candidate (ids unique)
    q_ok = live & (q > 0.0) & (q <= 1.0)
    p = torch.where(live, p, torch.zeros_like(p))
    if not block_verification:
        ok = q_ok & (coins * q < p)
        tau = torch.cumprod(ok.to(torch.int64), dim=1).sum(dim=1)
        scale = torch.ones(bs, dtype=torch.float32, device=dev)
    else:
        pp = []
        prev = torch.ones(bs, dtype=torch.float32, device=dev)
        for j in range(G):
            ratio = prev * p[:, j] / torch.where(q_ok[:, j], q[:, j], torch.ones_like(q[:, j]))
            prev = torch.where(q_ok[:, j], ratio.clamp(max=1.0), torch.zeros_like(prev))
            pp.append(prev)
        pp = torch.stack(pp, dim=1)
        level = pos[None, :] + 1
        need = (level < g[:, None]) & (pp > 0.0) & (pp < 1.0) & (coins < pp)
        mass = prims.masses(tl2, (mt, lt), t_temps, cand, qr, pp, need)
        h = torch.where(need, mass / (mass + (1.0 - pp)), pp)
        acc = (coins < h) & (level <= g[:, None])
        tau = torch.where(acc, level, torch.zeros_like(level)).amax(dim=1)
        scale = torch.where(tau > 0, pp.gather(1, (tau - 1).clamp(min=0)[:, None]).squeeze(1),
                            torch.ones(bs, dtype=torch.float32, device=dev))
    use_q = tau < g
    scale = torch.where(use_q, scale, torch.ones_like(scale))
    bonus = prims.sample(tl2, (mt, lt), t_temps, cand, qr, tau, scale, use_q)
    return tau.to(torch.int32), bonus.to(torch.int64), torch.zeros(bs, dtype=torch.int32, device=dev)


class TorchPrims:
    """Dense torch reference (CPU tests, GPU cross-checks): Q densified from the sparse rows."""

    def __init__(self, generator: Optional[torch.Generator] = None):
        self.gen = generator

    def stats(self, logits2d, temps, rows_per_request, need_rows):
        x = logits2d.float() / temps.repeat_interleave(rows_per_request)[:, None]
        m = x.amax(dim=-1)
        return m, torch.log(torch.exp(x - m[:, None]).sum(-1))

    @staticmethod
    def dense(tl2, st, t_temps, cand, qr):
        mt, lt = st
        bs, S = mt.shape
        G = cand.shape[1]
        V = tl2.shape[1]
        P = torch.exp(tl2.float().view(bs, S, V) / t_temps[:, None, None] - mt[..., None] - lt[..., None])
        Q = torch.zeros(bs, G, V, dtype=torch.float32, device=tl2.device)
        Q.scatter_(2, cand.clamp(0, V - 1), qr)
        return P, Q

    def masses(self, tl2, st, t_temps, cand, qr, pp, need):
        P, Q = self.dense(tl2, st, t_temps, cand, qr)
        bs, G = pp.shape
        out = torch.zeros(bs, G, dtype=torch.float32, device=pp.device)
        for j in range(G - 1):
            m = (pp[:, j, None] * P[:, j + 1] - Q[:, j + 1]).clamp(min=0).sum(-1)
            out[:, j] = torch.where(need[:, j], m, torch.zeros_like(m))
        return out

    def sample(self, tl2, st, t_temps, cand, qr, tau, scale, use_q):
        P, Q = self.dense(tl2, st, t_temps, cand, qr)
        bs, _, V = P.shape
        G = Q.shape[1]
        rows = torch.arange(bs, device=tau.device)
        Pr = P[rows, tau]
        Qr = Q[rows, tau.clamp(max=G - 1)]
        Qr = torch.where(use_q[:, None] & (Qr >= 0) & (Qr <= 1), Qr, torch.zeros_like(Qr))
        w = torch.nan_to_num((scale[:, None] * Pr - Qr).clamp(min=0), nan=0.0)
        dead = w.sum(-1) <= 0
        lanes = torch.arange(V, device=w.device)
        last = torch.where(Pr > 0, lanes, torch.full_like(lanes, -1)).amax(-1)
        fallback = torch.where(last >= 0, last, torch.full_like(last, V - 1))
        w[dead] = 0.0
        w[dead, fallback[dead]] = 1.0
        return torch.multinomial(w, 1, generator=self.gen).squeeze(1)


if triton is not None:

    @triton.jit
    def _sparse_q(cand_base, q_base, offs, K: tl.constexpr):
        # Q at every lane of the block: the selector's q where the lane is one of the K candidate ids, else 0
        qv = tl.zeros(offs.shape, tl.float32)
        for k in tl.static_range(K):
            cid = tl.load(cand_base + k)
            qk = tl.load(q_base + k)
            qv = tl.where(offs == cid, qv + qk, qv)
        return qv

    @triton.jit
    def _mass_sparse_kernel(
        t_ptr, mt_ptr, lt_ptr, tt_ptr, cand_ptr, q_ptr, pp_ptr, need_ptr, out_ptr, V, chunk, S, G, t_stride,
        num_splits, K: tl.constexpr, BLOCK: tl.constexpr,
    ):
        # program (r, split): r = b * G + j, level i = j + 1 (target row i, draft slot i)
        r = tl.program_id(0)
        split = tl.program_id(1)
        out = out_ptr + r * num_splits + split
        b = r // G
        i = r % G + 1
        # level G has no draft slot G (slots 0..G-1): never read past the request's sparse rows
        if (tl.load(need_ptr + r) == 0) | (i >= G):
            tl.store(out, 0.0)
        else:
            tt = tl.load(tt_ptr + b)
            ct = tl.load(mt_ptr + b * S + i)
            nt = tl.load(lt_ptr + b * S + i)
            s = tl.load(pp_ptr + r)
            tbase = t_ptr + (b * S + i).to(tl.int64) * t_stride
            cbase = cand_ptr + (b * G + i).to(tl.int64) * K
            qbase = q_ptr + (b * G + i).to(tl.int64) * K
            start = split * chunk
            end = tl.minimum(start + chunk, V)
            acc = tl.zeros([BLOCK], tl.float32)
            for v0 in range(start, end, BLOCK):
                offs = v0 + tl.arange(0, BLOCK)
                m = offs < end
                zt = tl.load(tbase + offs, mask=m, other=-float("inf")).to(tl.float32)
                pv = tl.exp(zt / tt - ct - nt)
                qv = _sparse_q(cbase, qbase, offs, K)
                qv = tl.where((qv >= 0.0) & (qv <= 1.0), qv, 0.0)
                acc += tl.where(m, tl.maximum(s * pv - qv, 0.0), 0.0)
            tl.store(out, tl.sum(acc, axis=0))

    @triton.jit
    def _race_sparse_kernel(
        t_ptr, mt_ptr, lt_ptr, tt_ptr, cand_ptr, q_ptr, row_ptr, scale_ptr, useq_ptr, seed, key_ptr, idx_ptr,
        lastp_ptr, V, chunk, S, G, t_stride, num_splits, K: tl.constexpr, BLOCK: tl.constexpr,
    ):
        # program (b, split): the best exponential-race key of w = max(s P - Q, 0) (Q only when use_q) over this
        # split of request b's bonus row, and the split's last lane with positive target mass (-1 if none)
        b = tl.program_id(0)
        split = tl.program_id(1)
        row = tl.load(row_ptr + b)
        drow = tl.minimum(row, G - 1)
        tt = tl.load(tt_ptr + b)
        ct = tl.load(mt_ptr + b * S + row)
        nt = tl.load(lt_ptr + b * S + row)
        s = tl.load(scale_ptr + b)
        uq = tl.load(useq_ptr + b) != 0
        tbase = t_ptr + (b * S + row).to(tl.int64) * t_stride
        cbase = cand_ptr + (b * G + drow).to(tl.int64) * K
        qbase = q_ptr + (b * G + drow).to(tl.int64) * K
        start = split * chunk
        end = tl.minimum(start + chunk, V)
        best_key = tl.full([BLOCK], -float("inf"), tl.float32)
        best_idx = tl.zeros([BLOCK], tl.int32) + (V - 1)
        last_p = tl.full([BLOCK], -1, tl.int32)
        for v0 in range(start, end, BLOCK):
            offs = v0 + tl.arange(0, BLOCK)
            m = offs < end
            zt = tl.load(tbase + offs, mask=m, other=-float("inf")).to(tl.float32)
            pv = tl.exp(zt / tt - ct - nt)
            qv = _sparse_q(cbase, qbase, offs, K)
            qv = tl.where(uq & (qv >= 0.0) & (qv <= 1.0), qv, 0.0)
            w = tl.maximum(s * pv - qv, 0.0)
            r0, r1, _r2, _r3 = tl.randint4x(seed, b * V + offs)
            w0 = r0.to(tl.uint32)
            w1 = r1.to(tl.uint32)
            upper = w0 >= 0x80000000
            f0 = tl.where(upper, ~w0, w0)
            f1 = tl.where(upper, ~w1, w1)
            small = (f0.to(tl.float32) * 2.3283064365386963e-10 + (f1.to(tl.float32) + 0.5) * 5.421010862427522e-20)
            e = tl.where(upper, -tl.log(small), -libdevice.log1p(-small))
            key = tl.where(m & (w > 0.0), tl.maximum(tl.log(w) - tl.log(e), -3.0e38), -float("inf"))
            last_p = tl.maximum(last_p, tl.where(m & (pv > 0.0), offs, -1))
            better = key > best_key
            best_key = tl.where(better, key, best_key)
            best_idx = tl.where(better, offs, best_idx)
        k = tl.max(best_key, axis=0)
        cand = tl.where(best_key == k, best_idx, V - 1)
        tl.store(key_ptr + b * num_splits + split, k)
        tl.store(idx_ptr + b * num_splits + split, tl.min(cand, axis=0))
        tl.store(lastp_ptr + b * num_splits + split, tl.max(last_p, axis=0))


def _splits(rows: int, vocab: int):
    num_splits = max(1, min(64, -(-TARGET_PROGRAMS // max(1, rows))))
    chunk = -(-(-(-vocab // num_splits)) // BLOCK_V) * BLOCK_V
    return -(-vocab // chunk), chunk


class TritonPrims:
    """CUDA primitives: inkling_vcap.fused_accept's _stats_kernel, and the sparse-q _mass / _race kernels."""

    _MIX = 0x9E3779B97F4A7C15

    def __init__(self):
        from inkling_vcap import fused_accept as fa

        self._stats = fa.TritonPrims()
        self._base = int(torch.randint(0, 2**62, (1,), device="cpu"))
        self._count = itertools.count(1)

    def _seed(self) -> int:
        return (self._base + next(self._count) * self._MIX) & 0x7FFFFFFFFFFFFFFF

    def stats(self, logits2d, temps, rows_per_request, need_rows):
        return self._stats.stats(logits2d, temps, rows_per_request, need_rows)

    def masses(self, tl2, st, t_temps, cand, qr, pp, need):
        mt, lt = (x.contiguous() for x in st)
        bs, G = pp.shape
        S = mt.shape[1]
        V = tl2.shape[1]
        K = cand.shape[2]
        rows = bs * G
        num_splits, chunk = _splits(rows, V)
        out = torch.empty(rows, num_splits, dtype=torch.float32, device=tl2.device)
        _mass_sparse_kernel[(rows, num_splits)](
            tl2, mt, lt, t_temps.contiguous(), cand, qr, pp.contiguous(), need.to(torch.int32).contiguous(), out,
            V, chunk, S, G, tl2.stride(0), num_splits, K=K, BLOCK=BLOCK_V,
        )
        return out.sum(-1).view(bs, G)

    def sample(self, tl2, st, t_temps, cand, qr, tau, scale, use_q):
        mt, lt = (x.contiguous() for x in st)
        bs = tau.shape[0]
        S = mt.shape[1]
        G = cand.shape[1]
        K = cand.shape[2]
        V = tl2.shape[1]
        assert bs * V < 2**31
        num_splits, chunk = _splits(bs, V)
        keys = torch.empty(bs, num_splits, dtype=torch.float32, device=tl2.device)
        idx = torch.empty(bs, num_splits, dtype=torch.int32, device=tl2.device)
        last_p = torch.empty(bs, num_splits, dtype=torch.int32, device=tl2.device)
        _race_sparse_kernel[(bs, num_splits)](
            tl2, mt, lt, t_temps.contiguous(), cand, qr, tau.to(torch.int32).contiguous(),
            scale.to(torch.float32).contiguous(), use_q.to(torch.int32).contiguous(), self._seed(), keys, idx,
            last_p, V, chunk, S, G, tl2.stride(0), num_splits, K=K, BLOCK=BLOCK_V,
        )
        best = keys.argmax(dim=1, keepdim=True)
        pick = idx.gather(1, best).squeeze(1).to(torch.int64)
        fallback = last_p.amax(dim=1).to(torch.int64)
        dead = (keys.amax(dim=1) == float("-inf")) & (fallback >= 0)
        return torch.where(dead, fallback, pick)
