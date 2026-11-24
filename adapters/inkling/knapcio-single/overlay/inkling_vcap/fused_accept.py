"""Sampled DSpark accept computed from the logits, without materialising the
[bs*8, V] target and [bs*7, V] draft probability tensors
(INKLING_VERIFY_CAP_FUSED_ACCEPT=1; default off).

The stock sampled accept (plain temperature sampling) runs these steps:
- SoftmaxTemp over the target logits: FlashInfer, two reads and one FP32 write;
- a BF16-to-FP32 copy of the draft logits;
- SoftmaxTemp over that copy: two reads and one write;
- the chain (or block-verification) kernel, which reads p(d_j), q(d_j) and, for the
  final sample, one row pair per request.

At c24 (T = 0.8) that is about 1.07 GB and 5 ms per step (R7S profile). Here:

  1. per row (max m, log-sum l) of z/T (`_stats_kernel`, a split online max/sum-exp
     plus a [rows, splits] combine): one read of every target row 0..g and every
     draft row 0..g-1 a request uses (rows past its cut are skipped);
  2. p(d_j) = exp(z_t/T - m_t - l_t), q(d_j) = exp(z_d/T_d - m_d - l_d) (the max and the
     centred log-normaliser are kept apart, as a softmax computes them): [bs, 7] gathers;
  3. the accept walk on [bs, 7] tensors:
     - chain: tau = the leading run of {u_j q_j < p_j};
     - block verification (Sun et al., Algorithm 2): the prefix probabilities, the
       residual masses r_i = sum max(p_i P_i - Q_i, 0) for the levels that can accept
       (`_mass_kernel`, which skips every other row), h_i and the backward walk;
  4. the bonus: one sample from max(s P_tau - Q_tau, 0), with s = 1 for chain and
     s = p_tau for block verification (P_tau alone when every live draft is
     accepted). It is drawn by an exponential race (`_race_kernel`):
     argmax_x log w(x) - log E_x with E_x ~ Exp(1) i.i.d., which picks x with
     probability w(x) / sum w. One pass, no normalisation, no CDF. E = -log(1 - V) with
     V from two 32-bit Philox words (resolution ~5e-20 near 0, so tail tokens down to a
     relative weight of ~1e-19 can still win). Philox key: a per-launch counter mixed
     into a base drawn once from torch's default generator (reproducible under
     torch.manual_seed; distinct per launch, no birthday collisions); counter = the
     request's flat vocabulary index (int32: bs * V < 2^31).

Under the verify cap the walk runs over the live drafts only (g = L - 1). The draft
then stops after g, and the bonus at tau = g comes from the target row g alone. That
is the stopped-chain form, exact by the same stopping-rule argument as the cap
(README, points 4 and "Block verification under the cap"). Nothing past row g is read.

Same law as the stock path: the formulas are the same. p and q come from lse instead
of a materialised softmax (FP32 rounding), and the bonus is drawn by a different (also
exact) sampling method from the same distribution. Eligible calls only: plain
temperature sampling (no top-k / top-p / min-p), a non-greedy draft, no vocab shard,
CUDA, not under graph capture. Anything else takes the stock path.
"""

from __future__ import annotations

import itertools
from typing import Optional

import torch

try:  # Triton is absent on the CPU test machine; the torch primitives cover the logic.
    import triton
    import triton.language as tl
except Exception:  # noqa: BLE001
    triton = None
    tl = None

BLOCK_V = 2048
TARGET_PROGRAMS = 1024


# ----------------------------------------------------------------------------- core
def accept_core(prims, *, candidates, target_logits, draft_logits, t_temps, d_temps,
                live_lens: Optional[torch.Tensor], block_verification: bool,
                coins: torch.Tensor):
    """(correct_len int32 [bs], bonus int64 [bs], cap_trim int32 [bs] (zeros)).
    candidates [bs, S] (anchor + G drafts), target_logits [bs*S, V],
    draft_logits [bs, G, V], t_temps / d_temps [bs] (> 0), live_lens [>= bs] or None
    (L, rows 0..L-1 live), coins [bs, G] uniform in [0, 1)."""
    bs, S = candidates.shape
    G = S - 1
    V = target_logits.shape[-1]
    dev = target_logits.device
    tl2 = target_logits.reshape(bs * S, V)
    dl2 = draft_logits.reshape(bs * G, V)
    if live_lens is None:
        g = torch.full((bs,), G, dtype=torch.int64, device=dev)
    else:
        g = (live_lens[:bs].to(device=dev, dtype=torch.int64) - 1).clamp(0, G)
    t_need = torch.arange(S, device=dev)[None, :] <= g[:, None]  # target rows 0..g
    pos = torch.arange(G, device=dev)
    live = pos[None, :] < g[:, None]  # draft j is verified (draft rows 0..g-1)
    mt, lt = prims.stats(tl2, t_temps, S, t_need.reshape(-1))
    md, ld = prims.stats(dl2, d_temps, G, live.reshape(-1))
    mt, lt, md, ld = mt.view(bs, S), lt.view(bs, S), md.view(bs, G), ld.view(bs, G)
    tok = candidates[:, 1:].to(torch.int64).clamp(0, V - 1)
    zt = tl2.view(bs, S, V)[:, :G].gather(2, tok[..., None]).squeeze(2).float()
    zd = dl2.view(bs, G, V).gather(2, tok[..., None]).squeeze(2).float()
    p = torch.exp(zt / t_temps[:, None] - mt[:, :G] - lt[:, :G])
    q = torch.exp(zd / d_temps[:, None] - md - ld)
    q_ok = live & (q > 0.0) & (q <= 1.0)
    p = torch.where(live, p, torch.zeros_like(p))
    if not block_verification:
        ok = q_ok & (coins * q < p)
        tau = torch.cumprod(ok.to(torch.int64), dim=1).sum(dim=1)
        scale = torch.ones(bs, dtype=torch.float32, device=dev)
    else:
        # prefix probabilities p_1..p_G (index i-1); 0 past the cut
        pp = []
        prev = torch.ones(bs, dtype=torch.float32, device=dev)
        for j in range(G):
            ratio = prev * p[:, j] / torch.where(q_ok[:, j], q[:, j], torch.ones_like(q[:, j]))
            prev = torch.where(q_ok[:, j], ratio.clamp(max=1.0), torch.zeros_like(prev))
            pp.append(prev)
        pp = torch.stack(pp, dim=1)  # [bs, G]: level i = j + 1
        level = pos[None, :] + 1
        # levels below the cut need their residual mass when they can accept
        need = (level < g[:, None]) & (pp > 0.0) & (pp < 1.0) & (coins < pp)
        mass = prims.masses(tl2, dl2, (mt, lt, md, ld), t_temps, d_temps, pp, need)
        h = torch.where(need, mass / (mass + (1.0 - pp)), pp)
        acc = (coins < h) & (level <= g[:, None])
        tau = torch.where(acc, level, torch.zeros_like(level)).amax(dim=1)
        scale = torch.where(tau > 0, pp.gather(1, (tau - 1).clamp(min=0)[:, None]).squeeze(1),
                            torch.ones(bs, dtype=torch.float32, device=dev))
    use_q = tau < g  # at tau == g the draft has stopped: the bonus is P_g alone
    scale = torch.where(use_q, scale, torch.ones_like(scale))
    bonus = prims.sample(tl2, dl2, (mt, lt, md, ld), t_temps, d_temps, tau, scale, use_q)
    return tau.to(torch.int32), bonus.to(torch.int64), torch.zeros(bs, dtype=torch.int32, device=dev)


class TorchPrims:
    """Dense torch reference of the three primitives (CPU tests, GPU cross-checks)."""

    def __init__(self, generator: Optional[torch.Generator] = None):
        self.gen = generator

    def stats(self, logits2d, temps, rows_per_request, need_rows):
        x = logits2d.float() / temps.repeat_interleave(rows_per_request)[:, None]
        m = x.amax(dim=-1)
        ls = torch.log(torch.exp(x - m[:, None]).sum(-1))
        return m, ls

    @staticmethod
    def _dense(tl2, dl2, st, t_temps, d_temps):
        mt, lt, md, ld = st
        bs, S = mt.shape
        G = md.shape[1]
        V = tl2.shape[1]
        P = torch.exp(tl2.float().view(bs, S, V) / t_temps[:, None, None]
                      - mt[..., None] - lt[..., None])
        Q = torch.exp(dl2.float().view(bs, G, V) / d_temps[:, None, None]
                      - md[..., None] - ld[..., None])
        return P, Q

    def masses(self, tl2, dl2, st, t_temps, d_temps, pp, need):
        P, Q = self._dense(tl2, dl2, st, t_temps, d_temps)
        bs, G = pp.shape
        out = torch.zeros(bs, G, dtype=torch.float32, device=pp.device)
        for j in range(G - 1):  # level i = j + 1: target row i, draft row i
            m = (pp[:, j, None] * P[:, j + 1] - Q[:, j + 1]).clamp(min=0).sum(-1)
            out[:, j] = torch.where(need[:, j], m, torch.zeros_like(m))
        return out

    def sample(self, tl2, dl2, st, t_temps, d_temps, tau, scale, use_q):
        P, Q = self._dense(tl2, dl2, st, t_temps, d_temps)
        bs, _, V = P.shape
        G = Q.shape[1]
        rows = torch.arange(bs, device=tau.device)
        Pr = P[rows, tau]
        Qr = Q[rows, tau.clamp(max=G - 1)]
        Qr = torch.where(use_q[:, None] & (Qr >= 0) & (Qr <= 1), Qr, torch.zeros_like(Qr))
        w = (scale[:, None] * Pr - Qr).clamp(min=0)
        w = torch.nan_to_num(w, nan=0.0)
        dead = w.sum(-1) <= 0
        w[dead, V - 1] = 1.0  # the stock kernels' fallback token
        return torch.multinomial(w, 1, generator=self.gen).squeeze(1)


# ----------------------------------------------------------------------------- Triton
if triton is not None:

    @triton.jit
    def _stats_kernel(
        x_ptr, temp_ptr, need_ptr, out_ptr, V, chunk, rows_per_request, stride,
        num_splits, BLOCK: tl.constexpr,
    ):
        # program (row, split): online (max, sum exp(x - max)) of x / T over the chunk;
        # rows not needed write (-inf, 0) without reading. Lanes that saw only -inf
        # keep a zero sum (the image's _softmax_temp_stats_kernel recurrence).
        row = tl.program_id(0)
        split = tl.program_id(1)
        out = out_ptr + (row * num_splits + split) * 2
        if tl.load(need_ptr + row) == 0:
            tl.store(out, -float("inf"))
            tl.store(out + 1, 0.0)
        else:
            t = tl.load(temp_ptr + row // rows_per_request)
            base = x_ptr + row.to(tl.int64) * stride
            start = split * chunk
            end = tl.minimum(start + chunk, V)
            m = tl.full([BLOCK], -float("inf"), tl.float32)
            acc = tl.zeros([BLOCK], tl.float32)
            for v0 in range(start, end, BLOCK):
                offs = v0 + tl.arange(0, BLOCK)
                x = tl.load(base + offs, mask=offs < end, other=-float("inf")).to(tl.float32)
                x = x / t
                m_new = tl.maximum(m, x)
                alive = m_new > -float("inf")
                acc = tl.where(alive, acc * tl.exp(m - m_new) + tl.exp(x - m_new), 0.0)
                m = m_new
            row_max = tl.max(m, axis=0)
            total = tl.sum(tl.where(m > -float("inf"), acc * tl.exp(m - row_max), 0.0), axis=0)
            tl.store(out, row_max)
            tl.store(out + 1, total)

    @triton.jit
    def _mass_kernel(
        t_ptr, d_ptr, mt_ptr, lt_ptr, md_ptr, ld_ptr, tt_ptr, td_ptr, pp_ptr, need_ptr,
        out_ptr, V, chunk, S, G, t_stride, d_stride, num_splits, BLOCK: tl.constexpr,
    ):
        # program (r, split): r = b * G + j, level i = j + 1 (target row i, draft row i)
        r = tl.program_id(0)
        split = tl.program_id(1)
        out = out_ptr + r * num_splits + split
        if tl.load(need_ptr + r) == 0:
            tl.store(out, 0.0)
        else:
            b = r // G
            i = r % G + 1
            tt = tl.load(tt_ptr + b)
            td = tl.load(td_ptr + b)
            ct = tl.load(mt_ptr + b * S + i)
            nt = tl.load(lt_ptr + b * S + i)
            cd = tl.load(md_ptr + b * G + i)
            nd = tl.load(ld_ptr + b * G + i)
            s = tl.load(pp_ptr + r)
            tbase = t_ptr + (b * S + i).to(tl.int64) * t_stride
            dbase = d_ptr + (b * G + i).to(tl.int64) * d_stride
            start = split * chunk
            end = tl.minimum(start + chunk, V)
            acc = tl.zeros([BLOCK], tl.float32)
            for v0 in range(start, end, BLOCK):
                offs = v0 + tl.arange(0, BLOCK)
                m = offs < end
                zt = tl.load(tbase + offs, mask=m, other=-float("inf")).to(tl.float32)
                zd = tl.load(dbase + offs, mask=m, other=-float("inf")).to(tl.float32)
                pv = tl.exp(zt / tt - ct - nt)
                qv = tl.exp(zd / td - cd - nd)
                qv = tl.where((qv >= 0.0) & (qv <= 1.0), qv, 0.0)
                acc += tl.where(m, tl.maximum(s * pv - qv, 0.0), 0.0)
            tl.store(out, tl.sum(acc, axis=0))

    @triton.jit
    def _race_kernel(
        t_ptr, d_ptr, mt_ptr, lt_ptr, md_ptr, ld_ptr, tt_ptr, td_ptr, row_ptr, scale_ptr,
        useq_ptr, seed, key_ptr, idx_ptr, V, chunk, S, G, t_stride, d_stride, num_splits,
        BLOCK: tl.constexpr,
    ):
        # program (b, split): the best exponential-race key of w = max(s P - Q, 0) (Q only
        # when use_q) over this split of request b's bonus row.
        b = tl.program_id(0)
        split = tl.program_id(1)
        row = tl.load(row_ptr + b)
        drow = tl.minimum(row, G - 1)
        tt = tl.load(tt_ptr + b)
        ct = tl.load(mt_ptr + b * S + row)
        nt = tl.load(lt_ptr + b * S + row)
        s = tl.load(scale_ptr + b)
        uq = tl.load(useq_ptr + b) != 0
        td = tl.load(td_ptr + b)
        cd = tl.where(uq, tl.load(md_ptr + b * G + drow), 0.0)
        nd = tl.where(uq, tl.load(ld_ptr + b * G + drow), 0.0)
        tbase = t_ptr + (b * S + row).to(tl.int64) * t_stride
        dbase = d_ptr + (b * G + drow).to(tl.int64) * d_stride
        start = split * chunk
        end = tl.minimum(start + chunk, V)
        best_key = tl.full([BLOCK], -float("inf"), tl.float32)
        best_idx = tl.zeros([BLOCK], tl.int32) + (V - 1)
        for v0 in range(start, end, BLOCK):
            offs = v0 + tl.arange(0, BLOCK)
            m = offs < end
            zt = tl.load(tbase + offs, mask=m, other=-float("inf")).to(tl.float32)
            pv = tl.exp(zt / tt - ct - nt)
            zd = tl.load(dbase + offs, mask=m & uq, other=-float("inf")).to(tl.float32)
            qv = tl.exp(zd / td - cd - nd)
            qv = tl.where(uq & (qv >= 0.0) & (qv <= 1.0), qv, 0.0)
            w = tl.maximum(s * pv - qv, 0.0)
            # E ~ Exp(1) = -log(1 - V), V uniform from two 32-bit words (fine near 0)
            r0, r1, _r2, _r3 = tl.randint4x(seed, b * V + offs)
            hi = r0.to(tl.uint32).to(tl.float32) * 2.3283064365386963e-10  # 2^-32
            lo = (r1.to(tl.uint32).to(tl.float32) + 0.5) * 5.421010862427522e-20  # 2^-64
            vv = tl.where(hi > 0.0, hi, lo)
            e = tl.where(vv < 1e-3, vv + 0.5 * vv * vv, -tl.log(1.0 - vv))
            key = tl.where(m & (w > 0.0), tl.log(w) - tl.log(e), -float("inf"))
            better = key > best_key
            best_key = tl.where(better, key, best_key)
            best_idx = tl.where(better, offs, best_idx)
        k = tl.max(best_key, axis=0)
        cand = tl.where(best_key == k, best_idx, V - 1)
        tl.store(key_ptr + b * num_splits + split, k)
        tl.store(idx_ptr + b * num_splits + split, tl.min(cand, axis=0))


def _splits(rows: int, vocab: int):
    num_splits = max(1, min(64, -(-TARGET_PROGRAMS // max(1, rows))))
    chunk = -(-(-(-vocab // num_splits)) // BLOCK_V) * BLOCK_V
    return -(-vocab // chunk), chunk


class TritonPrims:
    """The CUDA primitives: _stats_kernel, _mass_kernel, _race_kernel."""

    _MIX = 0x9E3779B97F4A7C15

    def __init__(self):
        # base key from torch's default CPU generator (reproducible under manual_seed);
        # each launch gets base + counter * golden (distinct keys, no collisions)
        self._base = int(torch.randint(0, 2**62, (1,), device="cpu"))
        self._count = itertools.count(1)

    def _seed(self) -> int:
        return (self._base + next(self._count) * self._MIX) & 0x7FFFFFFFFFFFFFFF

    def stats(self, logits2d, temps, rows_per_request, need_rows):
        rows, V = logits2d.shape
        num_splits, chunk = _splits(rows, V)
        out = torch.empty(rows, num_splits, 2, dtype=torch.float32, device=logits2d.device)
        if rows:
            _stats_kernel[(rows, num_splits)](
                logits2d, temps.to(torch.float32).contiguous(),
                need_rows.to(torch.int32).contiguous(), out, V, chunk, rows_per_request,
                logits2d.stride(0), num_splits, BLOCK=BLOCK_V,
            )
        part_max, part_sum = out[..., 0], out[..., 1]
        row_max = part_max.amax(dim=-1)
        alive = part_max > float("-inf")
        total = torch.where(alive, part_sum * torch.exp(part_max - row_max[:, None]),
                            torch.zeros_like(part_sum)).sum(-1)
        return row_max, torch.log(total)

    def masses(self, tl2, dl2, st, t_temps, d_temps, pp, need):
        mt, lt, md, ld = (x.contiguous() for x in st)
        bs, G = pp.shape
        S = mt.shape[1]
        V = tl2.shape[1]
        rows = bs * G
        num_splits, chunk = _splits(rows, V)
        out = torch.empty(rows, num_splits, dtype=torch.float32, device=tl2.device)
        _mass_kernel[(rows, num_splits)](
            tl2, dl2, mt, lt, md, ld, t_temps.contiguous(), d_temps.contiguous(),
            pp.contiguous(), need.to(torch.int32).contiguous(), out,
            V, chunk, S, G, tl2.stride(0), dl2.stride(0), num_splits, BLOCK=BLOCK_V,
        )
        return out.sum(-1).view(bs, G)

    def sample(self, tl2, dl2, st, t_temps, d_temps, tau, scale, use_q):
        mt, lt, md, ld = (x.contiguous() for x in st)
        bs = tau.shape[0]
        S = mt.shape[1]
        G = md.shape[1]
        V = tl2.shape[1]
        assert bs * V < 2**31
        num_splits, chunk = _splits(bs, V)
        keys = torch.empty(bs, num_splits, dtype=torch.float32, device=tl2.device)
        idx = torch.empty(bs, num_splits, dtype=torch.int32, device=tl2.device)
        _race_kernel[(bs, num_splits)](
            tl2, dl2, mt, lt, md, ld, t_temps.contiguous(), d_temps.contiguous(),
            tau.to(torch.int32).contiguous(), scale.to(torch.float32).contiguous(),
            use_q.to(torch.int32).contiguous(), self._seed(), keys, idx, V, chunk, S, G,
            tl2.stride(0), dl2.stride(0), num_splits, BLOCK=BLOCK_V,
        )
        best = keys.argmax(dim=1, keepdim=True)
        return idx.gather(1, best).squeeze(1).to(torch.int64)
