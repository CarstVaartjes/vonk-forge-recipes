"""The whole decode-step hook as TWO Triton launches on one stream (patch.step_ref is its reference; rule / history are
its spec): a split scan for the match position, then the per-row step.

Scan (grid bs x ceil(cap / CHUNK)): program (b, c) owns the candidate ends e in (L - 1 - (c + 1) CHUNK, L - 1 - c CHUNK]
(e >= KEY) of row b and, when it finds full KEY-gram hits, atomic_max-es the largest into best[b]. best[] is -1 between
steps: the step program of row b reads it and writes -1 back. The largest over all chunks is the largest over the row,
so the result equals the single-program backward scan. The scan reads history positions <= L - 2 only and writes
nothing but best[]. It is launched only when FORCE.

Step: one program per batch row b (slot s, anchor position p, the draft block d[b, 0..G] with d[b, 0] = the anchor):
  ok      = host_ok[b] and 0 <= s < rows and 0 <= p < cap and cov[s] >= p
  history : hist[s, p] = anchor (ok rows)
  match   : only when FORCE (the batch is not bypassed) and ok: e* = best[b] (the largest e < L = p + 1 over full
            KEY-gram hits of h[L-KEY:L], h[L-1] being the anchor taken from the register), re-verified here (a value
            that is not a full hit in range is dropped, so a stale best[] can never force), the backward extension m (cap MAXLEN), the
            continuation c_j = h[e + j] for j < n = min(L - e, G) (position L - 1 again from the register)
  forced  : j < n and m >= FORCE_LEN and every c_0..c_j in [0, vocab)   (the copy stops at the first invalid id)
  drafts  : d[b, 1 + j] = c_j where forced (in place; the verify input aliases this buffer)
  history : hist[s, p + 1 + j] = d[b, 1 + j] (ok rows, positions < cap); cov[s] = min(cap, p + 1 + G) (ok rows)
  outputs : min_live[b] = n_forced + 1 (0 when none), forced_any[b]
  sample  : HAS_SAMPLE: the selector sample is copied into owned buffers, forced slots made point masses (the
            existing lane when c_j is a candidate, else lane 0 after a full-row miss)
  counters: cnt[0] += 1 (program 0: steps), cnt[1] += ok, cnt[2] += (n_forced > 0), cnt[3] += n_forced  (atomics)
Masked rows are skipped by masked stores (no sinks needed). Nothing reads the host.
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
CHUNK = 2048   # candidate ends per scan program (a multiple of TILE)

if triton is not None:

    @triton.jit
    def _scan_kernel(hist_ptr, cap, rows, cov_ptr, slots_ptr, p0_ptr, hok_ptr, draft_ptr, draft_stride, best_ptr,
                     KEY: tl.constexpr, CHUNK: tl.constexpr, TILE: tl.constexpr):
        b = tl.program_id(0)
        c = tl.program_id(1).to(tl.int64)
        s = tl.load(slots_ptr + b).to(tl.int64)
        p = tl.load(p0_ptr + b).to(tl.int64)
        hok = tl.load(hok_ptr + b) != 0
        in_range = (s >= 0) & (s < rows)
        safe_s = tl.where(in_range, s, 0)
        cv = tl.load(cov_ptr + safe_s, mask=in_range, other=-1)
        ok = hok & in_range & (p >= 0) & (p < cap) & (cv >= p)
        L = p + 1
        top = L - 1 - c * CHUNK          # the largest candidate end this program owns
        bottom = top - CHUNK             # exclusive
        if ok & (L >= KEY + 1) & (top >= KEY):
            base = hist_ptr + safe_s * cap
            anchor = tl.load(draft_ptr + b * draft_stride).to(tl.int32).to(tl.int64)   # as the history stores it
            lanes = tl.arange(0, TILE).to(tl.int64)
            best = tl.full((), -1, tl.int64)
            t = top
            while (t > bottom) & (t >= KEY) & (best < 0):
                e = t - lanes
                hit = (e > bottom) & (e >= KEY)
                # i = 0: the last key token is the anchor (position L - 1, never read: e - 1 <= L - 2)
                v0 = tl.load(base + (e - 1), mask=hit, other=-1).to(tl.int64)
                hit = hit & (v0 == anchor)
                for i in tl.static_range(1, KEY):
                    qi = tl.load(base + (L - 1 - i)).to(tl.int64)
                    v = tl.load(base + (e - 1 - i), mask=hit, other=-1).to(tl.int64)
                    hit = hit & (v == qi)
                best = tl.max(tl.where(hit, e, -1), axis=0)
                t = t - TILE
            if best >= 0:
                tl.atomic_max(best_ptr + b, best, sem="relaxed")

    @triton.jit
    def _step_kernel(hist_ptr, cap, rows, cov_ptr, slots_ptr, p0_ptr, hok_ptr,
                     draft_ptr, draft_stride, vocab, force,
                     cand_in, cand_out, q_in, q_out, s_stride_b, s_stride_g,
                     min_live_ptr, any_ptr, cnt_ptr, best_ptr,
                     KEY: tl.constexpr, MAXLEN: tl.constexpr, FORCE_LEN: tl.constexpr, G: tl.constexpr,
                     GP: tl.constexpr, EP: tl.constexpr, K: tl.constexpr, KP: tl.constexpr,
                     HAS_SAMPLE: tl.constexpr):
        b = tl.program_id(0)
        s = tl.load(slots_ptr + b).to(tl.int64)
        p = tl.load(p0_ptr + b).to(tl.int64)
        hok = tl.load(hok_ptr + b) != 0
        in_range = (s >= 0) & (s < rows)
        safe_s = tl.where(in_range, s, 0)
        cv = tl.load(cov_ptr + safe_s, mask=in_range, other=-1)
        ok = hok & in_range & (p >= 0) & (p < cap) & (cv >= p)
        base = hist_ptr + safe_s * cap
        anchor = tl.load(draft_ptr + b * draft_stride).to(tl.int32).to(tl.int64)   # as the history stores it
        tl.store(base + p, anchor.to(tl.int32), mask=ok)
        L = p + 1
        # ---- match (the most recent full KEY-gram hit; h[L-1] is the anchor register)
        go = ok & (force != 0) & (L >= KEY + 1)
        best = tl.full((), -1, tl.int64)
        if go:
            # the scan's answer, re-verified (in range and a full KEY-gram hit; position L - 1 is the anchor register)
            cand = tl.load(best_ptr + b)
            good = (cand >= KEY) & (cand <= L - 1)
            safe = tl.where(good, cand, KEY)
            good = good & (tl.load(base + (safe - 1)).to(tl.int64) == anchor)
            for i in tl.static_range(1, KEY):
                qi = tl.load(base + (L - 1 - i)).to(tl.int64)
                good = good & (tl.load(base + (safe - 1 - i)).to(tl.int64) == qi)
            best = tl.where(good, cand, -1)
        tl.store(best_ptr + b, -1)       # -1 between steps (this program owns best[b])
        found = best >= 0
        ii = tl.arange(0, EP).to(tl.int64)
        pa = best - KEY - 1 - ii
        pb = L - KEY - 1 - ii
        live = found & (ii < MAXLEN - KEY) & (pa >= 0)
        va = tl.load(base + pa, mask=live, other=-1)
        vb = tl.load(base + pb, mask=live, other=-2)
        stop = ~(live & (va == vb))
        m = tl.where(found, KEY + tl.min(tl.where(stop, ii, MAXLEN - KEY), axis=0), 0)
        n = tl.where(found, tl.minimum(L - best, G), 0)
        j = tl.arange(0, GP).to(tl.int64)
        pos = best + j
        cj = tl.load(base + pos, mask=found & (j < n) & (pos < L - 1), other=-1).to(tl.int64)
        cj = tl.where(pos == L - 1, anchor, cj)
        valid = (j < n) & (cj >= 0) & (cj < vocab)
        first_bad = tl.min(tl.where(valid, GP, j), axis=0)
        forced = (m >= FORCE_LEN) & (j < first_bad) & (j < G)
        nf = tl.sum(forced.to(tl.int64), axis=0)
        # ---- drafts (in place) and their history
        dptr = draft_ptr + b * draft_stride + 1 + j
        d = tl.load(dptr, mask=j < G, other=0)
        nd = tl.where(forced, cj.to(d.dtype), d)
        tl.store(dptr, nd, mask=forced)
        hpos = p + 1 + j
        tl.store(base + hpos, nd.to(tl.int32), mask=ok & (j < G) & (hpos < cap))
        tl.store(cov_ptr + safe_s, tl.minimum(p + 1 + G, cap), mask=ok)
        tl.store(min_live_ptr + b, tl.where(nf > 0, nf + 1, 0))
        tl.store(any_ptr + b, nf > 0)
        # ---- owned selector sample
        if HAS_SAMPLE:
            k = tl.arange(0, KP).to(tl.int64)
            gk = (j[:, None] < G) & (k[None, :] < K)
            off = b * s_stride_b + j[:, None] * s_stride_g + k[None, :]
            ids = tl.load(cand_in + off, mask=gk, other=-1)
            qs = tl.load(q_in + off, mask=gk, other=0.0)
            hitk = (ids.to(tl.int64) == cj[:, None]) & gk
            has = tl.max(hitk.to(tl.int32), axis=1) > 0
            lane = tl.where(has, tl.argmax(hitk.to(tl.int32), axis=1), 0)
            ids2 = tl.where(forced[:, None] & (~has)[:, None] & (k[None, :] == 0), cj[:, None].to(ids.dtype), ids)
            one = (k[None, :] == lane[:, None]).to(qs.dtype)
            qs2 = tl.where(forced[:, None], one, qs)
            tl.store(cand_out + off, ids2, mask=gk)
            tl.store(q_out + off, qs2, mask=gk)
        # ---- counters
        one64 = tl.full((), 1, tl.int64)
        tl.atomic_add(cnt_ptr + 0, one64, mask=b == 0, sem="relaxed")
        tl.atomic_add(cnt_ptr + 1, one64, mask=ok, sem="relaxed")
        tl.atomic_add(cnt_ptr + 2, one64, mask=nf > 0, sem="relaxed")
        tl.atomic_add(cnt_ptr + 3, nf, mask=nf > 0, sem="relaxed")


class FusedStep:
    """Preallocated outputs and owned sample buffers for one worker; launch() runs one step."""

    def __init__(self, max_bs, gamma, device):
        self.max_bs, self.gamma, self.device = int(max_bs), int(gamma), device
        self.min_live = torch.zeros(self.max_bs, dtype=torch.int64, device=device)
        self.forced_any = torch.zeros(self.max_bs, dtype=torch.bool, device=device)
        self.ones = torch.ones(self.max_bs, dtype=torch.int8, device=device)
        self.counters = torch.zeros(4, dtype=torch.int64, device=device)
        self.best = torch.full((self.max_bs,), -1, dtype=torch.int64, device=device)
        # set when a submission raised (best[] may hold a scan result no step consumed); the next scan clears all of
        # best[] first, on the same stream, so a stale maximum can never hide a current hit
        self.dirty = False
        self.chunk = CHUNK
        self.last_scan = None   # (grid, args, meta) of the last prepared scan (GPU checks only)
        self._owned = {}

    def owned(self, cand, q):
        key = (tuple(cand.shape[1:]), cand.dtype, q.dtype)
        buf = self._owned.get(key)
        if buf is None:
            buf = self._owned[key] = (torch.empty((self.max_bs,) + tuple(cand.shape[1:]), dtype=cand.dtype,
                                                  device=self.device),
                                      torch.empty((self.max_bs,) + tuple(q.shape[1:]), dtype=q.dtype,
                                                  device=self.device))
        return buf

    def prepare(self, hist, slots, p0, host_ok, draft, vocab, force, sample):
        """Validate everything and build every argument and output view BEFORE anything is submitted. Returns
        (submit, outputs): submit() enqueues the scan (FORCE only; it writes nothing but best[]) and the mutating step; outputs = (min_live [bs], forced_any [bs],
        owned sample or None). Raises ValueError on any contract violation (nothing has been written then)."""
        bs = int(draft.shape[0])
        dev = draft.device
        if not (0 < bs <= self.max_bs) or draft.dim() != 2 or int(draft.shape[1]) != self.gamma + 1:
            raise ValueError(f"inkling_pcopy: draft block {tuple(draft.shape)} outside the fused step's shape")
        if not draft.is_contiguous() or draft.dtype not in (torch.int64, torch.int32) or draft.device != self.device:
            raise ValueError("inkling_pcopy: the draft block must be a contiguous int32/int64 tensor on the device")
        for name, t in (("slots", slots), ("p0", p0)):
            if t.dim() != 1 or int(t.shape[0]) != bs or t.device != dev or t.dtype not in (torch.int64, torch.int32):
                raise ValueError(f"inkling_pcopy: {name} must be an int [bs] tensor on the device")
        slots, p0 = slots.contiguous(), p0.contiguous()
        if host_ok is None:
            hok = self.ones
        else:
            if host_ok.dim() != 1 or int(host_ok.shape[0]) < bs or host_ok.dtype != torch.int8 or host_ok.device != dev:
                raise ValueError("inkling_pcopy: the host mask must be an int8 [>= bs] tensor on the device")
            hok = host_ok.contiguous()
        if hist.flat.device != dev or hist.flat.dtype != torch.int32 or hist.cov.dtype != torch.int64:
            raise ValueError("inkling_pcopy: history buffers on another device or of another dtype")
        if sample is not None:
            cand, q = sample
            if (cand.dim() != 3 or q.dim() != 3 or int(cand.shape[0]) < bs or int(q.shape[0]) < bs
                    or cand.shape[1:] != q.shape[1:] or int(cand.shape[1]) != self.gamma or int(cand.shape[2]) <= 0
                    or cand.device != dev or q.device != dev):
                raise ValueError("inkling_pcopy: selector sample shape / device outside the fused step's contract")
            cand, q = cand[:bs].contiguous(), q[:bs].contiguous()
            co, qo = self.owned(cand, q)
            co, qo = co[:bs], qo[:bs]
            K = int(cand.shape[2])
            sb, sg = cand.stride(0), cand.stride(1)
            if co.stride() != cand.stride() or qo.stride() != q.stride() or q.stride() != cand.stride():
                raise ValueError("inkling_pcopy: owned sample buffers do not share the sample's strides")
        else:
            cand = q = co = qo = draft  # unused pointers (HAS_SAMPLE = False)
            K, sb, sg = 1, 0, 0
        outputs = (self.min_live[:bs], self.forced_any[:bs], None if sample is None else (co, qo))
        args = (hist.flat, hist.cap, hist.rows, hist.cov, slots, p0, hok,
                draft, draft.stride(0), int(vocab), int(bool(force)),
                cand, co, q, qo, sb, sg,
                self.min_live, self.forced_any, self.counters, self.best)
        meta = dict(KEY=rule.KEY, MAXLEN=rule.MAXLEN, FORCE_LEN=rule.FORCE, G=self.gamma,
                    GP=triton.next_power_of_2(max(self.gamma, 1)), EP=triton.next_power_of_2(rule.MAXLEN - rule.KEY),
                    K=K, KP=triton.next_power_of_2(max(K, 1)), HAS_SAMPLE=sample is not None)

        do_scan = bool(force)
        if do_scan:
            chunk = int(self.chunk)
            if chunk <= 0 or chunk % TILE:
                raise ValueError(f"inkling_pcopy: scan chunk {chunk} is not a positive multiple of {TILE}")
            scan_args = (hist.flat, hist.cap, hist.rows, hist.cov, slots, p0, hok, draft, draft.stride(0), self.best)
            scan_grid = (bs, triton.cdiv(int(hist.cap), chunk))
            scan_meta = dict(KEY=rule.KEY, CHUNK=chunk, TILE=TILE)
            self.last_scan = (scan_grid, scan_args, scan_meta)

        def submit():
            try:
                if do_scan:
                    if self.dirty:
                        self.best.fill_(-1)
                        self.dirty = False
                    _scan_kernel[scan_grid](*scan_args, **scan_meta)
                _step_kernel[(bs,)](*args, **meta)
            except BaseException:
                self.dirty = True
                raise

        return submit, outputs

    def launch(self, hist, slots, p0, host_ok, draft, vocab, force, sample):
        submit, outputs = self.prepare(hist, slots, p0, host_ok, draft, vocab, force, sample)
        submit()
        return outputs
