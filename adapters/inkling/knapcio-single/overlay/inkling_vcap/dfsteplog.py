"""Full-width DFLASH capture: per request-step walked-row features, accepted drafts, per-visit expert unions, and
per-step GPU phase timings ("inkling-dflash-steplog/1"). Measurement only: it never changes what the engine computes,
and a capture failure never stops serving (it disables the capture, logs once, and the server keeps running).

Per decode step (host call order inside DFlashWorkerV2.forward_batch_generation):
  mark_draft_start(bs) (_stage_draft_sampler) takes a preallocated staging slot (no wait: when the writer is behind,
                                              the step is dropped as backpressure; serving never stalls), records e0 (draft phase start)
  start(...)          (_plan_verify)          snapshots request ids, pre-verify context lengths, request temperatures
                                              and the walked-row features of this step's draft (DflashFeatures, written
                                              in the draft graph), resets the device route cursor and sets the device
                                              real batch size, records e1 (verify phase start)
  route(ids)          (Inkling2Gate, target verify; recorded inside the verify graph) journals the real rows' routed
                                              ids per MoE visit: v3_kernels.record_route = a write kernel that only
                                              reads the cursor + a separate one-warp cursor advance (stream-ordered)
  finish(accept_len)  (_accept_block)         records e2 (verify phase end: target verify + logits adjustments +
                                              accept), enqueues the device->pinned copies, then zeroes the device real
                                              batch size (verify replays of dropped steps journal nothing), records e3
                                              (copies complete) and hands the slot to the writer thread
  end_step()          (forward_batch_generation exit) drops a step that never reached _accept_block
The writer waits for e3, computes the prefix unions (widths 1..8 per visit) and the timings, and writes atomic npz
chunks; after every flush and every idle second it rewrites status-<run>.json {logged, written, idle, ...} so the window
script can confirm the final drain before stopping the server.

Timings are instrumented GPU phase durations on the forward stream: t_draft_ms = e0->e1 (aux join, draft forward with
the selector walk and the feature writes, feature snapshot), t_verify_ms = e1->e2 (verify preparation, target verify
replay, logits adjustments, accept); both can include GPU idle while the host prepares work. t_start_ms is e0 relative
to an event recorded at attach (one GPU clock: consecutive differences = full step period); host_t = time.time() at
_plan_verify.

Columns per request-step: rid, context, greedy, temp (request temperature, 0 for greedy), t_eff (walk temperature: 1
for greedy rows), rows [G, K] (the walked selector rows, final scores), m [G], gap [G] (from rows at t_eff, as the
policy computes them), accepted (0..G drafts, full width). Per step: step, bs, run, union [MAX_VISITS, stride]
(-1 = unused visit), visits (MoE visits journaled; 0 = journal missing, invalid or partial (an unwritten real row); a
complete Inkling-2-Mini step has 48), walks (selector walks recorded between this step's draft start and its verify; 1 = the features are this step's),
t_start_ms, t_draft_ms, t_verify_ms, host_t. For steps with bs <= RAW_IDS_MAX_BS also ids [n, MAX_VISITS, stride,
MAX_TOPK] uint8 (routed expert ids per request / visit / row / lane, 255 = empty) and ids_req (their row index into the
per-request columns of the same file). Replays must use only steps with walks == 1 and visits == the model's count.

Lattice / target fields, per request-step: cand_ids [G, K] int32, scores [G, K, K]
fp16 (the full lattice: slot 0 rows all equal the anchor's), unary [G, K] (unary logits after the draft's unary
transform, before the pair correction), lse [G] (log-sum-exp of the same transform over the whole projected candidate
plane: tail mass = 1 - sum_k exp(unary_k - lse); NaN when the plane was not produced), drafted [G], committed [S]
(the step's out_tokens row; the first commit_len are emitted), commit_len, tgt_p1 [S] / tgt_gap [S] (target top-1
probability and top-1 minus top-2 of softmax(next_token_logits) at T = 1 per verify row, computed after e2; NaN when the
logits were not materialized), prompt_len, prompt_hash (hash of the prompt token ids, per request). Category and the
inside-thinking bit are derived offline (phase timestamps of the window script, the rebuilt per-rid outputs).
tgt_cand_p [G, K]: the target's probabilities (request temperature) of verify row j on slot j's candidate ids, sampled
requests without top-k / top-p / min-p filtering only (NaN otherwise): input of the exact block-verification replay
(tools/bvexpect.py). filtered: the request's target was filtered before sampling.
"""

from __future__ import annotations

import atexit
import json
import os
import queue
import socket
import threading
import time
import uuid
from pathlib import Path

import numpy as np
import torch

FORMAT = "inkling-dflash-steplog/1"
MAX_VISITS = 64
MAX_TOPK = 8
CHUNK_STEPS = int(os.environ.get("INKLING_VERIFY_CAP_STEPLOG_CHUNK", "64"))
QUEUE_DEPTH = int(os.environ.get("INKLING_VERIFY_CAP_STEPLOG_SLOTS", "32"))  # pinned staging slots (~0.9 MB each at cap 48)
# Raw routed ids are kept for steps with bs <= RAW_IDS_MAX_BS (uint8, 255 = empty): exact unions of mixed per-request
# widths in the replay. Env INKLING_VERIFY_CAP_STEPLOG_RAW_BS overrides (0 = off).
RAW_IDS_MAX_BS = int(os.environ.get("INKLING_VERIFY_CAP_STEPLOG_RAW_BS", "8"))


def _log(msg):
    from inkling_vcap.state import log

    log(f"steplog: {msg}")


def prefix_unions(ids, stride):
    """[MAX_VISITS, stride] int16: distinct experts over the first w rows of every request, w = 1..stride, per visit.
    ids [visits, bs, stride, MAX_TOPK] int16, -1 = empty lane. (policy3 steplog.prefix_unions.)"""
    visits = ids.shape[0]
    counts = np.full((MAX_VISITS, stride), -1, dtype=np.int16)
    if visits == 0:
        return counts
    visit, req, pos, lane = np.nonzero(ids >= 0)
    experts = ids[visit, req, pos, lane]
    if experts.size == 0:
        return counts
    first = np.full((visits, int(experts.max()) + 1), stride, dtype=np.int16)
    np.minimum.at(first, (visit, experts), pos)
    counts[:visits] = np.count_nonzero(first[..., None] <= np.arange(stride), axis=1)
    return counts


class _Slot:
    __slots__ = ("rows", "routes", "control", "seq", "seq_dev", "e0", "e1", "e2", "e3")

    def __init__(self, capacity, ncols, stride, cuda, device):
        self.rows = torch.empty((capacity, ncols), dtype=torch.float32, pin_memory=cuda)
        self.routes = torch.empty(MAX_VISITS * capacity * stride * MAX_TOPK, dtype=torch.int16, pin_memory=cuda)
        self.control = torch.empty(2, dtype=torch.int32, pin_memory=cuda)
        # walk counter before the draft / before the verify: their difference is the walks of this step
        self.seq_dev = torch.zeros(2, dtype=torch.int64, device=device)
        self.seq = torch.empty(2, dtype=torch.int64, pin_memory=cuda)
        if cuda:
            self.e0, self.e1, self.e2 = (torch.cuda.Event(enable_timing=True) for _ in range(3))
            self.e3 = torch.cuda.Event()
        else:
            self.e0 = self.e1 = self.e2 = self.e3 = None


class DfStepLog:
    def __init__(self, directory, *, stride, top_k, capacity, device, features):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.run = f"{socket.gethostname()}-{os.getpid()}-{uuid.uuid4().hex[:12]}"
        self.stride, self.gamma, self.top_k = int(stride), int(stride) - 1, int(top_k)
        self.capacity = int(capacity)
        self.device = torch.device(device)
        self.cuda = self.device.type == "cuda"
        self.features = features
        g, k, S = self.gamma, self.top_k, self.stride
        # float32 staging row per request-step (token / candidate ids < 2**24 are exact in float32).
        # META_COLS are filled at _plan_verify (draft side), FINISH_COLS at _accept_block.
        self.meta_cols = [("rows", g * k), ("m", g), ("gap", g), ("t_eff", 1), ("greedy", 1), ("temp", 1),
                          ("context", 1), ("cand", g * k), ("scores", g * k * k), ("unary", g * k), ("lse", g),
                          ("drafted", g), ("filtered", 1)]
        self.finish_cols = [("accepted", 1), ("committed", S), ("commit_len", 1), ("tgt_p1", S), ("tgt_gap", S),
                            ("tgt_cand_p", g * k)]
        self.col = {}
        off = 0
        for name, width in self.meta_cols + self.finish_cols:
            self.col[name] = slice(off, off + width)
            off += width
        self.ncols = off
        self.prompt_cache = {}  # rid -> (prompt_len, prompt_hash): hashed once per request on the host
        # A certified head may leave next_token_logits unwritten (CERT_WIDE_DFLASH) or narrowed: target top-1 / gap
        # are then recorded as NaN instead of reading a stale or partial buffer (fusion review).
        on = lambda v: os.environ.get(v, "").strip().lower() not in ("", "0", "off", "false", "no", "none")  # noqa: E731
        self.target_stats_ok = not any(on(v) for v in ("INKLING_CERT_HEAD", "INKLING_CERT_WIDE",
                                                       "INKLING_CERT_WIDE_DFLASH", "CERT_WIDE_DFLASH"))
        # Graph-referenced device storage: allocated once and kept for the process (also after a failure).
        self.control = torch.zeros(2, dtype=torch.int32, device=self.device)  # [visit cursor, real bs]
        self.journal = torch.full((MAX_VISITS * self.capacity * self.stride * MAX_TOPK,), -1, dtype=torch.int16,
                                  device=self.device)
        self.origin = None
        if self.cuda:
            self.origin = torch.cuda.Event(enable_timing=True)
            self.origin.record()
        self.failed = None    # first capture error; from then on every hook is a no-op
        self.slot = None      # taken at mark_draft_start
        self.marked = False   # mark_draft_start ran for the current step (with or without a slot)
        self.pending = None   # (rids, metadata, host_t) between start and finish
        self.active = False
        self.step = 0         # steps handed to the writer
        self.step_ids = 0     # every decode step seen (handed or dropped): the 'step' column, gaps = drops
        self.written = 0      # steps written to npz (writer thread)
        self.chunk = 0
        self.dropped = {"no_draft_mark": 0, "no_accept": 0, "over_capacity": 0, "no_rids": 0, "backpressure": 0}
        self.missing_routes = 0
        self.partial_routes = 0
        self.stale_features = 0
        self.visits_seen = {}
        self._writer_error = None
        self._closed = False
        self._warned = set()
        self.free = queue.Queue()
        self.ready = queue.Queue()
        for _ in range(QUEUE_DEPTH):
            self.free.put(_Slot(self.capacity, self.ncols, self.stride, self.cuda, self.device))
        self.thread = threading.Thread(target=self._write, name="inkling-dflash-steplog", daemon=True)
        self.thread.start()
        atexit.register(self.close)

    # ------------------------------------------------------------ failure isolation
    def fail(self, where, exc):
        """Disable the capture for the rest of the process; serving continues. Recorded graph kernels stay (their
        storage is kept alive) but journal nothing once the device real batch size is 0."""
        if self.failed is None:
            self.failed = f"{where}: {exc!r}"
            _log(f"CAPTURE DISABLED ({self.failed}); serving continues without the capture")
        self.pending = None
        self.active = False
        self.marked = False
        if self.slot is not None:
            self.free.put(self.slot)
            self.slot = None
        try:
            self.control[1].zero_()
        except Exception:  # noqa: BLE001
            pass
        # status-<run>.json is rewritten by the writer thread within a second (no file I/O on this thread)

    def _guard(self, where, fn, *args):
        if self.failed is not None:
            return
        try:
            fn(*args)
        except Exception as exc:  # noqa: BLE001 - never let the capture kill the scheduler
            self.fail(where, exc)

    def _log_once(self, key, msg):
        if key not in self._warned:
            self._warned.add(key)
            _log(msg)

    def _drop(self, why):
        self.dropped[why] += 1
        self.step_ids += 1  # a dropped step still consumes a step id (completeness = complete / ids)
        self._log_once(why, f"dropping a step ({why}); counts so far {self.dropped}")
        if self.slot is not None:
            self.free.put(self.slot)
            self.slot = None
        self.pending = None
        self.active = False
        self.marked = False
        self.control[1].zero_()  # no journal writes until the next start

    # ------------------------------------------------------------ scheduler thread: public hooks
    def mark_draft_start(self, bs):
        self._guard("mark_draft_start", self._mark_draft_start, int(bs))

    def start(self, *, batch, prefix_lens):
        self._guard("start", self._start, batch, prefix_lens)

    def finish(self, *, accept_len, commit_lens=None, out_tokens=None, logits=None):
        self._guard("finish", self._finish, accept_len, commit_lens, out_tokens, logits)

    def end_step(self):
        """forward_batch_generation returned: a step that never reached _accept_block (a megakernel tail / greedy
        accept path) is dropped here, so no state leaks into the next forward."""
        self._guard("end_step", self._end_step)

    def route(self, ids):
        if self.failed is not None:
            return
        capturing = self.cuda and torch.cuda.is_current_stream_capturing()
        # CUDA: always launched, also on eager calls outside a logged step (the target graph warm-up forwards then
        # JIT-compile both kernels before the capture); outside a step the device real batch size is 0, so they
        # journal nothing. The kernels read the device cursor / real batch size at every replay.
        if not (self.cuda or self.active):
            return
        if (ids.ndim != 2 or ids.shape[0] == 0 or ids.shape[0] % self.stride or not 0 < ids.shape[1] <= MAX_TOPK
                or ids.shape[0] > self.capacity * self.stride):
            if capturing or not self.active:
                # A graph shape this capture cannot journal (e.g. above capacity): nothing is recorded into it;
                # steps replaying it get visits = 0 and are excluded by the replay.
                self._log_once(f"route-shape-{tuple(ids.shape)}", f"verify graph with routes {tuple(ids.shape)} "
                               "is not journaled (outside the capture's capacity / layout)")
                return
            self.fail("route", RuntimeError(f"unexpected target MoE routes {tuple(ids.shape)}"))
            return
        if self.cuda:
            try:
                from inkling_vcap.v3_kernels import record_route

                record_route(ids, self.journal, self.control, self.stride, MAX_TOPK, MAX_VISITS)
            except Exception as exc:  # noqa: BLE001
                if capturing:
                    raise  # never leave a half-recorded graph capture behind silently
                self.fail("route", exc)
            return
        if not self.active:
            return
        cursor, bs = self.control.tolist()
        if cursor < MAX_VISITS and 0 < bs and bs * self.stride <= ids.shape[0]:
            dest = self.journal[:MAX_VISITS * bs * self.stride * MAX_TOPK]
            dest = dest.view(MAX_VISITS, bs * self.stride, MAX_TOPK)[cursor]
            dest.fill_(-1)
            dest[:, :ids.shape[1]].copy_(ids[:bs * self.stride])
        self.control[0].add_(1)

    # ------------------------------------------------------------ implementations
    def _end_step(self):
        if self.pending is not None or self.slot is not None:
            self._drop("no_accept")
        self.marked = False

    def _mark_draft_start(self, bs):
        if self.pending is not None or self.slot is not None:
            self._drop("no_accept")  # the previous step never reached _accept_block
        self.marked = True
        if bs > self.capacity or bs > self.features.capacity:
            self.dropped["over_capacity"] += 1
            self.step_ids += 1
            self._log_once("over_capacity", f"batch {bs} above the capture capacity {self.capacity}: not captured")
            self.control[1].zero_()
            return
        if self._writer_error is not None:
            raise RuntimeError(f"writer failed: {self._writer_error!r}")
        try:
            slot = self.free.get_nowait()  # never wait on the scheduler thread
        except queue.Empty:
            self.dropped["backpressure"] += 1
            self.step_ids += 1
            self._log_once("backpressure", f"writer behind: a step was not captured (counts {self.dropped})")
            self.control[1].zero_()
            return
        self.slot = slot
        slot.seq_dev[0].copy_(self.features.seq[0])
        if self.cuda:
            slot.e0.record()

    def _start(self, batch, prefix_lens):
        if self.cuda and torch.cuda.is_current_stream_capturing():
            raise RuntimeError("step boundary inside a CUDA graph capture")
        marked, self.marked = self.marked, False
        if self.slot is None:
            if not marked:
                self.dropped["no_draft_mark"] += 1
                self.step_ids += 1
                self._log_once("no_draft_mark", "a verify without a draft mark was not captured")
            self.control[1].zero_()
            return
        bs = int(prefix_lens.numel())
        reqs = getattr(batch, "reqs", None)
        if reqs is None or len(reqs) != bs:
            self._drop("no_rids")
            return
        rids = tuple(str(req.rid) for req in reqs)
        f = self.features
        dev = f.m.device
        greedy = f.greedy[:bs]
        temps = getattr(getattr(batch, "sampling_info", None), "temperatures", None)
        if temps is None:
            temp = torch.zeros(bs, dtype=torch.float32, device=dev)
        else:
            temp = temps.reshape(-1)[:bs].to(device=dev, dtype=torch.float32)
        temp = torch.where(greedy, torch.zeros_like(temp), temp)
        context = prefix_lens.reshape(-1)[:bs].to(device=dev, dtype=torch.float32)
        g, k = self.gamma, self.top_k
        nan = lambda width: torch.full((bs, width), float("nan"), device=dev)  # noqa: E731
        rows = f.rows[:bs].reshape(bs, -1) if f.rows is not None else nan(g * k)
        if getattr(f, "lattice", False):
            lattice = (f.cand[:bs].reshape(bs, -1).to(torch.float32), f.scores[:bs].reshape(bs, -1).to(torch.float32),
                       f.unary[:bs * g].reshape(bs, -1), f.lse[:bs * g].reshape(bs, -1),
                       f.drafted[:bs].to(torch.float32))
        else:
            lattice = (nan(g * k), nan(g * k * k), nan(g * k), nan(g), nan(g))
        # requests whose target is filtered (top-k / top-p / min-p) before sampling: their raw softmax is not the
        # verified law, so the exact block-verification replay input (tgt_cand_p) is not recorded for them
        filt = torch.zeros(bs, dtype=torch.bool, device=dev)
        si = getattr(batch, "sampling_info", None)
        for name, off in (("top_ks", None), ("top_ps", 1.0), ("min_ps", 0.0)):
            v = getattr(si, name, None)
            if isinstance(v, torch.Tensor) and v.numel() >= bs:
                v = v.reshape(-1)[:bs].to(device=dev)
                if name == "top_ks":
                    filt |= (v > 0) & (v < int(1 << 30))
                elif name == "top_ps":
                    filt |= v.to(torch.float32) < 1.0
                else:
                    filt |= v.to(torch.float32) > 0.0
        # A fresh tensor: the next draft overwrites the feature buffers.
        meta = torch.cat((rows, f.m[:bs], f.gap[:bs], f.t_eff[:bs, None], greedy[:, None].to(torch.float32),
                          temp[:, None], context[:, None]) + lattice + (filt[:, None].to(torch.float32),), dim=1)
        prompts = [self._prompt_key(req) for req in reqs]
        self.pending = (rids, meta, time.time(), prompts)
        self.slot.seq_dev[1].copy_(f.seq[0])
        self.control[0].zero_()
        self.control[1].fill_(bs)
        self.active = True
        if self.cuda:
            self.slot.e1.record()

    def _prompt_key(self, req):
        """(prompt_len, prompt_hash), hashed once per Req object: kept on the object (a reused rid is a new Req);
        fallback cache keyed by (rid, id(req), len) when the object refuses attributes."""
        hit = getattr(req, "_inkling_dfsl_prompt", None)
        if hit is not None:
            return hit
        ids = getattr(req, "origin_input_ids", None) or ()
        key = (req.rid, id(req), len(ids))
        hit = self.prompt_cache.get(key)
        if hit is None:
            hit = (len(ids), hash(tuple(ids)) & 0x7FFFFFFFFFFFFFFF)
            try:
                setattr(req, "_inkling_dfsl_prompt", hit)
            except Exception:  # noqa: BLE001 - __slots__ / read-only: use the cache
                if len(self.prompt_cache) > 4096:
                    self.prompt_cache.clear()
                self.prompt_cache[key] = hit
        return hit

    def _target_stats(self, logits, bs, chunk=32):
        """[bs, S] target top-1 probability and top-1 minus top-2 probability of every verify row (softmax at T = 1
        of the logits the accept reads); NaN when the logits are not materialized (a resolved argmax)."""
        S = self.stride
        dev = self.features.m.device
        if logits is None or logits.dim() != 2 or logits.shape[0] < bs * S:
            nanv = torch.full((bs, S), float("nan"), device=dev)
            return nanv, nanv.clone()
        rows = bs * S
        p1 = torch.empty(rows, dtype=torch.float32, device=dev)
        gap = torch.empty(rows, dtype=torch.float32, device=dev)
        for start in range(0, rows, chunk):
            end = min(rows, start + chunk)
            x = logits[start:end].float()
            lse = torch.logsumexp(x, dim=-1)
            top = x.topk(2, dim=-1).values
            a, b = torch.exp(top[:, 0] - lse), torch.exp(top[:, 1] - lse)
            p1[start:end].copy_(a)
            gap[start:end].copy_(a - b)
        return p1.view(bs, S), gap.view(bs, S)

    def _target_cand_p(self, logits, bs, meta, chunk=4):
        """[bs, G, K] target probabilities (softmax at the request temperature) of verify row j on slot j's candidate
        ids, for sampled requests (greedy rows NaN: unused, and a certified greedy batch has no full logits).
        Exact block-verification replay input (tools/bvexpect.py)."""
        G, K, S = self.gamma, self.top_k, self.stride
        dev = self.features.m.device
        out = torch.full((bs, G, K), float("nan"), device=dev)
        if logits is None or logits.dim() != 2 or logits.shape[0] < bs * S or not getattr(self.features, "lattice", False):
            return out
        temp = meta[:, self.col["temp"]][:, 0]
        greedy = (meta[:, self.col["greedy"]][:, 0] > 0.5) | (meta[:, self.col["filtered"]][:, 0] > 0.5)
        t = torch.where(temp > 0, temp, torch.ones_like(temp))
        cand = self.features.cand[:bs]
        rows = logits[:bs * S].view(bs, S, -1)
        for b0 in range(0, bs, chunk):
            b1 = min(bs, b0 + chunk)
            x = rows[b0:b1, :G].float()
            x = (x - x.amax(dim=-1, keepdim=True)) / t[b0:b1, None, None]  # shift before scaling (low temperatures)
            lse = torch.logsumexp(x, dim=-1, keepdim=True)
            out[b0:b1] = torch.exp(x.gather(-1, cand[b0:b1]) - lse)
        return torch.where(greedy[:, None, None], torch.full_like(out, float("nan")), out)

    def _finish(self, accept_len, commit_lens=None, out_tokens=None, logits=None):
        if self.pending is None:
            return
        rids, meta, host_t, prompts = self.pending
        self.pending = None
        self.active = False
        slot, self.slot = self.slot, None
        bs = len(rids)
        if accept_len.numel() < bs:
            self.free.put(slot)
            raise RuntimeError(f"accept_len has {accept_len.numel()} entries for {bs} requests")
        if self.cuda:
            slot.e2.record()  # verify phase end: the reductions below are outside the timed phases
        dev, S = meta.device, self.stride
        f32 = dict(device=dev, dtype=torch.float32)
        acc = accept_len.reshape(-1)[:bs].to(**f32)[:, None]
        committed = (out_tokens.reshape(bs, -1)[:, :S].to(**f32) if out_tokens is not None
                     else torch.full((bs, S), float("nan"), device=dev))
        clen = (commit_lens.reshape(-1)[:bs].to(**f32)[:, None] if commit_lens is not None
                else torch.full((bs, 1), float("nan"), device=dev))
        p1, gap = self._target_stats(logits if self.target_stats_ok else None, bs)
        cp = self._target_cand_p(logits, bs, meta).reshape(bs, -1)
        data = torch.cat((meta, acc, committed, clen, p1, gap, cp), dim=1)
        slot.rows[:bs].copy_(data, non_blocking=self.cuda)
        size = MAX_VISITS * bs * self.stride * MAX_TOPK
        slot.routes[:size].copy_(self.journal[:size], non_blocking=self.cuda)
        slot.control.copy_(self.control, non_blocking=self.cuda)
        slot.seq.copy_(slot.seq_dev, non_blocking=self.cuda)
        self.control[1].zero_()  # after the copies (stream order): replays of dropped steps journal nothing
        if self.cuda:
            slot.e3.record()
        self.ready.put((self.step_ids, rids, host_t, slot, prompts))
        self.step += 1
        self.step_ids += 1

    # ------------------------------------------------------------ writer thread
    def _write_status(self, idle=False):
        try:
            status = dict(format=FORMAT, run=self.run, logged=self.step, step_ids=self.step_ids, written=self.written,
                          chunks=self.chunk,
                          idle=bool(idle), failed=self.failed,
                          writer_error=None if self._writer_error is None else repr(self._writer_error),
                          dropped=dict(self.dropped), missing_routes=self.missing_routes,
                          partial_routes=self.partial_routes, stale_features=self.stale_features,
                          visits={str(k): v for k, v in self.visits_seen.items()}, time=time.time())
            path = self.directory / f"status-{self.run}.json"
            tmp = path.with_suffix(".tmp")
            tmp.write_text(json.dumps(status))
            tmp.replace(path)
        except Exception:  # noqa: BLE001 - status is best effort
            pass

    def _write(self):
        records = []
        while True:
            try:
                item = self.ready.get(timeout=1.0)
            except queue.Empty:
                if records:
                    self._flush_safe(records)
                    records.clear()
                self._write_status(idle=True)
                continue
            if item is None:
                if records:
                    self._flush_safe(records)
                self._write_status(idle=True)
                return
            step, rids, host_t, slot, prompts = item
            rec = None
            try:
                rec = self._digest(step, rids, host_t, slot, prompts)
            except Exception as exc:  # noqa: BLE001
                self._writer_failed(exc)
            finally:
                self.free.put(slot)
            if rec is not None:
                records.append(rec)
            if len(records) >= CHUNK_STEPS:
                self._flush_safe(records)
                records.clear()

    def _writer_failed(self, exc):
        if self._writer_error is None:
            self._writer_error = exc
            _log(f"writer error ({exc!r}); the capture stops at the next step, serving continues")

    def _digest(self, step, rids, host_t, slot, prompts):
        if self._writer_error is not None:
            return None
        if slot.e3 is not None:
            while not slot.e3.query():
                time.sleep(0.0005)
            t_start = self.origin.elapsed_time(slot.e0)
            t_draft = slot.e0.elapsed_time(slot.e1)
            t_verify = slot.e1.elapsed_time(slot.e2)
        else:
            t_start = t_draft = t_verify = float("nan")
        n = len(rids)
        cursor, cbs = (int(x) for x in slot.control.tolist())
        # cursor > MAX_VISITS: a verify path sharded the real rows; cbs != n: not this step's journal
        visits = cursor if (0 <= cursor <= MAX_VISITS and cbs == n) else 0
        size = visits * n * self.stride * MAX_TOPK
        ids = slot.routes[:size].numpy().reshape(visits, n, self.stride, MAX_TOPK)
        union = prefix_unions(ids, self.stride)
        if visits == 0:
            self.missing_routes += 1
            if self.missing_routes == 1:
                _log(f"no valid target route journal for step {step} (cursor {cursor}, bs {cbs} vs {n}); "
                     "its unions are marked missing")
        elif (ids[:, :, :, 0] < 0).any():
            # an unwritten real row: the step's routes are not trustworthy -> visits = 0 (excluded by the replay)
            self.partial_routes += 1
            if self.partial_routes == 1:
                _log(f"step {step}: a journaled visit has unwritten rows; the step is marked visits = 0")
            visits = 0
            union = np.full_like(union, -1)
            ids = ids[:0]
        self.visits_seen[visits] = self.visits_seen.get(visits, 0) + 1
        walks = int(slot.seq[1]) - int(slot.seq[0])
        if walks != 1:
            self.stale_features += 1
            if self.stale_features == 1:
                _log(f"{walks} selector walks in the draft of step {step} (expected 1); flagged by 'walks'")
        raw = None
        if n <= RAW_IDS_MAX_BS and visits > 0:
            raw = np.full((n, MAX_VISITS, self.stride, MAX_TOPK), 255, dtype=np.uint8)
            raw[:, :visits] = np.where(ids >= 0, ids, 255).astype(np.uint8).transpose(1, 0, 2, 3)
        return (step, rids, slot.rows[:n].numpy().copy(), union, visits, walks, t_start, t_draft, t_verify, host_t,
                raw, prompts)

    def _flush_safe(self, records):
        try:
            self._flush(records)
            self.written += len(records)
        except Exception as exc:  # noqa: BLE001
            self._writer_failed(exc)
        self._write_status()

    def _flush(self, records):
        (steps, rids, rows, unions, visits, walks, t_start, t_draft, t_verify, host_t, raws, prompts) = zip(*records)
        g, k = self.gamma, self.top_k
        data = np.concatenate(rows)
        offsets = np.cumsum([0] + [len(r) for r in rids])
        raw_req = [np.arange(offsets[i], offsets[i + 1]) for i, r in enumerate(raws) if r is not None]
        raw_ids = [r for r in raws if r is not None]
        raw_req = np.concatenate(raw_req) if raw_req else np.zeros(0, dtype=np.int64)
        raw_ids = (np.concatenate(raw_ids) if raw_ids
                   else np.zeros((0, MAX_VISITS, self.stride, MAX_TOPK), dtype=np.uint8))
        path = self.directory / f"dflash1-{self.run}-{self.chunk:06d}.npz"
        tmp = path.with_suffix(".tmp")
        with tmp.open("wb") as out:
            c = self.col
            prm = np.asarray([p for batch in prompts for p in batch], dtype=np.int64).reshape(-1, 2)
            np.savez_compressed(
                out, format=FORMAT, run=self.run, stride=self.stride, top_k=k,
                step=np.asarray(steps, dtype=np.int64), bs=np.asarray([len(r) for r in rids], dtype=np.int32),
                rid=np.asarray([rid for batch_ids in rids for rid in batch_ids]),
                rows=data[:, c["rows"]].reshape(-1, g, k), m=data[:, c["m"]], gap=data[:, c["gap"]],
                t_eff=data[:, c["t_eff"]][:, 0], greedy=data[:, c["greedy"]][:, 0] > 0.5, temp=data[:, c["temp"]][:, 0],
                context=data[:, c["context"]][:, 0].astype(np.int64), accepted=data[:, c["accepted"]][:, 0].astype(np.int8),
                cand_ids=np.nan_to_num(data[:, c["cand"]], nan=-1).astype(np.int32).reshape(-1, g, k),
                scores=data[:, c["scores"]].astype(np.float16).reshape(-1, g, k, k),
                unary=data[:, c["unary"]].reshape(-1, g, k), lse=data[:, c["lse"]],
                drafted=np.nan_to_num(data[:, c["drafted"]], nan=-1).astype(np.int32),
                committed=np.nan_to_num(data[:, c["committed"]], nan=-1).astype(np.int32),
                commit_len=np.nan_to_num(data[:, c["commit_len"]][:, 0], nan=-1).astype(np.int8),
                tgt_p1=data[:, c["tgt_p1"]], tgt_gap=data[:, c["tgt_gap"]],
                tgt_cand_p=data[:, c["tgt_cand_p"]].reshape(-1, g, k),
                prompt_len=prm[:, 0].astype(np.int32), prompt_hash=prm[:, 1],
                union=np.stack(unions), visits=np.asarray(visits, dtype=np.int16),
                walks=np.asarray(walks, dtype=np.int32), t_start_ms=np.asarray(t_start, dtype=np.float64),
                t_draft_ms=np.asarray(t_draft, dtype=np.float32), t_verify_ms=np.asarray(t_verify, dtype=np.float32),
                host_t=np.asarray(host_t, dtype=np.float64), ids=raw_ids, ids_req=raw_req.astype(np.int64),
            )
        tmp.replace(path)
        self.chunk += 1

    def close(self):
        if self._closed:
            return
        self._closed = True
        joined = True
        if self.thread.is_alive():
            self.ready.put(None)
            self.thread.join(timeout=30)
            joined = not self.thread.is_alive()
        self._write_status(idle=joined)
        _log(f"{'closed' if joined else 'close TIMED OUT (writer still busy)'}: run={self.run} logged={self.step} "
             f"written={self.written} chunks={self.chunk} dropped={self.dropped} missing_routes={self.missing_routes} "
             f"partial_routes={self.partial_routes} stale_features={self.stale_features} failed={self.failed}")
