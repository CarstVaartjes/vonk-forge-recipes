"""inkling_pcopy hooks (DFlashWorkerV2, KVCacheConfigurator; see the package docstring).

  KVCacheConfigurator._dflash_draft_kv_ring_bytes  charges the history buffer to the KV budget with the ring
  DFlashWorkerV2._init_draft_kv_ring              allocates the history (after the stock ring), checks the refusals
  DFlashWorkerV2.forward_batch_generation         resets the step state; extend batches seed the history first
  DFlashWorkerV2._plan_verify                     outermost: anchor, match, override, cap minimum, block write
  DFlashWorkerV2._observe_acceptance              forced rows grade no drafts (the estimator never learns from them)

Any refusal or hook exception makes the lever inert for the process (logged once): from then on every step is the
stock step (drafts untouched). A failure after a step's override has started is re-raised: a half-applied override is
never verified.
"""

import os
import sys

import torch

from . import history as hmod
from . import fused as fmod
from . import matcher, rule

CONFIG_MODULE = "sglang.srt.mem_cache.kv_cache_configurator"
WORKER_MODULE = "sglang.srt.speculative.dflash_worker_v2"
MODULES = (CONFIG_MODULE, WORKER_MODULE)
MARK = "_inkling_pcopy"
CAP_PAD = 256   # history positions past the context length (a verify block and slack)

STATE = {"key": 0, "min_ctx": 8192, "inert": "", "charged": None,
         "stats": {"steps": 0, "forced_rows": 0, "forced_slots": 0, "bypassed": 0, "seeded": 0, "gated": 0}}


def _log(msg):
    sys.stderr.write(f"[inkling_pcopy pid={os.getpid()}] {msg}\n")
    sys.stderr.flush()


def _disable(worker, why):
    if worker is not None and getattr(worker, "_inkling_pcopy", None) is not None:
        worker._inkling_pcopy = None
    if not STATE["inert"]:
        STATE["inert"] = why
        _log(f"prompt copy inert: {why}")


# ------------------------------------------------------------------------------------------------- budget charge
def make_ring_bytes(orig):
    def _dflash_draft_kv_ring_bytes(self):
        ring_bytes = orig(self)
        if ring_bytes <= 0:
            return ring_bytes
        from sglang.srt.runtime_context import get_schedule

        max_num_reqs = max(1, get_schedule().max_running_requests // self.attn_dp_size)
        rows = self._req_pool_rows(max_num_reqs) + self._req_pool_pre_alloc_size() + 2
        cap = int(self.model_config.context_len) + CAP_PAD
        hist = hmod.History.bytes_for(rows, cap)
        STATE["charged"] = hist
        _log(f"history {hist / 2**20:.1f} MiB charged to the KV budget ({rows} rows x {cap} positions)")
        return ring_bytes + hist

    _dflash_draft_kv_ring_bytes.__wrapped__ = orig
    setattr(_dflash_draft_kv_ring_bytes, MARK, True)
    return _dflash_draft_kv_ring_bytes


# ------------------------------------------------------------------------------------------------- refusals / init
def refusal(worker):
    """Why this server cannot run the lever (None: it can)."""
    from sglang.srt.runtime_context import get_disagg, get_parallel, get_schedule

    if getattr(worker, "selector", None) is None:
        return "the DFLASH selector draft is not in use"
    if getattr(worker, "draft_kv_ring", None) is None:
        return "no DFLASH draft KV ring"
    if getattr(worker, "_verify_planner", None) is not None:
        return "the stock ragged verify planner is on (static verify mode required)"
    if getattr(get_disagg(), "disaggregation_mode", "null") not in (None, "null"):
        return "PD disaggregation"
    if getattr(get_disagg(), "enable_pd_role_switch", False):
        return "PD role switching"
    if getattr(get_parallel(), "attn_dp_enabled", False):
        return "DP attention (decode-as-extend)"
    if getattr(get_schedule(), "enable_mixed_chunk", False):
        return "mixed chunked prefill"
    if getattr(worker, "_megakernel_sampled", None) is not None:
        return "the megakernel sampled accept"
    mod = sys.modules.get(type(worker).__module__)
    if mod is not None and float(getattr(mod, "SIMULATE_ACC_LEN", 0) or 0) > 0:
        return "simulated acceptance"
    vcap = sys.modules.get("inkling_vcap.state")
    if vcap is not None and getattr(getattr(vcap, "STATE", None), "steplog", None) is not None:
        return "the verify-cap steplog capture"
    rs = sys.modules.get("inkling_rsfix")
    rstate = (getattr(rs, "STATE", {}) or {}) if rs is not None else {}
    parts = set(rstate.get("parts", ()) or ())
    applied = set(rstate.get("applied", ()) or ())   # target modules patched (selector_kernel, selector_model, ...)
    if not {"selector", "reject"} <= parts or not {"reject", "selector_kernel", "selector_model"} <= applied:
        return (f"inkling_rsfix selector + reject parts not applied (parts {sorted(parts)}, "
                f"targets {sorted(applied)})")
    return None


def make_init_ring(orig):
    def _init_draft_kv_ring(self):
        orig(self)
        self._inkling_pcopy = None
        try:
            why = refusal(self)
            if why is not None:
                _disable(self, why)
                return
            rows = int(self.model_runner.req_to_token_pool.req_to_token.shape[0])
            cap = int(self.model_runner.req_to_token_pool.req_to_token.shape[1]) + CAP_PAD
            need = hmod.History.bytes_for(rows, cap)
            if STATE["charged"] is None or need > STATE["charged"]:
                _disable(self, f"the history ({need} B) was not charged to the KV budget ({STATE['charged']})")
                return
            hist = hmod.History(rows, cap, self.device)
            mc = self.model_runner.model_config
            vocab = int(getattr(mc, "token_id_vocab_size", None) or mc.vocab_size)   # the real (unpadded) vocabulary
            self._inkling_pcopy = {"hist": hist, "forced_any": None, "vocab": vocab, "phase": "done"}
            _log(f"prompt copy armed: key {rule.KEY}, force at {rule.FORCE}, {rows} rows x {cap} positions, "
                 f"{hist.nbytes / 2**20:.1f} MiB")
        except Exception as exc:  # noqa: BLE001
            _disable(self, f"init failed: {exc!r}")

    _init_draft_kv_ring.__wrapped__ = orig
    setattr(_init_draft_kv_ring, MARK, True)
    return _init_draft_kv_ring


# ------------------------------------------------------------------------------------------------- per forward
def _streaming(req):
    session = getattr(req, "session", None)
    return session is not None and bool(getattr(session, "streaming", False))


def _gated(req):
    """True for a request whose prompt is shorter than the minimum context: it is never matched or forced (on short
    prompts the copy misfires more often than it helps), and its history is not kept."""
    return len(req.origin_input_ids) < STATE["min_ctx"]


def make_forward(orig):
    def forward_batch_generation(self, batch, *args, **kwargs):
        st = getattr(self, "_inkling_pcopy", None)
        if st is not None:
            st["forced_any"] = None
            _set_min_live(None)
            mode = getattr(batch, "forward_mode", None)
            if mode is not None and (mode.is_extend() or getattr(batch, "is_extend_in_batch", False)):
                try:
                    gen = self.model_runner.req_to_token_pool.req_generation
                    entries = []
                    for i, req in enumerate(batch.reqs):
                        if _streaming(req) or _gated(req):
                            continue
                        slot = int(req.kv.req_pool_idx)
                        prefix = int(batch.prefix_lens[i])
                        end = prefix + int(batch.extend_lens[i])   # the batch's own snapshot of this extend
                        entries.append((slot, int(gen[slot]), prefix, end, req.full_untruncated_fill_ids))
                    STATE["stats"]["seeded"] += len(st["hist"].seed(entries))
                except Exception as exc:  # noqa: BLE001
                    _disable(self, f"history seed failed: {exc!r}")
        return orig(self, batch, *args, **kwargs)

    forward_batch_generation.__wrapped__ = orig
    setattr(forward_batch_generation, MARK, True)
    return forward_batch_generation


def _set_min_live(t):
    vstate = sys.modules.get("inkling_vcap.state")
    st = getattr(vstate, "STATE", None) if vstate is not None else None
    if st is not None:
        st.df_min_live = t


def _host_rows(self, batch):
    """[bs] host bool: the row's history belongs to its current allocation and the row may be matched (the request's
    slot must also be the slot the batch hands the kernels)."""
    gen = self.model_runner.req_to_token_pool.req_generation
    hist = self._inkling_pcopy["hist"]
    cpu_idx = getattr(batch, "req_pool_indices_cpu", None)
    ok = []
    for i, req in enumerate(batch.reqs):
        slot = getattr(req.kv, "req_pool_idx", None)
        good = (slot is not None and 0 <= int(slot) < hist.rows and not _streaming(req) and not _gated(req)
                and hist.gen[int(slot)] == int(gen[int(slot)])
                and (cpu_idx is None or int(cpu_idx[i]) == int(slot)))
        ok.append(bool(good))
    return ok


def _host_mask(hist, ok, bs, device):
    """The host mask on the device without a pageable copy: all-true built on the device; otherwise a pinned copy
    kept alive (with an event) until it has run."""
    if all(ok) and len(ok) == bs:
        return torch.ones(bs, dtype=torch.bool, device=device)
    vals = ok[:bs] + [False] * max(0, bs - len(ok))
    if device.type != "cuda":
        return torch.tensor(vals, dtype=torch.bool)
    t = torch.tensor([int(v) for v in vals], dtype=torch.int8, pin_memory=True)   # pinned, int8: no conversion
    d = t.to(device, non_blocking=True)
    ev = torch.cuda.Event()
    ev.record()
    hist._inflight.append((ev, (t, d)))
    return d


def _bypass(self, batch):
    """Why this batch is not forced (its history is still written), or None."""
    if getattr(batch, "has_grammar", False):
        return "grammar"
    si = getattr(batch, "sampling_info", None)
    if si is not None:
        if getattr(si, "has_custom_logit_processor", False):
            return "custom logit processor"
        if getattr(si, "need_min_p_sampling", False):
            return "min_p"
        if not getattr(si, "is_all_greedy", True) and getattr(self, "_selector_sample", None) is None:
            return "sampled rows without a selector sample"
    return None


def step_ref(self, batch, prefix_lens):
    """The reference step (CPU tests; the GPU check's oracle for the fused one). Anchor, match, then prepare everything (new drafted slots, owned sample, cap minimum, history write) before the
    one in-place copy into the aliased draft buffer. Device work only; no host sync."""
    st = self._inkling_pcopy
    hist = st["hist"]
    rows_ok = _host_rows(self, batch)
    if not any(rows_ok):                                               # no row may be matched: the step stays stock
        STATE["stats"]["gated"] += 1
        return None
    bs = int(prefix_lens.numel())
    draft = self._draft_block_tokens_buf[:bs]
    gamma = int(draft.shape[1]) - 1
    p0 = prefix_lens.reshape(-1)[:bs].to(torch.int64)
    slots = batch.req_pool_indices.reshape(-1)[:bs].to(torch.int64)
    host_ok = _host_mask(hist, rows_ok, bs, p0.device)
    ok = hist.covered(slots, p0, host_ok.bool())
    hist.write_block(slots, p0, draft[:, :1], ok, offset=0)            # the anchor joins the context
    STATE["stats"]["steps"] += 1
    why = _bypass(self, batch)
    if why is not None:
        STATE["stats"]["bypassed"] += 1
        hist.write_block(slots, p0, draft[:, 1:], ok, offset=1)
        return None
    m, n, cont = matcher.match(hist.hist, slots, p0 + 1, ok, gamma)
    valid = (cont >= 0) & (cont < st["vocab"])
    run = torch.cumprod(valid.to(torch.int64), dim=1).bool()           # stop the copy at the first invalid id
    forced = rule.force_mask(m, n, gamma) & ok[:, None] & run
    new_draft = torch.where(forced, cont.to(draft.dtype), draft[:, 1:])
    sample = getattr(self, "_selector_sample", None)
    new_sample = None if sample is None else rule.override_sample(sample[0][:bs], sample[1][:bs], cont, forced)
    any_row = forced.any(dim=1)
    min_live = torch.where(any_row, forced.sum(dim=1) + 1, torch.zeros_like(n))
    hist.write_block(slots, p0, new_draft, ok, offset=1)               # drafted slots as they will be verified
    st["phase"] = "override"   # only non-fallible assignments from here
    draft[:, 1:].copy_(new_draft)
    if new_sample is not None:
        self._selector_sample = new_sample
    st["forced_any"] = any_row
    _set_min_live(min_live)
    return forced


COUNTER_EVERY = 1024
COUNTER_NAMES = ("steps", "covered_rows", "forced_rows", "forced_slots")


def _counters(st, fs):
    """Every COUNTER_EVERY steps, one non_blocking D2H of the device counters into pinned memory; the line is logged
    once its event has completed (event.query, never a wait)."""
    st["n"] = st.get("n", 0) + 1
    pend = st.get("cnt_pending")
    if pend is not None and pend[0].query():
        vals = pend[1].tolist()
        st["cnt_pending"] = None
        _log("counters: " + ", ".join(f"{k} {v}" for k, v in zip(COUNTER_NAMES, vals)))
    if st["n"] % COUNTER_EVERY == 0 and st.get("cnt_pending") is None:
        host = st.get("cnt_host")
        if host is None:
            host = st["cnt_host"] = torch.zeros(4, dtype=torch.int64, pin_memory=True)
        host.copy_(fs.counters, non_blocking=True)
        ev = torch.cuda.Event()
        ev.record()
        st["cnt_pending"] = (ev, host)


def _distinct_slots(batch, bs):
    cpu = getattr(batch, "req_pool_indices_cpu", None)
    ids = cpu.tolist()[:bs] if cpu is not None else [int(r.kv.req_pool_idx) for r in batch.reqs[:bs]]
    return len(set(ids)) == len(ids)


def step(self, batch, prefix_lens):
    """The decode-step hook: the fused Triton step on CUDA (fused.py: a split scan, then the per-row step), the
    reference step otherwise. Argument checks and every output view are built before the phase is armed; what can
    still fail after that (JIT compilation, the launches) propagates and the stock verify never runs on a possibly
    rewritten block; counter polling and reaping after the submission are best effort."""
    st = self._inkling_pcopy
    hist = st["hist"]
    if not hist.hist.is_cuda or fmod.triton is None:
        return step_ref(self, batch, prefix_lens)
    rows_ok = _host_rows(self, batch)
    if not any(rows_ok):                                               # no row may be matched: no device work at all
        STATE["stats"]["gated"] += 1
        return None
    bs = int(prefix_lens.numel())
    draft = self._draft_block_tokens_buf[:bs]
    fs = st.get("fused")
    if fs is None:
        fs = st["fused"] = fmod.FusedStep(hmod.SINKS, int(draft.shape[1]) - 1, draft.device)
    if not _distinct_slots(batch, bs):
        raise ValueError("inkling_pcopy: duplicate request slots in one batch")
    p0 = prefix_lens.reshape(-1)[:bs]
    slots = batch.req_pool_indices.reshape(-1)[:bs]
    host_ok = None if (len(rows_ok) == bs and all(rows_ok)) else _host_mask(hist, rows_ok, bs, draft.device)
    why = _bypass(self, batch)
    sample = getattr(self, "_selector_sample", None) if why is None else None
    submit, (min_live, any_row, owned) = fs.prepare(hist, slots, p0, host_ok, draft, st["vocab"], why is None, sample)
    st["phase"] = "override"   # armed before the mutating kernel is submitted
    submit()
    if owned is not None:
        self._selector_sample = owned
    if why is None:
        st["forced_any"] = any_row
        _set_min_live(min_live)
    else:
        STATE["stats"]["bypassed"] += 1
    STATE["stats"]["steps"] += 1
    _after(st, fs, hist)
    return None


def _after(st, fs, hist):
    """Counter polling and pinned-buffer reaping after a submitted step: best effort (diagnostics and housekeeping
    only); a failure here never reaches the verify of a step whose outputs are already published."""
    if not st.get("counters_off"):
        try:
            _counters(st, fs)
        except Exception as exc:  # noqa: BLE001
            st["counters_off"] = True
            _log(f"counters off: {type(exc).__name__}")
    try:
        hist.reap()
    except Exception as exc:  # noqa: BLE001
        if not st.get("reap_warned"):
            st["reap_warned"] = True
            _log(f"history reap failed (retried next step): {type(exc).__name__}")


def make_plan_verify(orig):
    def _plan_verify(self, *, batch, draft_input, prefix_lens):
        try:
            st = getattr(self, "_inkling_pcopy", None)
            if st is not None:
                st["phase"] = "pre"
                try:
                    step(self, batch, prefix_lens)
                except Exception as exc:  # noqa: BLE001
                    # Before the in-place copy the drafts are untouched and the step stays stock; a failure in the
                    # copy itself must never reach the verify.
                    if st.get("phase") == "override":
                        raise
                    st["forced_any"] = None
                    _set_min_live(None)
                    _disable(self, f"step failed: {exc!r}")
                st["phase"] = "done"
            return orig(self, batch=batch, draft_input=draft_input, prefix_lens=prefix_lens)
        finally:
            _set_min_live(None)

    _plan_verify.__wrapped__ = orig
    setattr(_plan_verify, MARK, True)
    return _plan_verify


def make_observe(orig):
    def _observe_acceptance(self, *args, **kwargs):
        st = getattr(self, "_inkling_pcopy", None)
        mask = st.get("forced_any") if st is not None else None
        est = getattr(self, "_acceptance_estimator", None)
        if mask is None or est is None:
            return orig(self, *args, **kwargs)
        observe = est.observe

        def masked(*, features, num_correct_drafts, num_graded_drafts):
            m = mask[: num_graded_drafts.shape[0]]
            return observe(features=features, num_correct_drafts=num_correct_drafts,
                           num_graded_drafts=num_graded_drafts.masked_fill(m, 0))

        had = "observe" in getattr(est, "__dict__", {})
        prior = est.__dict__.get("observe") if had else None
        est.observe = masked
        try:
            return orig(self, *args, **kwargs)
        finally:
            if had:
                est.observe = prior
            else:
                del est.observe

    _observe_acceptance.__wrapped__ = orig
    setattr(_observe_acceptance, MARK, True)
    return _observe_acceptance


# ------------------------------------------------------------------------------------------------- patching
def _need(cls, names, where, hint):
    if cls is None:
        raise RuntimeError(f"inkling_pcopy: {where} is missing" + hint)
    for name in names:
        if getattr(cls, name, None) is None:
            raise RuntimeError(f"inkling_pcopy: {where}.{name} is missing" + hint)


def patch_config(mod, hint=""):
    cls = getattr(mod, "KVCacheConfigurator", None)
    _need(cls, ["_dflash_draft_kv_ring_bytes"], "KVCacheConfigurator", hint)
    if getattr(cls._dflash_draft_kv_ring_bytes, MARK, False):
        return False
    cls._dflash_draft_kv_ring_bytes = make_ring_bytes(cls._dflash_draft_kv_ring_bytes)
    return True


def patch_worker(mod, hint=""):
    cls = getattr(mod, "DFlashWorkerV2", None)
    names = ["_init_draft_kv_ring", "forward_batch_generation", "_plan_verify", "_observe_acceptance"]
    _need(cls, names, "DFlashWorkerV2", hint)
    if getattr(cls._plan_verify, MARK, False):
        return False
    cls._init_draft_kv_ring = make_init_ring(cls._init_draft_kv_ring)
    cls.forward_batch_generation = make_forward(cls.forward_batch_generation)
    cls._plan_verify = make_plan_verify(cls._plan_verify)
    cls._observe_acceptance = make_observe(cls._observe_acceptance)
    return True
