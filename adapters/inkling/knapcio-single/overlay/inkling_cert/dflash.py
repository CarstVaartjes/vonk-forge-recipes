"""inkling_cert on the DFLASH worker (SPEC=dflash2): the certified head (<= 16 rows) on greedy verify steps.

DSpark arms the certified head from its verify epilogue (patches.patch_verify). The DFLASH worker has no such
epilogue: its accept runs eagerly after the target verify graph, as `argmax(next_token_logits)` on all-greedy
batches whose logits nothing else reads (DFlashWorkerV2._can_skip_verify_logits: no grammar, no logprobs, no
logit adjustments, no selector sample). This module:

  DFlashWorkerV2.forward_batch_generation  records (worker, batch) for the step (a context variable)
  TpModelWorker.forward_batch_generation   on that step's target verify (is_verify=True): cert.on =
                                           batch_ok(...) and _can_skip_verify_logits(batch, sampling_info), written
                                           on the stream before the replay, and reset to 0 right after it, so no
                                           other graph replay (prefill, the next step) ever reads a stale 1

The narrow certified head (<= 16 rows: c1 / c2 with 8-row verify blocks) writes pico's exact logits on every
candidate tile and -inf elsewhere, so the stock softcap / mask / argmax over next_token_logits returns the same
token as the full head, bit for bit, also for ragged verify layouts.

The wide head (17..INKLING_CERT_WIDE_MAX_ROWS rows) returns the certified IDs and writes the logits buffer only on a
fallback. Its IDs are handed to the accept instead of argmax(next_token_logits):
  patches._wide_logits                  (capture) WIDE_IDS[(logits data_ptr, rows)] = that graph's ID buffer
  DecodeCudaGraphRunner._process_output_after_replay  (target verify, every replay) REPLAY["ids"] = the ID buffer
                                        of the graph that just replayed, looked up by its (padded) output logits
  DFlashWorkerV2._accept_block          on an armed step whose replay produced IDs: target_predict = ids[:bs * block]
                                        (the stock greedy branch; with the verify cap's cutoff, AcceptGreedy)
Not with SGLANG_RAGGED_VERIFY_MODE=compact (rows not request-major): refused at patch time.
"""

import contextvars
import functools
import sys

import torch

WORKER = "sglang.srt.speculative.dflash_worker_v2"
TP_WORKER = "sglang.srt.managers.tp_worker"
_STEP = contextvars.ContextVar("inkling_cert_dflash_step", default=None)
COUNTS = {"steps": 0, "armed": 0, "live_gather": 0, "wide_injected": 0}
WIDE_IDS = {}            # (data_ptr, rows) of a captured target-verify logits buffer -> [certified-ID buffers]
REPLAY = {"ids": None, "wide": False}  # the replayed target-verify graph's IDs; wide: it contained the wide head
ACTIVE = {"dflash": False}  # set when the DFLASH worker is patched (this process serves DFLASH)


def _state():
    from . import patches

    return patches


def note_wide_ids(out, ids):
    """patches._wide_logits, at capture (the IDs also ride on the returned logits object as _inkling_certwide_ids)."""
    WIDE_IDS.setdefault((out.data_ptr(), int(out.shape[0])), []).append(ids)


def _replayed_ids(logits):
    """(ids, wide) for a target-verify replay's padded output logits. The graph's static output object carries the
    attribute when the logits processor returned our view unchanged; otherwise the (buffer, rows) registry is used
    if exactly one captured graph matches. A wide graph whose IDs cannot be identified is fatal (its logits buffer
    is not written on certified steps)."""
    if not isinstance(logits, torch.Tensor) or logits.dim() != 2:
        return None, False
    ids = getattr(logits, "_inkling_certwide_ids", None)
    if ids is not None:
        return ids, True
    found = WIDE_IDS.get((logits.data_ptr(), int(logits.shape[0])))
    if not found:
        return None, False
    if len(found) == 1:
        return found[0], True
    return None, True


def patch_runner(mod):
    """Record which captured wide-certified graph (if any) a target-verify replay ran."""
    p = _state()
    cls = getattr(mod, "DecodeCudaGraphRunner", None)
    p._need(cls, ["_process_output_after_replay"], "DecodeCudaGraphRunner")
    orig = cls._process_output_after_replay
    if getattr(orig, "_inkling_cert", False):
        return

    @functools.wraps(orig)
    def _process_output_after_replay(self, output, forward_batch):
        output = orig(self, output, forward_batch)
        if self.capture_forward_mode.is_target_verify() and not self.model_runner.is_draft_worker:
            REPLAY["ids"], REPLAY["wide"] = _replayed_ids(getattr(output, "next_token_logits", None))
        return output

    _process_output_after_replay._inkling_cert = True
    cls._process_output_after_replay = _process_output_after_replay
    p.STATE["patched"].append("DecodeCudaGraphRunner._process_output_after_replay (DFLASH wide IDs)")


def dead_rows_to_anchor(x2, cert):
    """Under the verify cap (inkling_vcap on DFLASH: STATE.src_index, live row r <=> src_index[r] == r, dead rows
    -> their request's anchor row, identity on uncapped steps), give every dead row its anchor's hidden row before
    the certified head, so dead rows (whose hidden states the cap's expert remap made meaningless) add no candidate
    tiles of their own and cannot force a fallback. Captured into the verify graph; selected on the device:
    with cert.on == 0 (the full head serves, e.g. sampled steps) the rows stay as they are, bit for bit.
    Exact for every row an accept reads: live rows keep their own hidden rows, and the cap's cutoff never reads a
    dead row's token (_accept_block's cutoff_verify_lens)."""
    if not ACTIVE["dflash"]:
        return x2
    vs = sys.modules.get("inkling_vcap.state")
    st = getattr(vs, "STATE", None)
    src = getattr(st, "src_index", None)
    stride = int(getattr(st, "stride", 0) or 0)
    m = x2.shape[0]
    if (not isinstance(src, torch.Tensor) or not src.is_cuda or src.dtype != torch.int64 or src.numel() < m
            or src.device != x2.device or stride <= 0 or m % stride):
        return x2
    # src_index addresses the uniform request-major layout (stride rows per request); a compact ragged verify packs
    # rows differently, so the gather is only used without it
    try:
        from sglang.srt.speculative.ragged_verify import ragged_verify_compact_enabled

        if ragged_verify_compact_enabled():
            return x2
    except ImportError:
        return x2
    COUNTS["live_gather"] += 1
    gathered = x2.index_select(0, src[:m])
    return torch.where(cert.on.view(1, 1) != 0, gathered, x2)


def patch_worker(mod):
    p = _state()
    p._need(mod, ["DFlashWorkerV2"], WORKER)
    cls = mod.DFlashWorkerV2
    p._need(cls, ["forward_batch_generation", "_can_skip_verify_logits"], f"{WORKER}.DFlashWorkerV2")
    if getattr(cls.forward_batch_generation, "_inkling_cert", False):
        return
    p._need(cls, ["_accept_block"], f"{WORKER}.DFlashWorkerV2")
    if p.STATE["cfg"].get("wide"):
        from sglang.srt.speculative.ragged_verify import ragged_verify_compact_enabled

        if ragged_verify_compact_enabled():
            raise p.ops.CertFatal("inkling_cert: INKLING_CERT_WIDE with SGLANG_RAGGED_VERIFY_MODE=compact under DFLASH "
                                  "(the certified IDs are request-major)")
    fbg, accept = cls.forward_batch_generation, cls._accept_block
    ACTIVE["dflash"] = True

    @functools.wraps(fbg)
    def forward_batch_generation(self, batch, *args, **kwargs):
        token = _STEP.set((self, batch, {"armed": False}))
        try:
            return fbg(self, batch, *args, **kwargs)
        finally:
            _STEP.reset(token)

    @functools.wraps(accept)
    def _accept_block(self, *args, **kwargs):
        ids = REPLAY["ids"]
        step = _STEP.get()
        armed = step is not None and step[2]["armed"]
        if armed and REPLAY["wide"] and ids is None:
            raise p.ops.CertFatal("inkling_cert: a wide-certified verify graph replayed but its IDs are ambiguous")
        if (ids is not None and armed and kwargs.get("target_predict") is None
                and kwargs.get("verify_shard") is None):
            n = int(kwargs["bs"]) * int(self.block_size)
            if ids.shape[0] < n:
                raise p.ops.CertFatal("inkling_cert: certified IDs shorter than the verify block")
            kwargs["target_predict"] = ids[:n]
            COUNTS["wide_injected"] += 1
        return accept(self, *args, **kwargs)

    forward_batch_generation._inkling_cert = True
    _accept_block._inkling_cert = True
    cls.forward_batch_generation = forward_batch_generation
    cls._accept_block = _accept_block
    p.STATE["patched"].append("DFlashWorkerV2.forward_batch_generation/_accept_block")


def patch_tp_worker(mod):
    p = _state()
    p._need(mod, ["TpModelWorker"], TP_WORKER)
    cls = mod.TpModelWorker
    p._need(cls, ["forward_batch_generation"], f"{TP_WORKER}.TpModelWorker")
    orig = cls.forward_batch_generation
    if getattr(orig, "_inkling_cert_dflash", False):
        return

    @functools.wraps(orig)
    def forward_batch_generation(self, *args, **kwargs):
        step = _STEP.get()
        cert = p.STATE["cert"]
        if step is None or cert is None or not kwargs.get("is_verify", False):
            return orig(self, *args, **kwargs)
        worker, batch, flags = step
        si = getattr(batch, "sampling_info", None)
        # mode "check" serves pico's full head and builds the certified result beside it (cert.run_check)
        armed = bool(p.STATE["mode"] in ("1", "check") and p.STATE["disabled"] is None and p.batch_ok(batch, si)
                     and worker._can_skip_verify_logits(batch, si))
        COUNTS["steps"] += 1
        COUNTS["armed"] += armed
        p.STATE["calls"] += 1
        every = p.STATE["cfg"]["log_every"]
        if every and p.STATE["calls"] % every == 0:
            p._report(cert)
        flags["armed"] = armed
        REPLAY["ids"], REPLAY["wide"] = None, False  # set again only by a captured target-verify replay
        cert.set_on(armed)
        try:
            out = orig(self, *args, **kwargs)
        finally:
            cert.set_on(0)  # enqueued after the replay: no later replay reads a stale 1
        if not armed:
            REPLAY["ids"], REPLAY["wide"] = None, False
        return out

    forward_batch_generation._inkling_cert_dflash = True
    cls.forward_batch_generation = forward_batch_generation
    p.STATE["patched"].append("TpModelWorker.forward_batch_generation (DFLASH verify arming)")
