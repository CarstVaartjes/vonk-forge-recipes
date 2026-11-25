"""Prompt-end checkpoint recency (INKLING_PROMPT_TOUCH=1): LRU order only, nothing the model computes changes.

A request's prompt-end checkpoint (the conv state at the end of its prompt and the sliding window behind it) gets its
LRU recency when it is inserted, at prefill time. The final insert of the finished request refreshes only the window
behind the OUTPUT end (SWA refreshes the end node's window ancestors), and the request's lock moves there. So once the
request finishes, its prompt-end window is the oldest evictable window in the tree, older than every stale chunk
window that later prefills left (their states are gone after the per-path cap; the windows can never serve a match).
Under window-pool pressure the prompt-end windows go first, in prefill order: the state survives without its window,
the next turn of that conversation (which shares exactly the prompt when the template drops the reasoning) falls
back to the root and re-prefills everything, and each such re-prefill evicts the next-oldest prompt-end window.

This module bumps the recency of the prompt path right after the final insert, with the cache's own touch_prefix
(the tree walk the scheduler uses for waiting requests: no allocation, no lock, no state slot), keyed by the prompt up
to the deepest depth the request published by the end of its prefill. Requests without such a depth (whole-prompt
cache hits, retracted since, streaming sessions, PD) are left alone. Scope: the hybrid window + state tree.
"""

import os
import sys

from . import _hook

URC = "sglang.srt.mem_cache.unified_radix_cache"
SWA = "sglang.srt.mem_cache.unified_cache.components.swa"
STATE = {"calls": 0, "skipped": 0, "inert": "", "min": 4096}
MARK = "_inkling_prompt_ckpt"


GUARD = BaseException   # caught at every overlay boundary (orig() is never inside one)
SHUTDOWN = (SystemExit, KeyboardInterrupt)   # marked inert, then re-raised: shutdown is never swallowed


def _log(msg):
    try:
        sys.stderr.write(f"[inkling_ptouch pid={os.getpid()}] {msg}\n")
        sys.stderr.flush()
    except SHUTDOWN:
        raise
    except BaseException:  # noqa: BLE001  (diagnostics are best effort)
        pass


def _inert(why):
    """why: a fixed literal. Marked inert first, then logged best effort."""
    STATE["inert"] = why
    _log("INERT: " + why)


def _premise(mod):
    """The SWA window refresh happens at the END node of a match / insert only (not on the walk down)."""
    try:
        import inspect
        src = inspect.getsource(mod.SWAComponent.refresh_lru)
        if "reset_node_and_window_ancestors_mru" not in src or "WALKDOWN" not in src:
            _inert("SWA refresh_lru changed (the window refresh is no longer end-node only)")
    except GUARD as exc:  # noqa: BLE001
        _inert("premise check failed")
        if isinstance(exc, SHUTDOWN):
            raise


def _hybrid(cache):
    names = {type(c).__name__ for c in getattr(cache, "_components_tuple", ())}
    return "SWAComponent" in names and "MambaComponent" in names


def wrap(mod):
    try:
        _wrap(mod)
    except GUARD as exc:  # noqa: BLE001
        _inert("install failed")
        if isinstance(exc, SHUTDOWN):
            raise


def _wrap(mod):
    try:
        cls = mod.UnifiedRadixCache
        orig = cls._checkpoint_into_tree
        need = ("touch_prefix", "disable")
        if not callable(getattr(cls, "touch_prefix", None)):
            _inert(f"UnifiedRadixCache lacks {need[0]}")
            return
        from sglang.srt.mem_cache.common import _req_radix_key
    except GUARD as exc:  # noqa: BLE001
        _inert("install failed")
        if isinstance(exc, SHUTDOWN):
            raise
        return
    if getattr(orig, "_inkling_ptouch", False):
        return

    def _checkpoint_into_tree(self, req, *, up_to, **kwargs):
        pre, failed, pending = None, False, None
        try:
            if not STATE["inert"] and not self.disable and _hybrid(self) and not getattr(req, "session_id", None):
                pre = (bool(req.finished()), len(req.origin_input_ids), int(req.retraction_count))
        except GUARD as exc:  # noqa: BLE001
            STATE["inert"] = "pre-read failed"                # no call that could raise before orig()
            pre, failed = None, True
            if isinstance(exc, SHUTDOWN):
                pending = exc
        out = orig(self, req, up_to=up_to, **kwargs)          # exactly once, outside the overlay's handlers
        if failed:
            _inert("pre-read failed")
        if pending is not None:
            raise pending                                     # the original insert ran; shutdown goes on
        if pre is None:
            return out
        try:
            if STATE["inert"]:
                return out
            finished, prompt, retractions = pre
            if not finished:
                # the checkpoint that completes the prompt: the deepest depth published on the prompt path so far
                depth = int(req.kv.cache_inserted_len)
                if up_to >= prompt >= STATE["min"] and 0 < depth <= prompt:
                    setattr(req, MARK, (depth, retractions))
            else:
                from sglang.srt.managers.req_types import FINISH_ABORT
                mark = getattr(req, MARK, None)
                if (mark is not None and mark[1] == retractions and int(req.retraction_count) == retractions
                        and not isinstance(req.finished_reason, FINISH_ABORT)):
                    self.touch_prefix(_req_radix_key(self, req, req.origin_input_ids, max_prefix_len=mark[0]))
                    STATE["calls"] += 1
                    if STATE["calls"] in (1, 24, 64) or STATE["calls"] % 1024 == 0:
                        _log(f"prompt paths touched after the final insert: {STATE['calls']} "
                             f"(skipped {STATE['skipped']})")
                else:
                    STATE["skipped"] += 1
        except GUARD as exc:  # noqa: BLE001
            _inert("touch failed")
            if isinstance(exc, SHUTDOWN):
                raise
        return out

    _checkpoint_into_tree._inkling_ptouch = True
    _checkpoint_into_tree.__wrapped__ = orig
    cls._checkpoint_into_tree = _checkpoint_into_tree
    _log(f"armed: prompt paths (prompts >= {STATE['min']} tokens) touched after the final insert")


def install():
    try:
        STATE["min"] = int(os.environ.get("INKLING_PROMPT_TOUCH_MIN", "4096") or 4096)
        _hook.when_imported(SWA, _premise)
        _hook.when_imported(URC, wrap)
    except GUARD as exc:  # noqa: BLE001
        _inert("install failed")
        if isinstance(exc, SHUTDOWN):
            raise
