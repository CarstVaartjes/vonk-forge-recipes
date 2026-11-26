"""Prefix-hit tail recompute for the DFlash2 draft ring (SPEC=dflash2), as an import-time overlay.

    INKLING_DFLASH_PREFIX_TAIL=2048        (default: unset / 0 -> install() does nothing)
    INKLING_DFLASH_PREFIX_TAIL_MAX_INPUT=4096   only inputs of at most this many tokens are bounded (default 4096)

DFlash2's draft KV is a ring over the last 2,048 positions, filled from the target's tapped hidden states as the
target prefills or verifies them. A radix-cache hit skips the target prefill of the matched prefix, so the ring holds
only the uncached suffix and the drafter proposes with little or no context ("prefix-hit starvation"; speedbench
primes every prompt, and multi-turn traffic re-hits its prefix every turn). This bounds the prefix match so that the
last N input tokens are always prefilled again: Req._compute_max_prefix_len (the bound SGLang's own logprob_start_len
path uses for match_prefix, the hicache prefetch and the storage linker) returns at most input_len - N. The cache
itself picks the longest reusable prefix within the bound (page / state-checkpoint aligned), exactly as for a
logprob request. When the uncached suffix is already >= N tokens nothing changes.

Inkling's hybrid conv-window state is cached at checkpoints, not at every position: a bound that falls between
checkpoints resumes from the nearest stored state below it, which on the FIRST bounded hit of a prompt is usually the
root (a full re-prefill; measured: 32K / 128K hits 0.2 -> 6.5 s / 0.4 -> 35 s TTFT, with no acceptance gain at those
lengths). That re-prefill plants a state at the bound (SGLang's branching checkpoint), so a later hit of the same
prompt redoes only ~N tokens. The plant is not reliable across turns (it needs the chunk-grid alignment of the
extend; the per-path state cap drops the older one) and it replaces the request's own end-of-prompt checkpoint, so
the next turn re-prefills from the bound even when it is not bounded itself. Short-turn agent loops on 5-8K
histories measured -7 % end to end; so only inputs of at most INKLING_DFLASH_PREFIX_TAIL_MAX_INPUT tokens (default
4096) are bounded: single-shot prompts up to 4K keep the gain (speedbench longctx +16 % decode tok/s, +6 % end to
end), agent histories above 4K run exactly as without the bound. Raise it (e.g. 8192) for long-output or
document-QA traffic, where a turn of ~1-2K output tokens pays the re-prefill back.

Scope: the bound applies only when the server's speculative algorithm is DFLASH (read once from SGLang's runtime
context on the first lookup; any other algorithm, or a context that cannot be read, leaves every request stock), never
to a dLLM request, and never to a streaming-session continuation (its committed slot KV must not shrink). The stock
bound and its lookup are pinned by source hash (_sources.py): a different image refuses at start-up (exit 78; set
DFLASH_PREFIX_TAIL=0 to run without the tail).

(B) INKLING_DFLASH_TAIL_SNAPSHOTS=8 (snapshot.py, bpatch.py): instead of recomputing, the draft window of a prompt's
end is copied out of the ring when its prefill completes and copied back into the ring of a later request that hits
the same prefix (draft-only state; the target path is untouched). With INKLING_DFLASH_TAIL_DEFER=0, independent of
(A): (A) still bounds prompts of at most MAX_INPUT tokens and (B) fills the ring below the bound.

(A defers to B) INKLING_DFLASH_TAIL_DEFER=1 (with both on): before (A) lowers the bound, a snapshot of the same prefix
that ends past (A)'s bound and already covers the draft window (A) would recompute is reserved (snapshot.reserve);
the request then keeps the stock bound, capped at the snapshot's end, and (B) restores the window instead of the target
recomputing it. Without such a snapshot (A) applies, decided before the radix lookup runs. The lookup decides the actual
reusable endpoint; the reservation is checked after it (snapshot.holds) and before the restore (snapshot.claim); see
snapshot.py for what a late mismatch does (the target keeps a valid cache hit; only the draft window can be shorter).
Not for logprob requests, multi-rank processes or hierarchical / external KV storage (the tail applies as before).
0 runs exactly as (A) + (B) independently.

Exactness: the recomputed positions produce the same KV / states as a cold prefill of those positions (FP order of
the prefill chunking only, the prefill allowance); the drafter is draft-only. Cost: TTFT on the bounded hits (above).
"""

import os
import sys

from . import _hook, _sources

ENV = "INKLING_DFLASH_PREFIX_TAIL"
MODULE = "sglang.srt.managers.schedule_batch"
MARK = "_inkling_dftail"
OFF_HINT = " (set DFLASH_PREFIX_TAIL=0 DFLASH_TAIL_SNAPSHOTS=0 to run without the DFlash2 prefix tail and snapshots)"
ENV_DEFER = "INKLING_DFLASH_TAIL_DEFER"
STATE = {"installed": False, "tail": 0, "max_input": 4096, "bounded": 0, "defer": False, "deferred": 0,
         "defer_dropped": 0}
# None: not resolved yet; True / False: the server's speculative algorithm is / is not DFLASH
SCOPE = {"dflash": None, "why": "", "logged": False}


def _log(msg):
    sys.stderr.write(f"[inkling_dftail pid={os.getpid()}] {msg}\n")
    sys.stderr.flush()


def tail_from_env(env=None):
    env = os.environ if env is None else env
    raw = env.get(ENV, "").strip()
    if raw in ("", "0"):
        return 0
    try:
        n = int(raw)
    except ValueError as exc:
        raise RuntimeError(f"inkling_dftail: {ENV}={raw!r} is not an integer") from exc
    if not 0 <= n <= 1 << 20:
        raise RuntimeError(f"inkling_dftail: {ENV}={n} out of range")
    return n


def max_input_from_env(env=None):
    env = os.environ if env is None else env
    raw = env.get(ENV + "_MAX_INPUT", "").strip()
    if not raw:
        return 4096
    try:
        n = int(raw)
    except ValueError as exc:
        raise RuntimeError(f"inkling_dftail: {ENV}_MAX_INPUT={raw!r} is not an integer") from exc
    if n < 0:
        raise RuntimeError(f"inkling_dftail: {ENV}_MAX_INPUT={n} is negative")
    return n


def defer_from_env(env=None):
    env = os.environ if env is None else env
    raw = env.get(ENV_DEFER, "").strip()
    if raw not in ("", "0", "1"):
        raise RuntimeError(f"inkling_dftail: {ENV_DEFER}={raw!r} must be 0 or 1")
    return raw == "1"


def _spec_algorithm():
    """The serving process's speculative algorithm (SGLang runtime context; the scheduler publishes it at start-up)."""
    from sglang.srt.runtime_context import get_spec

    return get_spec().speculative_algorithm


def _in_scope():
    """True only once the speculative algorithm has been read as DFLASH. Resolved on the first lookup and cached;
    a context that cannot be read yet leaves the request stock and is tried again on the next one."""
    if SCOPE["dflash"] is not None:
        return SCOPE["dflash"]
    try:
        alg = _spec_algorithm()
    except Exception as exc:  # noqa: BLE001  (inert, never fails a request)
        if not SCOPE["logged"]:
            SCOPE["logged"] = True
            _log(f"inert for now: cannot read the speculative algorithm ({exc!r})")
        return False
    name = getattr(alg, "name", alg)
    SCOPE["dflash"] = isinstance(name, str) and name.strip().upper() == "DFLASH"
    SCOPE["why"] = repr(name)
    _log(f"speculative algorithm {name!r}: " + ("prefix tail active" if SCOPE["dflash"] else "inert (not DFLASH)"))
    return SCOPE["dflash"]


def _reservation(req, input_len, capped):
    """(A defers to B) The lookup's reservation: planned on the first capping bound of a lookup (PENDING), then the
    same answer for every later bound of that request until its next lookup or its restore."""
    from . import snapshot as sn

    d = getattr(req, "__dict__", None)
    if d is None:
        return None
    res = d.get(sn.RES_ATTR)
    if res is sn.PENDING:
        try:
            res = sn.reserve(req, input_len, capped)
        except Exception as exc:  # noqa: BLE001  (never fails a request: the tail applies)
            res = None
            if not SCOPE.get("defer_logged"):
                SCOPE["defer_logged"] = True
                _log(f"tail deferral skipped: {exc!r}")
        d[sn.RES_ATTR] = res
        if res is not None:
            STATE["deferred"] += 1
    if isinstance(res, sn.Reservation) and res.input_len == int(input_len) and res.end > capped:
        return res
    return None


def make_bound(orig, tail, max_input=4096, defer=False):
    def _compute_max_prefix_len(self, input_len):
        bound = orig(self, input_len)
        if tail <= 0 or int(input_len) > max_input:
            return bound
        is_dllm = getattr(self, "is_dllm", None)
        if callable(is_dllm) and is_dllm():
            return bound
        session = getattr(self, "session", None)
        if session is not None and getattr(session, "streaming", False):
            return bound  # a streaming continuation reuses its slot's committed KV, which must not shrink
        capped = max(0, int(input_len) - tail)
        if capped < bound and _in_scope():
            res = _reservation(self, input_len, capped) if defer else None
            if res is not None:
                return min(bound, res.end)  # > capped; B restores the window below the hit
            STATE["bounded"] += 1
            return capped
        return bound

    _compute_max_prefix_len.__wrapped__ = orig
    setattr(_compute_max_prefix_len, MARK, True)
    return _compute_max_prefix_len


def _plain_tree(tree_cache):
    """The device-only radix tree this recipe runs (SGLang's UnifiedRadixCache itself, no subclass, with no
    hierarchical / external KV storage attached): storage-backed trees plan their staged loads outside this lookup."""
    return (type(tree_cache).__name__ == "UnifiedRadixCache" and getattr(tree_cache, "cache_controller", None) is None
            and getattr(tree_cache, "linker", None) is None)


def make_lookup(orig):
    """(A defers to B) Req.init_next_round_input: a lookup with a tree cache plans its reservation afresh (the previous
    one is released). After the radix lookup the reservation must still hold at the actual endpoint; if it does not
    (nothing touches the slots during a match, so only an unexpected state gets here) it is dropped and the request
    keeps its valid cache hit: the match is not repeated, because a match stages hybrid-state slots and copies."""
    from . import snapshot as sn

    def init_next_round_input(self, tree_cache=None, *args, **kwargs):
        if tree_cache is None:
            return orig(self, tree_cache, *args, **kwargs)
        sn.drop(self.__dict__.get(sn.RES_ATTR))
        self.__dict__[sn.RES_ATTR] = sn.PENDING if _plain_tree(tree_cache) else None
        try:
            out = orig(self, tree_cache, *args, **kwargs)
        except BaseException:
            sn.drop(self.__dict__.get(sn.RES_ATTR))
            self.__dict__[sn.RES_ATTR] = None
            raise
        res = self.__dict__.get(sn.RES_ATTR)
        if not isinstance(res, sn.Reservation):
            self.__dict__[sn.RES_ATTR] = None
            return out
        try:
            ok = sn.holds(res, self)
        except Exception:  # noqa: BLE001
            ok = False
        if not ok:
            sn.drop(res)
            self.__dict__[sn.RES_ATTR] = None
            STATE["defer_dropped"] += 1
        return out

    init_next_round_input.__wrapped__ = orig
    setattr(init_next_round_input, MARK, True)
    return init_next_round_input


def patch(mod):
    cls = getattr(mod, "Req", None)
    orig = getattr(cls, "_compute_max_prefix_len", None) if cls is not None else None
    if orig is None:
        raise RuntimeError("inkling_dftail: schedule_batch.Req._compute_max_prefix_len is missing" + OFF_HINT)
    if getattr(orig, MARK, False):
        return False
    if STATE["defer"]:
        lookup = getattr(cls, "init_next_round_input", None)
        if lookup is None:
            raise RuntimeError("inkling_dftail: schedule_batch.Req.init_next_round_input is missing" + OFF_HINT)
        cls.init_next_round_input = make_lookup(lookup)
    cls._compute_max_prefix_len = make_bound(orig, STATE["tail"], STATE["max_input"], STATE["defer"])
    return True


def install(env=None, root=None):
    """(A) INKLING_DFLASH_PREFIX_TAIL and (B) INKLING_DFLASH_TAIL_SNAPSHOTS are independent; either arms the overlay."""
    if STATE["installed"]:
        return STATE["tail"]
    from . import bpatch, snapshot

    tail = tail_from_env(env)
    max_input = max_input_from_env(env)
    slots = snapshot.slots_from_env(env)
    defer = defer_from_env(env) and bool(tail) and bool(slots)
    files = (_sources.PART_A if tail else ()) + (_sources.PART_B if slots else ())
    if files:
        _sources.check(root, OFF_HINT, files)
        targets = ([MODULE] if tail else []) + (list(bpatch.MODULES) if slots else [])
        loaded = [m for m in targets if m in sys.modules]
        if loaded:
            raise RuntimeError(f"inkling_dftail: install before {loaded} are imported" + OFF_HINT)
    STATE.update(installed=True, tail=tail, max_input=max_input, defer=defer)
    snapshot.STATE["slots"] = slots
    snapshot.STATE["defer"] = defer
    if tail:
        _hook.when_imported(MODULE, patch)
        _log(f"armed: a prefix-cache match leaves the last {tail} input tokens to be prefilled again "
             f"(inputs of at most {max_input} tokens; DFLASH only, not streaming sessions)"
             + ("; deferred to a draft-tail snapshot that covers the window" if defer else ""))
    if slots:
        _hook.when_imported(bpatch.CONFIG_MODULE, lambda mod: bpatch.patch_config(mod, slots, OFF_HINT))
        _hook.when_imported(bpatch.RING_MODULE, lambda mod: bpatch.patch_ring(mod, OFF_HINT))
        _hook.when_imported(bpatch.WORKER_MODULE, lambda mod: bpatch.patch_worker(mod, slots, OFF_HINT))
        _hook.when_imported(bpatch.UPDATER_MODULE, lambda mod: bpatch.patch_updater(mod, OFF_HINT))
        _log(f"armed: draft-tail snapshots, {slots} slots (prompt-end draft windows restored on prefix-cache hits)")
    return tail
