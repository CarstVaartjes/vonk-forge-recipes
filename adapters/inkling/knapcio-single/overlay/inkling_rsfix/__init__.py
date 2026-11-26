"""inkling_rsfix: exact final-token sampling in SGLang's speculative reject sampler.

    INKLING_RSFIX=1              on: all parts (unset / 0: nothing is imported)
    INKLING_RSFIX=reject,markov  only the listed parts (reject, markov, blockwise, selector)

Parts (each pins the stock source by sha256 and refuses to patch a different text):

- reject (sglang.kernels.ops.speculative.reject_sampling; DSpark and DFLASH sampled
  verification through chain_speculative_sampling_triton): the residual / bonus CDF pass
  of speculative_sampling_classic_kernel and of the split kernels picked the first lane
  with ``total_cumsum > target_u``, with no range or positive-mass condition, and fell back
  to VOCAB_SIZE - 1. FP32 rounding between the normalizer (tl.sum, or the block-
  verification residual mass) and the scan (tl.cumsum) let a padded vocabulary row
  (p = 0) or a lane >= VOCAB_SIZE be emitted. Fixed: first lane with
  ``(total_cumsum > target_u) & in range & (val > 0)``; no match -> last lane with positive
  residual mass; no positive residual lane at all -> last lane with positive target mass.
  See reject.py.
- markov (dspark.markov_sample, the DSpark Markov draft sampler): SGLang PR #51681,
  Philox words that round to u == 1 gave infinite exponential noise. See markov.py.
- blockwise (dspark.blockwise_rejection, off unless SGLANG_OPT_DSV41_BLOCKWISE_REJECTION):
  SGLang PR #51692, log-sum precision and the log(u) + log(q) accept rounding. See
  blockwise.py (a residual-cancellation case remains; the route is off in the recipe).
- selector (DFLASH / DFlash2 selector walk: dflash._selector_walk_kernel,
  CandidateSelector.sample_path's CPU path, lilicorr._selector_walk_torch): a sampled slot
  whose coin passed the scan's end was clamped to the last candidate, which can have
  q = 0. Fixed: first lane with cumsum > u and q > 0, else the last lane with q > 0;
  greedy rows unchanged. See selector.py.

Exactness: the target distribution is unchanged in exact arithmetic (the fixed rule picks
the same lane whenever the stock rule picks a lane with positive mass); only rounding
events change, and they no longer emit a zero-mass or out-of-range token. Accept
decisions, block verification and the RNG streams are untouched by `reject`. `markov`
changes a seeded draft draw only where the corrected upper-half noise changes the winner.

Patch: module attributes rebound right after each target module is imported (post-import
hook); the stock launchers look the kernels up as module globals at call time. CUDA-graph
safety as stock (the split path's extra per-row int32 is written one launch before use).
"""

import functools
import os
import sys
import types
import weakref

from . import _hook, _sources

PARTS = ("reject", "markov", "blockwise", "selector")
MARK = "_inkling_rsfix"

# part -> its target modules (keys of _PLAN / _sources.MODULES / _sources.EXPECTED)
PART_TARGETS = {
    "reject": ("reject",),
    "markov": ("markov",),
    "blockwise": ("blockwise",),
    "selector": ("selector_kernel", "selector_model", "selector_lilicorr"),
}


def _targets(chosen_parts):
    return tuple(t for p in chosen_parts for t in PART_TARGETS[p])


# target -> (fixed module, names rebound on the stock module, names the stock module must
# NOT have (new helpers), class methods {Class.method: (fixed name, "classmethod" |
# "function")}, injected stock dependencies (None: the fixed module's _STOCK_DEPS))
_PLAN = {
    "reject": (
        "inkling_rsfix.reject",
        ("speculative_sampling_classic_kernel", "_split_sum_kernel", "_split_cdf_kernel",
         "_split_scratch", "_chain_speculative_sampling_split"),
        (),
        {},
        None,
    ),
    "markov": (
        "inkling_rsfix.markov",
        ("_markov_sample_partial_kernel", "_select_block"),
        ("_exponential_from_philox",),
        {"MarkovSampleStep.torch": ("markov_torch", "classmethod")},
        None,
    ),
    "blockwise": (
        "inkling_rsfix.blockwise",
        ("_stats_kernel", "_rejection_kernel", "_resample_kernel",
         "blockwise_rejection_triton", "blockwise_rejection_torch"),
        ("_log_normalizer_parts",),
        {},
        None,
    ),
    "selector_kernel": ("inkling_rsfix.selector", ("_selector_walk_kernel",), (), {}, ()),
    "selector_model": (
        "inkling_rsfix.selector",
        (),
        (),
        {"CandidateSelector.sample_path": ("sample_path", "function")},
        ("selector_walk_triton", "_follow_maps"),
    ),
    "selector_lilicorr": ("inkling_rsfix.selector", ("_selector_walk_torch",), (), {}, ()),
}

# Module-level constants the fixed code relies on: (kind, value), kind "constexpr" for a
# triton.language.constexpr, "int" for a plain int.
_CONSTS = {
    "reject": {},
    "markov": {"_NO_INDEX": ("constexpr", 2147483647), "_UNIFORM_SCALE": ("constexpr", 2.3283064365386963e-10),
               "_PHILOX_ROUNDS": ("int", 10)},
    "blockwise": {"_TL_RAND_MIN": ("constexpr", 4.6566127342e-10), "_STATS_BLOCK": ("int", 8192),
                  "_RESAMPLE_BLOCK": ("int", 1024)},
    "selector_kernel": {},
    "selector_model": {},
    "selector_lilicorr": {},
}

# Every module object rsfix has patched (history per instance: a reload of any of them is
# refused, whatever is bound in it at that moment).
_PATCHED = weakref.WeakSet()

STATE = {"installed": False, "parts": (), "applied": [], "stock": {}, "fatal": None}


class RsfixFatal(BaseException):
    """BaseException: SGLang's model registry swallows Exception during model imports."""


def _log(msg):
    print(f"[inkling_rsfix] {msg}", file=sys.stderr, flush=True)


_OFF = ("", "0", "off", "false", "no", "none")
_ALL = ("1", "on", "true", "yes", "all")


def parts(env=None):
    """The parts INKLING_RSFIX selects, in PARTS order ((): off)."""
    env = os.environ if env is None else env
    raw = env.get("INKLING_RSFIX", "").strip().lower()
    if raw in _OFF:
        return ()
    if raw in _ALL:
        return PARTS
    chosen = [p.strip() for p in raw.split(",") if p.strip()]
    bad = [p for p in chosen if p not in PARTS]
    if bad or not chosen:
        raise RuntimeError(f"inkling_rsfix: INKLING_RSFIX={raw!r} (use 1, 0 or a subset of {','.join(PARTS)})")
    return tuple(p for p in PARTS if p in chosen)


def enabled(env=None):
    return bool(parts(env))


def check_sources(root=None, which=None):
    """RuntimeError unless every selected target file matches the pinned image text."""
    for target in _targets(which or PARTS):
        _sources.check_part(target, root)


def _jit_type():
    try:
        from triton.runtime.jit import JITFunction
    except Exception:  # noqa: BLE001  (CPU tests: no triton)
        return None
    return JITFunction


def _is_stock(obj, mod):
    """True when ``obj`` (a plain function, a genuine Triton JITFunction or a classmethod
    of one) was defined by the code of ``mod`` itself: its Python function's globals are
    the module's namespace. A wrapper installed by someone else fails: functools.wraps
    keeps __module__ (and may copy a JIT object's .fn), but the wrapper is neither a
    JITFunction nor a function of ``mod``."""
    if isinstance(obj, (classmethod, staticmethod)):
        obj = obj.__func__
    jit = _jit_type()
    fn = obj.fn if jit is not None and type(obj) is jit else obj
    return (isinstance(fn, types.FunctionType) and fn.__globals__ is vars(mod)
            and not getattr(fn, MARK, False))


def _constexpr_type():
    try:
        from triton.language import constexpr
    except Exception:  # noqa: BLE001  (CPU tests: no triton)
        return None
    return constexpr


def _const_ok(value, kind, want):
    if kind == "int":
        return type(value) is int and value == want
    cx = _constexpr_type()
    is_cx = type(value) is cx if cx is not None else type(value).__name__ == "constexpr"
    return is_cx and getattr(value, "value", None) == want


# Kind of every injected stock dependency (constants are also value-checked by _CONSTS).
_DEP_KINDS = {
    "_split_chain_walk": "jit",
    "_NO_INDEX": "const", "_UNIFORM_SCALE": "const", "_philox": "function", "_round_to_step_dtype": "jit",
    "_RESAMPLE_BLOCK": "const", "_STATS_BLOCK": "const", "_TL_RAND_MIN": "const", "RejectionNoise": "class",
    "_insert_kernel": "jit", "_log1p_neg_stable": "jit", "_murmur3_uniform32": "jit", "_step_seed": "jit",
    # selector_model: the launcher imported by name from the kernels module, and its own helper
    "selector_walk_triton": "from:sglang.kernels.ops.speculative.dflash", "_follow_maps": "compiled",
}


def _code_name(obj):
    fn = getattr(obj, "fn", obj)
    return getattr(getattr(fn, "__code__", None), "co_name", None)


def _dep_ok(name, obj, mod, consts):
    """An injected dependency must be of its expected kind, the stock module's own, and the
    function of that name (code name; for a torch.compile wrapper, the wrapped function's)."""
    kind = _DEP_KINDS.get(name)
    if kind in ("jit", "function") or (kind or "").startswith("from:"):
        if _code_name(obj) != name:
            return False
    if kind == "class":
        return isinstance(obj, type) and obj.__module__ == mod.__name__
    if kind == "jit":
        jit = _jit_type()
        return jit is not None and type(obj) is jit and _is_stock(obj, mod)
    if kind == "function":
        return isinstance(obj, types.FunctionType) and _is_stock(obj, mod)
    if kind == "const":
        return name in consts
    if kind == "compiled":
        # The bare stock function, or torch.compile's own wrapper of it: a function whose
        # code is Dynamo's (functools.wraps copies names, never __code__) and whose
        # _torchdynamo_orig_callable is the stock function.
        if isinstance(obj, types.FunctionType) and _is_stock(obj, mod):
            return _code_name(obj) == name
        code = getattr(obj, "__code__", None)
        dynamo = code is not None and code.co_filename.replace("\\", "/").endswith("torch/_dynamo/eval_frame.py")
        orig = getattr(obj, "_torchdynamo_orig_callable", None)
        return (isinstance(obj, types.FunctionType) and dynamo and isinstance(orig, types.FunctionType)
                and _is_stock(orig, mod) and _code_name(orig) == name)
    if kind and kind.startswith("from:"):
        src = sys.modules.get(kind[5:])
        return (src is not None and isinstance(obj, types.FunctionType) and getattr(src, name, None) is obj
                and _is_stock(obj, src))
    return False


def _is_patched(part, mod):
    """True when every name ``part`` rebinds is bound, on ``mod``, to the very object of
    rsfix's fixed module (identity, not a copyable marker); class methods must be a
    classmethod whose __func__ is rsfix's function."""
    fixed_name, names, _, methods, _ = _PLAN[part]
    fixed_mod = sys.modules.get(fixed_name)
    if fixed_mod is None:
        return False
    for n in names:
        if getattr(mod, n, None) is not getattr(fixed_mod, n, object()):
            return False
    for qual, (attr, kind) in methods.items():
        cls_name, meth = qual.split(".")
        raw = vars(getattr(mod, cls_name, object)).get(meth)
        fn = raw.__func__ if kind == "classmethod" and isinstance(raw, classmethod) else (
            raw if kind == "function" else None)
        if fn is None or fn is not getattr(fixed_mod, attr, object()):
            return False
    return True


def _refuse_reload(part, mod):
    """Before every execution of a target: a module object rsfix has ever patched is being
    reloaded. Refuse before the stock definitions run, so the patched bindings stay (also
    when someone else has wrapped a patched binding since)."""
    if mod in _PATCHED or _is_patched(part, mod):
        raise RsfixFatal(f"inkling_rsfix: reloading {mod.__name__} is not supported (it is patched)")


def apply(part, mod):
    """Rebind ``part``'s fixed objects on the imported stock module ``mod``. Returns
    {"stock": {...}, "fixed": {...}} (attribute name -> object; ``Class.method`` keys
    hold the raw classmethod), so a check can switch between them with use()."""
    import importlib

    fixed_name, names, new_names, methods, deps = _PLAN[part]
    path = getattr(mod, "__file__", None)
    try:
        with open(path, encoding="utf-8") as fh:
            text = fh.read()
    except (OSError, TypeError) as exc:
        raise RsfixFatal(f"inkling_rsfix: cannot read {mod.__name__} source ({path})") from exc
    bad = _sources.mismatches(part, text)
    if bad:
        raise RsfixFatal(f"inkling_rsfix: {path} differs from the pinned image: {bad}")
    for name, (kind, want) in _CONSTS[part].items():
        if not hasattr(mod, name) or not _const_ok(getattr(mod, name), kind, want):
            raise RsfixFatal(f"inkling_rsfix: {mod.__name__}.{name} is not the {kind} {want!r}")
    stock = {}
    for name in names:
        obj = getattr(mod, name, None)
        if obj is None or not _is_stock(obj, mod):
            raise RsfixFatal(f"inkling_rsfix: {mod.__name__}.{name} is missing or not the module's own")
        stock[name] = obj
    for name in new_names:
        if hasattr(mod, name):
            raise RsfixFatal(f"inkling_rsfix: {mod.__name__}.{name} already exists (upstream fix present?)")
    for qual, (_attr, kind) in methods.items():
        cls_name, meth = qual.split(".")
        raw = vars(getattr(mod, cls_name)).get(meth)
        is_kind = isinstance(raw, classmethod) if kind == "classmethod" else isinstance(raw, types.FunctionType)
        if not is_kind or not _is_stock(raw, mod):
            raise RsfixFatal(f"inkling_rsfix: {mod.__name__}.{qual} is not the stock {kind}")
        stock[qual] = raw
    # The fixed module imports nothing from sglang; its stock dependencies are bound here,
    # before any of its kernels can run.
    fixed_mod = importlib.import_module(fixed_name)
    for dep in (fixed_mod._STOCK_DEPS if deps is None else deps):
        if not hasattr(mod, dep) or not _dep_ok(dep, getattr(mod, dep), mod, _CONSTS[part]):
            raise RsfixFatal(f"inkling_rsfix: {mod.__name__}.{dep} is missing or not the module's own")
        setattr(fixed_mod, dep, getattr(mod, dep))
    fixed = {name: getattr(fixed_mod, name) for name in names + new_names}
    for obj in fixed.values():
        setattr(getattr(obj, "fn", obj), MARK, True)
    for qual, (attr, kind) in methods.items():
        fn = getattr(fixed_mod, attr)
        setattr(fn, MARK, True)
        fixed[qual] = classmethod(fn) if kind == "classmethod" else fn
    use(mod, fixed)
    _PATCHED.add(mod)
    return {"stock": stock, "fixed": fixed}


def use(mod, table):
    """Set every attribute of ``table`` (from apply()) on ``mod``; a None value deletes it."""
    for key, obj in table.items():
        target, name = mod, key
        if "." in key:
            cls_name, name = key.split(".")
            target = getattr(mod, cls_name)
        if obj is None:
            if hasattr(target, name):
                delattr(target, name)
        else:
            setattr(target, name, obj)


def _on_import(part, mod):
    """Runs after every execution of the target module (persistent hook): a module that an
    outer loader discarded after a successful patch is re-executed fresh and patched again.
    (A reload of a patched module is refused before execution by _refuse_reload.)"""
    if _is_patched(part, mod):
        return
    if STATE["fatal"] is not None:
        raise RsfixFatal(f"inkling_rsfix: an earlier patch attempt failed: {STATE['fatal']}")
    try:
        tables = apply(part, mod)
    except BaseException as exc:  # noqa: BLE001
        # A failed import leaves the module out of sys.modules and the hook stays armed;
        # the latch makes every retry fail the same way instead of loading stock code.
        STATE["fatal"] = repr(exc)
        if isinstance(exc, RsfixFatal):
            raise
        raise RsfixFatal(f"inkling_rsfix: patching {part} failed: {exc!r}") from exc
    STATE["stock"][part] = tables["stock"]
    if part not in STATE["applied"]:
        STATE["applied"].append(part)


def install(env=None, root=None):
    """Call once at interpreter start-up (sitecustomize), before sglang is imported and
    before other threads run imports: a first import that is already resolving its loader
    on another thread when the hooks are registered would not be wrapped."""
    if STATE["installed"]:
        return bool(STATE["parts"])
    chosen = parts(env)
    if chosen:
        check_sources(root, chosen)
        targets = _targets(chosen)
        loaded = [_sources.MODULES[t][0] for t in targets if _sources.MODULES[t][0] in sys.modules]
        if loaded:
            # A cached stock module would stay in service if a late patch failed.
            raise RuntimeError(f"inkling_rsfix: install before these modules are imported: {loaded}")
        for target in targets:
            _hook.when_imported(_sources.MODULES[target][0], functools.partial(_on_import, target))
            _hook.before_exec(_sources.MODULES[target][0], functools.partial(_refuse_reload, target))
    STATE["parts"] = chosen
    STATE["installed"] = True
    if chosen:
        _log(f"enabled: parts={','.join(chosen)} (final-token lane rule: in range, positive mass, "
             "last-positive fallback; Philox fold; blockwise log ratio; selector walk lane rule)")
    return bool(chosen)
