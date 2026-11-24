"""The SGLang patch points of inkling_pico12 (see README.md for the full list and the exactness argument).

  unquant      UnquantizedLinearMethod: a per-layer subclass whose apply / apply_into send eligible calls of a
               packed layer to pico; sm12x_fp32_lm_head(_applies): the FP32 LM head (logits_processor imports
               both from this module at call time, so replacing the module attributes reaches it)
  inkling2_attn  Inkling2Attention._project_sources: the stacked q/k/v case (three distinct [M, K] sources)
  inkling2     Inkling2ForConditionalGeneration.post_load_weights: code the weights after every (re)load
  inkling2_mtp Inkling2ForConditionalGenerationMTP.load_weights (draft linears), project_through_lm_head
               (the draft's BF16 projection through the target head)

Every patch falls back to the function it replaced for anything it does not take.
"""

import copy
import functools
import importlib
import os
import sys
import time

import torch

from . import build, ops

UNQUANT = "sglang.srt.layers.quantization.unquant"
ATTN = "sglang.srt.models.inkling_common.inkling2_attn"
MODEL = "sglang.srt.models.inkling2"
MTP = "sglang.srt.models.inkling2_mtp"
DSPARK = "sglang.srt.models.dspark"

STATE = {
    "levers": (),
    "cfg": None,
    "log": None,
    "patched": [],
    "qkvr_composed": None,
}
HEADS = {}  # weight data_ptr -> Pack of a BF16 LM head
EXIT = os._exit  # tests replace it


def _log(msg):
    (STATE["log"] or (lambda m: sys.stderr.write(m + "\n")))(msg)


def _need(owner, names, where):
    missing = [n for n in names if getattr(owner, n, None) is None]
    if missing:
        raise ops.Pico12Fatal(f"inkling_pico12: patch target {where} lacks {missing}")


def _guarded(fn):
    """Run our own load-time work; any failure ends the process with code 78 (a half-armed overlay must not serve)."""

    @functools.wraps(fn)
    def run(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except (KeyboardInterrupt, SystemExit):
            raise
        except BaseException as exc:  # noqa: BLE001
            _log(f"FATAL in {fn.__name__}: {exc!r}")
            sys.stderr.flush()
            EXIT(78)
            raise

    return run


def _ext():
    if ops.EXT is None:
        cfg = STATE["cfg"]
        ops.EXT = build.load(cfg["cache_dir"], cfg["std"], verbose=cfg["verbose"])
    return ops.EXT


def _range_ext():
    if ops.RANGE_EXT is None:
        cfg = STATE["cfg"]
        ops.RANGE_EXT = build.load_range(cfg["cache_dir"], cfg["std"], verbose=cfg["verbose"])
    return ops.RANGE_EXT


# ---- quant method subclass ----------------------------------------------------------------------------

_METHOD_CLASSES = {}


def method_class(base):
    """Subclass of an unquantized linear method class whose apply / apply_into run pico for a packed layer."""
    if getattr(base, "_pico12_class", False):
        return base
    sub = _METHOD_CLASSES.get(base)
    if sub is not None:
        return sub
    base_apply = base.apply
    base_apply_into = getattr(base, "apply_into", None)

    def apply(self, layer, x, bias=None):
        pack = layer.__dict__.get("_pico12")
        if (
            pack is not None
            and bias is None
            and ops.eligible(x, pack)
            and ops.weight_current(layer.weight, pack)
        ):
            return ops.linear(x, pack, x.dtype)
        return base_apply(self, layer, x, bias)

    ns = {"apply": apply, "_pico12_class": True, "__module__": __name__}
    if base_apply_into is not None:

        def apply_into(self, layer, x, output, bias=None):
            pack = layer.__dict__.get("_pico12")
            if (
                pack is not None
                and bias is None
                and x.dim() == 2
                and ops.eligible(x, pack)
                and ops.out_ok(output, x.shape[0], pack, torch.bfloat16, x.device)
                and ops.weight_current(layer.weight, pack)
            ):
                ops.gemv(ops.as_rows(x, pack.k), pack, output)
                return output
            return base_apply_into(self, layer, x, output, bias=bias)

        ns["apply_into"] = apply_into
    sub = type("Pico12" + base.__name__, (base,), ns)
    _METHOD_CLASSES[base] = sub
    return sub


def _unquantized_class():
    mod = sys.modules.get(UNQUANT) or importlib.import_module(UNQUANT)
    cls = getattr(mod, "UnquantizedLinearMethod", None)
    if cls is None:
        raise ops.Pico12Fatal(f"inkling_pico12: {UNQUANT}.UnquantizedLinearMethod is missing")
    return cls


def linear_reason(layer):
    """None if pico can take this linear layer, else why not."""
    w = getattr(layer, "weight", None)
    if not isinstance(w, torch.Tensor) or w.dim() != 2 or w.dtype != torch.bfloat16:
        return "weight is not a BF16 matrix"
    if getattr(layer, "bias", None) is not None:
        return "has a bias"
    method = getattr(layer, "quant_method", None)
    if method is None or not isinstance(method, _unquantized_class()):
        return f"quant method {type(method).__name__}"
    return ops.shape_reason(*w.shape)


def register_linear(layer, label):
    """Code layer.weight (in place on a reload) and give the layer the pico method. Returns (pack, reason)."""
    why = linear_reason(layer)
    if why is not None:
        return None, why
    pack = ops.encode(layer.weight, label, old=layer.__dict__.get("_pico12"))
    layer.__dict__["_pico12"] = pack
    method = layer.quant_method
    if not getattr(type(method), "_pico12_class", False):
        new = copy.copy(method)
        new.__class__ = method_class(type(method))
        for attr in ("apply", "apply_into"):  # instance-level wrappers (kernel API debug logging) would shadow ours
            new.__dict__.pop(attr, None)
        layer.quant_method = new
    return pack, None


# ---- unquant: method base + LM head ---------------------------------------------------------------------


def make_lm_head(stock_fn, stock_applies):
    def pack_for(x, weight, out):
        if not isinstance(weight, torch.Tensor) or weight.dim() != 2:
            return None
        pack = HEADS.get(weight.data_ptr())
        if pack is None or tuple(weight.shape) != (pack.n, pack.k) or not ops.eligible(x, pack):
            return None
        if out is not None and not ops.out_ok(out, ops.rows_of(x), pack, torch.float32, x.device):
            return None
        return pack

    def sm12x_fp32_lm_head_applies(x, weight, out):
        return stock_applies(x, weight, out) or pack_for(x, weight, out) is not None

    def sm12x_fp32_lm_head(x, weight, out):
        pack = pack_for(x, weight, out)
        if pack is None:
            return stock_fn(x, weight, out)
        return ops.linear(x, pack, torch.float32, out=out)

    sm12x_fp32_lm_head.__wrapped__ = stock_fn
    sm12x_fp32_lm_head_applies.__wrapped__ = stock_applies
    sm12x_fp32_lm_head._pico12 = sm12x_fp32_lm_head_applies._pico12 = True
    return sm12x_fp32_lm_head, sm12x_fp32_lm_head_applies


def patch_unquant(mod):
    levers = STATE["levers"]
    _need(mod, ["UnquantizedLinearMethod", "is_batch_invariant_mode_enabled"], UNQUANT)
    _need(mod.UnquantizedLinearMethod, ["apply", "apply_into"], f"{UNQUANT}.UnquantizedLinearMethod")
    ops._batch_invariant = mod.is_batch_invariant_mode_enabled
    if "lmhead" in levers:
        _need(mod, ["sm12x_fp32_lm_head", "sm12x_fp32_lm_head_applies"], UNQUANT)
        if not getattr(mod.sm12x_fp32_lm_head, "_pico12", False):
            fn, applies = make_lm_head(mod.sm12x_fp32_lm_head, mod.sm12x_fp32_lm_head_applies)
            mod.sm12x_fp32_lm_head, mod.sm12x_fp32_lm_head_applies = fn, applies
            STATE["patched"].append("unquant.sm12x_fp32_lm_head")


# ---- inkling2_attn: stacked qkvr -------------------------------------------------------------------------


def _qkvr_gemv_first(mod):
    """If inkling_kernels' qkvr_gemv lever is on and has not patched _project_sources yet, apply it now so that
    the pico wrapper ends up outermost whatever order the post-import hooks run in. Returns its name or None."""
    ik = sys.modules.get("inkling_kernels")
    if ik is None or "qkvr_gemv" not in tuple(getattr(ik, "STATE", {}).get("levers", ())):
        return None
    cls = mod.Inkling2Attention
    if getattr(cls._project_sources, "inkling_cfg", None) is None:
        from inkling_kernels import qkvr_gemv

        qkvr_gemv.patch(mod, ik.qkvr_config_from_env())
    return "qkvr_gemv"


def make_project_sources(inner, stacked_fn):
    def _project_sources(self, sources):
        plan = self.__dict__.get("_pico12_qkvr")
        if plan is not None:
            q, k, v = sources.q, sources.k, sources.v
            if (
                len({id(q), id(k), id(v)}) == 3
                and ops.stacked_eligible((q, k, v), plan)
                and ops.weight_current(self.qkvr.weight, plan.full)
            ):
                out = ops.stacked_project(stacked_fn(sources), q.shape[0], plan)
                return ops.split_like_stock(out, self.split_sizes)
        return inner(self, sources)

    _project_sources.__wrapped__ = inner
    _project_sources.inkling_pico12 = True
    # inkling_kernels.qkvr_gemv.patch() returns early when it sees its config: keep it visible through us
    _project_sources.inkling_cfg = getattr(inner, "inkling_cfg", None)
    return _project_sources


def patch_attn(mod):
    _need(mod, ["Inkling2Attention", "_stacked", "AttentionSources"], ATTN)
    cls = mod.Inkling2Attention
    _need(cls, ["_project_sources", "_project_qkvr"], f"{ATTN}.Inkling2Attention")
    if getattr(cls._project_sources, "inkling_pico12", False):
        return
    STATE["qkvr_composed"] = _qkvr_gemv_first(mod)
    cls._project_sources = make_project_sources(cls._project_sources, mod._stacked)
    STATE["patched"].append(
        "Inkling2Attention._project_sources"
        + (" (outside inkling_kernels.qkvr_gemv)" if STATE["qkvr_composed"] else "")
    )


# ---- inkling2: code the target's weights after every load -----------------------------------------------


def _mb(n):
    return f"{n / 1e6:.1f} MB"


def _settings_text():
    s = ops.SET
    text = f"max_m={s.max_m} stacked_max_m={s.stacked_max_m} wk={s.wk} pdl={int(s.pdl)}"
    return text + (f" stack1 (kv_mt={s.stack1_kv_mt} fuse={int(s.stack1_fuse)})" if s.stack1 else "")


def _ext_text(t_ext):
    info = build.INFO
    if not info:
        return "ext stand-in"
    how = "reused" if info.get("cached") else "built"
    return f"ext {how} in {t_ext:.1f} s ({info.get('build_dir')})"


def prepare_target(model):
    levers = STATE["levers"]
    t0 = time.monotonic()
    _ext()
    if ops.SET.stack1 and "dense" in levers and 3 * ops.SET.stacked_max_m > ops.MAX_ROWS:
        _range_ext()
    t_ext = time.monotonic() - t0
    total = raw = 0
    parts, skipped = [], {}
    if "dense" in levers:
        attn_cls = getattr(sys.modules.get(ATTN), "Inkling2Attention", None)
        if attn_cls is None:
            raise ops.Pico12Fatal(f"inkling_pico12: {ATTN} not imported at weight load")
        n_q = n_kv = n_wo = 0
        for name, mod in model.named_modules():
            if not isinstance(mod, attn_cls):
                continue
            full, why = register_linear(mod.qkvr, f"{name}.qkvr")
            if full is None:
                skipped[f"{name}.qkvr"] = why
                mod.__dict__.pop("_pico12_qkvr", None)
            else:
                n_q += 1
                total, raw = total + full.nbytes, raw + full.raw_bytes
                bounds = ops.kv_bounds(mod.split_sizes)
                old = mod.__dict__.get("_pico12_qkvr")
                kv = None
                if bounds is not None and ops.SET.stacked_max_m > 0 and ops.stacked_needs_kv(ops.SET.stacked_max_m):
                    kv = ops.encode(
                        mod.qkvr.weight[bounds[0] : bounds[1]],
                        f"{name}.qkvr[{bounds[0]}:{bounds[1]}]",
                        old=old.kv if old is not None else None,
                    )
                    n_kv += 1
                    total += kv.nbytes
                if ops.SET.stacked_max_m > 0:
                    lo, hi = bounds if bounds is not None else (0, 0)
                    mod.__dict__["_pico12_qkvr"] = ops.QkvrPlan(full, kv, lo, hi)
            pack, why = register_linear(mod.wo_ud, f"{name}.wo_ud")
            if pack is None:
                skipped[f"{name}.wo_ud"] = why
            else:
                n_wo += 1
                total, raw = total + pack.nbytes, raw + pack.raw_bytes
        if n_q + n_wo == 0:
            raise ops.Pico12Fatal("inkling_pico12: dense is on but no Inkling2Attention qkvr / wo_ud was packable")
        parts.append(f"dense: qkvr {n_q} (+kv {n_kv}) wo_ud {n_wo}")
    if "lmhead" in levers or "drafthead" in levers:
        lm_head = getattr(getattr(model, "llm", None), "lm_head", None)
        w = getattr(lm_head, "weight", None)
        if not isinstance(w, torch.Tensor) or w.dim() != 2 or w.dtype != torch.bfloat16:
            raise ops.Pico12Fatal(f"inkling_pico12: lm head weight is {type(w).__name__} {getattr(w, 'dtype', '')}, "
                                  "not a BF16 matrix")
        old = next(iter(HEADS.values()), None)
        if old is not None and (old.n, old.k) != tuple(w.shape):
            old = None
        pack = ops.encode(w, "lm_head", old=old)
        HEADS.clear()
        HEADS[pack.wptr] = pack
        total, raw = total + pack.nbytes, raw + pack.raw_bytes
        users = [x for x in ("lmhead", "drafthead") if x in levers]
        parts.append(f"head [{pack.n}x{pack.k}] for {'+'.join(users)}")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    skip = ""
    if skipped:
        first = "; ".join(f"{k}: {v}" for k, v in list(skipped.items())[:3])
        skip = f"; skipped {len(skipped)} ({first}{' ...' if len(skipped) > 3 else ''})"
    _log(
        f"active (target): {', '.join(parts)}; packed {_mb(total)} extra beside {_mb(raw)} of BF16 originals "
        f"(kept){skip}; patched {', '.join(STATE['patched']) or 'nothing'}; {_ext_text(t_ext)}; "
        f"encode {time.monotonic() - t0 - t_ext:.1f} s; {_settings_text()}"
    )
    STATE["target_bytes"] = total


def patch_model(mod):
    _need(mod, ["Inkling2ForConditionalGeneration"], MODEL)
    cls = mod.Inkling2ForConditionalGeneration
    _need(cls, ["post_load_weights", "load_weights"], f"{MODEL}.Inkling2ForConditionalGeneration")
    orig = cls.post_load_weights
    if getattr(orig, "_pico12", False):
        return

    @functools.wraps(orig)
    def post_load_weights(self, *args, **kwargs):
        result = orig(self, *args, **kwargs)
        _guarded(prepare_target)(self)
        return result

    post_load_weights._pico12 = True
    cls.post_load_weights = post_load_weights
    STATE["patched"].append("Inkling2ForConditionalGeneration.post_load_weights")


# ---- inkling2_mtp: draft linears + the draft's head projection ------------------------------------------


def prepare_draft(model):
    t0 = time.monotonic()
    _ext()
    layers = getattr(model, "layers", None)
    if layers is None:
        raise ops.Pico12Fatal("inkling_pico12: draft model has no .layers")
    n, total, skipped = 0, 0, {}
    for name, mod in layers.named_modules():
        if getattr(mod, "quant_method", None) is None or not isinstance(getattr(mod, "weight", None), torch.Tensor):
            continue
        pack, why = register_linear(mod, f"draft.layers.{name}")
        if pack is None:
            skipped[name] = why
        else:
            n, total = n + 1, total + pack.nbytes
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    skip = f"; skipped {len(skipped)} ({'; '.join(f'{k}: {v}' for k, v in list(skipped.items())[:3])})" if skipped else ""
    _log(f"active (draft): {n} linears packed {_mb(total)} extra{skip}; encode {time.monotonic() - t0:.1f} s; "
         f"{_settings_text()}")


_SPEC_UNSET = object()
_spec_dtype = [_SPEC_UNSET]


def _dspark_head_dtype():
    """--speculative-dspark-lm-head-dtype (its own FP8 / MXFP8 head path wins when set)."""
    if _spec_dtype[0] is _SPEC_UNSET:
        try:
            spec = importlib.import_module(DSPARK).get_spec()
            _spec_dtype[0] = getattr(spec, "speculative_dspark_lm_head_dtype", None)
        except Exception:  # noqa: BLE001  no runtime context (offline checks): not cached
            return None
    return _spec_dtype[0]


def make_project_through_lm_head(inner):
    def project_through_lm_head(hidden, lm_head):
        w = getattr(lm_head, "weight", None)
        if isinstance(w, torch.Tensor) and w.dim() == 2:
            pack = HEADS.get(w.data_ptr())
            if (
                pack is not None
                and tuple(w.shape) == (pack.n, pack.k)
                and getattr(lm_head, "_idb_head", None) is None  # inkling_draftbytes head8: fewer bytes, it wins
                and ops.eligible(hidden, pack)
                and _dspark_head_dtype() is None
            ):
                return ops.linear(hidden, pack, w.dtype)
        return inner(hidden, lm_head)

    project_through_lm_head.__wrapped__ = inner
    project_through_lm_head._pico12 = True
    return project_through_lm_head


def patch_mtp(mod):
    levers = STATE["levers"]
    _need(mod, ["Inkling2ForConditionalGenerationMTP"], MTP)
    if "draft" in levers:
        cls = mod.Inkling2ForConditionalGenerationMTP
        _need(cls, ["load_weights"], f"{MTP}.Inkling2ForConditionalGenerationMTP")
        orig = cls.load_weights
        if not getattr(orig, "_pico12", False):

            @functools.wraps(orig)
            def load_weights(self, *args, **kwargs):
                result = orig(self, *args, **kwargs)
                _guarded(prepare_draft)(self)
                return result

            load_weights._pico12 = True
            cls.load_weights = load_weights
            STATE["patched"].append("Inkling2ForConditionalGenerationMTP.load_weights")
    if "drafthead" in levers:
        _need(mod, ["project_through_lm_head"], MTP)
        if not getattr(mod.project_through_lm_head, "_pico12", False):
            mod.project_through_lm_head = make_project_through_lm_head(mod.project_through_lm_head)
            STATE["patched"].append("inkling2_mtp.project_through_lm_head")
