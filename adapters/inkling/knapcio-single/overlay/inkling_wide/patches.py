"""The SGLang patch points of inkling_wide (README.md has the list and the exactness argument).

  unquant        UnquantizedLinearMethod: a per-layer subclass (of whatever method class the layer has then,
                 e.g. inkling_pico12's) whose apply / apply_into send calls of MIN_M..MAX_M rows of a registered
                 layer to the wide GEMM; sm12x_fp32_lm_head(_applies): the FP32 LM head at MIN_M..MAX_M rows
  inkling2_attn  Inkling2Attention._project_sources: the stacked q/k/v case (three distinct [M, K] sources)
  inkling2       Inkling2ForConditionalGeneration.post_load_weights: register the weights after every (re)load
                 (after inkling_pico12 coded them, when it is armed: install inkling_wide after it)

Every patch falls back to the function it replaced for anything it does not take.
"""

import copy
import functools
import importlib
import sys
import time

import torch

from . import build, ops

UNQUANT = "sglang.srt.layers.quantization.unquant"
ATTN = "sglang.srt.models.inkling_common.inkling2_attn"
MODEL = "sglang.srt.models.inkling2"

STATE = {"levers": (), "cfg": None, "log": None, "patched": []}
HEADS = {}  # weight data_ptr -> Wide of the BF16 LM head
EXIT = None  # tests replace it; default os._exit


def _log(msg):
    (STATE["log"] or (lambda m: sys.stderr.write(m + "\n")))(msg)


def _exit(code):
    import os

    (EXIT or os._exit)(code)


def _need(owner, names, where):
    missing = [n for n in names if getattr(owner, n, None) is None]
    if missing:
        raise ops.WideFatal(f"inkling_wide: patch target {where} lacks {missing}")


def _guarded(fn):
    """Run load-time work; any failure ends the process with code 78 (a half-armed overlay must not serve)."""

    @functools.wraps(fn)
    def run(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except (KeyboardInterrupt, SystemExit):
            raise
        except BaseException as exc:  # noqa: BLE001
            _log(f"FATAL in {fn.__name__}: {exc!r}")
            sys.stderr.flush()
            _exit(78)
            raise

    return run


def _ext():
    if ops.EXT is None:
        cfg = STATE["cfg"]
        ext = build.load(cfg["cache_dir"], verbose=cfg["verbose"])
        ext.prepare(torch.cuda.current_device())
        ops.EXT = ext
    return ops.EXT


_MARKERS = ("inkling_cfg", "inkling_pico12", "inkling_prefill_qkvr_split")


def _forward_markers(new, inner):
    """Keep the other _project_sources overlays' markers visible through our wrapper (as inkling_prefill does), so
    that none of them wraps a second time."""
    for attr in _MARKERS:
        if hasattr(inner, attr):
            setattr(new, attr, getattr(inner, attr))


# ---- quant method subclass ----------------------------------------------------------------------------

_METHOD_CLASSES = {}


def method_class(base):
    """Subclass of a linear method class whose apply / apply_into run the wide GEMM for a registered layer."""
    if getattr(base, "_wide_class", False):
        return base
    sub = _METHOD_CLASSES.get(base)
    if sub is not None:
        return sub
    base_apply = base.apply
    base_apply_into = getattr(base, "apply_into", None)

    def apply(self, layer, x, bias=None):
        entry = layer.__dict__.get("_wide")
        if entry is not None and bias is None and ops.eligible(x, entry) and ops.weight_current(layer.weight, entry):
            return ops.linear(x, entry, x.dtype)
        return base_apply(self, layer, x, bias)

    ns = {"apply": apply, "_wide_class": True, "__module__": __name__}
    if base_apply_into is not None:

        def apply_into(self, layer, x, output, bias=None):
            entry = layer.__dict__.get("_wide")
            if (
                entry is not None
                and bias is None
                and x.dim() == 2
                and ops.eligible(x, entry)
                and ops.out_ok(output, x.shape[0], entry, torch.bfloat16, x.device)
                and ops.weight_current(layer.weight, entry)
            ):
                ops.gemm(ops.as_rows(x, entry.k), entry, output, x.shape[0])
                return output
            return base_apply_into(self, layer, x, output, bias=bias)

        ns["apply_into"] = apply_into
    sub = type("Wide" + base.__name__, (base,), ns)
    _METHOD_CLASSES[base] = sub
    return sub


def _unquantized_class():
    mod = sys.modules.get(UNQUANT) or importlib.import_module(UNQUANT)
    cls = getattr(mod, "UnquantizedLinearMethod", None)
    if cls is None:
        raise ops.WideFatal(f"inkling_wide: {UNQUANT}.UnquantizedLinearMethod is missing")
    return cls


def linear_reason(layer):
    w = getattr(layer, "weight", None)
    if not isinstance(w, torch.Tensor) or w.dim() != 2 or w.dtype != torch.bfloat16:
        return "weight is not a BF16 matrix"
    if getattr(layer, "bias", None) is not None:
        return "has a bias"
    method = getattr(layer, "quant_method", None)
    if method is None or not isinstance(method, _unquantized_class()):
        return f"quant method {type(method).__name__}"
    if not w.is_contiguous():
        return "weight not contiguous"
    return ops.shape_reason(*w.shape)


def _pico_wants(layer):
    """True if inkling_pico12 is armed for the target's dense layers (its pack must then exist)."""
    pico = sys.modules.get("inkling_pico12")
    return pico is not None and "dense" in tuple(getattr(pico, "STATE", {}).get("levers", ()))


def register_linear(layer, label, kind):
    """Register layer.weight for the wide GEMM and give the layer the wide method. Returns (entry, reason)."""
    why = linear_reason(layer)
    if why is not None:
        layer.__dict__.pop("_wide", None)
        return None, why
    pack = layer.__dict__.get("_pico12")
    if pack is None and ops.SET.fmt == "auto" and _pico_wants(layer):
        raise ops.WideFatal(
            f"inkling_wide: {label} has no inkling_pico12 pack although inkling_pico12 dense is armed: install "
            "inkling_wide after inkling_pico12 (its post_load_weights wrapper must run after pico's)"
        )
    entry = ops.make(layer.weight, label, kind, pack)
    layer.__dict__["_wide"] = entry
    method = layer.quant_method
    if not getattr(type(method), "_wide_class", False):
        new = copy.copy(method)
        new.__class__ = method_class(type(method))
        for attr in ("apply", "apply_into"):  # instance-level wrappers would shadow ours
            new.__dict__.pop(attr, None)
        layer.quant_method = new
    return entry, None


# ---- unquant: LM head ---------------------------------------------------------------------------------


def make_lm_head(inner_fn, inner_applies):
    def entry_for(x, weight, out):
        if not isinstance(weight, torch.Tensor) or weight.dim() != 2:
            return None
        entry = HEADS.get(weight.data_ptr())
        if entry is None or tuple(weight.shape) != (entry.n, entry.k) or not ops.eligible(x, entry):
            return None
        if out is not None and not ops.out_ok(out, ops.rows_of(x), entry, torch.float32, x.device):
            return None
        if not ops.weight_current(weight, entry):
            return None
        return entry

    def sm12x_fp32_lm_head_applies(x, weight, out):
        return entry_for(x, weight, out) is not None or inner_applies(x, weight, out)

    def sm12x_fp32_lm_head(x, weight, out):
        entry = entry_for(x, weight, out)
        if entry is None:
            return inner_fn(x, weight, out)
        return ops.linear(x, entry, torch.float32, out=out)

    sm12x_fp32_lm_head.__wrapped__ = inner_fn
    sm12x_fp32_lm_head_applies.__wrapped__ = inner_applies
    sm12x_fp32_lm_head._wide = sm12x_fp32_lm_head_applies._wide = True
    return sm12x_fp32_lm_head, sm12x_fp32_lm_head_applies


def patch_unquant(mod):
    levers = STATE["levers"]
    _need(mod, ["UnquantizedLinearMethod", "is_batch_invariant_mode_enabled"], UNQUANT)
    _need(mod.UnquantizedLinearMethod, ["apply", "apply_into"], f"{UNQUANT}.UnquantizedLinearMethod")
    ops._batch_invariant = mod.is_batch_invariant_mode_enabled
    if "lmhead" in levers:
        _need(mod, ["sm12x_fp32_lm_head", "sm12x_fp32_lm_head_applies"], UNQUANT)
        if not getattr(mod.sm12x_fp32_lm_head, "_wide", False):
            fn, applies = make_lm_head(mod.sm12x_fp32_lm_head, mod.sm12x_fp32_lm_head_applies)
            mod.sm12x_fp32_lm_head, mod.sm12x_fp32_lm_head_applies = fn, applies
            STATE["patched"].append("unquant.sm12x_fp32_lm_head")


# ---- inkling2_attn: stacked qkvr ------------------------------------------------------------------------


def make_project_sources(inner):
    def _project_sources(self, sources):
        plan = self.__dict__.get("_wide_qkvr")
        if plan is not None:
            entry, regions = plan
            q, k, v = sources.q, sources.k, sources.v
            if (
                len({id(q), id(k), id(v)}) == 3
                and ops.stacked_eligible((q, k, v), entry, regions)
                and ops.weight_current(self.qkvr.weight, entry)
            ):
                out = ops.stacked_project(q, k, v, entry, regions)
                return ops.split_like_stock(out, self.split_sizes)
        return inner(self, sources)

    _project_sources.__wrapped__ = inner
    _project_sources.inkling_wide = True
    _forward_markers(_project_sources, inner)
    return _project_sources


def patch_attn(mod):
    _need(mod, ["Inkling2Attention", "AttentionSources"], ATTN)
    cls = mod.Inkling2Attention
    _need(cls, ["_project_sources", "_project_qkvr"], f"{ATTN}.Inkling2Attention")
    if getattr(cls._project_sources, "inkling_wide", False):
        return
    cls._project_sources = make_project_sources(cls._project_sources)
    STATE["patched"].append("Inkling2Attention._project_sources")


# ---- inkling2: register the target's weights after every load -------------------------------------------


def _mb(n):
    return f"{n / 1e6:.1f} MB"


def _settings_text():
    s = ops.SET
    caps = ",".join(f"{k}={v}" for k, v in s.max_m_kind.items())
    return (f"min_m={s.min_m} max_m={s.max_m} ({caps}) scope={s.scope} fmt={s.fmt} pdl={int(s.pdl)} mg={s.mg} "
            f"pfd={s.pfd} "
            f"splitk={s.splitk or 'auto'}")


def prepare_target(model):
    levers = STATE["levers"]
    t0 = time.monotonic()
    _ext()
    t_ext = time.monotonic() - t0
    entries, parts, skipped = [], [], {}
    device = None
    if "dense" in levers:
        attn_cls = getattr(sys.modules.get(ATTN), "Inkling2Attention", None)
        if attn_cls is None:
            raise ops.WideFatal(f"inkling_wide: {ATTN} not imported at weight load")
        n_q = n_wo = n_st = n_x = 0
        for name, mod in model.named_modules():
            if not isinstance(mod, attn_cls):
                continue
            e, why = register_linear(mod.qkvr, f"{name}.qkvr", "qkvr")
            if e is None:
                skipped[f"{name}.qkvr"] = why
                mod.__dict__.pop("_wide_qkvr", None)
            else:
                n_q += 1
                n_x += e.fmt == ops.BF16X
                entries.append(e)
                device = e.weight.device
                regions = ops.regions_of(mod.split_sizes)
                if regions is not None:
                    mod.__dict__["_wide_qkvr"] = (e, regions)
                    n_st += 1
                else:
                    mod.__dict__.pop("_wide_qkvr", None)
            e, why = register_linear(mod.wo_ud, f"{name}.wo_ud", "wo")
            if e is None:
                skipped[f"{name}.wo_ud"] = why
            else:
                n_wo += 1
                n_x += e.fmt == ops.BF16X
                entries.append(e)
                device = e.weight.device
        if n_q + n_wo == 0:
            raise ops.WideFatal("inkling_wide: dense is on but no Inkling2Attention qkvr / wo_ud was registered")
        parts.append(f"dense: qkvr {n_q} (stacked {n_st}) wo_ud {n_wo}, bf16x {n_x}/{n_q + n_wo}")
    if "lmhead" in levers:
        lm_head = getattr(getattr(model, "llm", None), "lm_head", None)
        w = getattr(lm_head, "weight", None)
        if not isinstance(w, torch.Tensor) or w.dim() != 2 or w.dtype != torch.bfloat16:
            raise ops.WideFatal(f"inkling_wide: lm head weight is {type(w).__name__} {getattr(w, 'dtype', '')}, "
                                "not a BF16 matrix")
        pico_heads = getattr(sys.modules.get("inkling_pico12.patches"), "HEADS", {}) or {}
        pack = pico_heads.get(w.data_ptr())
        pico = sys.modules.get("inkling_pico12")
        pico_head = pico is not None and {"lmhead", "drafthead"} & set(getattr(pico, "STATE", {}).get("levers", ()))
        if pack is None and ops.SET.fmt == "auto" and pico_head:
            raise ops.WideFatal("inkling_wide: the LM head has no inkling_pico12 pack although inkling_pico12 "
                                "lmhead is armed: install inkling_wide after inkling_pico12")
        e = ops.make(w, "lm_head", "head", pack)
        HEADS.clear()
        HEADS[e.wptr] = e
        entries.append(e)
        device = e.weight.device
        parts.append(f"head [{e.n}x{e.k}] {'bf16x' if e.fmt == ops.BF16X else 'raw'}")
    if device is not None and device.type == "cuda":
        ops.SET.sms = torch.cuda.get_device_properties(device).multi_processor_count
    ws_bytes = ops.ensure_workspace(entries, ops.SET.min_m, ops.SET.max_m, device) if device is not None else 0
    skip = ""
    if skipped:
        first = "; ".join(f"{k}: {v}" for k, v in list(skipped.items())[:3])
        skip = f"; skipped {len(skipped)} ({first}{' ...' if len(skipped) > 3 else ''})"
    info = build.INFO
    how = f"ext {'reused' if info.get('cached') else 'built'} in {t_ext:.1f} s ({info.get('build_dir')})" if info \
        else "ext stand-in"
    _log(
        f"active (target): {', '.join(parts)}; split-K workspace {_mb(ws_bytes)}{skip}; patched "
        f"{', '.join(STATE['patched']) or 'nothing'}; {how}; {_settings_text()}"
    )


def is_in_scope(forward_batch):
    """Decode or target-verify forward (the forward modes wide is built for)."""
    mode = getattr(forward_batch, "forward_mode", None)
    if mode is None:
        return False
    try:
        return bool(mode.is_decode() or mode.is_target_verify())
    except AttributeError:
        return False


def make_forward(inner):
    """Mark the target forward's scope for ops.in_range (restored afterwards, also on error)."""

    @functools.wraps(inner)
    def forward(self, input_ids, positions, forward_batch, *args, **kwargs):
        prev = ops.IN_SCOPE[0]
        ops.IN_SCOPE[0] = is_in_scope(forward_batch)
        try:
            return inner(self, input_ids, positions, forward_batch, *args, **kwargs)
        finally:
            ops.IN_SCOPE[0] = prev

    forward._wide = True
    return forward


def patch_model(mod):
    _need(mod, ["Inkling2ForConditionalGeneration"], MODEL)
    cls = mod.Inkling2ForConditionalGeneration
    _need(cls, ["post_load_weights", "forward"], f"{MODEL}.Inkling2ForConditionalGeneration")
    if not getattr(cls.forward, "_wide", False):
        cls.forward = make_forward(cls.forward)
        STATE["patched"].append("Inkling2ForConditionalGeneration.forward (scope)")
    orig = cls.post_load_weights
    if getattr(orig, "_wide", False):
        return

    @functools.wraps(orig)
    def post_load_weights(self, *args, **kwargs):
        result = orig(self, *args, **kwargs)
        _guarded(prepare_target)(self)
        return result

    post_load_weights._wide = True
    cls.post_load_weights = post_load_weights
    STATE["patched"].append("Inkling2ForConditionalGeneration.post_load_weights")
