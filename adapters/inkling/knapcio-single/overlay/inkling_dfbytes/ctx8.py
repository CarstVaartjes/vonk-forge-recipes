"""ctx8: the DFlash2 drafter's context projections on FlashInfer's b12x MXFP8 GEMM (draft-only).

After every verify the DFLASH worker writes the committed rows' target features into the draft ring, eagerly:
  DFlashDraftModel.project_target_hidden          fc [2304, 9216] (BF16 nn.Linear), then hidden_norm
  FusedKVMaterializeHelper.materialize            the 5 layers' stacked K/V projection flat_kv_weight [10240, 2304]
                                                  (BF16 torch.mm into the helper's workspace), then the fused K
                                                  RMSNorm + RoPE kernel and one ring write per layer
With INKLING_DFLASH_CTX8=1 both GEMMs read MXFP8 copies of their weights (inkling_draftbytes.quant "mx": FP8 e4m3,
one E8M0 scale per 32 inputs, the mx8 format; scales swizzled once by flashinfer.block_scale_interleave) through
flashinfer.mxfp8_quantize(x) + mm_mxfp8(backend="b12x") (W8A8, as mx8w) for 1..INKLING_DFLASH_CTX8_MAX_M rows (default
256). Larger calls (long prefills) and inputs the quantizer cannot take (not a contiguous, 16-byte aligned 2-D CUDA BF16
[rows, K]) run the stock function unchanged. On the MXFP8 path everything after the GEMM is the stock code: the same
hidden_norm call, and in materialize the stock body line by line with only the torch.mm replaced (it writes into the
same workspace rows).

Draft-only: the K/V the drafter attends to change by the quantization of two projections; the target forward, the
verify and the accept are untouched. Greedy output is the target's argmax chain (up to the schedule-dependent near-ties
the project documents: a different acceptance changes verify widths and batch composition); sampled verification keeps
the target law: ctx8 changes the drafter's hidden states and so the values of q, but the accept still reads the q of
the proposal actually drawn together with the target's probabilities for the verified rows (which ctx8 does not
change, schedule-dependent rounding aside). Proposals and acceptance can change; they are measured.

The three stock functions this relies on (materialize is reproduced, __init__ builds the weight that is copied,
project_target_hidden is reproduced) are pinned by the sha256 of their source in the pinned image; any difference
refuses the overlay at start-up. Any other set-up problem (an unsupported fc or weight, a FlashInfer error while
building or warming the copies) leaves that projection on the stock path and logs why; a GEMM error in serving does the
same for the rest of the process; a CUDA error (sticky, not recoverable) is raised as DfbytesFatal. The copies are
derived weights: once a copy is built, the drafter and the target model both declare _derived_weight_cache_error, so
SGLang refuses every online weight update (each runner checks its own model before writing, so no runner is written
first; the stock flat_kv_weight is a build-once copy too). The worker's optional aux CUDA graphs
(SGLANG_OPT_DFLASH_AUX_GRAPH) are not supported: install() leaves ctx8 off when they are requested.
"""

import ast
import functools
import hashlib

import torch

from . import STATE, DfbytesFatal, _log

MARK = "_inkling_dfbytes_ctx8"
OFF_HINT = " (DFLASH_CTX8=0 keeps the stock BF16 context projections)"
UPDATE_ERROR = ("Online weight updates of the DFlash2 drafter are not supported with its MXFP8 context projections "
                "(inkling_dfbytes ctx8: derived weight copies); restart with DFLASH_CTX8=0 to use them.")

# module -> qualified name -> sha256 of its exact source lines (decorators included) in the pinned image
PINS = {
    "sglang.kernels.ops.speculative.fused_kv_materialize": {
        "FusedKVMaterializeHelper.__init__": "66ca4b9c6c9303cde57af1fd31472b4d37a12bba5d1537be370ed238d70860bd",
        "FusedKVMaterializeHelper.materialize": "68b5fc5e9b3487f4ff283ce5bcb1602d30937f566bebd4ed6af7b9ad62b6f3ef",
    },
    "sglang.srt.models.dflash": {
        "DFlashDraftModel.project_target_hidden": "7e5dd00e4f3e4f7b0f908f581fa4e75cdffcb6395ba8f2ea69672a60cea12e62",
    },
    "sglang.srt.speculative.dflash_worker_v2": {
        # wrapped: runs after the draft model is loaded, with the target worker attached; builds the fused helper
        "DFlashWorkerV2._init_fused_kv_helper": "fb3376f81d478b6113a78dab292ae249497c72e2458f3c5268a54af432c64f4e",
    },
}


# ------------------------------------------------------------------ source pins
def segment(text, qualname):
    """Exact source lines of a top-level function / class method (decorators included), or None when the name is
    missing or defined more than once."""
    node = ast.parse(text)
    for part in qualname.split("."):
        hits = [n for n in getattr(node, "body", [])
                if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and n.name == part]
        if len(hits) != 1:
            return None
        node = hits[0]
    start = min([node.lineno] + [d.lineno for d in node.decorator_list])
    lines = text.splitlines(keepends=True)
    return "".join(lines[start - 1: node.end_lineno])


def digest(text, qualname):
    seg = segment(text, qualname)
    return None if seg is None else hashlib.sha256(seg.encode("utf-8")).hexdigest()


def mismatches(text, pins):
    bad = []
    for name, want in pins.items():
        got = digest(text, name)
        if got is None:
            bad.append((name, "missing or defined twice"))
        elif got != want:
            bad.append((name, f"sha256 {got[:12]} != pinned {want[:12]}"))
    return bad


def check_module(mod):
    path = getattr(mod, "__file__", None)
    try:
        with open(path, encoding="utf-8") as fh:
            text = fh.read()
    except (OSError, TypeError) as exc:
        raise DfbytesFatal(f"inkling_dfbytes: ctx8 cannot read {mod.__name__} source{OFF_HINT}") from exc
    bad = mismatches(text, PINS[mod.__name__])
    if bad:
        raise DfbytesFatal(f"inkling_dfbytes: ctx8: {path} differs from the pinned image: {bad}{OFF_HINT}")


def _require(owner, names, where):
    missing = [n for n in names if not hasattr(owner, n)]
    if missing:
        raise DfbytesFatal(f"inkling_dfbytes: ctx8: {where} lacks {missing}{OFF_HINT}")


# ------------------------------------------------------------------ MXFP8 copies and the GEMM
class Ctx8Weight:
    """MXFP8 copy of a BF16 [N, K] weight: q FP8 [N, K] (the mx8 pack) and sf, its E8M0 scales swizzled for b12x."""

    def __init__(self, pack, sf):
        self.pack = pack
        self.sf = sf

    @property
    def q(self):
        return self.pack.q

    @property
    def n(self):
        return self.pack.n

    @property
    def k(self):
        return self.pack.k

    def nbytes(self):
        return self.pack.nbytes() + self.sf.numel()  # FP8 bytes + linear E8M0 scales (kept for refresh) + swizzled


def ineligible_weight(weight):
    """None if b12x mm_mxfp8 can serve this weight (BF16 2-D CUDA, N and K multiples of 128), else the reason."""
    if not isinstance(weight, torch.Tensor) or weight.dim() != 2:
        return "not a 2-D tensor"
    if weight.dtype != torch.bfloat16 or not weight.is_cuda:
        return f"{weight.dtype} on {weight.device} (needs CUDA BF16)"
    n, k = weight.shape
    if n % 128 or k % 128:
        return f"shape {tuple(weight.shape)} (N and K must be multiples of 128)"
    return None


def build(weight, old=None):
    """Quantize weight [N, K] BF16 to MXFP8 (inkling_draftbytes.quant "mx") and swizzle its scales. With ``old`` the
    existing tensors are refilled in place (their addresses stay valid for anything captured)."""
    from flashinfer import block_scale_interleave
    from inkling_draftbytes.quant import quantize

    with torch.no_grad():
        pack = quantize(weight.detach(), "mx", out=None if old is None else old.pack)
        sf = block_scale_interleave(pack.scale.contiguous())
        if old is not None and old.sf.shape == sf.shape and old.sf.dtype == sf.dtype and old.sf.device == sf.device:
            old.sf.copy_(sf)
            return old
    return Ctx8Weight(pack, sf)


def why_not(x, w):
    """None if the b12x path takes input x [rows, K] for w, else a short reason ('rows' when only the row count is
    outside 1..max_m)."""
    if not isinstance(x, torch.Tensor) or x.dim() != 2:
        return "not 2-D"
    if not x.is_cuda or x.dtype != torch.bfloat16:
        return f"{x.dtype} on {x.device}"
    if x.device != w.q.device:
        return "device"
    if x.shape[1] != w.k:
        return f"K {x.shape[1]} != {w.k}"
    if not x.is_contiguous() or x.data_ptr() % 16:
        return "not contiguous / 16-byte aligned"
    if not 0 < x.shape[0] <= STATE["ctx8_max_m"]:
        return "rows"
    return None


_FALLBACK_LOGGED = set()
_FIRST_LOGGED = set()


def _note(kind, x, reason):
    if reason is None:
        if kind not in _FIRST_LOGGED:
            _FIRST_LOGGED.add(kind)
            _log(f"ctx8: first {kind} call on b12x MXFP8 ({x.shape[0]} rows)")
        n = STATE["ctx8_fc_calls" if kind == "fc" else "ctx8_kv_calls"] + 1
        if n in (1000, 10000, 100000):  # prefill appends are few: these counts mean decode appends run on b12x
            _log(f"ctx8: {n} {kind} calls on b12x MXFP8 ({STATE.get('ctx8_disabled', 0)} fallbacks)")
        return
    if reason != "rows" and (kind, reason) not in _FALLBACK_LOGGED:
        _FALLBACK_LOGGED.add((kind, reason))
        _log(f"ctx8: {kind} call kept BF16: {reason} (logged once)")


def _plan_rows(w):
    """One row count per b12x tile plan over 1..max_m (the compiled-kernel cache is keyed by the plan, not by M), or
    every row count when the plan selector cannot be read."""
    rows = range(1, STATE["ctx8_max_m"] + 1)
    try:
        from flashinfer.gemm.kernels.dense_blockscaled_gemm_sm120_b12x import _select_default_dense_gemm_plan
        from flashinfer.utils import get_device_sm_count

        sms = get_device_sm_count(w.q.device)
        first = {}
        for m in rows:
            plan = _select_default_dense_gemm_plan(m, w.n, w.k, sms, is_mxfp8=True, expected_m=m)
            first.setdefault((tuple(plan.mma_tiler_mn), bool(plan.swap_ab)), m)
        return sorted(first.values())
    except Exception:  # noqa: BLE001 - a different FlashInfer: warm every row count
        return list(rows)


def warmup(w):
    """Run the b12x path at load time once per tile plan over 1..max_m (the production call shapes: out=None and a
    workspace slice), so no b12x kernel compiles while serving: these two GEMMs run eagerly, outside the draft graphs
    whose capture compiles mx8w's plans. Returns (seconds, row counts run)."""
    import time

    t0 = time.time()
    rows = _plan_rows(w)
    with torch.no_grad():
        x = torch.zeros((STATE["ctx8_max_m"], w.k), dtype=torch.bfloat16, device=w.q.device)
        out = torch.empty((STATE["ctx8_max_m"], w.n), dtype=torch.bfloat16, device=w.q.device)
        for m in rows:
            gemm(x[:m], w)
            gemm(x[:m], w, out=out[:m])
        torch.cuda.synchronize(w.q.device)
    return time.time() - t0, rows


# Only these exception types may fall back to BF16: what FlashInfer / CuTe DSL raise for an unsupported shape, tactic
# or missing piece. Everything else (RuntimeError, which carries CUDA / driver errors, torch.AcceleratorError, OOM, ...)
# is treated as fatal: a CUDA fault can leave the context unusable, so it must not be retried on another kernel.
_RECOVERABLE = (ValueError, NotImplementedError, TypeError, ImportError, AttributeError, KeyError, AssertionError)
_CUDA_MARKERS = ("CUDA", "cuda", "cudaError", "CUBLAS", "illegal", "misaligned", "launch", "device-side assert",
                 "out of memory")


def fatal(exc):
    """True unless exc (and every exception chained to it) is a recoverable type whose text names no CUDA fault."""
    seen = set()
    e = exc
    while e is not None and id(e) not in seen:
        seen.add(id(e))
        if not isinstance(e, _RECOVERABLE):
            return True
        text = str(e)
        if any(m in text for m in _CUDA_MARKERS):
            return True
        e = e.__cause__ or e.__context__
    return False


def _recoverable_or_raise(kind, exc):
    """Return if exc may fall back to the stock path; else raise (DfbytesFatal passes the worker's except Exception)."""
    if fatal(exc):
        raise DfbytesFatal(f"inkling_dfbytes: ctx8: {kind}: not a recoverable kernel error "
                           f"({type(exc).__name__}: {str(exc)[:300]}){OFF_HINT}") from exc
    if _capturing():
        raise exc


def _capturing():
    return torch.cuda.is_available() and torch.cuda.is_current_stream_capturing()


def _disable(owner, kind, exc):
    """A b12x error in serving: this projection goes back to the stock path for the rest of the process."""
    owner.__dict__.pop("_idf_c8", None)
    STATE["ctx8_disabled"] = STATE.get("ctx8_disabled", 0) + 1
    _log(f"ctx8: {kind} b12x call failed ({type(exc).__name__}: {str(exc)[:200]}); stock BF16 from now on")


def _declare_derived(module):
    if isinstance(module, torch.nn.Module) and getattr(module, "_derived_weight_cache_error", None) is None:
        module._derived_weight_cache_error = UPDATE_ERROR


def gemm(x2, w, out=None):
    """[M, K] BF16 -> [M, N] BF16: flashinfer.mxfp8_quantize + mm_mxfp8(backend="b12x"), default tile plan (no
    autotuning, no tuned-tactic lookup: the default plan per shape)."""
    from flashinfer import mm_mxfp8, mxfp8_quantize
    from flashinfer.autotuner import autotune

    xq, xs = mxfp8_quantize(x2, is_sf_swizzled_layout=True)
    # the default tactic only (no autotuner cache lookup), as the pinned image's fixed-tactic MXFP8 path does: the
    # load-time warm-up then covers every kernel these calls can use
    with autotune(False, skip_ops={"mxfp8_gemm"}):
        return mm_mxfp8(xq, w.q.t(), xs, w.sf, out=out, out_dtype=torch.bfloat16, backend="b12x")


# ------------------------------------------------------------------ fc (sglang.srt.models.dflash)
def prepare_fc(model):
    """MXFP8 copy of the context fc (after load_weights), or None (stock fc, reason logged)."""
    fc = getattr(model, "fc", None)
    if getattr(model, "quantizable", False) or getattr(model, "fc_tp_columns", None) is not None:
        reason = "a quantized or TP-split fc"
    elif not isinstance(fc, torch.nn.Linear) or fc.bias is not None:
        reason = f"the fc is {type(fc).__name__} (needs an nn.Linear without bias)"
    else:
        reason = ineligible_weight(fc.weight)
    if reason:
        if fc is not None:
            fc.__dict__.pop("_idf_c8", None)
        _log(f"ctx8: context fc stays BF16: {reason}")
        return None
    old = fc.__dict__.get("_idf_c8")
    try:
        w = build(fc.weight, old)
        secs, warm_rows = warmup(w) if old is None else (0.0, [])
    except Exception as exc:  # noqa: BLE001 - keep the stock fc
        _recoverable_or_raise("context fc set-up", exc)
        fc.__dict__.pop("_idf_c8", None)
        _log(f"ctx8: context fc stays BF16: set-up failed ({type(exc).__name__}: {str(exc)[:200]})")
        return None
    fc.__dict__["_idf_c8"] = w
    _declare_derived(model)
    STATE["ctx8_fc"] += 1
    _log(f"ctx8: context fc {tuple(fc.weight.shape)} -> {w.nbytes() / 1e6:.1f} MB MXFP8"
         + (" (refreshed in place)" if old is not None
            else f", {len(warm_rows)} tile plans over rows 1..{STATE['ctx8_max_m']} warmed in {secs:.1f} s"))
    return w


def patch_dflash(mod):
    _require(mod, ["DFlashDraftModel"], mod.__name__)
    base = mod.DFlashDraftModel
    _require(base, ["load_weights", "project_target_hidden"], "DFlashDraftModel")
    if getattr(base.project_target_hidden, MARK, False):
        return False
    check_module(mod)
    orig_load = base.load_weights
    orig_project = base.project_target_hidden

    @functools.wraps(orig_load)
    def load_weights(self, weights):
        out = orig_load(self, weights)
        prepare_fc(self)
        return out

    @functools.wraps(orig_project)
    def project_target_hidden(self, target_hidden):
        fc = getattr(self, "fc", None)
        w = None if fc is None else fc.__dict__.get("_idf_c8")
        if w is None or getattr(self, "quantizable", False) or getattr(self, "fc_tp_columns", None) is not None:
            return orig_project(self, target_hidden)
        reason = why_not(target_hidden, w)
        if reason is not None:
            _note("fc", target_hidden, reason)
            return orig_project(self, target_hidden)
        try:
            projected = gemm(target_hidden, w)
        except Exception as exc:  # noqa: BLE001
            _recoverable_or_raise("fc", exc)
            _disable(fc, "fc", exc)
            return orig_project(self, target_hidden)
        _note("fc", target_hidden, None)
        STATE["ctx8_fc_calls"] += 1
        # stock: projected = self.fc(target_hidden); return self.hidden_norm(projected)
        return self.hidden_norm(projected)

    setattr(load_weights, MARK, True)
    setattr(project_target_hidden, MARK, True)
    base.load_weights = load_weights
    base.project_target_hidden = project_target_hidden
    return True


# ------------------------------------------------------------------ stacked K/V (fused_kv_materialize)
def prepare_kv(helper, layers=()):
    """MXFP8 copy of the helper's stacked K/V weight (after the stock __init__), or None (stock, reason logged)."""
    reason = ineligible_weight(getattr(helper, "flat_kv_weight", None))
    if reason:
        _log(f"ctx8: stacked context K/V stays BF16: {reason}")
        return None
    try:
        w = build(helper.flat_kv_weight)
        secs, warm_rows = warmup(w)
    except Exception as exc:  # noqa: BLE001 - keep the stock projection (and the stock fused helper)
        _recoverable_or_raise("stacked context K/V set-up", exc)
        _log(f"ctx8: stacked context K/V stays BF16: set-up failed ({type(exc).__name__}: {str(exc)[:200]})")
        return None
    helper.__dict__["_idf_c8"] = w
    for layer in list(layers)[:1]:  # a drafter module: the updater scans model.modules()
        _declare_derived(layer)
    STATE["ctx8_kv"] += 1
    _log(f"ctx8: stacked context K/V {tuple(helper.flat_kv_weight.shape)} ({helper.n_layers} layers) -> "
         f"{w.nbytes() / 1e6:.1f} MB MXFP8, {len(warm_rows)} tile plans over rows 1..{STATE['ctx8_max_m']} warmed in "
         f"{secs:.1f} s")
    return w


class _Fallback(Exception):
    pass


def materialize_c8(mod, self, w, ctx_hidden, positions, write_layer_kv):
    """FusedKVMaterializeHelper.materialize of the pinned image, line by line, with the projection GEMM on b12x.
    The caller checked: 1 <= rows <= max_m, ctx_hidden a contiguous aligned CUDA BF16 [rows, K] on the weight's device
    and flat_kv_weight BF16, so the stock device / dtype conversions of ctx_hidden are no-ops."""
    total_ctx = ctx_hidden.shape[0]
    if total_ctx == 0:
        return

    if positions.ndim != 1:
        positions = positions.reshape(-1)
    if positions.numel() != total_ctx:
        raise ValueError(
            "positions must match ctx_hidden token count for fused KV materialization: "
            f"positions={positions.numel()}, total_ctx={total_ctx}."
        )

    if ctx_hidden.device != self.device:
        ctx_hidden = ctx_hidden.to(self.device, non_blocking=True)
    if ctx_hidden.dtype != self.flat_kv_weight.dtype:
        ctx_hidden = ctx_hidden.to(self.flat_kv_weight.dtype)
    if positions.device != self.device:
        positions = positions.to(
            device=self.device, dtype=torch.int64, non_blocking=True
        )
    elif positions.dtype != torch.int64:
        positions = positions.to(torch.int64)

    max_position = (
        self.max_position_hint
        if self.max_position_hint is not None
        else int(positions.max().item())
    )
    cos_sin_cache = self._ensure_rope_cache(max_position)

    self._ensure_workspace(total_ctx, ctx_hidden.dtype)
    assert self._proj_workspace is not None
    assert self._k_workspace is not None
    assert self._v_workspace is not None

    # stock: torch.mm(ctx_hidden, self.flat_kv_weight.t(), out=proj_out_2d) (or its fallbacks)
    try:
        proj_out_2d = gemm(ctx_hidden, w, out=self._proj_workspace[:total_ctx])
    except Exception as exc:  # noqa: BLE001 - nothing has been written but the workspace: rerun stock
        _recoverable_or_raise("K/V", exc)
        _disable(self, "K/V", exc)
        raise _Fallback() from exc

    proj_out = proj_out_2d.view(total_ctx, self.n_layers, self.layer_out_dim)
    tmp_k = self._k_workspace[:, :total_ctx]
    tmp_v = self._v_workspace[:, :total_ctx]
    cache_k, cache_v = mod._fused_norm_rope_stacked(
        proj_out,
        self.k_norm_weights,
        self.eps_values,
        cos_sin_cache,
        positions,
        self.num_kv_heads,
        self.head_dim,
        self.rotary_dim,
        k_out=tmp_k,
        v_out=tmp_v,
    )
    for layer_idx in range(self.n_layers):
        write_layer_kv(layer_idx, cache_k[layer_idx], cache_v[layer_idx])


def patch_fused_kv(mod):
    _require(mod, ["FusedKVMaterializeHelper", "_fused_norm_rope_stacked"], mod.__name__)
    cls = mod.FusedKVMaterializeHelper
    _require(cls, ["__init__", "materialize", "_ensure_rope_cache", "_ensure_workspace"], "FusedKVMaterializeHelper")
    if getattr(cls.materialize, MARK, False):
        return False
    check_module(mod)
    orig_init = cls.__init__
    orig_materialize = cls.materialize

    @functools.wraps(orig_init)
    def __init__(self, *args, **kwargs):
        orig_init(self, *args, **kwargs)
        layers = kwargs.get("layers", args[0] if args else ())
        prepare_kv(self, layers)

    @functools.wraps(orig_materialize)
    def materialize(self, ctx_hidden, positions, write_layer_kv):
        w = self.__dict__.get("_idf_c8")
        if w is None or getattr(self.flat_kv_weight, "dtype", None) != torch.bfloat16:
            return orig_materialize(self, ctx_hidden, positions, write_layer_kv)
        reason = why_not(ctx_hidden, w)
        if reason is not None:
            _note("K/V", ctx_hidden, reason)
            return orig_materialize(self, ctx_hidden, positions, write_layer_kv)
        try:
            materialize_c8(mod, self, w, ctx_hidden, positions, write_layer_kv)
        except _Fallback:
            return orig_materialize(self, ctx_hidden, positions, write_layer_kv)
        _note("K/V", ctx_hidden, None)
        STATE["ctx8_kv_calls"] += 1

    setattr(__init__, MARK, True)
    setattr(materialize, MARK, True)
    cls.__init__ = __init__
    cls.materialize = materialize
    return True


# ------------------------------------------------------------------ worker (dflash_worker_v2): refuse online updates
def mark_online_updates(worker):
    """Once a copy is built, the drafter and the target declare _derived_weight_cache_error: every SGLang online update
    path checks its runner's model before writing, so the update is refused before any runner is written."""
    draft = getattr(worker, "draft_model", None)
    helper = getattr(worker, "_fused_kv_helper", None)
    fc = getattr(draft, "fc", None)
    built = (fc is not None and "_idf_c8" in fc.__dict__) or (helper is not None and "_idf_c8" in helper.__dict__)
    if not built:
        return False
    target = getattr(getattr(getattr(worker, "target_worker", None), "model_runner", None), "model", None)
    marked = [m for m in (draft, target) if isinstance(m, torch.nn.Module)]
    for m in marked:
        _declare_derived(m)
    _log(f"ctx8: online weight updates refused while the MXFP8 context copies are in use ({len(marked)} models marked)")
    return True


def patch_worker(mod):
    _require(mod, ["DFlashWorkerV2"], mod.__name__)
    cls = mod.DFlashWorkerV2
    _require(cls, ["_init_fused_kv_helper"], "DFlashWorkerV2")
    if getattr(cls._init_fused_kv_helper, MARK, False):
        return False
    check_module(mod)
    orig = cls._init_fused_kv_helper

    @functools.wraps(orig)
    def _init_fused_kv_helper(self, *args, **kwargs):
        out = orig(self, *args, **kwargs)
        mark_online_updates(self)
        return out

    setattr(_init_fused_kv_helper, MARK, True)
    cls._init_fused_kv_helper = _init_fused_kv_helper
    return True
