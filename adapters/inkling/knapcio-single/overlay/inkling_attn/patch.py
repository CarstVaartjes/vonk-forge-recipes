"""Patch points of inkling_attn (prefill / extend attention with the relative bias computed in-kernel).

Stock prefill path per layer visit (inkling2_attn.py Inkling2Attention._forward, the non-compact branch):
  rel_bias = self.rel_logits_proj(r)            # cuBLAS einsum -> BF16 [t, 24, rel_extent]
  attn(q, k, v, forward_batch, return_lse=True, rel_bias=rel_bias)
    -> FlashAttentionBackend.forward_extend -> flash_attn_with_kvcache (SM120)
    -> flash_attn_varlen_func -> _flash_attn_varlen_func (FA4: shear copy + attention with the sheared bias)

With this overlay, inside an eligible visit (prefill/extend only, see _visit_for):
  1. Inkling2Attention._forward is wrapped to mark the visit (a context variable; nothing else changes);
  2. RelLogitsProj.forward returns r itself (the [t, H, 16] rel-clip output, no einsum) and records it in the
     visit: r travels to the backend in the rel_bias slot, so every narrowing SGLang applies to rel_bias
     (graph_utils.attention_input_scope / _attention_on_real_rows slice rows) applies to r the same way;
  3. flash_attention_v4_sm120._flash_attn_varlen_func (the module global the SM120 wrapper calls) is wrapped:
     a rel_bias that is (a row prefix of) the visit's r runs ops.prefill_attention with the layer's proj,
     returning FA4's (out, lse). Any call shape the kernel does not cover falls back to the stock call with
     the materialised einsum bias (same math as stock). An r-shaped rel_bias (width 16) that reaches FA4
     outside its visit raises instead of being misread.

Breakable prefill graphs replay attention as a deferred eager call outside the visit, so they are excluded at
the visit check (as inkling_prefill.bias_layout does). Decode, target verify and draft extend never enter a visit.

Coexistence with inkling_prefill.bias_layout: both mark visits and wrap FA4's varlen entry, and bias_layout
wraps RelLogitsProj.forward. This package dispatches one level earlier, at RelLogitsProj.__call__ (before
nn.Module.__call__ reaches any forward wrapper), so in an eligible visit r is returned whatever order the two
packages were installed in; bias_layout's forward wrapper is not reached, its varlen wrapper sees a width-16
rel_bias it did not make and passes it through, and layers this package does not take (e.g. global layers with
only prefill_swa armed) still get bias_layout's sheared bias. Should anything else consume an eligible visit
first, the _forward wrapper notices (the visit's r stays unset) and logs it once.

Device route: the visit requires a CUDA device of compute capability 12.x, the condition under which
FlashAttentionBackend._load_fa4_kernels uses the patched SM120 module, and no context-parallel extend.
"""

import contextvars
import os
import sys
import weakref

ATTN_MODULE = "sglang.srt.models.inkling_common.inkling2_attn"
COMMON_ATTN_MODULE = "sglang.srt.models.inkling_common.attn"
SM120_MODULE = "sglang.kernels.ops.attention.flash_attention_v4_sm120"
BREAKABLE_MODULE = "sglang.srt.model_executor.runner_backend_utils.breakable_cuda_graph.context"
CP_MODULE = "sglang.srt.layers.cp.utils"
MARK = "inkling_attn_prefill"
D_REL = 16
HEAD_DIM = 128

_CTX = contextvars.ContextVar("inkling_attn_ctx", default=None)
STATS = {"kernel": 0, "fallback": 0, "shadowed": 0, "gated": 0, "gate_fallback": 0}
# INKLING_ATTN_GATE=1: sliding-window prefill visits run the gated epilogue (gate_sink inside the attention kernel,
# bit-identical to SGLang's gate_sink on the FP32 out + LSE). The visit records the gate / sinks its
# _project_sources returns (an instance-level hook for the duration of the visit); the FA4 call then launches the
# gated kernel (BF16 y + LSE; the FP32 out stays unwritten) and the visit's gate_sink call returns y.
# INKLING_ATTN_GATE=all (or 2): global prefill visits too (BF16 and KV12S packed readers, via gate_begin / gate_commit).
_GATE_ENV = os.environ.get("INKLING_ATTN_GATE", "").strip().lower()
GATE_ALL = _GATE_ENV in ("all", "2")
GATE = GATE_ALL or _GATE_ENV in ("1", "on", "true", "yes", "swa")


def _kernel_api_logging():
    """SGLang's kernel API logging (SGLANG_KERNEL_API_LOGLEVEL > 0, parsed like kernel_api_logging._get_env) logs and
    may dump the FA4 wrapper's returned FP32 out before gate_sink runs; a gated launch leaves that buffer unwritten,
    so the gated epilogue stays off while it is enabled."""
    try:
        return int(os.environ.get("SGLANG_KERNEL_API_LOGLEVEL", "0") or 0) != 0
    except ValueError:
        return False


GATE_OFF_FOR_LOGGING = GATE and _kernel_api_logging()
if GATE_OFF_FOR_LOGGING:
    GATE = False
    GATE_ALL = False
_LOGGED = set()


def _log_once(key, msg):
    if key in _LOGGED:
        return
    _LOGGED.add(key)
    sys.stderr.write(f"[inkling_attn pid={os.getpid()}] {msg}\n")
    sys.stderr.flush()


if GATE_OFF_FOR_LOGGING:
    _log_once("gate_off", "gated epilogue OFF: SGLANG_KERNEL_API_LOGLEVEL is set (it reads the FA4 output buffer)")


class _Visit:
    __slots__ = ("owner", "local", "window_left", "r", "proj", "gate", "sinks", "tokens", "gated")

    def __init__(self, owner, local, window_left, proj):
        self.owner = owner
        self.local = local
        self.window_left = window_left
        self.proj = proj
        self.r = None
        self.gate = self.sinks = self.tokens = None  # from the visit's _project_sources (INKLING_ATTN_GATE)
        self.gated = None  # (FP32 out buffer left unwritten, BF16 y) of a gated launch, until gate_sink takes y


class use_visit:
    """Context manager for direct calls (tools / tests): inside it, the patched varlen wrapper treats `r`
    (or a row prefix of it) passed as rel_bias as this package's marker for a layer with `proj`."""

    def __init__(self, r, proj, window_left=None):
        local = window_left is not None and window_left >= 0
        self._visit = _Visit(None, local, window_left if local else None, proj)
        self._visit.r = r
        _PRODUCED.append(weakref.ref(r))
        del _PRODUCED[:-256]
        self._token = None

    def __enter__(self):
        self._token = _CTX.set(self._visit)
        return self._visit

    def __exit__(self, *exc):
        _CTX.reset(self._token)
        return False


_SM12X = {}


def _is_sm12x(device):
    """FlashAttentionBackend._load_fa4_kernels takes the SM120 FA4 module (the one patched here) iff major == 12."""
    import torch

    idx = device.index if device.index is not None else torch.cuda.current_device()
    if idx not in _SM12X:
        # GB10 only: the extension is built for sm_121a, and capability 12.x is what makes the backend use the
        # patched SM120 FA4 module
        _SM12X[idx] = tuple(torch.cuda.get_device_capability(idx)) == (12, 1)
    return _SM12X[idx]


def _visit_for(self, forward_batch, attn, get_attn_backend, get_model, in_breakable, kinds, cp_active=None):
    import torch

    local = bool(getattr(self, "is_local", False))
    if ("swa" if local else "global") not in kinds:
        return None
    if in_breakable is not None and in_breakable():
        return None
    if cp_active is not None and cp_active(forward_batch):
        return None
    mode = forward_batch.forward_mode
    if not mode.is_extend() or mode.is_decode() or mode.is_target_verify():
        return None
    for name in ("is_draft_extend", "is_draft_extend_v2"):
        fn = getattr(mode, name, None)
        if fn is not None and fn():
            return None
    if getattr(attn, "pack_gqa", None):
        return None
    try:
        if get_model().kv_cache_dtype == "mxfp8":
            return None
    except Exception:  # noqa: BLE001
        return None
    contains = getattr(forward_batch, "contains_mm_inputs", None)
    if contains is not None and contains():
        return None
    backend = get_attn_backend()
    if getattr(backend, "fa_skip_kv_cache", False) or getattr(backend, "use_mla", False):
        return None
    meta = getattr(backend, "forward_metadata", None)
    if meta is None:
        return None
    for name in ("swa_spec_metadata", "image_token_ranges", "local_attn_metadata"):
        if getattr(meta, name, None) is not None:
            return None
    if getattr(self, "head_dim", None) != HEAD_DIM or getattr(self, "d_rel", None) != D_REL:
        return None
    heads, kv_heads = self.num_tp_heads, self.num_tp_kv_heads
    if kv_heads <= 0 or heads % kv_heads:
        return None
    proj = getattr(getattr(self, "rel_logits_proj", None), "proj", None)
    rel = self.rel_extent
    if not (
        isinstance(proj, torch.Tensor)
        and proj.is_cuda
        and _is_sm12x(proj.device)
        and proj.dtype == torch.bfloat16
        and proj.dim() == 2
        and tuple(proj.shape) == (D_REL, rel)
        and proj.is_contiguous()
    ):
        return None
    window = getattr(attn, "sliding_window_size", None)
    if local:
        if window is None or window < 0 or window + 1 != rel:
            return None
    elif window not in (None, -1):
        return None
    return _Visit(self, local, window if local else None, proj)


def make_forward(prev, get_attn_backend, get_model, in_breakable, kinds, cp_active=None):
    def _forward(self, hidden_states, positions, forward_batch, *, attn, attn_padded):
        visit = _visit_for(self, forward_batch, attn, get_attn_backend, get_model, in_breakable, kinds, cp_active)
        if visit is None:
            if GATE and _CTX.get() is not None:  # an ineligible nested visit must not see the outer one
                token = _CTX.set(None)
                try:
                    return prev(self, hidden_states, positions, forward_batch, attn=attn, attn_padded=attn_padded)
                finally:
                    _CTX.reset(token)
            return prev(self, hidden_states, positions, forward_batch, attn=attn, attn_padded=attn_padded)
        token = _CTX.set(visit)
        hooked = False
        if GATE and (visit.local or GATE_ALL) and "_project_sources" not in self.__dict__:
            project = self._project_sources  # the full (class-level, wrapped) chain

            def _project_sources(sources):
                res = project(sources)
                if isinstance(res, tuple) and len(res) == 6:
                    visit.gate, visit.sinks = res[4], res[5]
                    visit.tokens = res[0].shape[0]
                return res

            self.__dict__["_project_sources"] = _project_sources  # instance attribute shadows the method
            hooked = True
        try:
            result = prev(self, hidden_states, positions, forward_batch, attn=attn, attn_padded=attn_padded)
        finally:
            if hooked:
                self.__dict__.pop("_project_sources", None)
            _CTX.reset(token)
        if visit.gated is not None:
            raise RuntimeError("inkling_attn: a gated attention output never reached gate_sink")
        if visit.r is None:
            STATS["shadowed"] += 1
            _log_once(
                "shadowed",
                "an eligible visit never reached RelLogitsProj.__call__ with an r this package takes "
                "(something else produced the bias); that visit ran without this package",
            )
        return result

    _forward.__wrapped__ = prev
    setattr(_forward, MARK, True)
    return _forward


_PRODUCED = []  # weak references to the last r markers handed out (bounded)


def make_rel_call(prev):
    """RelLogitsProj.__call__: inside an eligible visit return r itself (the marker) before nn.Module.__call__,
    i.e. before every RelLogitsProj.forward wrapper, whatever order the overlays were installed in."""
    import torch

    def __call__(self, *args, **kwargs):
        r_out = args[0] if args else kwargs.get("r_out")
        log_scaling_tau = args[1] if len(args) > 1 else kwargs.get("log_scaling_tau")
        visit = _CTX.get()
        if (
            visit is not None
            and visit.r is None
            and log_scaling_tau is None
            and visit.owner is not None
            and visit.owner.rel_logits_proj is self
            and self.proj is visit.proj
            and isinstance(r_out, torch.Tensor)
            and r_out.dim() == 3
            and r_out.is_cuda
            and r_out.dtype == torch.bfloat16
            and r_out.shape[1] == visit.owner.num_tp_heads
            and r_out.shape[2] == D_REL
            and r_out.stride(2) == 1
            and len(args) + len(kwargs) <= 2
        ):
            visit.r = r_out
            _PRODUCED.append(weakref.ref(r_out))
            del _PRODUCED[:-256]
            return r_out
        return prev(self, *args, **kwargs)

    __call__.__wrapped__ = prev
    setattr(__call__, MARK, True)
    return __call__


def _is_produced(bias):
    for ref in _PRODUCED:
        r = ref()
        if r is not None and r.device == bias.device and _same_storage_rows(bias, r):
            return True
    return False


def _same_storage_rows(bias, r):
    return (
        bias.dim() == 3
        and bias.data_ptr() == r.data_ptr()
        and bias.shape[1:] == r.shape[1:]
        and bias.stride() == r.stride()
        and bias.shape[0] <= r.shape[0]
    )


def packed_rel_projection(bias):
    """Resolve a fused projection marker intercepted by a packed attention reader."""
    visit = _CTX.get()
    if visit is not None and visit.r is not None and _same_storage_rows(bias, visit.r):
        return visit.proj
    if bias.shape[-1] == D_REL and _is_produced(bias):
        raise RuntimeError("inkling_attn: packed reader received a marker outside its visit")
    return None


def _kernel_call(visit, bias, kwargs):
    """Arguments for ops.prefill_attention, or None when this call is outside what the kernel covers."""
    import torch

    q = kwargs.get("q")
    k = kwargs.get("k")
    v = kwargs.get("v")
    page_table = kwargs.get("page_table")
    cu_q = kwargs.get("cu_seqlens_q")
    seq_k = kwargs.get("seqused_k")
    out = kwargs.get("out")
    for name in (
        "qv", "cu_seqlens_k", "seqused_q", "learnable_sink", "score_mod", "mask_mod", "aux_tensors",
        "q_descale", "k_descale", "v_descale", "sfq", "sfk", "sfv", "gather_kv_indices", "block_sparse_tensors",
        "aux_scalars",
    ):
        if kwargs.get(name) is not None:
            return None
    if kwargs.get("softcap") not in (None, 0.0) or kwargs.get("rel_bias_presheared"):
        return None
    if not kwargs.get("causal", False) or kwargs.get("num_splits", 1) not in (0, 1) or kwargs.get("pack_gqa"):
        return None
    window = tuple(kwargs.get("window_size", (None, None)))
    if visit.local:
        if window != (visit.window_left, 0):
            return None
    elif window not in ((None, None), (-1, -1), (None, 0), (-1, 0)):
        return None
    scale = kwargs.get("softmax_scale")
    if scale is None:
        return None
    if not all(isinstance(t, torch.Tensor) for t in (q, k, v, page_table, cu_q, seq_k)):
        return None
    if k.dim() != 4 or k.shape[1] <= 0 or k.shape[2] <= 0 or cu_q.numel() < 2 or q.dim() != 3 or q.shape[1] <= 0:
        return None
    if not (
        q.dtype == k.dtype == v.dtype == torch.bfloat16
        and q.dim() == 3
        and q.shape[2] == HEAD_DIM
        and q.shape[1] == bias.shape[1]
        and q.shape[0] == bias.shape[0]
        and k.dim() == 4
        and k.shape == v.shape
        and k.shape[3] == HEAD_DIM
        and k.shape[1] % 64 == 0
        and q.shape[1] % k.shape[2] == 0
        and page_table.dtype == cu_q.dtype == seq_k.dtype == torch.int32
        and page_table.dim() == 2
        and page_table.stride(1) == 1
        and cu_q.is_contiguous()
        and seq_k.is_contiguous()
    ):
        return None
    if out is not None and not (
        out.dtype == torch.float32 and tuple(out.shape) == (q.shape[0], q.shape[1], HEAD_DIM) and out.stride(2) == 1
    ):
        return None
    if out is None:
        return None  # FA4 would return q.dtype; Inkling always passes an FP32 out
    dev = q.device
    if not all(t.device == dev for t in (k, v, page_table, cu_q, seq_k, bias, out, visit.proj)):
        return None
    if not (
        q.stride(2) == 1 and q.stride(0) % 2 == 0 and q.stride(1) % 2 == 0 and q.data_ptr() % 4 == 0
        and all(c.stride(3) == 1 and c.stride(0) % 8 == 0 and c.stride(1) % 8 == 0 and c.stride(2) % 8 == 0
                and c.data_ptr() % 16 == 0 for c in (k, v))
        and bias.stride(0) % 2 == 0 and bias.stride(1) % 2 == 0 and bias.data_ptr() % 4 == 0
        and out.stride(0) % 2 == 0 and out.stride(1) % 2 == 0 and out.data_ptr() % 8 == 0
        and cu_q.dim() == 1 and seq_k.dim() == 1 and seq_k.numel() >= cu_q.numel() - 1
        and page_table.shape[0] >= cu_q.numel() - 1
    ):
        return None
    return dict(
        q=q, k_cache=k, v_cache=v, page_table=page_table, cu_seqlens_q=cu_q, seqused_k=seq_k, r=bias,
        proj=visit.proj, softmax_scale=float(scale), window_left=visit.window_left if visit.local else None, out=out,
    )


def make_varlen(stock):
    import torch

    def _flash_attn_varlen_func(*args, **kwargs):
        bias = kwargs.get("rel_bias")
        if bias is not None and not args:
            visit = _CTX.get()
            if visit is not None and visit.r is not None and _same_storage_rows(bias, visit.r):
                call = _kernel_call(visit, bias, kwargs)
                if call is not None:
                    from . import ops

                    STATS["kernel"] += 1
                    if GATE and (visit.local or GATE_ALL) and visit.gated is None and _gate_ok(visit, call):
                        gate, sinks = visit.gate, visit.sinks
                        y = torch.empty(tuple(gate.shape), dtype=torch.bfloat16, device=gate.device)
                        _, lse = ops.prefill_attention(**call, gate=gate, sinks=sinks, y=y)
                        visit.gated = (call["out"], y)
                        STATS["gated"] += 1
                        kind = "SWA" if visit.local else "global"
                        _log_once(f"gated_first_{kind}", f"first gated {kind} prefill launch ({call['q'].shape[0]} rows)")
                        return call["out"], lse  # out stays unwritten: only gate_sink consumes it, and returns y
                    return ops.prefill_attention(**call)
                STATS["fallback"] += 1
                _log_once("fallback", "a marked call is outside the kernel's coverage; running stock FA4 for it")
                kwargs["rel_bias"] = torch.einsum("thd,de->the", bias, visit.proj)
                kwargs.pop("rel_bias_presheared", None)
                return stock(**kwargs)
        if bias is not None and bias.dim() == 3 and bias.shape[-1] == D_REL and _is_produced(bias):
            raise RuntimeError(
                "inkling_attn: an r-shaped rel_bias (width 16) reached FA4 outside the layer visit that made it"
            )
        return stock(*args, **kwargs)

    _flash_attn_varlen_func.__wrapped__ = stock
    setattr(_flash_attn_varlen_func, MARK, True)
    return _flash_attn_varlen_func


def _gate_ok(visit, call):
    """The gated launch is taken only for a whole-visit call (the q rows _project_sources returned, so no
    real-rows narrowing between the visit and the backend) with gate / sinks in the layouts the kernel reads."""
    import torch

    gate, sinks, q, out = visit.gate, visit.sinks, call["q"], call["out"]
    if gate is None or sinks is None or visit.tokens != q.shape[0]:
        return False
    t, heads = q.shape[0], q.shape[1]
    ok = (
        gate.dtype == sinks.dtype == torch.bfloat16
        and gate.dim() == 2 and tuple(gate.shape) == (t, heads * HEAD_DIM)
        and gate.stride(1) == 1 and gate.stride(0) % 8 == 0 and gate.data_ptr() % 16 == 0
        and sinks.dim() == 2 and tuple(sinks.shape) == (t, heads) and sinks.stride(1) == 1
        and gate.device == sinks.device == q.device
        and tuple(out.shape[:2]) == (t, heads)
    )
    if not ok:
        STATS["gate_fallback"] += 1
        _log_once("gate_fallback", "gated epilogue: unexpected gate / sinks layout; stock gate_sink used")
    return ok


def gate_begin(q, out):
    """For a packed (KV12S) global prefill reader: the gate / sinks / y kwargs of a gated launch when this visit's
    gate_sink can take the kernel's y (INKLING_ATTN_GATE=all, whole-visit rows, kernel layouts), else None. The
    caller launches with them and then calls gate_commit(g, out)."""
    import torch

    if not GATE_ALL:
        return None
    visit = _CTX.get()
    if visit is None or visit.local or visit.gated is not None or out is None:
        return None
    if not _gate_ok(visit, {"q": q, "out": out}):
        return None
    y = torch.empty(tuple(visit.gate.shape), dtype=torch.bfloat16, device=visit.gate.device)
    return {"gate": visit.gate, "sinks": visit.sinks, "y": y, "_visit": visit}


def gate_commit(g, out):
    """After the gated launch(es) of gate_begin's kwargs: the visit's gate_sink returns y (out stays unwritten)."""
    visit = g["_visit"]
    visit.gated = (out, g["y"])
    STATS["gated"] += 1
    _log_once("gated_first_global", f"first gated global prefill launch ({visit.tokens} rows, packed reader)")


def make_gate_sink(stock):
    """inkling2_attn's gate_sink: after a gated launch in this visit, return its BF16 y (the FP32 out it was handed
    was never written, so anything else is an error); every other call is the stock function."""

    def gate_sink(output, gate, lse, sinks, *, output_dtype, head_groups=None):
        visit = _CTX.get()
        done = visit.gated if visit is not None else None
        if done is None:
            return stock(output, gate, lse, sinks, output_dtype=output_dtype, head_groups=head_groups)
        out, y = done
        visit.gated = None
        if (output.data_ptr() != out.data_ptr() or output.numel() != out.numel() or head_groups is not None
                or gate is not visit.gate or sinks is not visit.sinks or tuple(y.shape) != tuple(gate.shape)):
            raise RuntimeError("inkling_attn: gate_sink did not receive the gated visit's attention output")
        import torch

        if output_dtype != torch.bfloat16:
            raise RuntimeError("inkling_attn: gated epilogue writes BF16, gate_sink asked for another dtype")
        return y

    gate_sink.__wrapped__ = stock
    setattr(gate_sink, MARK, True)
    return gate_sink


def patch_sm120(module):
    stock = getattr(module, "_flash_attn_varlen_func", None)
    if stock is None:
        raise RuntimeError(f"inkling_attn: {SM120_MODULE}._flash_attn_varlen_func is missing")
    if getattr(stock, MARK, False):
        return False
    module._flash_attn_varlen_func = make_varlen(stock)
    return True


def patch_common_attn(module):
    import inspect

    cls = getattr(module, "RelLogitsProj", None)
    fwd = getattr(cls, "forward", None) if cls is not None else None
    if fwd is None:
        raise RuntimeError(f"inkling_attn: {COMMON_ATTN_MODULE}.RelLogitsProj.forward is missing")
    params = list(inspect.signature(inspect.unwrap(fwd)).parameters)
    if params[:3] != ["self", "r_out", "log_scaling_tau"]:
        raise RuntimeError(f"inkling_attn: RelLogitsProj.forward signature changed: {params}")
    prev = cls.__call__
    if getattr(prev, MARK, False):
        return False
    cls.__call__ = make_rel_call(prev)
    return True


def patch_attn(module, kinds):
    import importlib

    cls = getattr(module, "Inkling2Attention", None)
    prev = getattr(cls, "_forward", None) if cls is not None else None
    get_attn_backend = getattr(module, "get_attn_backend", None)
    get_model = getattr(module, "get_model", None)
    if prev is None or get_attn_backend is None or get_model is None:
        raise RuntimeError(
            f"inkling_attn: {ATTN_MODULE}.Inkling2Attention._forward, get_attn_backend or get_model is missing"
        )
    if getattr(prev, MARK, False):
        return False
    in_breakable = getattr(importlib.import_module(BREAKABLE_MODULE), "is_in_breakable_cuda_graph", None)
    if in_breakable is None:
        raise RuntimeError(f"inkling_attn: {BREAKABLE_MODULE}.is_in_breakable_cuda_graph is missing")
    cp_active = getattr(importlib.import_module(CP_MODULE), "is_cp_active", None)
    if cp_active is None:
        raise RuntimeError(f"inkling_attn: {CP_MODULE}.is_cp_active is missing")
    cls._forward = make_forward(prev, get_attn_backend, get_model, in_breakable, frozenset(kinds), cp_active)
    if GATE_OFF_FOR_LOGGING:
        _log_once("gate_off", "gated epilogue OFF: SGLANG_KERNEL_API_LOGLEVEL is set (it reads the FA4 output buffer)")
    if GATE:
        gs = getattr(module, "gate_sink", None)
        if gs is None:
            raise RuntimeError(f"inkling_attn: {ATTN_MODULE}.gate_sink is missing (INKLING_ATTN_GATE)")
        if not getattr(gs, MARK, False):
            module.gate_sink = make_gate_sink(gs)
        _log_once("gate", "gated epilogue on: " + ("sliding-window and global" if GATE_ALL else "sliding-window")
                  + " prefill visits run gate_sink inside the attention kernel")
    return True
