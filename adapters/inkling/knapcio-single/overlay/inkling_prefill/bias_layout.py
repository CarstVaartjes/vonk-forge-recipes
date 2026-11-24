"""C2a bias_layout: write the relative-position bias straight into FA4's sheared layout.

Stock prefill path per layer visit (inkling2_attn.py Inkling2Attention._forward, the
non-compact branch):
  1. rel_bias = RelLogitsProj(r): cuBLAS einsum thd,de->the, BF16 [t, 24, 1024]
     (8K chunk: 403 MB written);
  2. FA4 (flash_attn/cute/interface.py) shears it: ShearingBias reads those 403 MB
     and writes a [round_up(t, 64), 24, 1280] copy (503 MB) in the column-aligned
     layout the SM120 attention mainloop loads by TMA;
  3. the attention kernel reads the sheared copy.
Steps 1-2 cost ~6 ms per visit at an 8K chunk (bench: einsum 2.0 ms; FA4 with bias
7.7 ms vs 1.8 ms without it on SWA layers).

This lever fuses 1 and 2: one Triton kernel computes r . proj (tl.dot, K = 16,
FP32 accumulate, BF16 round, like the einsum) and stores each value at its sheared
position, plus the -inf / left pads, into the same per-stream workspace the stock
shear uses (interface._shear_bias_empty). FA4 is then called with
rel_bias_presheared=True, which its interface already supports (it skips the prep
and shear kernels and asserts the [round_up(t, tile_m), heads, rel_extent + 256]
shape); only SGLang's SM120 wrapper drops that keyword, so the wrapper's
_flash_attn_varlen_func global is wrapped to forward it for the buffer this lever
made in the current layer visit.

Buffer identity: SGLang narrows rel_bias to the real rows before the backend call
(layers/attention/graph_utils.py attention_input_scope inside prefill graphs, and
_attention_on_real_rows), which yields a new view. So the varlen wrapper recognises
the buffer by storage pointer against the one recorded in the active visit (a
context variable set by the _forward wrapper, so nothing can match outside the
visit that made it) and passes the [round_up(q rows, 64)] row prefix of the full
buffer, the shape FA4 asserts for presheared bias. A rel-bias of the sheared width
that reaches FA4 outside its visit raises instead of being misread as unsheared.
Breakable prefill graphs (which replay attention as a deferred eager call, outside
the visit) are excluded at the visit check.

Layout: inkling_prefill._layout (shear_params) mirrors ShearingBias for SM120's
bias tile (tile_m 64, tile_n 128, pack_gqa off); tests/prefill proves the
element-wise description equal to a literal transcription of the CuTe kernel, and
tools/gpu_check_prefill.py compares the sheared buffer and the attention output
bitwise against the stock path.

Scope: prefill/extend forwards only (not decode, target verify or draft extend),
BF16 KV (not mxfp8), no multimodal image masks, no SWA spec metadata, main
attention with pack_gqa off; everything else runs stock. Patch points:
  Inkling2Attention._forward  (marks the eligible visit, no other change)
  RelLogitsProj.forward       (returns the sheared tensor inside a marked visit)
  flash_attention_v4_sm120._flash_attn_varlen_func (adds rel_bias_presheared)

CUDA graphs: static grid (tokens, heads); request lookup by binary search over the
backend's cu_seqlens_q on the device, so the same graph serves any request mix.
"""

import contextvars

ATTN_MODULE = "sglang.srt.models.inkling_common.inkling2_attn"
COMMON_ATTN_MODULE = "sglang.srt.models.inkling_common.attn"
SM120_MODULE = "sglang.kernels.ops.attention.flash_attention_v4_sm120"
INTERFACE_MODULE = "sglang.kernels.ops.attention.flash_attn.cute.interface"
MARK = "inkling_prefill_bias_layout"
_PRODUCED_WIDTHS = set()
SM120_BIAS_TILE_M = 64
BLOCK_T = 32
BLOCK_C = 128

_CTX = contextvars.ContextVar("inkling_prefill_bias_ctx", default=None)


class _Visit:
    __slots__ = ("owner", "local", "window_left", "cu_seqlens_q", "seqlens_k", "sheared")

    def __init__(self, owner, local, window_left, cu_seqlens_q, seqlens_k):
        self.owner = owner
        self.local = local
        self.window_left = window_left
        self.cu_seqlens_q = cu_seqlens_q
        self.seqlens_k = seqlens_k
        self.sheared = None


class use_presheared:
    """Context manager for direct FA4 calls (tools/gpu_check_prefill.py): inside it,
    the patched varlen wrapper treats `bias` (or a row prefix view of it) as
    presheared."""

    def __init__(self, bias):
        self._visit = _Visit(None, None, None, None, None)
        self._visit.sheared = bias
        self._token = None

    def __enter__(self):
        self._token = _CTX.set(self._visit)
        return self._visit.sheared

    def __exit__(self, *exc):
        _CTX.reset(self._token)
        return False


def _round_up(n, m):
    return -(-n // m) * m


def sheared_rel_bias(r, proj, cu_seqlens_q, seqlens_k, local, window_left, empty=None):
    """[round_up(T, 64), H, rel_extent + 256] sheared bias of r [T, H, 16] . proj."""
    import torch
    import triton

    from . import _triton_bias

    t, heads, d = r.shape
    rel = proj.shape[1]
    pad = rel + 256
    shape = (_round_up(t, SM120_BIAS_TILE_M), heads, pad)
    out = empty(shape, r.dtype, r.device) if empty is not None else torch.empty(
        shape, dtype=r.dtype, device=r.device
    )
    if r.stride(-1) != 1:
        r = r.contiguous()
    n_req = cu_seqlens_q.numel() - 1
    if t and n_req > 0:
        _triton_bias._sheared_rel_bias[(triton.cdiv(t, BLOCK_T), heads)](
            r,
            proj,
            out,
            cu_seqlens_q,
            seqlens_k,
            n_req,
            r.stride(0),
            r.stride(1),
            out.stride(0),
            out.stride(1),
            D=d,
            REL=rel,
            PAD=pad,
            LOCAL=bool(local),
            WL=int(window_left) if local else 0,
            TM=SM120_BIAS_TILE_M,
            BT=BLOCK_T,
            BC=BLOCK_C,
            SEARCH=max(1, n_req.bit_length()),
        )
    _PRODUCED_WIDTHS.add(pad)
    return out


def _visit_for(self, forward_batch, attn, get_attn_backend, get_model, in_breakable=None):
    import torch

    if in_breakable is not None and in_breakable():
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
    cu_q = getattr(meta, "cu_seqlens_q", None)
    seq_k = getattr(meta, "cache_seqlens_int32", None)
    if not (
        isinstance(cu_q, torch.Tensor)
        and isinstance(seq_k, torch.Tensor)
        and cu_q.dtype == seq_k.dtype == torch.int32
        and cu_q.is_cuda
        and seq_k.numel() >= cu_q.numel() - 1
    ):
        return None
    window = getattr(attn, "sliding_window_size", None)
    local = bool(self.is_local)
    rel = self.rel_extent
    if local:
        if window is None or window < 0 or window + 1 != rel:
            return None
    elif window not in (None, -1):
        return None
    if rel % 128:
        return None
    return _Visit(self, local, window if local else 0, cu_q, seq_k)


def make_forward(prev, get_attn_backend, get_model, in_breakable=None):
    def _forward(self, hidden_states, positions, forward_batch, *, attn, attn_padded):
        visit = _visit_for(self, forward_batch, attn, get_attn_backend, get_model, in_breakable)
        if visit is None:
            return prev(
                self, hidden_states, positions, forward_batch, attn=attn, attn_padded=attn_padded
            )
        token = _CTX.set(visit)
        try:
            return prev(
                self, hidden_states, positions, forward_batch, attn=attn, attn_padded=attn_padded
            )
        finally:
            _CTX.reset(token)

    _forward.__wrapped__ = prev
    setattr(_forward, MARK, True)
    return _forward


def make_rel_forward(prev, empty):
    import torch

    def forward(self, r_out, log_scaling_tau=None):
        visit = _CTX.get()
        if (
            visit is not None
            and log_scaling_tau is None
            and visit.owner.rel_logits_proj is self
            and r_out.dim() == 3
            and r_out.is_cuda
            and r_out.dtype == torch.bfloat16
            and self.proj.dtype == torch.bfloat16
            and self.proj.is_contiguous()
            and r_out.shape[2] == self.proj.shape[0] == 16
            and self.proj.shape[1] == visit.owner.rel_extent
        ):
            out = sheared_rel_bias(
                r_out,
                self.proj,
                visit.cu_seqlens_q,
                visit.seqlens_k,
                visit.local,
                visit.window_left,
                empty=empty,
            )
            visit.sheared = out
            return out
        return prev(self, r_out, log_scaling_tau)

    forward.__wrapped__ = prev
    setattr(forward, MARK, True)
    return forward


def make_varlen(stock):
    def _flash_attn_varlen_func(*args, **kwargs):
        bias = kwargs.get("rel_bias")
        if bias is not None:
            visit = _CTX.get()
            full = visit.sheared if visit is not None else None
            if (
                full is not None
                and bias.data_ptr() == full.data_ptr()
                and bias.dim() == 3
                and bias.shape[1:] == full.shape[1:]
                and bias.stride() == full.stride()
            ):
                q = kwargs.get("q")
                rows = _round_up(q.shape[0], SM120_BIAS_TILE_M)
                if q.dim() != 3 or rows > full.shape[0]:
                    raise RuntimeError(
                        f"inkling_prefill.bias_layout: {q.shape[0]} query rows exceed the "
                        f"sheared bias ({full.shape[0]} rows)"
                    )
                kwargs["rel_bias"] = full[:rows]
                kwargs["rel_bias_presheared"] = True
                kwargs.pop("rel_bias_prep_cache", None)
            elif bias.dim() == 3 and bias.shape[-1] in _PRODUCED_WIDTHS:
                raise RuntimeError(
                    "inkling_prefill.bias_layout: a rel-bias of the sheared width "
                    f"{bias.shape[-1]} reached FA4 outside the layer visit that made it"
                )
        return stock(*args, **kwargs)

    _flash_attn_varlen_func.__wrapped__ = stock
    setattr(_flash_attn_varlen_func, MARK, True)
    return _flash_attn_varlen_func


def patch_sm120(module):
    stock = getattr(module, "_flash_attn_varlen_func", None)
    if stock is None:
        raise RuntimeError(f"inkling_prefill.bias_layout: {SM120_MODULE}._flash_attn_varlen_func is missing")
    if getattr(stock, MARK, False):
        return False
    module._flash_attn_varlen_func = make_varlen(stock)
    return True


def patch_common_attn(module):
    import importlib

    cls = getattr(module, "RelLogitsProj", None)
    prev = getattr(cls, "forward", None) if cls is not None else None
    if prev is None:
        raise RuntimeError(f"inkling_prefill.bias_layout: {COMMON_ATTN_MODULE}.RelLogitsProj.forward is missing")
    if getattr(prev, MARK, False):
        return False
    interface = importlib.import_module(INTERFACE_MODULE)
    empty = getattr(interface, "_shear_bias_empty", None)
    if empty is None:
        raise RuntimeError(f"inkling_prefill.bias_layout: {INTERFACE_MODULE}._shear_bias_empty is missing")
    cls.forward = make_rel_forward(prev, empty)
    return True


BREAKABLE_MODULE = "sglang.srt.model_executor.runner_backend_utils.breakable_cuda_graph.context"


def patch_attn(module):
    import importlib

    cls = getattr(module, "Inkling2Attention", None)
    prev = getattr(cls, "_forward", None) if cls is not None else None
    get_attn_backend = getattr(module, "get_attn_backend", None)
    get_model = getattr(module, "get_model", None)
    if prev is None or get_attn_backend is None or get_model is None:
        raise RuntimeError(
            f"inkling_prefill.bias_layout: {ATTN_MODULE}.Inkling2Attention._forward, "
            "get_attn_backend or get_model is missing"
        )
    if getattr(prev, MARK, False):
        return False
    in_breakable = getattr(importlib.import_module(BREAKABLE_MODULE), "is_in_breakable_cuda_graph", None)
    if in_breakable is None:
        raise RuntimeError(f"inkling_prefill.bias_layout: {BREAKABLE_MODULE}.is_in_breakable_cuda_graph is missing")
    cls._forward = make_forward(prev, get_attn_backend, get_model, in_breakable)
    return True


def sheared_reference(r, proj, cu_seqlens_q, seqlens_k, local, window_left):
    """Slow torch reference of sheared_rel_bias (any device; rows of tokens past
    cu_seqlens_q[-1] and the round-up rows are left NaN). Uses the stock einsum
    for the pre-bias values and _layout.shear_params for the placement."""
    import torch

    from . import _layout

    t, heads, _ = r.shape
    rel = proj.shape[1]
    pad = rel + 256
    pre = torch.einsum("thd,de->the", r, proj)
    out = torch.full((_round_up(t, SM120_BIAS_TILE_M), heads, pad), float("nan"), dtype=r.dtype, device=r.device)
    cu = [int(v) for v in cu_seqlens_q.tolist()]
    sk_all = [int(v) for v in seqlens_k.tolist()]
    left_pad = float("-inf") if local else 0.0
    for b in range(len(cu) - 1):
        sq = cu[b + 1] - cu[b]
        for m in range(sq):
            left, right, _ = _layout.shear_params(m, sq, sk_all[b], local, window_left, rel, SM120_BIAS_TILE_M)
            row = out[cu[b] + m]
            row[:, right:] = float("-inf")
            row[:, :left] = left_pad
            if right > left:
                idx = torch.arange(right - 1 - left, -1, -1, device=r.device)  # columns left..right-1
                row[:, left:right] = pre[cu[b] + m][:, idx]
    return out
