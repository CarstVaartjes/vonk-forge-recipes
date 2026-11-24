"""Patch points of inkling_vattn (DSpark target-verify attention).

Stock verify path per layer visit (models/inkling_common/inkling2_attn.py Inkling2Attention._forward):
  compact = ... verify and SGLANG_OPT_INKLING2_VERIFY_REL_BIAS_KERNEL and tokens <= VERIFY_REL_BIAS_MAX_TOKENS (32)
  compact:  _decode_prologue(padded=False) -> attention_prologue -> rel bias (tokens, 24, 1024) BF16
            -> backend forward_extend (rel_bias_compact_decode) -> kernels inkling2_attn.rel_bias_partials
            -> decode_partials (split-KV Triton) -> combine_gate_sink (epilogue)
  else:     _decode_prologue(padded=True) -> q / bias padded to 32 heads -> FA4 (shear copy + mainloop, pack_gqa)
            -> gate_sink

cap:  models module global VERIFY_REL_BIAS_MAX_TOKENS := max_tokens (read by _forward at call time), so the
      compact path (stock prologue, stock _decode_partials with its own launch heuristics, stock combine) takes
      every verify batch up to max_tokens rows. Nothing else changes.
bias: the models-module name attention_prologue (bound by `from ... import attention_prologue`) is wrapped: a
      compact VERIFY call (verify_windows given, padded_group == heads // kv_heads, no shear, more than
      bias_min_tokens rows: at <= 32 rows the stock kernel is as fast) on SM12x runs the
      stock prologue with rel_proj = [I_16 | 0] (16, 128), which stores the BF16 rel-clip output r itself in
      rel[..., :16] (the K = 16 dot with a 0/1 matrix is exact), and records (rel, real proj) keyed by rel's data
      pointer. The wrapped rel_bias_partials takes the record of its rel_bias (same pointer, shape, live tensor)
      and runs ops.inkernel_bias_partials with the real proj. A 128-wide rel bias without a record raises.

Both wrappers are Python-only at call time (no device sync), so they are captured into the verify CUDA graphs
(bs 1..48) like the stock calls; the per-shape launch config is a pure function of static shapes.
"""

from contextvars import ContextVar
import inspect
import os
import sys

MODEL_MODULE = "sglang.srt.models.inkling_common.inkling2_attn"
OPS_MODULE = "sglang.kernels.ops.attention.inkling2_attn"

PROLOGUE_PARAMS = (
    "q_in", "k_in", "v_in", "r_in", "heads", "kv_heads", "d_rel", "padded_group", "k_sconv", "v_sconv",
    "k_cache", "v_cache", "metadata", "q_scale", "r_scale", "rel_proj", "seq_lens", "local_window",
    "qk_threshold", "rel_threshold", "clip_eps", "norm_eps", "k_dynamic", "v_dynamic", "positions",
    "tau_floor", "tau_alpha", "track_mask", "track_indices", "verify_windows", "shear",
)
PARTIALS_PARAMS = (
    "q", "k_cache", "v_cache", "page_table", "seqused_k", "rel_bias", "softmax_scale", "out",
    "window_size_left", "verify",
)
I_HEADS, I_KV_HEADS, I_D_REL, I_PADDED_GROUP, I_REL_PROJ = 4, 5, 6, 7, 15
R_WIDTH = 128  # the prologue's rel extent must be a multiple of 128

BYPASS_BIAS = ContextVar("inkling_vattn_bypass_bias", default=False)

STATS = {"verify_inkernel": 0, "passthrough": 0, "eye_in_capture": 0}
# Prologue r-markers awaiting their attention call: data_ptr -> (weakref to the rel tensor, proj). Several may be
# pending at once (A prologue, B prologue, A attention); each is consumed by its own attention call.
_PENDING = {}
_PENDING_MAX = 64
_EYE = {}
_SM12 = {}


def _log(msg):
    sys.stderr.write(f"[inkling_vattn pid={os.getpid()}] {msg}\n")
    sys.stderr.flush()


def _check_params(fn, expected, what):
    params = tuple(inspect.signature(fn).parameters)
    if params != expected:
        raise RuntimeError(f"inkling_vattn: {what} signature changed: {params}")


def _is_sm12(device):
    import torch

    key = device.index
    if key not in _SM12:
        _SM12[key] = device.type == "cuda" and torch.cuda.get_device_capability(device)[0] == 12
    return _SM12[key]


def _eye_proj(proj):
    """[I | 0] of shape (d_rel, 128) in proj's dtype / device: rel = dot(r, eye) = r in columns [0, d_rel);
    None while the current stream is capturing and the matrix does not exist yet."""
    import torch

    key = (proj.device, proj.dtype, proj.shape[0])
    eye = _EYE.get(key)
    if eye is None:
        # Never create it inside a capture: the fill would only run when that graph replays (and the memory
        # would belong to its private pool). SGLang's full-graph backend runs eager warm-up steps first.
        if torch.cuda.is_current_stream_capturing():
            STATS["eye_in_capture"] += 1
            if STATS["eye_in_capture"] == 1:
                _log("verify visit inside a CUDA-graph capture before any eager one; bias lever stays stock there")
            return None
        eye = torch.zeros((proj.shape[0], R_WIDTH), dtype=proj.dtype, device=proj.device)
        eye[:, : proj.shape[0]] = torch.eye(proj.shape[0], dtype=proj.dtype, device=proj.device)
        torch.cuda.synchronize(proj.device)  # once per device: ready before any stream or graph reads it
        _EYE[key] = eye
    return eye


def _arg(args, kwargs, index, name):
    return args[index] if len(args) > index else kwargs[name]


def _record(rel, proj):
    import weakref

    if len(_PENDING) >= _PENDING_MAX:
        for ptr in [p for p, (ref, _) in _PENDING.items() if ref() is None]:
            del _PENDING[ptr]
        if len(_PENDING) >= _PENDING_MAX:
            raise RuntimeError("inkling_vattn: too many prologue r-markers without their attention call")
    _PENDING[rel.data_ptr()] = (weakref.ref(rel), proj)


def _take(rel_bias):
    entry = _PENDING.get(rel_bias.data_ptr())
    if entry is None:
        return None
    rel = entry[0]()
    if rel is None or rel.shape[-1] != rel_bias.shape[-1] or rel.shape[1] != rel_bias.shape[1]:
        del _PENDING[rel_bias.data_ptr()]
        return None
    del _PENDING[rel_bias.data_ptr()]
    return entry[1]


def make_prologue(orig, min_tokens, max_tokens):
    def attention_prologue(*args, **kwargs):
        verify_windows = kwargs.get("verify_windows")
        shear = kwargs.get("shear")
        if BYPASS_BIAS.get() or len(args) > 22 or verify_windows is None or shear is not None:
            return orig(*args, **kwargs)
        q_in = args[0] if args else kwargs["q_in"]
        heads = _arg(args, kwargs, I_HEADS, "heads")
        kv_heads = _arg(args, kwargs, I_KV_HEADS, "kv_heads")
        d_rel = _arg(args, kwargs, I_D_REL, "d_rel")
        padded_group = _arg(args, kwargs, I_PADDED_GROUP, "padded_group")
        proj = _arg(args, kwargs, I_REL_PROJ, "rel_proj")
        if (
            padded_group != heads // kv_heads
            or d_rel != 16
            or proj.shape[0] != d_rel
            or proj.shape[1] <= R_WIDTH
            or not proj.is_contiguous()
            or proj.dtype != q_in.dtype
            or not min_tokens <= q_in.shape[0] <= max_tokens
            or not _is_sm12(q_in.device)
        ):
            return orig(*args, **kwargs)
        eye = _eye_proj(proj)
        if eye is None:  # first use inside a CUDA-graph capture: stay stock (no eager warm-up ran)
            return orig(*args, **kwargs)
        if len(args) > I_REL_PROJ:
            args = args[:I_REL_PROJ] + (eye,) + args[I_REL_PROJ + 1:]
        else:
            kwargs = dict(kwargs, rel_proj=eye)
        q, k, v, rel = orig(*args, **kwargs)
        assert rel.shape[-1] == R_WIDTH
        _record(rel, proj)
        return q, k, v, rel

    attention_prologue.__wrapped__ = orig
    attention_prologue._inkling_vattn = True
    return attention_prologue


def make_partials(orig):
    from . import ops

    def rel_bias_partials(q, k_cache, v_cache, page_table, seqused_k, rel_bias, softmax_scale, out, *,
                          window_size_left, verify):
        proj = _take(rel_bias)
        if proj is not None:
            assert verify and rel_bias.shape[0] >= q.shape[0]
            STATS["verify_inkernel"] += 1
            return ops.inkernel_bias_partials(q, k_cache, v_cache, page_table, seqused_k, rel_bias, proj,
                                              softmax_scale, window_size_left=window_size_left)
        if rel_bias.shape[-1] == R_WIDTH:  # this model's extent is 1024: an r marker that lost its record
            raise RuntimeError("inkling_vattn: a 128-wide rel bias (an r marker) reached rel_bias_partials "
                               "without its prologue record")
        STATS["passthrough"] += 1
        return orig(q, k_cache, v_cache, page_table, seqused_k, rel_bias, softmax_scale, out,
                    window_size_left=window_size_left, verify=verify)

    rel_bias_partials.__wrapped__ = orig
    rel_bias_partials._inkling_vattn = True
    return rel_bias_partials


def patch_model(module, levers, max_tokens, bias_min_tokens):
    if getattr(module, "VERIFY_REL_BIAS_MAX_TOKENS", None) != 32:
        raise RuntimeError("inkling_vattn: VERIFY_REL_BIAS_MAX_TOKENS is not the stock 32")
    if "cap" in levers:
        module.VERIFY_REL_BIAS_MAX_TOKENS = max_tokens
    if "bias" in levers:
        orig = module.attention_prologue
        if getattr(orig, "_inkling_vattn", False):
            return
        _check_params(orig, PROLOGUE_PARAMS, "attention_prologue")
        module.attention_prologue = make_prologue(orig, bias_min_tokens, module.VERIFY_REL_BIAS_MAX_TOKENS)


def patch_ops(module):
    orig = module.rel_bias_partials
    if getattr(orig, "_inkling_vattn", False):
        return
    _check_params(orig, PARTIALS_PARAMS, "rel_bias_partials")
    module.rel_bias_partials = make_partials(orig)


def register(levers, max_tokens, bias_min_tokens):
    from . import _hook

    def ops_cb(module):
        patch_ops(module)
        if os.environ.get("INKLING_VATTN_VERBOSE"):
            _log(f"patched {OPS_MODULE}")

    def model_cb(module):
        patch_model(module, levers, max_tokens, bias_min_tokens)
        if os.environ.get("INKLING_VATTN_VERBOSE"):
            _log(f"patched {MODEL_MODULE}")

    if "bias" in levers:
        _hook.when_imported(OPS_MODULE, ops_cb)
    _hook.when_imported(MODEL_MODULE, model_cb)
