"""inkling_prologue: fused K / V short conv + L2 clip / value norm in the prefill attention prologue (SWA layers).

    INKLING_PROLOGUE=1   on (unset / 0: nothing is imported)

Patch: Inkling2Attention._prologue. For an EXTEND visit of a sliding-window layer with >= 2048 rows (the stock
short-conv prefill kernel's own threshold), static conv weights with residual, head_dim 128, no k-clip scale and a
weightless value norm, k_sconv + k_l2_clip and v_sconv + v_norm each run as one kernel (_triton._sconv_post_kernel);
the conv-cache prepare / update calls and the q / rel clips are the stock code. Everything else runs the stock method.
Bit-identical to the stock conv + clip / RMSNorm: the head row sums follow the CuTe kernels' reduction order (_triton
_row_sumsq, SUM_ORDER; tools/gpu_check_prologue.py requires 0 ulp). CUDA-graph safe (no host sync).
"""

import os
import sys

from . import _hook

STATE = {"installed": False, "applied": [], "fused": 0, "stock": 0}

# Source digests (sha256[:16] of the function / class source, the same scheme as inkling_pfuse) of everything the fused
# kernel copies or relies on: a changed image refuses to install instead of running the old conv / clip / norm math.
_TARGETS = (
    ("srt/models/inkling_common/inkling2_attn.py", "Inkling2Attention._prologue", "bee4e4209010ba08"),
    ("srt/models/inkling_common/inkling2_attn.py", "QKL2Clip", "ab220241fb563e1e"),
    ("kernels/ops/layernorm/inkling2_qk_l2_clip.py", "qk_l2_clip_fwd_cutedsl", "70d55d47c05648fc"),
    ("kernels/ops/layernorm/inkling2_qk_l2_clip.py", "QKL2ClipFwd", "b0be913c00270b77"),
    ("kernels/ops/layernorm/inkling2_cute_utils.py", "qk_l2_clip_row_stats", "a43c63be8515eb1c"),
    ("kernels/ops/layernorm/inkling2_cute_utils.py", "rmsnorm_row_rstd", "0e862090ca9b2e32"),
    ("kernels/ops/layernorm/inkling2_cute_utils.py", "min_f32", "382f3ee5dc9c5396"),
    ("kernels/ops/layernorm/inkling2_rms_norm.py", "rms_norm", "23afb3d646391ccb"),
    ("kernels/ops/layernorm/inkling2_rms_norm.py", "RMSNormFwd", "a8e486ef1d67daf3"),
    ("srt/models/inkling_common/inkling2_norm.py", "RMSNorm", "a95658ccfd18f291"),
    ("kernels/ops/mamba/inkling2_sconv.py", "_sconv_prefill_kernel", "87e75ec4b699b7e6"),
    ("kernels/ops/mamba/inkling2_sconv.py", "short_convolution", "bef47a776e5ded4b"),
    ("srt/models/inkling_common/inkling2_sconv.py", "ShortConvolution.forward", "e0818ad6a45c4763"),
    ("srt/models/inkling_common/sconv.py", "ShortConvolution._conv_state", "62310e046cdb423e"),
    ("srt/models/inkling_common/sconv.py", "ShortConvolution._sconv_cache", "16dfc5b448de3104"),
    ("srt/models/inkling_common/sconv.py", "ShortConvolution._prepare_extend_sconv_cache", "0e34a081b2d8aba0"),
    ("srt/models/inkling_common/sconv.py", "ShortConvolution._weight_2d", "b574dfaeced0ec64"),
    ("srt/models/inkling_common/kernels/sconv.py", "update_sconv_cache", "95a7c24bd11cb7ed"),
    ("kernels/numerics.py", "rsqrt_approx_f32", "7691326f1c1add3b"),
)


def check_sources(root=None):
    from inkling_pfuse import sglang_root, source_digest  # the shared digest scheme (no sglang / torch import)

    root = root or sglang_root()
    texts = {}
    for rel, qualname, digest in _TARGETS:
        path = os.path.join(root, rel)
        if path not in texts:
            try:
                with open(path, encoding="utf-8") as f:
                    texts[path] = f.read()
            except OSError as exc:
                raise RuntimeError(f"inkling_prologue: patch target {path} is missing") from exc
        got = source_digest(texts[path], qualname)
        if got != digest:
            raise RuntimeError(f"inkling_prologue: {rel}:{qualname} changed (sha {got} != {digest}); refusing to patch")


def mixed_chunk_on():
    """True when the server runs with --enable-mixed-chunk (a MIXED batch replays through an EXTEND prefill graph,
    so decode rows would see prefill-only math). None when the server args are not published yet."""
    try:
        from sglang.srt.runtime_context import get_server_args

        return bool(getattr(get_server_args(), "enable_mixed_chunk", False))
    except Exception:  # noqa: BLE001
        return None
MIN_ROWS = 2048
HEAD = 128


def _log(msg):
    print(f"[inkling_prologue] {msg}", file=sys.stderr, flush=True)


class PrologueFatal(BaseException):
    pass


# The CuTe reduction tree of the stock clip / RMSNorm row sum (see _triton._row_sumsq); fixed by
# tools/gpu_check_prologue.py, which requires 0 ulp against the stock pair for these values.
SUM_ORDER = {1: (0, 0), 2: (0, 0)}  # post -> (PER, XOR): sequential lane fold, descending butterfly (CuTe)


def conv_post(x, weight, cache, md, post, thr, eps, order=None, ss=None):
    """Stock short_convolution (static weights, residual, prefill kernel) followed by the K clip (post 1) or the
    weightless value RMSNorm (post 2) over 128-wide heads, in one kernel. x [T, H*128] BF16."""
    import torch
    import triton

    from . import _triton

    width = weight.shape[1]
    per, xor = SUM_ORDER[post] if order is None else order
    out = torch.empty_like(x, memory_format=torch.contiguous_format)
    _triton._sconv_post_kernel[(triton.cdiv(x.shape[0], 16), triton.cdiv(x.shape[1], HEAD))](
        x, weight, cache, md["safe_idx"], md["cache_mask"], md["cu"], md["si"], out, x.shape[0],
        x.stride(0), *weight.stride(), *cache.stride(), x.shape[1], width, (width + 1) ** -0.5, post,
        float(thr), float(eps), 16, HEAD, per, xor, int(per >= 0), ss, ss is not None, enable_fp_fusion=False)
    return out


def _conv_post(sconv, x, forward_batch, post, thr, eps, update_sconv_cache):
    meta = sconv._conv_state(forward_batch)
    cache = sconv._sconv_cache()
    sconv._prepare_extend_sconv_cache(forward_batch, cache, x, meta.track_conv_indices)
    out = conv_post(x, sconv._weight_2d(), cache, meta.precomputed, post, thr, eps)
    update_sconv_cache(x=x, sconv_cache=cache, cache_indices=meta.cache_indices,
                       has_initial_state=meta.has_initial_state, query_start_loc=meta.query_start_loc)
    return out


def _eligible(self, q, k, v, forward_batch):
    import torch

    mode = forward_batch.forward_mode
    if not self.is_local or getattr(mode, "name", None) != "EXTEND":  # plain EXTEND only (not MIXED / DLLM / verify)
        return False
    for name in ("is_draft_extend", "is_draft_extend_v2"):
        fn = getattr(mode, name, None)
        if fn is not None and fn():
            return False
    if torch.compiler.is_compiling() or k.shape[0] < MIN_ROWS or self.head_dim != HEAD:
        return False
    for sconv in (self.k_sconv, self.v_sconv):
        if sconv.is_dynamic or not sconv.use_residual:
            return False
    kc, vn = self.k_l2_clip, self.v_norm
    if getattr(kc, "scale", None) is not None or vn is None or vn.weight is not None or vn.pre_weight is not None:
        return False
    if getattr(vn, "output_dtype", None) not in (None, torch.bfloat16) or k.shape[1] % HEAD or v.shape[1] % HEAD:
        return False
    return all(t.dtype == torch.bfloat16 and t.dim() == 2 and t.stride(-1) == 1 and t.is_cuda for t in (k, v))


def make_prologue(stock, update_sconv_cache):
    def _prologue(self, q, k, v, r, sources, positions, forward_batch):
        if not _eligible(self, q, k, v, forward_batch):
            STATE["stock"] += 1
            return stock(self, q, k, v, r, sources, positions, forward_batch)
        STATE["fused"] += 1
        if STATE["fused"] == 1:
            if mixed_chunk_on():
                raise PrologueFatal("inkling_prologue: --enable-mixed-chunk is on; refusing")
            _log(f"first fused prologue call ({k.shape[0]} rows)")
        tokens = q.shape[0]
        kc = self.k_l2_clip
        k = _conv_post(self.k_sconv, k, forward_batch, 1, kc.threshold, kc.eps, update_sconv_cache)
        v = _conv_post(self.v_sconv, v, forward_batch, 2, 0.0, self.v_norm.eps, update_sconv_cache)
        q = self.q_l2_clip(q.reshape(tokens, self.num_tp_heads, self.head_dim), output_scale=None)
        r = self.rel_l2_clip(r.reshape(tokens, self.num_tp_heads, self.d_rel), output_scale=None)
        return (q, k.view(tokens, self.num_tp_kv_heads, self.head_dim),
                v.view(tokens, self.num_tp_kv_heads, self.head_dim), r)

    _prologue.__wrapped__ = stock
    _prologue._inkling_prologue = True
    return _prologue


def _patch(mod):
    if mixed_chunk_on():
        raise PrologueFatal("inkling_prologue: --enable-mixed-chunk is on; refusing (decode rows would get prefill math)")
    cls = getattr(mod, "Inkling2Attention", None)
    stock = getattr(cls, "_prologue", None)
    if stock is None:
        raise PrologueFatal("inkling_prologue: Inkling2Attention._prologue is missing")
    if getattr(stock, "_inkling_prologue", False):
        return
    import inspect

    src = inspect.getsource(stock)
    for needle in ("k = self.k_sconv(", "v = self.v_sconv(", "dynamic_input=None if self.is_local else sources.k",
                   "q = self.q_l2_clip(", "k = self.k_l2_clip(k.reshape(tokens, self.num_tp_kv_heads, self.head_dim))",
                   "v = self.v_norm(v)", "r = self.rel_l2_clip(", "return q, k, v, r"):
        if needle not in src:
            raise PrologueFatal(f"inkling_prologue: _prologue changed (missing {needle!r})")
    from sglang.srt.models.inkling_common.kernels.sconv import update_sconv_cache

    cls._prologue = make_prologue(stock, update_sconv_cache)
    STATE["applied"].append("Inkling2Attention._prologue")


def install(env=None):
    if STATE["installed"]:
        return bool(STATE["applied"]) or None
    STATE["installed"] = True
    env = os.environ if env is None else env
    if env.get("INKLING_PROLOGUE", "").strip().lower() in ("", "0", "off", "false", "no"):
        return False
    check_sources()
    _hook.when_imported("sglang.srt.models.inkling_common.inkling2_attn", _patch)
    _log("enabled: SWA prefill K conv + L2 clip and V conv + value norm fused (bit-identical to the stock conv + clip / RMSNorm)")
    return True
