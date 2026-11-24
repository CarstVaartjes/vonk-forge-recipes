"""Inkling-2-Mini prefill glue fusion overlay for SGLang on DGX Spark (GB10, sm_121).

    INKLING_PFUSE=loopmix,resnorm,sconvres   (or "all")      unset / empty: install() does nothing, nothing is imported.
    INKLING_PFUSE_MIN_ROWS=512                         smallest extend batch (rows) served by the fused path
    INKLING_PFUSE_LOOP_NORM=fused|stock                loopmix attention side: RMSNorm inside the mix kernel (two
                                                       passes) or FP32 mix output + the stock RMSNorm

Levers (both patch Inkling2DecoderLayer._forward for EXTEND / MIXED batches of >= MIN_ROWS rows; decode, target
verify and small batches run the stock method):

loopmix   the loop message mixes of the core layers' passes 2-3 (24 of 48 visits): attention-residual state mix,
          bank mix, controller norm, the stock split-K down projection, controller mix and (attention side) the
          layer RMSNorm in 3 kernels without any FP32 [T, D] intermediate on the attention side
resnorm   every other visit (except layer 0): attention-residual mix + RMSNorm in one kernel, the FP32 mix output
          is never written
sconvres  mlp short convolution + residual update in one kernel (chunks >= 2048 rows, where stock uses its
          _sconv_prefill_kernel); bit-identical to stock

Exactness: same weights, same per-element formulas, same BF16 rounding points as the stock kernels; the order of
the FP32 row reductions (attn-res sums of squares and query dots, RMS statistics) differs (rounding level, see
README.md and tools/gpu_check_pfuse.py).

install() checks every patch target in the installed sglang sources by hashing the function sources (without
importing sglang); any difference raises RuntimeError, which the combined sitecustomize turns into exit 78.
"""

import ast
import hashlib
import importlib.util
import os
import sys

from . import _hook

LEVERS = ("loopmix", "resnorm", "sconvres")
MIN_ROWS_FLOOR = 64

# (file under the sglang package, qualified name, sha256[:16] of the function / class source)
_TARGETS = (
    ("srt/models/inkling2.py", "Inkling2DecoderLayer._forward", "cdb3a67939277e1f"),
    ("srt/models/inkling2.py", "Inkling2DecoderLayer._attention_input", "2e6ea7084228ad09"),
    ("srt/models/inkling2.py", "Inkling2CausalLLM._run_loop", "fe79d40a475dd3c9"),
    ("srt/models/inkling_common/inkling2_attn_res.py", "block_attn_res", "2666f662da354650"),
    ("srt/models/inkling_common/inkling2_attn_res.py", "Inkling2AttnResState.attention_input", "50e3b14c62986935"),
    ("srt/models/inkling_common/inkling2_attn_res.py", "Inkling2AttnResState.mlp_input", "847ccbaee50a121d"),
    ("srt/models/inkling_common/inkling2_attn_res.py", "Inkling2AttnRes.scaled_query", "f18530961d907d6a"),
    ("srt/models/inkling_common/inkling2_attn_res.py", "Inkling2AttnRes.attn_query", "3a36319a2d39b5c6"),
    ("srt/models/inkling_common/inkling2_attn_res.py", "Inkling2AttnRes.is_block_boundary", "5b6ec507ddb15855"),
    ("srt/models/inkling_common/inkling2_loop.py", "LoopMessageBank.mix", "b22570dc5e9b43ac"),
    ("srt/models/inkling_common/inkling2_loop.py", "LoopMessageBank.__init__", "94004b3c50136318"),
    ("srt/models/inkling_common/inkling2_loop.py", "stack_controller_weights", "16ca0e1fd9ee875a"),
    ("srt/models/inkling_common/inkling2_loop.py", "scaled_loop_queries", "f52afc5e44f44dee"),
    ("srt/models/inkling_common/inkling2_norm.py", "RMSNorm.forward", "e04e5d4a8f1718da"),
    ("kernels/ops/elementwise/inkling2_loop.py", "message_controllers", "5e7196cf066beebb"),
    ("kernels/ops/elementwise/inkling2_loop.py", "_controller_down", "98e1a531e9cf78b1"),
    ("kernels/ops/elementwise/inkling2_loop.py", "_controller_mix", "f84dc37e145d1ad3"),
    ("kernels/ops/elementwise/inkling2_loop.py", "_controller_norm", "acc8b3b1dade5f05"),
    ("kernels/ops/attention/inkling2_attn_res.py", "attn_res", "9f5f6c90ca01b9f9"),
    ("kernels/ops/attention/inkling2_attn_res.py", "AttnResFwdKernel", "5e8084f639532464"),
    ("kernels/ops/layernorm/inkling2_rms_norm.py", "rms_norm", "23afb3d646391ccb"),
    ("kernels/ops/layernorm/inkling2_rms_norm.py", "RMSNormFwd", "a8e486ef1d67daf3"),
    ("kernels/numerics.py", "rsqrt_approx_f32", "7691326f1c1add3b"),
    ("srt/models/inkling_common/inkling2_sconv.py", "ShortConvolution.forward", "e0818ad6a45c4763"),
    ("kernels/ops/mamba/inkling2_sconv.py", "_sconv_prefill_kernel", "87e75ec4b699b7e6"),
    ("kernels/ops/mamba/inkling2_sconv.py", "short_convolution", "bef47a776e5ded4b"),
    ("srt/models/inkling_common/inkling2_attn_res.py", "Inkling2AttnResState.finalize_layer", "961f89e2c7458bcc"),
    ("kernels/ops/elementwise/inkling2_residual_add.py", "_residual_add_kernel", "fa519ada0fb86aa8"),
    ("kernels/ops/elementwise/inkling2_residual_add.py", "residual_add", "a75923e24e039605"),
    ("srt/models/inkling_common/kernels/sconv.py", "update_sconv_cache", "95a7c24bd11cb7ed"),
    ("srt/models/inkling_common/sconv.py", "ShortConvolution._prepare_extend_sconv_cache", "0e34a081b2d8aba0"),
    ("srt/models/inkling_common/sconv.py", "ShortConvolution._weight_2d", "b574dfaeced0ec64"),
)

STATE = {"installed": False, "levers": (), "applied": []}


def _log(msg):
    sys.stderr.write(f"[inkling_pfuse pid={os.getpid()}] {msg}\n")
    sys.stderr.flush()


def parse_levers(value):
    items = [v.strip() for v in value.split(",") if v.strip()]
    items = [v for v in items if v.lower() not in ("0", "off", "none", "false", "no")]
    if "all" in items:
        items = list(LEVERS) + [v for v in items if v != "all"]
    levers = tuple(dict.fromkeys(items))
    unknown = [v for v in levers if v not in LEVERS]
    if unknown:
        raise RuntimeError(f"inkling_pfuse: unknown INKLING_PFUSE entries {unknown}; known: {', '.join(LEVERS)}, all")
    return levers


def _min_rows():
    raw = os.environ.get("INKLING_PFUSE_MIN_ROWS", "512").strip()
    try:
        value = int(raw)
    except ValueError as exc:
        raise RuntimeError(f"inkling_pfuse: INKLING_PFUSE_MIN_ROWS={raw!r} is not an integer") from exc
    if value < MIN_ROWS_FLOOR:
        raise RuntimeError(f"inkling_pfuse: INKLING_PFUSE_MIN_ROWS={value} must be >= {MIN_ROWS_FLOOR}")
    return value


def sglang_root():
    spec = importlib.util.find_spec("sglang")  # does not execute sglang
    if spec is None or not spec.submodule_search_locations:
        raise RuntimeError("inkling_pfuse: the sglang package is not importable")
    return list(spec.submodule_search_locations)[0]


def source_digest(text, qualname):
    node = ast.parse(text)
    for part in qualname.split("."):
        node = next(
            (n for n in ast.iter_child_nodes(node)
             if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name == part),
            None,
        )
        if node is None:
            return None
    return hashlib.sha256(ast.get_source_segment(text, node).encode()).hexdigest()[:16]


def check_sources(root=None):
    root = root or sglang_root()
    texts = {}
    for rel, qualname, digest in _TARGETS:
        path = os.path.join(root, rel)
        if path not in texts:
            try:
                with open(path, encoding="utf-8") as f:
                    texts[path] = f.read()
            except OSError as exc:
                raise RuntimeError(f"inkling_pfuse: patch target {path} is missing") from exc
        got = source_digest(texts[path], qualname)
        if got != digest:
            raise RuntimeError(
                f"inkling_pfuse: {rel}:{qualname} changed (sha {got} != {digest}); refusing to patch"
            )


def install(levers=None, root=None):
    """Activate the levers named in INKLING_PFUSE (or `levers`). Idempotent."""
    if STATE["installed"]:
        return STATE["levers"]
    levers = parse_levers(os.environ.get("INKLING_PFUSE", "") if levers is None else ",".join(levers))
    if not levers:
        STATE["installed"] = True
        return ()
    min_rows = _min_rows()
    check_sources(root)
    from . import layer

    loop_norm = os.environ.get("INKLING_PFUSE_LOOP_NORM", "fused").strip().lower()
    if loop_norm not in ("fused", "stock"):
        raise RuntimeError(f"inkling_pfuse: INKLING_PFUSE_LOOP_NORM={loop_norm!r} must be fused or stock")
    layer.CFG.update(levers=levers, min_rows=min_rows, loop_norm=loop_norm)

    def callback(module):
        layer.patch(module)
        STATE["applied"].append(f"layer@{layer.TARGET_MODULE}")
        if os.environ.get("INKLING_PFUSE_VERBOSE"):
            _log(f"applied {','.join(levers)} to {layer.TARGET_MODULE}")

    _hook.when_imported(layer.TARGET_MODULE, callback)
    STATE.update(installed=True, levers=levers)
    _log(f"enabled {','.join(levers)} (min_rows={min_rows})")
    return levers
