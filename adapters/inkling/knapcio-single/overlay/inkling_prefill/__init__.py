"""Inkling-2-Mini prefill overlay for SGLang on DGX Spark (GB10, sm_121).

    INKLING_PREFILL=qkvr_split,bias_layout,quant_once,partial_bf16   (or "all")
    unset / empty: install() does nothing and nothing is imported.

Each lever is meant to be bit-identical to the stock path it replaces;
tools/gpu_check_prefill.py checks that on the Spark. Order by value per effort
(8K-chunk estimates from the prefill bench, docs/prefill/PREFILL.md):

qkvr_split    Inkling2Attention._project_sources: the 28 stacked-source qkvr
              projections compute only the kept columns (-7.9 % TTFT at 8K).
              Env INKLING_PREFILL_QKVR_MIN_M (128), INKLING_PREFILL_QKVR_OUT
              (floor 128). Outputs are views of one [M, 8600] buffer (the stock
              row stride the fused attention prologue asserts). Composes with inkling_kernels.qkvr_gemv,
              which keeps M <= 16 (decode/verify).
bias_layout   rel-bias computed straight into FA4's sheared layout; FA4 called
              with rel_bias_presheared (~-7 %).
quant_once    NVFP4 MoE input quantized once per token, then gathered (~-1.5 %).
              Env INKLING_PREFILL_QUANT_MIN_T (256).
partial_bf16  MoE combine writes BF16 instead of FP32 + cast (~-1.2 %).

install() parses the env, checks the patch targets in the installed sglang sources
(without importing sglang), registers post-import hooks, and prints one line.
Problems raise RuntimeError (the combined sitecustomize then exits with code 78);
a target attribute missing at import time makes that import fail.
INKLING_PREFILL_VERBOSE=1 logs each applied patch.
"""

import importlib.util
import os
import sys

from . import _hook

LEVERS = ("qkvr_split", "bias_layout", "quant_once", "partial_bf16")

ATTN = "srt/models/inkling_common/inkling2_attn.py"
MOE = "srt/models/inkling_common/inkling2_moe.py"

_SOURCE_CHECKS = {
    "qkvr_split": (
        (ATTN, (
            "class Inkling2Attention(",
            "def _project_sources(self, sources):",
            "self.qkvr(_stacked(sources))",
            "q, _, _, r, gate, sinks = q_proj",
            "self.split_sizes = [",
        )),
        ("srt/layers/quantization/unquant.py", (
            "def get_bf16_gemm_backend(",
            "return F.linear(x, layer.weight, bias)",
        )),
    ),
    "bias_layout": (
        (ATTN, (
            "def _forward(self, hidden_states, positions, forward_batch, *, attn, attn_padded):",
            "rel_bias = self.rel_logits_proj(r)",
            "attention_kwargs = dict(rel_bias=rel_bias, **bias_kwargs)",
            "from sglang.srt.model_executor.forward_context import get_attn_backend",
        )),
        ("srt/models/inkling_common/attn.py", (
            "class RelLogitsProj(",
            "log_scaling_tau: torch.Tensor | None = None",
        )),
        ("kernels/ops/attention/flash_attention_v4_sm120.py", (
            "_flash_attn_varlen_func,",
            "result = _flash_attn_varlen_func(",
            'rel_bias_kwargs["rel_bias"] = rel_bias',
        )),
        ("kernels/ops/attention/flash_attn/cute/interface.py", (
            "if rel_bias is not None and rel_bias_presheared:",
            "_round_up_to_tile(total_q, tile_m)",
            "def _shear_bias_empty(shape, dtype, device):",
            "rel_extent_padded = rel_extent + 256",
        )),
        ("kernels/ops/attention/flash_attn/cute/shearing_bias.py", (
            "n_block_for_rel0 = (n_idx_right - 1) // 128",
            "self.rel_extent_padded // 128 - (attn_n_block_max - n_block_for_rel0)",
            "-Float32.inf if const_expr(window_size_left is not None) else 0.0",
        )),
        ("kernels/ops/attention/fa4_sm120/runtime.py", (
            "tile_m, tile_n = 64, 128",
        )),
        ("srt/layers/attention/flashattention_backend.py", (
            'kwargs["rel_bias"] = rel_bias',
        )),
        ("srt/model_executor/runner_backend_utils/breakable_cuda_graph/context.py", (
            "def is_in_breakable_cuda_graph(",
        )),
    ),
    "quant_once": (
        ("kernels/ops/moe/inkling2_experts_sm120.py", (
            "def nvfp4_routed_quantize(x, dst2src, top_k, offsets):",
            "def _nvfp4_outputs(rows, width, experts, device):",
            "scale_rows = triton.cdiv(rows + 127 * experts, 128) * 128",
        )),
        ("kernels/jit/csrc/inkling2/routed_quant.cuh", (
            "(offsets[expert] - offsets[0] + expert * 127) / 128 * 128) * scale_k",
            "((local / 128) * (scale_k / 4) + col / 4) * 512 + (local % 32) * 16 + ((local / 32) % 4) * 4 + col % 4",
        )),
        ("kernels/jit/csrc/inkling2/routed_quant_sm120.cuh", (
            "int64_t(dst2src[row] / top_k) * K",
            "if (row == s_offsets[expert + 1] - 1) zero_expert_tail(",
        )),
        (MOE, ("experts_sm120.nvfp4_routed_quantize(",)),
    ),
    "partial_bf16": (
        ("kernels/ops/moe/inkling2_moe.py", (
            "def _routed_partial(",
            "total += tl.where(inside, x.to(tl.bfloat16).to(tl.float32), 0.0)",
            "tl.store(Y + src * N + cols, total, cols < N)",
            "def _nvfp4_row_scale(",
            "def _row_expert(",
        )),
        (MOE, (
            "def _forward_sm120(self, x, weights, ids, combine):",
            "out = routed_partial(down, offsets, src2dst, top_k, nvfp4_scales=scales).to(",
        )),
    ),
}

STATE = {"installed": False, "levers": (), "applied": []}

# K5 GPU check: the split is bit-equal to the stacked cuBLAS GEMM from 128 rows per
# source up, but not at 64 (cuBLAS picks another kernel for the small k/v GEMMs).
QKVR_MIN_M_FLOOR = 128


def _log(msg):
    sys.stderr.write(f"[inkling_prefill pid={os.getpid()}] {msg}\n")
    sys.stderr.flush()


def parse_levers(value):
    items = [v.strip() for v in value.split(",") if v.strip()]
    if any(v.lower() in ("0", "off", "none", "false", "no") for v in items):
        items = [v for v in items if v.lower() not in ("0", "off", "none", "false", "no")]
    if "all" in items:
        items = list(LEVERS) + [v for v in items if v != "all"]
    levers = tuple(dict.fromkeys(items))
    unknown = [v for v in levers if v not in LEVERS]
    if unknown:
        raise RuntimeError(
            f"inkling_prefill: unknown INKLING_PREFILL entries {unknown}; known: {', '.join(LEVERS)}, all"
        )
    return levers


def _int_env(name, default, lo, hi):
    raw = os.environ.get(name, str(default)).strip()
    try:
        value = int(raw)
    except ValueError as exc:
        raise RuntimeError(f"inkling_prefill: {name}={raw!r} is not an integer") from exc
    if not lo <= value <= hi:
        raise RuntimeError(f"inkling_prefill: {name}={value} must be in [{lo}, {hi}]")
    return value


def config_from_env():
    out = os.environ.get("INKLING_PREFILL_QKVR_OUT", "strided").strip().lower()
    if out != "strided":
        raise RuntimeError(
            f"inkling_prefill: INKLING_PREFILL_QKVR_OUT={out!r} is not supported; only 'strided' "
            "(one buffer, stock row stride) is safe: separate outputs break the fused "
            "attention prologue's shared-stride assertion in decode/verify"
        )
    return {
        "qkvr_split": {"min_m": _int_env("INKLING_PREFILL_QKVR_MIN_M", 128, QKVR_MIN_M_FLOOR, 1 << 20), "out": out},
        "quant_once": {"min_t": _int_env("INKLING_PREFILL_QUANT_MIN_T", 256, 2, 1 << 20)},
    }


def sglang_root():
    spec = importlib.util.find_spec("sglang")  # does not execute sglang
    if spec is None or not spec.submodule_search_locations:
        raise RuntimeError("inkling_prefill: the sglang package is not importable")
    return list(spec.submodule_search_locations)[0]


def check_sources(levers, root=None):
    root = root or sglang_root()
    for lever in levers:
        for rel, needles in _SOURCE_CHECKS[lever]:
            path = os.path.join(root, rel)
            try:
                with open(path, encoding="utf-8") as f:
                    text = f.read()
            except OSError as exc:
                raise RuntimeError(f"inkling_prefill: {lever} patch target {path} is missing") from exc
            missing = [n for n in needles if n not in text]
            if missing:
                raise RuntimeError(f"inkling_prefill: {lever} patch target {path} lacks {missing}")


def _attach(lever, module_name, fn):
    def callback(module):
        fn(module)
        STATE["applied"].append(f"{lever}@{module_name}")
        if os.environ.get("INKLING_PREFILL_VERBOSE"):
            _log(f"applied {lever} to {module_name}")

    _hook.when_imported(module_name, callback)


def _register(levers, cfg):
    if "qkvr_split" in levers:
        from . import qkvr_split

        _attach("qkvr_split", qkvr_split.TARGET_MODULE,
                lambda m: qkvr_split.patch(m, cfg["qkvr_split"]))
    if "bias_layout" in levers:
        from . import bias_layout

        _attach("bias_layout", bias_layout.SM120_MODULE, bias_layout.patch_sm120)
        _attach("bias_layout", bias_layout.COMMON_ATTN_MODULE, bias_layout.patch_common_attn)
        _attach("bias_layout", bias_layout.ATTN_MODULE, bias_layout.patch_attn)
    if "quant_once" in levers:
        from . import quant_once

        _attach("quant_once", quant_once.TARGET_MODULE,
                lambda m: quant_once.patch(m, cfg["quant_once"]))
    if "partial_bf16" in levers:
        from . import partial_bf16

        _attach("partial_bf16", partial_bf16.MODEL_MODULE, partial_bf16.patch)


def describe(levers, cfg):
    parts = []
    for lever in levers:
        if lever == "qkvr_split":
            parts.append("qkvr_split(min_m={min_m},out={out})".format(**cfg["qkvr_split"]))
        elif lever == "quant_once":
            parts.append("quant_once(min_t={min_t})".format(**cfg["quant_once"]))
        else:
            parts.append(lever)
    return " ".join(parts)


def install(levers=None, root=None):
    """Activate the levers named in INKLING_PREFILL (or `levers`). Idempotent."""
    if STATE["installed"]:
        return STATE["levers"]
    levers = parse_levers(os.environ.get("INKLING_PREFILL", "") if levers is None else ",".join(levers))
    if not levers:
        STATE["installed"] = True
        return ()
    cfg = config_from_env()
    check_sources(levers, root)
    _register(levers, cfg)
    STATE.update(installed=True, levers=levers)
    _log("enabled " + describe(levers, cfg))
    return levers
