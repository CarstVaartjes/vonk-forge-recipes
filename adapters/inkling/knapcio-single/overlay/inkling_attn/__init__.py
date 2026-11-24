"""Inkling-2-Mini prefill attention overlay for SGLang on DGX Spark (GB10, sm_121).

    INKLING_ATTN=prefill_swa[,prefill_global]   (or "all")
    unset / empty: install() does nothing and nothing is imported.

prefill_swa     sliding-window layers (40 of the 48 layer visits) at prefill / extend: one CUDA kernel computes
                the relative-position bias inside the attention mainloop (from r [t, 24, 16] and proj [16, 1024])
                instead of the stock einsum -> [t, 24, 1024] BF16 tensor -> FA4 shear copy -> FA4 with bias.
prefill_global  the same for the 8 global (causal) visits; the bias is non-zero only within 1024 keys of the
                diagonal, so most key tiles of a long prefix run without bias work.

Decode, target verify and draft extend are untouched. Same math as stock (BF16-rounded bias, FP32 scores,
BF16 P, FP32 output and LSE); differences are FP32 accumulation order only (see README.md for the measured
error against an FP32 reference, side by side with stock FA4's).

install() parses the env, checks the patch targets in the installed sglang sources (without importing sglang),
registers post-import hooks and prints one line. Problems raise RuntimeError (the combined sitecustomize then
exits with code 78). The CUDA extension is built (or reused from INKLING_ATTN_CACHE, default
~/.cache/inkling-attn) when the Inkling attention module is imported, before any CUDA graph capture.
INKLING_ATTN_VERBOSE=1 logs each applied patch.
"""

import importlib.util
import os
import sys

from . import _hook

LEVERS = ("prefill_swa", "prefill_global")

ATTN = "srt/models/inkling_common/inkling2_attn.py"

_SOURCE_CHECKS = (
    (ATTN, (
        "class Inkling2Attention(",
        "def _forward(self, hidden_states, positions, forward_batch, *, attn, attn_padded):",
        "rel_bias = self.rel_logits_proj(r)",
        "attention_kwargs = dict(rel_bias=rel_bias, **bias_kwargs)",
        "from sglang.srt.model_executor.forward_context import get_attn_backend",
        "from sglang.srt.runtime_context import get_model, get_parallel, get_platform",
        "self.rel_extent = config.sliding_window_size if is_local else config.rel_extent",
        "sliding_window_size=config.sliding_window_size - 1 if is_local else -1",
        "out_dtype=torch.float32",
        "pack_gqa=False",
        "return_lse=True,",
        "r = self.rel_l2_clip(",
    )),
    ("srt/models/inkling_common/attn.py", (
        "class RelLogitsProj(",
        "log_scaling_tau: torch.Tensor | None = None",
        "self.proj = nn.Parameter(torch.empty(d_rel, rel_extent), requires_grad=False)",
        'return torch.einsum("thd,de->the", r, self.proj)',
    )),
    ("kernels/ops/attention/flash_attention_v4_sm120.py", (
        "_flash_attn_varlen_func,",
        "result = _flash_attn_varlen_func(",
        "seqused_k=cache_seqlens,",
        'rel_bias_kwargs["rel_bias"] = rel_bias',
        "if window_size == (-1, -1):",
    )),
    ("srt/layers/attention/flashattention_backend.py", (
        'kwargs["rel_bias"] = rel_bias',
        "softmax_lse.T",
        "cache_seqlens=cache_seqlens,",
    )),
    ("kernels/ops/attention/fa4_sm120/flash_fwd.py", (
        "acc_S[i] = acc_S[i] * base_softmax_scale",
        "sBias[row, col, consumer_state.index].to(",
    )),
    ("kernels/ops/attention/flash_attn/cute/shearing_bias.py", (
        "-Float32.inf if const_expr(window_size_left is not None) else 0.0",
    )),
    ("kernels/ops/attention/flash_attn/cute/interface.py", (
        "else (num_head, total_q)",
    )),
    ("srt/model_executor/runner_backend_utils/breakable_cuda_graph/context.py", (
        "def is_in_breakable_cuda_graph(",
    )),
    ("srt/layers/cp/utils.py", (
        "def is_cp_active(forward_batch) -> bool:",
    )),
    ("srt/layers/attention/flashattention_backend.py", (
        "if device_capability[0] == 12:",
        "from sglang.kernels.ops.attention.flash_attention_v4_sm120 import (",
    )),
)

STATE = {"installed": False, "levers": (), "applied": []}


def _log(msg):
    sys.stderr.write(f"[inkling_attn pid={os.getpid()}] {msg}\n")
    sys.stderr.flush()


def parse_levers(value):
    items = [v.strip() for v in value.split(",") if v.strip()]
    items = [v for v in items if v.lower() not in ("0", "off", "none", "false", "no")]
    if "all" in items:
        items = list(LEVERS) + [v for v in items if v != "all"]
    levers = tuple(dict.fromkeys(items))
    unknown = [v for v in levers if v not in LEVERS]
    if unknown:
        raise RuntimeError(f"inkling_attn: unknown INKLING_ATTN entries {unknown}; known: {', '.join(LEVERS)}, all")
    return levers


def sglang_root():
    spec = importlib.util.find_spec("sglang")  # does not execute sglang
    if spec is None or not spec.submodule_search_locations:
        raise RuntimeError("inkling_attn: the sglang package is not importable")
    return list(spec.submodule_search_locations)[0]


def check_sources(root=None):
    root = root or sglang_root()
    for rel, needles in _SOURCE_CHECKS:
        path = os.path.join(root, rel)
        try:
            with open(path, encoding="utf-8") as f:
                text = f.read()
        except OSError as exc:
            raise RuntimeError(f"inkling_attn: patch target {path} is missing") from exc
        missing = [n for n in needles if n not in text]
        if missing:
            raise RuntimeError(f"inkling_attn: patch target {path} lacks {missing}")


def _attach(module_name, fn):
    def callback(module):
        fn(module)
        STATE["applied"].append(module_name)
        if os.environ.get("INKLING_ATTN_VERBOSE"):
            _log(f"patched {module_name}")

    _hook.when_imported(module_name, callback)


def _register(levers):
    from . import patch

    kinds = tuple(lv.split("_", 1)[1] for lv in levers)  # "swa" / "global"

    def attn_cb(module):
        patch.patch_attn(module, kinds)
        if not os.environ.get("INKLING_ATTN_LAZY_BUILD"):
            from . import build

            build.load()
            info = build.INFO
            _log(f"extension {'reused' if info['cached'] else 'built'} in {info['seconds']:.1f} s ({info['tag']})")

    _attach(patch.SM120_MODULE, patch.patch_sm120)
    _attach(patch.COMMON_ATTN_MODULE, patch.patch_common_attn)
    _attach(patch.ATTN_MODULE, attn_cb)


def install(levers=None, root=None):
    """Activate the levers named in INKLING_ATTN (or `levers`). Idempotent."""
    if STATE["installed"]:
        return STATE["levers"]
    levers = parse_levers(os.environ.get("INKLING_ATTN", "") if levers is None else ",".join(levers))
    if not levers:
        STATE["installed"] = True
        return ()
    check_sources(root)
    _register(levers)
    STATE.update(installed=True, levers=levers)
    _log("enabled " + " ".join(levers))
    return levers
