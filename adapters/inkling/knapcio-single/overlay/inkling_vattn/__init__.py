"""Inkling-2-Mini DSpark target-verify attention overlay for SGLang on DGX Spark (GB10, sm_121).

    INKLING_VATTN=1                 default levers (cap,bias)
    INKLING_VATTN=cap[,bias]        explicit levers
    unset / empty / 0: install() does nothing and nothing is imported.

cap   Verify attention of every batch size up to INKLING_VATTN_MAX_TOKENS rows (default 384 = 48 requests x 8)
      goes through the compact split-KV relative-bias kernel (the stock path the engine uses up to 32 rows,
      i.e. 4 requests) instead of q / bias padded to 4 heads per KV group + FA4's shear copy + FA4 + gate_sink.
      Only the cap constant changes; prologue, kernel, launch heuristics and epilogue are the stock ones.
bias  In that compact verify path the relative bias is computed inside the attention kernel from the rel-clip
      output r and the layer's rel_logits_proj; the prologue writes r (16 of 128 columns) instead of the
      (tokens, 24, 1024) BF16 bias, and the kernel no longer reads that tensor. Only above
      INKLING_VATTN_BIAS_MIN_TOKENS - 1 = 32 rows by default (5+ requests), where it is measured faster.

Requires SGLANG_OPT_INKLING2_VERIFY_REL_BIAS_KERNEL=1 (the stock switch of the compact verify path).
Decode (one token per request), prefill and draft extend are untouched. Same math as stock; see README.md.
install() checks the patch targets in the installed sglang sources (without importing sglang), registers
post-import hooks and prints one line. Problems raise RuntimeError (sitecustomize then exits with code 78).
"""

import importlib.util
import os
import sys

from . import _hook

LEVERS = ("cap", "bias")
DEFAULT_LEVERS = ("cap", "bias")
DEFAULT_MAX_TOKENS = 384
DEFAULT_BIAS_MIN_TOKENS = 33  # bias lever only above the stock cap (32 rows): measured no gain at 1-4 requests

MODEL_ATTN = "srt/models/inkling_common/inkling2_attn.py"
_SOURCE_CHECKS = (
    (MODEL_ATTN, (
        "VERIFY_REL_BIAS_MAX_TOKENS = 32",
        "and tokens <= VERIFY_REL_BIAS_MAX_TOKENS",
        "envs.SGLANG_OPT_INKLING2_VERIFY_REL_BIAS_KERNEL.get()",
        "from sglang.kernels.ops.attention.inkling2_attn_prologue import attention_prologue",
        "return attention_prologue(",
        "self.padded_group if padded else group,",
        "self.rel_logits_proj.proj,",
        "q, k, v, rel_bias = self._decode_prologue(\n                q, k, v, r, sources, positions, forward_batch, False, verify=verify\n            )",
        "rel_bias_compact_decode=True,",
        "partial(\n                    combine_gate_sink, gate=gate, sinks=sinks\n                ),",
    )),
    ("srt/layers/attention/flashattention_backend.py", (
        "from sglang.kernels.ops.attention.inkling2_attn import rel_bias_partials",
        "elif rel_bias_compact_decode:\n                assert forward_batch.forward_mode.is_target_verify()",
        "                        rel_bias,\n                        layer.scaling,\n                        _fa_out,\n"
        "                        window_size_left=(\n                            window_size[0] if window_size[0] >= 0 else None\n"
        "                        ),\n                        verify=True,",
    )),
    ("kernels/ops/attention/inkling2_attn.py", (
        "def rel_bias_partials(",
        "from sglang.kernels.ops.attention.inkling2_decode_attention_sm90 import (\n        decode_partials,\n    )",
        "def combine_gate_sink(o_partial, m_partial, l_partial, *, gate, sinks):",
    )),
    ("kernels/ops/attention/inkling2_decode_attention_sm90.py", (
        "def _decode_partials(",
        "    tl.extra.cuda.gdc_wait()\n    tl.extra.cuda.gdc_launch_dependents()\n    b = tl.program_id(0) // HEADS_K",
        "stat = (split * batch * QLEN + token) * (HEADS_K * GROUP) + qh",
        "        bias = tl.load(\n            Bias + token[:, None] * SB0 + qh[:, None] * SB1 + dist,\n"
        "            head_mask[:, None] & keep & (dist < EXTENT),\n            other=0,\n        ).to(tl.float32)",
        "        acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)",
    )),
    ("kernels/ops/attention/inkling2_attn_prologue.py", (
        "        r = r.to(Rel.dtype.element_ty)\n        if COMPACT:",
        "steps: tl.constexpr = (REL_EXTENT // 128 + REL_WORKERS - 1) // REL_WORKERS",
        "            rel = tl.dot(r, proj)\n            if COMPACT:\n                tl.store(rel_row + ext[None, :], rel, head_mask & in_extent)",
        "    rel_extent = rel_proj.shape[1]",
        "and kv_heads in (1, 2)",
    )),
)

STATE = {"installed": False, "levers": (), "max_tokens": DEFAULT_MAX_TOKENS, "applied": []}


def _log(msg):
    sys.stderr.write(f"[inkling_vattn pid={os.getpid()}] {msg}\n")
    sys.stderr.flush()


def _on(value):
    return value.strip().lower() not in ("", "0", "off", "none", "false", "no")


def parse_levers(value):
    if not _on(value):
        return ()
    items = [v.strip() for v in value.split(",") if v.strip()]
    if len(items) == 1 and items[0].lower() in ("1", "on", "true", "yes", "all"):
        return DEFAULT_LEVERS if items[0].lower() != "all" else LEVERS
    levers = tuple(dict.fromkeys(items))
    unknown = [v for v in levers if v not in LEVERS]
    if unknown:
        raise RuntimeError(f"inkling_vattn: unknown INKLING_VATTN entries {unknown}; known: {', '.join(LEVERS)}")
    if "bias" in levers and "cap" not in levers:
        # bias also works below the stock cap; allowed.
        pass
    return levers


def sglang_root():
    spec = importlib.util.find_spec("sglang")  # does not execute sglang
    if spec is None or not spec.submodule_search_locations:
        raise RuntimeError("inkling_vattn: the sglang package is not importable")
    return list(spec.submodule_search_locations)[0]


def check_sources(root=None):
    root = root or sglang_root()
    for rel, needles in _SOURCE_CHECKS:
        path = os.path.join(root, rel)
        try:
            with open(path, encoding="utf-8") as f:
                text = f.read()
        except OSError as exc:
            raise RuntimeError(f"inkling_vattn: patch target {path} is missing") from exc
        missing = [n for n in needles if n not in text]
        if missing:
            raise RuntimeError(f"inkling_vattn: patch target {path} lacks {missing}")


def install(levers=None, root=None):
    """Activate the levers named in INKLING_VATTN (or `levers`). Idempotent."""
    if STATE["installed"]:
        return STATE["levers"]
    levers = parse_levers(os.environ.get("INKLING_VATTN", "") if levers is None else ",".join(levers))
    if not levers:
        STATE["installed"] = True
        return ()
    if not _on(os.environ.get("SGLANG_OPT_INKLING2_VERIFY_REL_BIAS_KERNEL", "")):
        raise RuntimeError("inkling_vattn: needs SGLANG_OPT_INKLING2_VERIFY_REL_BIAS_KERNEL=1 (the compact verify path)")
    max_tokens = int(os.environ.get("INKLING_VATTN_MAX_TOKENS", DEFAULT_MAX_TOKENS))
    if max_tokens < 32:
        raise RuntimeError("inkling_vattn: INKLING_VATTN_MAX_TOKENS must be >= 32 (the stock cap)")
    bias_min = int(os.environ.get("INKLING_VATTN_BIAS_MIN_TOKENS", DEFAULT_BIAS_MIN_TOKENS))
    check_sources(root)
    from . import patch

    patch.register(levers, max_tokens, bias_min)
    STATE.update(installed=True, levers=levers, max_tokens=max_tokens)
    _log(f"enabled {','.join(levers)} max_tokens={max_tokens}" + (f" bias_min_tokens={bias_min}" if "bias" in levers else ""))
    return levers
