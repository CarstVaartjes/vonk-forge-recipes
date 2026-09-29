#!/usr/bin/env python3
"""Stage vLLM's stock weight copies through pinned memory (GB10 UMA).

safetensors 0.8 maps every shard MAP_PRIVATE|PROT_WRITE. A device copy that
reads straight from that mapping makes the GB10 driver pin the pages with
write intent, which copy-on-writes them into anonymous host memory for as long
as the mapping lives. vLLM's DeepseekV41 VL wrapper sorts the whole weight
stream first, so all 39 mappings live for the entire load and the copies
accumulate (~1.2 GiB per layer measured; both Sparks wedged 2026-09-11/12).

The EXL3 overlay stages its own copies (overlay/exl3.py h2d_copy). This patch
covers the stock loaders this model still uses: default_weight_loader (norms,
router gates, Engram q/k, heads), VocabParallelEmbedding (embed_tokens), the
linear layers (vision tower, indexer weights_proj), the attn_sink loader and
the DSpark draft's narrow loader. Each `X.copy_(Y)` becomes
`dsv41_h2d_copy(X, Y)`, which memcpy's a CPU source into a reusable pinned
buffer and DMAs from there; device or CPU->CPU copies are unchanged.
"""
from __future__ import annotations

from pathlib import Path
import sys

MARK = "dsv41-h2d-stage"

HELPER = '''
# [dsv41-h2d-stage]
_DSV41_PIN_POOL: dict = {}


def dsv41_h2d_copy(dest, src):
    """dest.copy_(src) that never DMAs from a (mmap-backed) CPU tensor."""
    if src.device.type != "cpu" or dest.device.type == "cpu" or src.numel() == 0:
        dest.copy_(src)
        return
    if src.dtype != dest.dtype:
        src = src.to(dtype=dest.dtype)
    numel = src.numel()
    buf = _DSV41_PIN_POOL.get(src.dtype)
    if buf is None or buf.numel() < numel:
        want = max(numel, 1 << 20, 0 if buf is None else buf.numel() * 2)
        buf = torch.empty(want, dtype=src.dtype, device="cpu", pin_memory=True)
        _DSV41_PIN_POOL[src.dtype] = buf
    stage = buf[:numel].view(src.shape)
    stage.copy_(src)
    dest.copy_(stage)
'''

SHIM = '''
# [dsv41-h2d-stage] lazy import: weight_utils imports the layers package.
def _dsv41_h2d_copy(dest, src):
    from vllm.model_executor.model_loader.weight_utils import dsv41_h2d_copy

    return dsv41_h2d_copy(dest, src)
'''

# (relative path, [(old, new), ...]); every old must occur exactly the given
# number of times (count) so a vLLM drift is loud, not silent.
JOBS = [
    (
        "model_executor/model_loader/weight_utils.py",
        [
            ("            param.data.copy_(loaded_weight.view(param.shape))\n",
             "            dsv41_h2d_copy(param.data, loaded_weight.view(param.shape))\n", 1),
            ("            param.data.copy_(loaded_weight)\n    except Exception:\n",
             "            dsv41_h2d_copy(param.data, loaded_weight)\n    except Exception:\n", 1),
        ],
        HELPER,
    ),
    (
        "model_executor/layers/linear.py",
        [
            # newline-anchored so the 8-space form does not also match the 16-space lines
            ("\n        param.data.copy_(loaded_weight)\n", "\n        _dsv41_h2d_copy(param.data, loaded_weight)\n", 1),
            ("\n        param_data.copy_(loaded_weight)\n", "\n        _dsv41_h2d_copy(param_data, loaded_weight)\n", 5),
            ("\n                param_data.copy_(loaded_weight)\n", "\n                _dsv41_h2d_copy(param_data, loaded_weight)\n", 2),
        ],
        SHIM,
    ),
    (
        "model_executor/layers/vocab_parallel_embedding.py",
        [
            ("            param.data.copy_(loaded_weight)\n", "            _dsv41_h2d_copy(param.data, loaded_weight)\n", 1),
            ("        param[: loaded_weight.shape[0]].data.copy_(loaded_weight)\n",
             "        _dsv41_h2d_copy(param[: loaded_weight.shape[0]].data, loaded_weight)\n", 1),
        ],
        SHIM,
    ),
    (
        "models/deepseek_v4_1/nvidia/model.py",
        [
            ("                    params_dict[name][:n].copy_(narrow_weight)\n",
             "                    _dsv41_h2d_copy(params_dict[name][:n], narrow_weight)\n", 1),
        ],
        SHIM,
    ),
    (
        "models/deepseek_v4_1/nvidia/dspark.py",
        [
            ("                    params_dict[name][: narrow.shape[0]].copy_(narrow)\n",
             "                    _dsv41_h2d_copy(params_dict[name][: narrow.shape[0]], narrow)\n", 1),
        ],
        SHIM,
    ),
]


def apply(text: str, edits, footer: str) -> tuple[str, str]:
    if MARK in text:
        return text, "skipped"
    for old, new, count in edits:
        have = text.count(old)
        if have != count:
            return text, f"missing:{old.strip()[:50]!r} count={have} want={count}"
    for old, new, _ in edits:
        text = text.replace(old, new)
    if not text.endswith("\n"):
        text += "\n"
    return text + footer, "applied"


def main() -> int:
    try:
        import vllm

        root = Path(vllm.__file__).resolve().parent
    except Exception as exc:
        print(f"WARN: vllm not importable; skip h2d stage patch ({exc})", file=sys.stderr)
        return 0
    rc = 0
    for rel, edits, footer in JOBS:
        path = root / rel
        if not path.is_file():
            print(f"WARN: h2d stage: {rel} missing", file=sys.stderr)
            rc = 1
            continue
        out, status = apply(path.read_text(), edits, footer)
        if status == "applied":
            path.write_text(out)
            print(f"h2d stage: applied ({rel})")
        elif status == "skipped":
            print(f"h2d stage: already applied ({rel})")
        else:
            print(f"WARN: h2d stage {rel}: {status}", file=sys.stderr)
            rc = 1
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
