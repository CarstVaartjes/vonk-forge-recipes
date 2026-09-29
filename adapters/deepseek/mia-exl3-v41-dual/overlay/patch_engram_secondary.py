#!/usr/bin/env python3
"""Stop vLLM from loading Engram embed tables as 47 GiB pinned Parameters.

The EXL3 tree has no layers.{1,14}.engram.embed.{weight,scale}. Those FP8
rows stay in native shards 47+48. On 2× GB10, pinning both tables (~94 GiB)
plus EXL3 weights OOMs during ParallelEngramEmbedding.__init__.

File-backed lookup (engram_file_backend + row_store) opens the shards itself.
This patch:
  * does not register secondary_weights (so the loader never iterates 95 GiB)
  * drops leftover 47/48 iterators if a previous overlay left them in place
  * reverts the old dsv41-engram-secondary insertion
"""
from __future__ import annotations

from pathlib import Path
import sys

MARK = "dsv41-engram-file"
OLD_MARK = "dsv41-engram-secondary"

LLM_OLD = "        self.set_moe_parameters()\n"

LLM_NEW = """        self.set_moe_parameters()
        # [dsv41-engram-file] n-gram tables are file-backed (row_store); do not
        # register secondary_weights that pin 47 GiB/layer on Spark UMA.
"""

PREV_LLM_NEW = """        self.set_moe_parameters()
        # [dsv41-engram-secondary] native FP8 n-gram tables live beside EXL3.
        _engram_dir = getattr(config, "engram_table_dir", None) or ""
        if _engram_dir:
            from vllm.model_executor.model_loader.default_loader import (
                DefaultModelLoader,
            )
            self.secondary_weights = [
                DefaultModelLoader.Source(
                    model_or_path=_engram_dir,
                    revision=None,
                    prefix="",
                    fall_back_to_pt=False,
                    allow_patterns_overrides=[
                        "model-00047-of-00048.safetensors",
                        "model-00048-of-00048.safetensors",
                    ],
                )
            ]
"""

VL_OLD = "        self.language_model.hf_to_vllm_mapper = WeightsMapper()\n"

VL_NEW = """        self.language_model.hf_to_vllm_mapper = WeightsMapper()
        # [dsv41-engram-file] embed tables are not loaded as Parameters.
"""

PREV_VL_NEW = """        self.language_model.hf_to_vllm_mapper = WeightsMapper()
        # [dsv41-engram-secondary] top-level model is what get_all_weights sees.
        _engram_dir = getattr(config, "engram_table_dir", None) or ""
        if _engram_dir:
            from vllm.model_executor.model_loader.default_loader import (
                DefaultModelLoader,
            )
            self.secondary_weights = [
                DefaultModelLoader.Source(
                    model_or_path=_engram_dir,
                    revision=None,
                    prefix="",
                    fall_back_to_pt=False,
                    allow_patterns_overrides=[
                        "model-00047-of-00048.safetensors",
                        "model-00048-of-00048.safetensors",
                    ],
                )
            ]
"""

LOADER_OLD = (
    "        return ((source.prefix + name, tensor) "
    "for (name, tensor) in weights_iterator)\n"
)

LOADER_NEW = """        names_tensors = (
            (source.prefix + name, tensor) for (name, tensor) in weights_iterator
        )
        # [dsv41-engram-file] never materialize embed tables from shards 47/48.
        _pats = source.allow_patterns_overrides or []
        if any("00047-of-00048" in str(_p) or "00048-of-00048" in str(_p) for _p in _pats):
            names_tensors = ((n, t) for (n, t) in names_tensors if False)
        return names_tensors
"""

PREV_LOADER_NEW = """        names_tensors = (
            (source.prefix + name, tensor) for (name, tensor) in weights_iterator
        )
        # [dsv41-engram-secondary] shards 47/48 also hold q/k/wkv; keep embed.
        _pats = source.allow_patterns_overrides or []
        if any("00047-of-00048" in str(_p) for _p in _pats):
            names_tensors = (
                (n, t) for (n, t) in names_tensors if ".engram.embed." in n
            )
        return names_tensors
"""


def apply_text(text: str, old: str, new: str, mark: str = MARK) -> tuple[str, str]:
    if mark in text:
        return text, "skipped"
    if text.count(old) != 1:
        return text, f"missing:{old[:40]!r} count={text.count(old)}"
    return text.replace(old, new, 1), "applied"


def migrate_text(text: str, prev: str, new: str) -> tuple[str, str]:
    if MARK in text and OLD_MARK not in text:
        return text, "skipped"
    if prev in text:
        return text.replace(prev, new, 1), "migrated"
    return text, "no-prev"


def _vllm_root() -> Path | None:
    try:
        import vllm

        return Path(vllm.__file__).resolve().parent
    except Exception:
        return None


def main() -> int:
    root = _vllm_root()
    if root is None:
        print("WARN: vllm not importable; skip engram patch", file=sys.stderr)
        return 0
    jobs = (
        (
            root / "models/deepseek_v4_1/nvidia/model.py",
            LLM_OLD,
            LLM_NEW,
            PREV_LLM_NEW,
            "llm",
        ),
        (
            root / "models/deepseek_v4_1/nvidia/vl_model.py",
            VL_OLD,
            VL_NEW,
            PREV_VL_NEW,
            "vl",
        ),
        (
            root / "model_executor/model_loader/default_loader.py",
            LOADER_OLD,
            LOADER_NEW,
            PREV_LOADER_NEW,
            "loader",
        ),
    )
    rc = 0
    for path, old, new, prev, label in jobs:
        if not path.is_file():
            print(f"WARN: {label} missing ({path})", file=sys.stderr)
            continue
        text = path.read_text()
        out, status = migrate_text(text, prev, new)
        if status == "no-prev":
            out, status = apply_text(text, old, new)
        if status in ("applied", "migrated"):
            path.write_text(out)
            print(f"engram file-backend: {status} {label} ({path})")
        elif status == "skipped":
            print(f"engram file-backend: already applied {label}")
        else:
            print(f"WARN: engram file-backend {label}: {status}", file=sys.stderr)
            rc = 1
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
