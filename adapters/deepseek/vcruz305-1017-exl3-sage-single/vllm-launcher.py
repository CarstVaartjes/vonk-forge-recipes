#!/usr/bin/env python3
"""Controller launcher for vcruz305's one-Spark DeepSeek-V4.1-Flash-1017 SAGE EXL3 1.62 bpw lane.

The engine arguments are the recipe's. Before exec it makes the engine's cache directories and builds the
cold-expert ATS alias side file once, as the kit's scripts/build_cold_alias.sh does: 1,418 of the coldest routed
experts and the embedding, gathered into one 64-byte-aligned file the engine maps read-only. The side file lives in
the persistent cache, so a restart of the same launch finds it; a manifest that differs from the shipped selection
is rebuilt in place, as the kit's serve.sh requires.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

MODEL = Path(os.environ.get("VLLM_EXL3_MODEL_DIR", "/models"))
ALIAS = Path(os.environ.get("VLLM_EXL3_ATS_ALIAS_DIR", "/outputs/cache/alias"))
TOOLS = Path("/opt/vonk/tools")
SIZE = 9228943360  # experts.bin of the shipped selection (kit README)


def selection_matches(selection: dict[str, list[int]]) -> bool:
    try:
        have = json.loads((ALIAS / "manifest.json").read_text()).get("experts") or {}
        size = (ALIAS / "experts.bin").stat().st_size
    except (OSError, ValueError):
        return False

    def norm(experts: dict[str, list[int]]) -> dict[str, list[int]]:
        return {key: sorted(int(item) for item in value) for key, value in experts.items()}

    return size == SIZE and norm(have) == norm(selection)


def main() -> None:
    for name in (
        "HOME",
        "XDG_CACHE_HOME",
        "VLLM_CACHE_ROOT",
        "TRITON_CACHE_DIR",
        "CUTE_DSL_CACHE_DIR",
        "B12X_COMPILE_CACHE_DIR",
    ):
        if os.environ.get(name):
            Path(os.environ[name]).mkdir(parents=True, exist_ok=True)
    for path in (MODEL / "config.json", MODEL / "model.safetensors.index.json"):
        if not path.is_file():
            raise SystemExit(f"immutable model artifact is missing: {path}")
    selection = json.loads((TOOLS / "cold-alias-selection.json").read_text())
    if not selection_matches(selection):
        ALIAS.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            [
                sys.executable,
                str(TOOLS / "ats_build_alias.py"),
                "--model-dir",
                str(MODEL),
                "--out",
                str(ALIAS),
                "--experts-json",
                str(TOOLS / "cold-alias-selection.json"),
                "--embed",
                "--align",
                "64",
            ],
            check=True,
        )
    args = sys.argv[1:]
    for index, arg in enumerate(args):
        if arg == "--max-model-len" and index + 1 < len(args):
            os.environ.setdefault("VLLM_DS41_ROPE_TRIM_MAX_POS", args[index + 1])
        elif arg.startswith("--max-model-len="):
            os.environ.setdefault("VLLM_DS41_ROPE_TRIM_MAX_POS", arg.split("=", 1)[1])
    os.execvp("vllm", ["vllm", *sys.argv[1:]])


if __name__ == "__main__":
    main()
