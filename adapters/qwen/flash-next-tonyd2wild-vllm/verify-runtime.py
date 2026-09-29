#!/usr/bin/env python3
"""Check the image packaging at build time (ours; upstream features are not tested here)."""

from __future__ import annotations

import ast
import sys
from pathlib import Path

PKG = Path("/usr/local/lib/python3.12/dist-packages/vllm")
NVIDIA = "models/qwen4_exp/nvidia"
# Files the upstream overlays replace: they must exist in the pinned base image.
BASE = (
    f"{NVIDIA}/ple_layer.py",
    f"{NVIDIA}/model_state.py",
    f"{NVIDIA}/mtp.py",
    f"{NVIDIA}/qsa.py",
    f"{NVIDIA}/ops/ple.py",
    f"{NVIDIA}/ops/qsa.py",
    "config/compilation.py",
    "platforms/interface.py",
    "model_executor/layers/quantization/modelopt.py",
)
# Files that must exist after the overlays are copied in, with a marker each.
OVERLAYS = {
    f"{NVIDIA}/ple_layer.py": "QWEN4EXP_PLE_STAGED",
    f"{NVIDIA}/ops/ple_mmap.py": "QWEN4EXP_PLE_MMAP_CHUNK",
    f"{NVIDIA}/model_state.py": "QWEN4EXP_PLE_STAGED",
    f"{NVIDIA}/mtp.py": "QWEN4EXP_DRAFT_VOCAB",
    "config/compilation.py": "qwen4_exp_ple_mmap_gather",
}


def main() -> None:
    stage = sys.argv[1] if len(sys.argv) == 2 else ""
    if stage == "base":
        for name in BASE:
            if not (PKG / name).is_file():
                raise SystemExit(f"pinned vLLM image is missing {PKG / name}")
    elif stage == "overlays":
        for name, marker in OVERLAYS.items():
            path = PKG / name
            text = path.read_text(encoding="utf-8")
            if marker not in text:
                raise SystemExit(f"overlay is missing or wrong: {path}")
            ast.parse(text, filename=str(path))
        for name in BASE:
            ast.parse((PKG / name).read_text(encoding="utf-8"), filename=name)
        if not Path("/opt/vonk/bin/qwen38-tonyd2wild-serve").is_file():
            raise SystemExit("Controller wrapper is missing")
    else:
        raise SystemExit("usage: verify-runtime.py base|overlays")


if __name__ == "__main__":
    main()
