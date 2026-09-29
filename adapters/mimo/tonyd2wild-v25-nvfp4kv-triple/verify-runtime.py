"""Verify the MiMo V2.5 TP3 mods and Ray runtime at image build time."""

from __future__ import annotations

import importlib
import os
from pathlib import Path

SITE = Path("/usr/local/lib/python3.12/dist-packages/vllm")
MARKERS = {
    SITE / "v1/attention/backends/triton_attn_diffkv.py": "nvfp4",
    SITE / "v1/attention/ops/wmma_decode.py": "wmma",
    SITE / "model_executor/layers/attention/attention.py": "nvfp4-kv-diffkv: live dtype",
    SITE / "model_executor/virtual_tp.py": "pad_or_narrow_weight",
}
WRAPPER = Path("/opt/vonk/bin/vllm")


def main() -> None:
    for path, marker in MARKERS.items():
        if not path.is_file() or marker not in path.read_text(encoding="utf-8"):
            raise SystemExit(f"MiMo V2.5 mod is missing or wrong: {path}")
    if not os.access(WRAPPER, os.X_OK):
        raise SystemExit("Controller vLLM wrapper is not executable")
    importlib.import_module("ray")


if __name__ == "__main__":
    main()
