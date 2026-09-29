"""Verify the MiMo overlays and audio libraries at image build time."""

from __future__ import annotations

import importlib
import os
from pathlib import Path

SITE = Path("/usr/local/lib/python3.12/dist-packages/vllm")
MARKERS = {
    SITE / "model_executor/models/mimo_v2.py": "_shard_fp8_qkv_proj",
    SITE / "model_executor/models/mimo_v2_omni.py": "SupportsEagle3",
    SITE / "v1/attention/backends/triton_attn_diffkv.py": "fp8",
}
WRAPPER = Path("/opt/vonk/bin/vllm")


def main() -> None:
    for path, marker in MARKERS.items():
        if not path.is_file() or marker not in path.read_text(encoding="utf-8"):
            raise SystemExit(f"MiMo overlay is missing or wrong: {path}")
    if not os.access(WRAPPER, os.X_OK):
        raise SystemExit("Controller vLLM wrapper is not executable")

    for name in ("av", "soundfile"):
        importlib.import_module(name)


if __name__ == "__main__":
    main()
