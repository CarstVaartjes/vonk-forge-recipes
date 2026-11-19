"""Verify the Hy4 flash dual-Spark adapter packaging at build time."""

from __future__ import annotations

import importlib
import os
from pathlib import Path

SITE = Path("/usr/local/lib/python3.12/dist-packages/vllm/models")
PATCHED = (
    SITE / "hy_v4_flash/nvidia/engram.py",
    SITE / "hy_v4_flash/nvidia/attention.py",
    SITE / "hy_v4_flash/nvidia/sm120.py",
    SITE / "hy_v4/nvidia/attention.py",
)
WRAPPER = Path("/opt/vonk/bin/vllm")
LAUNCHER = Path("/usr/local/bin/vllm")


def main() -> None:
    for path in (*PATCHED, WRAPPER, LAUNCHER):
        if not path.is_file():
            raise SystemExit(f"required Hy4 flash runtime file is missing: {path}")
    for path in (WRAPPER, LAUNCHER):
        if not os.access(path, os.X_OK):
            raise SystemExit(f"not executable: {path}")
    for module in (
        "vllm.models.hy_v4_flash.nvidia.sm120",
        "vllm.models.hy_v4_flash.nvidia.engram",
    ):
        importlib.import_module(module)


if __name__ == "__main__":
    main()
