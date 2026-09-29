"""Verify the Ray, b12x and NCCL layers of the M3 image at build time."""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path

NCCL = Path("/opt/nccl230/build/lib/libnccl.so.2")
WRAPPER = Path("/opt/vonk/bin/vllm")
VLLM = Path("/opt/venv/bin/vllm")
RAY = Path("/opt/venv/bin/ray")
PIP_NCCL = Path(
    "/opt/venv/lib/python3.12/site-packages/nvidia/nccl/lib/libnccl.so.2"
)


def main() -> None:
    if not NCCL.is_file() or NCCL.read_bytes()[:4] != b"\x7fELF":
        raise SystemExit(f"NCCL v2.30u1 library is missing or invalid: {NCCL}")
    if PIP_NCCL.resolve() != NCCL.resolve():
        raise SystemExit("the pip NCCL library is not re-pointed to v2.30u1")
    for path in (WRAPPER, VLLM, RAY):
        if not os.access(path, os.X_OK):
            raise SystemExit(f"required executable is missing: {path}")
    if importlib.util.find_spec("ray") is None:
        raise SystemExit("ray is not installed")
    scratch = importlib.util.find_spec("b12x.integration.paged_attention_scratch")
    if scratch is None or scratch.origin is None:
        raise SystemExit("b12x is not installed")
    if "copy_runtime_metadata" not in Path(scratch.origin).read_text(encoding="utf-8"):
        raise SystemExit("b12x lacks copy_runtime_metadata")


if __name__ == "__main__":
    main()
