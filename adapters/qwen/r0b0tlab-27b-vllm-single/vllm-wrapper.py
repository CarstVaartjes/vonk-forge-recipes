#!/usr/bin/env python3
"""Launch the pinned vLLM for the r0b0tlab Qwen3.8 27B NVFP4 single-Spark profiles."""

from __future__ import annotations

import os
import sys
from pathlib import Path

TARGET = Path("/models/target")
DRAFTER = Path("/models/drafter")

arguments = sys.argv[1:]
if str(TARGET) not in arguments:
    raise SystemExit("the immutable /models/target checkpoint argument is required")

for path in (
    TARGET / "config.json",
    TARGET / "model.safetensors.index.json",
    TARGET / "hf_quant_config.json",
):
    if not path.is_file():
        raise SystemExit(f"immutable model artifact is missing: {path}")

# The DFlash2 option mounts its draft next to the target; only require it when
# the launch actually names it.
if any(str(DRAFTER) in argument for argument in arguments):
    for name in ("config.json", "model.safetensors"):
        if not (DRAFTER / name).is_file():
            raise SystemExit(f"immutable model artifact is missing: {DRAFTER / name}")

vllm = next(
    (
        candidate
        for candidate in ("/usr/local/bin/vllm", "/opt/vllm/.venv/bin/vllm")
        if Path(candidate).is_file() and os.access(candidate, os.X_OK)
    ),
    None,
)
if vllm is None:
    raise SystemExit("the pinned vLLM executable is missing")
os.execv(vllm, (vllm, *arguments))
