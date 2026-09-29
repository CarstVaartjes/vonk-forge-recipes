#!/usr/bin/env python3
"""Run upstream's fail-closed runtime audit, then the pinned vLLM (upstream's entrypoint contract)."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

MODEL = Path("/models")
AUDIT = "/usr/local/bin/audit_runtime.py"

arguments = sys.argv[1:]
if str(MODEL) not in arguments:
    raise SystemExit("the immutable /models checkpoint argument is required")
for name in ("config.json", "model.safetensors.index.json"):
    if not (MODEL / name).is_file():
        raise SystemExit(f"immutable model artifact is missing: {MODEL / name}")

# Upstream's entrypoint admits every launch through the same audit.
subprocess.run([AUDIT], check=True)

vllm = next(
    (
        candidate
        for candidate in (
            "/opt/vllm/bin/vllm",
            "/usr/local/bin/vllm",
            shutil.which("vllm"),
        )
        if candidate and Path(candidate).is_file() and os.access(candidate, os.X_OK)
    ),
    None,
)
if vllm is None:
    raise SystemExit("the pinned vLLM executable is missing")
os.execv(vllm, (vllm, *arguments))
