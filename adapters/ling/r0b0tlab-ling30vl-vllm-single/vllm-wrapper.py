#!/usr/bin/env python3
"""Launch the pinned vLLM for this recipe; every engine option comes from the recipe."""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

MODEL = Path("/models")
arguments = sys.argv[1:]
if str(MODEL) not in arguments:
    raise SystemExit(f"the immutable {MODEL} checkpoint argument is required")
if not (MODEL / "config.json").is_file():
    raise SystemExit(f"immutable model artifact is missing: {MODEL / 'config.json'}")
vllm = next(
    (
        candidate
        for candidate in ("/usr/local/bin/vllm", "/opt/vllm/.venv/bin/vllm", shutil.which("vllm"))
        if candidate and Path(candidate).is_file() and os.access(candidate, os.X_OK)
    ),
    None,
)
if vllm is None:
    raise SystemExit("the pinned vLLM executable is missing")
os.execv(vllm, (vllm, *arguments))
