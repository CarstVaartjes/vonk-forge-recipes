"""Verify the Vonk packaging of the vLLM image at build time."""

from __future__ import annotations

import os
import shutil
from pathlib import Path

if not os.access("/opt/vonk/bin/vllm", os.X_OK):
    raise SystemExit("Vonk vLLM wrapper is not executable")
if shutil.which("vllm") is None:
    raise SystemExit("vllm is not on PATH in the base image")
for name in ("o200k_base.tiktoken", "cl100k_base.tiktoken"):
    if not (Path("/opt/vonk/harmony-reqs") / name).is_file():
        raise SystemExit(f"missing harmony encoding file: {name}")
