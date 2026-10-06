"""Verify the Vonk packaging of the vLLM image at build time."""

from __future__ import annotations

import os
import shutil

if not os.access("/opt/vonk/bin/vllm", os.X_OK):
    raise SystemExit("Vonk vLLM wrapper is not executable")
if shutil.which("vllm") is None:
    raise SystemExit("vllm is not on PATH in the base image")
if shutil.which("ray") is None:
    raise SystemExit("ray is not on PATH in the base image")
