"""Verify the Vonk packaging of the Inkling 2 Mini vLLM image at build time."""

from __future__ import annotations

import importlib.util
import os
import shutil
from pathlib import Path

if not os.access("/opt/vonk/bin/vllm", os.X_OK):
    raise SystemExit("Vonk vLLM wrapper is not executable")
if shutil.which("vllm") is None:
    raise SystemExit("vllm is not on PATH in the base image")
spec = importlib.util.find_spec("vllm")
if spec is None or spec.origin is None:
    raise SystemExit("the vllm package is not importable in the base image")
registry = Path(spec.origin).parent / "model_executor" / "models" / "registry.py"
if "Inkling2LoopForConditionalGeneration" not in registry.read_text(encoding="utf-8"):
    raise SystemExit("this vLLM build does not register Inkling 2 Mini")
