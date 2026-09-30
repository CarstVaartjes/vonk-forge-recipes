"""Verify the Vonk packaging of the SGLang image at build time."""

from __future__ import annotations

import importlib.util
import os
import importlib

if not os.access("/opt/vonk/bin/sglang-serve", os.X_OK):
    raise SystemExit("Vonk sglang-serve wrapper is not executable")
if importlib.util.find_spec("sglang") is None:
    raise SystemExit("sglang is not importable in the base image")
try:
    importlib.import_module("numpy")
    importlib.import_module("librosa")
except ImportError as exc:
    raise SystemExit(f"SGLang audio dependencies are not importable: {exc}") from exc
