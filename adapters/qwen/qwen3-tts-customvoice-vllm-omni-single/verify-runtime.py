"""Verify the Vonk packaging of the vLLM-Omni Qwen3-TTS image at build time."""

from __future__ import annotations

import hashlib
import importlib
import os
import sysconfig
from pathlib import Path

for executable in ("/opt/vonk/bin/vllm", "/usr/local/bin/vllm-omni"):
    if not os.access(executable, os.X_OK):
        raise SystemExit(f"{executable} is not executable")

# The deploy config the recipe passes is the creator's file, and the image bundles
# the same bytes (vllm_omni/deploy/qwen3_tts.yaml at the pinned release).
vendored = Path("/opt/vonk/qwen3_tts.yaml").read_bytes()
bundled = Path(sysconfig.get_paths()["purelib"], "vllm_omni/deploy/qwen3_tts.yaml")
if bundled.read_bytes() != vendored:
    raise SystemExit("the vendored qwen3_tts.yaml differs from the one the image bundles")
print("qwen3_tts.yaml sha256", hashlib.sha256(vendored).hexdigest())

# The server mounts POST /v1/audio/speech through this adapter.
importlib.import_module("vllm_omni.entrypoints.openai.tts_adapters.qwen3_tts")
