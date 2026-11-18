"""Verify the Vonk packaging of the vLLM Voxtral Realtime image at build time."""

from __future__ import annotations

import importlib
import importlib.metadata
import os
import shutil

if not os.access("/opt/vonk/bin/vllm", os.X_OK):
    raise SystemExit("Vonk vLLM wrapper is not executable")
if shutil.which("vllm") is None:
    raise SystemExit("vllm is not on PATH in the base image")

# The packages of vLLM's `audio` extra: vLLM decodes uploads with soundfile, then PyAV.
for name in ("av", "scipy", "soundfile", "soxr", "mistral_common"):
    importlib.import_module(name)

# The Voxtral Realtime recipe needs mistral-common >= 1.9.0 (the Voxtral Realtime tokenizer).
installed = importlib.metadata.version("mistral_common")
if tuple(int(part) for part in installed.split(".")[:2]) < (1, 9):
    raise SystemExit(f"mistral_common {installed} is older than 1.9.0")

# vLLM mounts /v1/audio/transcriptions only for a model class that supports transcription, and
# /v1/realtime (WebSocket) only for one that supports realtime.
from vllm.model_executor.models.interfaces import (
    supports_realtime,
    supports_transcription,
)
from vllm.model_executor.models.voxtral_realtime import VoxtralRealtimeGeneration

if not supports_transcription(VoxtralRealtimeGeneration):
    raise SystemExit("VoxtralRealtimeGeneration is not a transcription model")
if not supports_realtime(VoxtralRealtimeGeneration):
    raise SystemExit("VoxtralRealtimeGeneration is not a realtime model")
