"""Verify the Vonk packaging of the vLLM audio image at build time."""

from __future__ import annotations

import importlib
import os
import shutil

if not os.access("/opt/vonk/bin/vllm", os.X_OK):
    raise SystemExit("Vonk vLLM wrapper is not executable")
if shutil.which("vllm") is None:
    raise SystemExit("vllm is not on PATH in the base image")

# The packages of vLLM's `audio` extra: vLLM decodes uploads with soundfile, then PyAV.
for name in ("av", "scipy", "soundfile", "soxr", "mistral_common"):
    importlib.import_module(name)

# vLLM mounts /v1/audio/transcriptions only for a model class that supports transcription.
from vllm.model_executor.models.interfaces import supports_transcription
from vllm.model_executor.models.qwen3_asr import Qwen3ASRForConditionalGeneration

if not supports_transcription(Qwen3ASRForConditionalGeneration):
    raise SystemExit("Qwen3ASRForConditionalGeneration is not a transcription model")
