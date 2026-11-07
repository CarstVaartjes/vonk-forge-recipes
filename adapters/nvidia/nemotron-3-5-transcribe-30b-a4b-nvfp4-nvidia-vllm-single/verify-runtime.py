"""Verify the Vonk packaging of the NeMo Speech vLLM image at build time."""

from __future__ import annotations

import importlib
import importlib.metadata
import os
import shutil

if not os.access("/opt/vonk/bin/vllm", os.X_OK):
    raise SystemExit("Vonk vLLM wrapper is not executable")
if shutil.which("vllm") is None:
    raise SystemExit("vllm is not on PATH in the base image")

# The model card names the plugin vLLM must load (VLLM_PLUGINS=nemo_speechlm).
plugins = importlib.metadata.entry_points(group="vllm.general_plugins")
registered = [entry for entry in plugins if entry.name == "nemo_speechlm"]
if not registered:
    raise SystemExit("the nemo_speechlm vLLM plugin is not registered")
registered[0].load()

for name in ("av", "cv2", "soundfile", "librosa", "lhotse", "peft", "hydra", "omegaconf"):
    importlib.import_module(name)
# The word-timestamp middleware named by the model card's serve command.
importlib.import_module("nemo.collections.speechlm2.vllm.salm.ctc_serving")
