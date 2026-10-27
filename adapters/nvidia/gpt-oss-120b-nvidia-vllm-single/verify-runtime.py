"""Verify the Vonk packaging of the vLLM image at build time."""

from __future__ import annotations

import os
import shutil
from importlib import metadata
from pathlib import Path

if not os.access("/opt/vonk/bin/vllm", os.X_OK):
    raise SystemExit("Vonk vLLM wrapper is not executable")
if shutil.which("vllm") is None:
    raise SystemExit("vllm is not on PATH in the base image")
for name in ("o200k_base.tiktoken", "cl100k_base.tiktoken"):
    if not (Path("/opt/vonk/harmony-reqs") / name).is_file():
        raise SystemExit(f"missing harmony encoding file: {name}")
# On vLLM 0.21 (NGC 26.05.post1) the MXFP4 Marlin load exhausted an idle
# Spark's unified memory on 2026-10-21 and, with #47851 backported, on
# 2026-10-27. Upstream validated gpt-oss-120b on DGX Spark from v0.26.0
# (vllm-project/vllm#43969), so refuse an older base.
release = metadata.version("vllm").split("+")[0].split(".")
if (int(release[0]), int(release[1])) < (0, 26):
    raise SystemExit(f"vLLM {metadata.version('vllm')} predates 0.26.0")
