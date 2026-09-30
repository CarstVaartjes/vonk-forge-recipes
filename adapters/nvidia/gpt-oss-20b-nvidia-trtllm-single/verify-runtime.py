"""Verify the Vonk packaging of the NVIDIA TensorRT-LLM image at build time."""

from __future__ import annotations

import os
import shutil
from pathlib import Path

WRAPPER = Path("/opt/vonk/bin/trtllm-serve")
CONFIG = Path("/opt/vonk/trtllm/llm-api-config.yml")
HARMONY = Path("/opt/vonk/harmony-reqs")


def main() -> None:
    if not os.access(WRAPPER, os.X_OK):
        raise SystemExit("Vonk trtllm-serve wrapper is not executable")
    if shutil.which("trtllm-serve") is None:
        raise SystemExit("trtllm-serve is not on PATH in the base image")
    if not CONFIG.is_file() or not CONFIG.read_text(encoding="utf-8").strip():
        raise SystemExit("the playbook LLM API config is missing or empty")
    for name in ("o200k_base.tiktoken", "cl100k_base.tiktoken"):
        if not (HARMONY / name).is_file():
            raise SystemExit(f"missing harmony encoding file: {name}")


if __name__ == "__main__":
    main()
