"""Verify the Gemma 4 31B NVFP4 vLLM adapter packaging at build time."""

from __future__ import annotations

import os
import shutil
from pathlib import Path


def main() -> None:
    wrapper = Path("/opt/vonk/bin/vllm")
    if not wrapper.is_file() or not os.access(wrapper, os.X_OK):
        raise SystemExit("the vLLM wrapper is missing or not executable")
    if not Path("/opt/vonk/chat_template.jinja").is_file():
        raise SystemExit("the chat template is missing")
    if shutil.which("vllm") is None and not Path("/opt/vllm/.venv/bin/vllm").is_file():
        raise SystemExit("the vLLM executable is missing from the base image")


if __name__ == "__main__":
    main()
