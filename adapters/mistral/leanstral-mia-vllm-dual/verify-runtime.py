"""Verify the Leanstral vLLM adapter packaging at build time."""

from __future__ import annotations

import os
import shutil
from pathlib import Path


def main() -> None:
    wrapper = Path("/opt/vonk/bin/vllm")
    if not wrapper.is_file() or not os.access(wrapper, os.X_OK):
        raise SystemExit("the vLLM wrapper is missing or not executable")
    if shutil.which("vllm") is None and not Path("/opt/vllm/.venv/bin/vllm").is_file():
        raise SystemExit("the vLLM executable is missing from the base image")
    if shutil.which("ray") is None and not Path("/opt/vllm/.venv/bin/ray").is_file():
        raise SystemExit("the Ray CLI is missing from the adapter image")
    import ray

    if ray.__version__ != "2.58.0":
        raise SystemExit("Ray 2.58.0 is required")


if __name__ == "__main__":
    main()
