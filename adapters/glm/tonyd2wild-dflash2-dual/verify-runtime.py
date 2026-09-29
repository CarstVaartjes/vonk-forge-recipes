"""Verify our packaging of the publisher's GLM 5.3 SM121 DFlash2 image."""

from __future__ import annotations

import os
from pathlib import Path

WRAPPER = Path("/opt/vonk/bin/vllm")


def main() -> None:
    if not WRAPPER.is_file():
        raise SystemExit(f"required runtime file is missing: {WRAPPER}")
    if not os.access(WRAPPER, os.X_OK):
        raise SystemExit("Controller vLLM wrapper is not executable")


if __name__ == "__main__":
    main()
