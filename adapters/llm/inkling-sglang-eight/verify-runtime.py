"""Verify our packaging of the official SGLang DGX Spark image at build time."""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path


def main() -> None:
    if importlib.util.find_spec("sglang") is None:
        raise SystemExit("SGLang is not importable in the image")
    wrapper = Path("/opt/vonk/bin/sglang-serve")
    if not wrapper.is_file() or not os.access(wrapper, os.X_OK):
        raise SystemExit("Vonk SGLang launcher is missing or not executable")


if __name__ == "__main__":
    main()
