"""Verify the exact official SGLang DGX Spark image at build time."""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path

EXPECTED_COMMIT = "b7252cc6b0c78b25ecea7ee5efa91a6ae37d0f19"
REQUIRED_MODULES = ("sglang", "sglang.launch_server", "torch", "torchaudio", "transformers")


def main() -> None:
    if os.environ.get("SGLANG_BUILD_COMMIT") != EXPECTED_COMMIT:
        raise SystemExit("SGLang image source revision does not match the runtime authority")
    missing = [name for name in REQUIRED_MODULES if importlib.util.find_spec(name) is None]
    if missing:
        raise SystemExit(f"SGLang image is missing required modules: {', '.join(missing)}")
    wrapper = Path("/opt/vonk/bin/sglang-serve")
    if not wrapper.is_file() or not os.access(wrapper, os.X_OK):
        raise SystemExit("Vonk SGLang launcher is missing or not executable")


if __name__ == "__main__":
    main()
