"""Verify the Puzzle 75B single-Spark adapter packaging at build time."""

from __future__ import annotations

import os
from pathlib import Path

WRAPPER = Path("/opt/vonk/bin/vllm")


def main() -> None:
    if not WRAPPER.exists() or not os.access(WRAPPER, os.X_OK):
        raise SystemExit(f"vllm launcher is missing or not executable: {WRAPPER}")


if __name__ == "__main__":
    main()
