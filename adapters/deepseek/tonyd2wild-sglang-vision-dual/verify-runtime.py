"""Fail the image build unless our SGLang packaging is complete."""

from __future__ import annotations

import os
from pathlib import Path


def main() -> None:
    launcher = Path("/opt/vonk/bin/sglang-serve")
    if not launcher.is_file() or not os.access(launcher, os.X_OK):
        raise SystemExit("Vonk SGLang launcher is missing or not executable")


if __name__ == "__main__":
    main()
