"""Fail the image build unless our Inkling 2 Mini DGX Spark packaging is complete."""

from __future__ import annotations

import os
from pathlib import Path


def main() -> None:
    launcher = Path("/opt/vonk/bin/sglang-serve")
    if not launcher.is_file() or not os.access(launcher, os.X_OK):
        raise SystemExit("Vonk SGLang launcher is missing or not executable")
    overlay = Path("/opt/inkling-overlay")
    for relative in (
        "sitecustomize.py",
        "inkling_vcap/policies/r2-theta004.json",
        "inkling_draftbytes/hotvocab/inkling2-mini-hot-r2.json",
    ):
        if not (overlay / relative).is_file():
            raise SystemExit(f"overlay file is missing: {overlay / relative}")


if __name__ == "__main__":
    main()
