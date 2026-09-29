"""Verify the llama-server build and wrapper at image build time."""

from __future__ import annotations

import os
from pathlib import Path

SERVER = Path("/opt/llama.cpp/bin/llama-server")
WRAPPER = Path("/opt/vonk/bin/llama-server")


def main() -> None:
    for path in (SERVER, WRAPPER):
        if not path.is_file() or not os.access(path, os.X_OK):
            raise SystemExit(f"missing or not executable: {path}")


if __name__ == "__main__":
    main()
