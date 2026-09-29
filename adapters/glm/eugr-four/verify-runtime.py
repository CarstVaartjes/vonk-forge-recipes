"""Fail the adapter build unless the image contains the matched GLM stack."""

from __future__ import annotations

import os
from pathlib import Path

REQUIRED_FILES = (
    Path("/opt/vonk/lib/libnccl.so.2"),
    Path("/usr/local/bin/vllm"),
)


def main() -> None:
    for path in REQUIRED_FILES:
        if not path.is_file():
            raise SystemExit(f"required GLM runtime file is missing: {path}")
    if Path("/opt/vonk/lib/libnccl.so.2").read_bytes()[:4] != b"\x7fELF":
        raise SystemExit("NCCL runtime is not an ELF library")
    if not os.access("/usr/local/bin/vllm", os.X_OK):
        raise SystemExit("vLLM executable is not executable")


if __name__ == "__main__":
    main()
