"""Fail the adapter build unless the image has its Ray/vLLM entrypoints."""

from __future__ import annotations

import os
from pathlib import Path


def _one_executable(*candidates: str) -> None:
    if not any(
        Path(path).is_file() and os.access(path, os.X_OK) for path in candidates
    ):
        raise SystemExit(f"required runtime executable is missing: {candidates}")


def main() -> None:
    _one_executable("/opt/vllm/.venv/bin/vllm", "/usr/local/bin/vllm")
    _one_executable("/opt/vllm/.venv/bin/ray", "/usr/local/bin/ray")
    nccl = Path("/opt/vonk/lib/libnccl.so.2")
    if not nccl.is_file() or nccl.read_bytes()[:4] != b"\x7fELF":
        raise SystemExit("NCCL runtime is missing or is not an ELF library")


if __name__ == "__main__":
    main()
