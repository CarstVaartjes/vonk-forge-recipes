"""Verify the consolidated GLM 5.3 SM121 adapter at build time."""

from __future__ import annotations

import importlib
import os
from pathlib import Path

REQUIRED = (
    Path("/opt/vonk/lib/libnccl.so.2"),
    Path("/opt/vonk/bin/vllm"),
    Path("/opt/vonk/templates/glm53-chat-template-mm.jinja"),
)
# The wrapper's Ray launch: it starts the first of these and imports ray in its
# own python3 interpreter, the one running this check.
RAY = (Path("/opt/vllm/.venv/bin/ray"), Path("/usr/local/bin/ray"))


def _first_executable(candidates: tuple[Path, ...]) -> Path:
    for candidate in candidates:
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return candidate
    raise SystemExit(f"required GLM runtime executable is missing: {candidates}")


def main() -> None:
    for path in REQUIRED:
        if not path.is_file():
            raise SystemExit(f"required GLM runtime file is missing: {path}")
    if REQUIRED[0].read_bytes()[:4] != b"\x7fELF":
        raise SystemExit("NCCL runtime is not an ELF library")
    if not os.access(REQUIRED[1], os.X_OK):
        raise SystemExit("vLLM wrapper is not executable")
    _first_executable(RAY)
    importlib.import_module("ray")


if __name__ == "__main__":
    main()
