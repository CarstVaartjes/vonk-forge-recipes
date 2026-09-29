"""Verify the consolidated GLM 5.3 SM121 adapter at build time."""

from __future__ import annotations

import os
from pathlib import Path

REQUIRED = (
    Path("/opt/vonk/bin/vllm"),
    Path("/opt/vonk/templates/glm53-chat-template-mm.jinja"),
)


def main() -> None:
    for path in REQUIRED:
        if not path.is_file():
            raise SystemExit(f"required GLM runtime file is missing: {path}")
    if not os.access(REQUIRED[0], os.X_OK):
        raise SystemExit("vLLM wrapper is not executable")


if __name__ == "__main__":
    main()
