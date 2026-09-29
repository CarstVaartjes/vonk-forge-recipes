"""Verify the Hy3 dual-Spark adapter packaging at build time."""

from __future__ import annotations

import os
from pathlib import Path

SITE = Path("/usr/local/lib/python3.12/dist-packages/vllm")
REASONING = SITE / "reasoning/hy_v3_reasoning_parser.py"
TOOLS = SITE / "tool_parsers/hy_v3_tool_parser.py"
WRAPPER = Path("/opt/vonk/bin/vllm")
LAUNCHERS = (Path("/usr/local/bin/vllm"), Path("/usr/local/bin/ray"))


def main() -> None:
    for path in (REASONING, TOOLS, WRAPPER, *LAUNCHERS):
        if not path.is_file():
            raise SystemExit(f"required Hy3 runtime file is missing: {path}")
    for path in (WRAPPER, *LAUNCHERS):
        if not os.access(path, os.X_OK):
            raise SystemExit(f"not executable: {path}")
    for path in (REASONING, TOOLS):
        if ":opensource>" not in path.read_text():
            raise SystemExit(f"the :opensource token patch is missing from {path}")


if __name__ == "__main__":
    main()
