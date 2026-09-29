"""Verify the Hy3 parser patch and Controller wrapper at image build time."""

from __future__ import annotations

import os
from pathlib import Path

SITE = Path("/usr/local/lib/python3.12/dist-packages/vllm")
MARKERS = {
    SITE / "reasoning/hy_v3_reasoning_parser.py": "<think:opensource>",
    SITE / "tool_parsers/hy_v3_tool_parser.py": "<tool_calls:opensource>",
}
WRAPPER = Path("/opt/vonk/bin/vllm")
VLLM = Path("/usr/local/bin/vllm")


def main() -> None:
    for path, marker in MARKERS.items():
        if not path.is_file() or marker not in path.read_text(encoding="utf-8"):
            raise SystemExit(f"Hy3 parser patch is missing or wrong: {path}")
    if not os.access(WRAPPER, os.X_OK):
        raise SystemExit("Controller vLLM wrapper is not executable")
    if not os.access(VLLM, os.X_OK):
        raise SystemExit("vLLM executable is not executable")


if __name__ == "__main__":
    main()
