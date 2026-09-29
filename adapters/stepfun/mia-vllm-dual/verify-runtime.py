"""Verify the Step 3.7 Flash dual-Spark adapter packaging at build time."""

from __future__ import annotations

import os
from pathlib import Path

SITE = Path("/usr/local/lib/python3.12/dist-packages/vllm")
DRAFTER = SITE / "model_executor/models/step3p5_mtp.py"
WRAPPER = Path("/opt/vonk/bin/vllm")
LAUNCHER = Path("/usr/local/bin/vllm")


def main() -> None:
    for path in (DRAFTER, WRAPPER, LAUNCHER):
        if not path.is_file():
            raise SystemExit(f"required Step 3.7 runtime file is missing: {path}")
    for path in (WRAPPER, LAUNCHER):
        if not os.access(path, os.X_OK):
            raise SystemExit(f"not executable: {path}")
    if "local_mtp_unquant_hack" not in DRAFTER.read_text():
        raise SystemExit(f"the MTP drafter patch is missing from {DRAFTER}")


if __name__ == "__main__":
    main()
