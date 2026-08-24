"""Apply the minimal MOVA inference-only import patch to one exact source tree."""

from __future__ import annotations

import hashlib
from pathlib import Path


TARGET = Path("/opt/mova-source/mova/diffusion/pipelines/__init__.py")
EXPECTED_SHA256 = "fd3b5c2624db282717b11f885ebff4713fb5d7e707bf80c7e6237e786d78bea4"
REPLACEMENT = b'from .pipeline_mova import MOVA\n\n__all__ = ["MOVA"]\n'


def main() -> None:
    current = TARGET.read_bytes()
    digest = hashlib.sha256(current).hexdigest()
    if digest != EXPECTED_SHA256:
        raise SystemExit(f"refusing to patch unexpected MOVA pipeline initializer: {digest}")
    TARGET.write_bytes(REPLACEMENT)


if __name__ == "__main__":
    main()
