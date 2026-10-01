#!/usr/bin/env python3
"""Fail closed if our packaging of the Qwen3.8-27B EXL3 image is incomplete.

Only files we ship and the engine's importability are checked; upstream
behaviour is not.
"""

import os
from pathlib import Path

required = (
    "/opt/exl3/tools/serve_openai.py",
    "/opt/vonk/bin/exllamav3-serve",
)
missing = [path for path in required if not Path(path).is_file()]
if missing:
    raise SystemExit(f"incomplete EXL3 runtime: {missing}")
if not os.access("/opt/vonk/bin/exllamav3-serve", os.X_OK):
    raise SystemExit("/opt/vonk/bin/exllamav3-serve is not executable")

try:
    # torch first: the compiled extension needs libc10/libtorch already loaded.
    import torch  # noqa: F401, I001
    import exllamav3_ext  # noqa: F401
    import aiohttp  # noqa: F401
except Exception as exc:  # pragma: no cover - exercised by image build
    raise SystemExit(f"incomplete EXL3 runtime: {exc}") from exc
