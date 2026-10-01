#!/usr/bin/env python3
"""Fail closed if our packaging of the Qwen3.8-Flash-Next EXL3 image is incomplete.

Only files we ship and the engine's importability are checked; upstream
behaviour is not. The fork-kernel check mirrors upstream's verify_runtime: a
stock exllamav3 wheel left in the environment lacks these kernels.
"""

import os
from pathlib import Path

required = (
    "/opt/tabbyAPI/main.py",
    "/opt/vonk/tabby-config.yml",
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
    import triton  # noqa: F401
    import uvloop  # noqa: F401
except Exception as exc:  # pragma: no cover - exercised by image build
    raise SystemExit(f"incomplete EXL3 runtime: {exc}") from exc

absent = [name for name in ("gr_mix_int8", "exl3_moe_mixedk") if not hasattr(exllamav3_ext, name)]
if absent:
    raise SystemExit(f"exllamav3 is not the vcruz305 fork runtime; missing kernels: {absent}")
