#!/usr/bin/env python3
"""Fail closed if our packaging of the Inkling 2 Mini EXL3 image is incomplete.

Only files we ship and the engine's importability are checked; upstream
behaviour is not. The fork checks mirror upstream's verify_runtime (env.sh): a
stock exllamav3 wheel left in the environment lacks these symbols, the
Inkling2Loop architecture and the MTP drafter with speculative sampling. The
build runs without a GPU, so only the compiled extension is imported and
torch.cuda is not asked; the Python sources are read as text.
"""

import importlib.util

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
    import exllamav3_ext as ext
    import uvloop  # noqa: F401
except Exception as exc:  # pragma: no cover - exercised by image build
    raise SystemExit(f"incomplete EXL3 runtime: {exc}") from exc

absent = [name for name in ("InkToken", "exl3_mlin", "MoeRow") if not hasattr(ext, name)]
if absent:
    raise SystemExit(f"exllamav3 is not the vcruz305 fork runtime; missing symbols: {absent}")
spec = importlib.util.find_spec("exllamav3")
if spec is None or not spec.submodule_search_locations:
    raise SystemExit("incomplete EXL3 runtime: exllamav3 is not installed")
package = Path(next(iter(spec.submodule_search_locations)))
if "Inkling2Loop" not in (package / "architecture" / "architectures.py").read_text():
    raise SystemExit("Inkling2Loop architecture is not registered in this exllamav3")
if "spec_sampling" not in (package / "architecture" / "inkling2_mtp.py").read_text():
    raise SystemExit("the MTP drafter predates the pinned commit (no speculative sampling)")
