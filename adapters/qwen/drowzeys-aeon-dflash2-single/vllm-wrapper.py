#!/usr/bin/env python3
"""Bind Vonk's mounts to drowzeys' AEON DFlash2 1M-YaRN launch.

Upstream bind-mounts a YaRN-extended config.json over the DFlash2 drafter
(vLLM does not stretch the draft rope past 262144 by itself). Model mounts are
read-only, so this wrapper mirrors the drafter directory with symlinks and the
upstream config file, exactly like the other drowzeys wrappers mirror a
checkpoint with one patched file.
"""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

DRAFTER_SOURCE = Path("/models/drafter")
DRAFTER_VIEW = Path("/outputs/drafter-yarn-1m")
DRAFTER_CONFIG = Path("/opt/vonk/overlay/dflash2_config_yarn_1m.json")


def _prepare_drafter_view() -> None:
    if not DRAFTER_SOURCE.is_dir():
        raise SystemExit(f"drafter checkpoint is missing: {DRAFTER_SOURCE}")
    DRAFTER_VIEW.mkdir(parents=True, exist_ok=True)
    for path in DRAFTER_SOURCE.rglob("*"):
        relative = path.relative_to(DRAFTER_SOURCE)
        destination = DRAFTER_VIEW / relative
        if relative == Path("config.json"):
            continue
        if path.is_dir():
            destination.mkdir(parents=True, exist_ok=True)
            continue
        if destination.exists() or destination.is_symlink():
            continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.symlink_to(path)
    temporary = DRAFTER_VIEW / ".config.json.tmp"
    shutil.copyfile(DRAFTER_CONFIG, temporary)
    temporary.replace(DRAFTER_VIEW / "config.json")


_prepare_drafter_view()
vllm = shutil.which("vllm")
if vllm is None:
    raise SystemExit("the vllm executable is missing from the adapter image")
os.execv(vllm, (vllm, *sys.argv[1:]))
