"""The catalog build outputs, generated once per test process.

``catalog-index.json``, ``qualification/qualification-index.json`` and
``packages/`` are build outputs of the recipe sources and are not committed.
Tests read them from one temporary directory written by the same
``tools/build-catalog-index --output-dir`` command CI and publication use, so
no test depends on (or mutates) outputs a developer built in the checkout.
"""

from __future__ import annotations

import atexit
import shutil
import subprocess
import sys
import tempfile
from functools import cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


@cache
def generated_root() -> Path:
    directory = Path(tempfile.mkdtemp(prefix="vonk-recipes-generated-"))
    atexit.register(shutil.rmtree, directory, ignore_errors=True)
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "tools/build-catalog-index"),
            "--output-dir",
            str(directory),
        ],
        cwd=ROOT,
        check=True,
        stdout=subprocess.DEVNULL,
    )
    return directory


GENERATED = generated_root()
