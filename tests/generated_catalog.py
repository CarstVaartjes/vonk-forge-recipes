"""The catalog build outputs, generated once per test process.

``catalog-index.json``, ``qualification/qualification-index.json`` and
``packages/`` are build outputs of the recipe sources and are not committed.
Tests read them from one temporary directory written by the same
``tools/build-catalog-index --output-dir`` command CI and publication use, so
no test depends on (or mutates) outputs a developer built in the checkout.
"""

from __future__ import annotations

import atexit
import os
import shutil
import subprocess
import sys
import tempfile
from functools import cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


# Provenance arguments are fixed: no test reads the checkout history.
SOURCE_COMMIT = "0" * 40
SOURCE_DATE_EPOCH = 0
SHARED_ENV = "VONK_RECIPES_GENERATED_DIR"


@cache
def generated_root() -> Path:
    """Build once per test run: pytest-xdist workers reuse the controller's.

    The controller imports this module before it starts its workers, so the
    directory it exports is inherited by every worker. Tests only read it.
    """
    shared = os.environ.get(SHARED_ENV)
    if shared and Path(shared).is_dir():
        return Path(shared)
    directory = Path(tempfile.mkdtemp(prefix="vonk-recipes-generated-"))
    os.environ[SHARED_ENV] = str(directory)
    atexit.register(shutil.rmtree, directory, ignore_errors=True)
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "tools/build-catalog-index"),
            "--source-commit",
            SOURCE_COMMIT,
            "--source-date-epoch",
            str(SOURCE_DATE_EPOCH),
            "--output-dir",
            str(directory),
        ],
        cwd=ROOT,
        check=True,
        stdout=subprocess.DEVNULL,
    )
    return directory


GENERATED = generated_root()
