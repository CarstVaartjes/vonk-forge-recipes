"""The catalog build outputs, generated once per test process.

``catalog-index.json``, ``qualification/qualification-index.json`` and
``packages/`` are build outputs of the recipe sources and are not committed.
Tests read them from one temporary directory written by the same
``tools/build-catalog-index --output-dir`` command CI and publication use, so
no test depends on (or mutates) outputs a developer built in the checkout.

The build runs with a failing ``git`` first on ``PATH`` that leaves a marker:
the build takes its provenance as arguments and must never call git.
CI runs this module as a script beside its release builds and exports the
directory, so the suite reuses that build: ``generated_catalog.py <directory>``.
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


GIT_MARKER = "git-was-called"


def build_generated(directory: Path) -> None:
    """Build the outputs into ``directory`` with a failing git first on PATH."""
    directory.mkdir(parents=True, exist_ok=True)  # the marker's home exists first
    stub_dir = Path(tempfile.mkdtemp(prefix="vonk-recipes-no-git-"))
    try:
        stub = stub_dir / "git"
        stub.write_text(f"#!/bin/sh\ntouch '{directory / GIT_MARKER}'\nexit 1\n")
        stub.chmod(0o755)
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
            env={**os.environ, "PATH": f"{stub_dir}{os.pathsep}{os.environ['PATH']}"},
        )
    finally:
        shutil.rmtree(stub_dir, ignore_errors=True)


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
    build_generated(directory)
    return directory


if __name__ == "__main__":
    build_generated(Path(sys.argv[1]))
    raise SystemExit(0)

GENERATED = generated_root()
