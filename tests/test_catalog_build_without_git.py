"""The catalog build takes its provenance as arguments and never runs git."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from tests.generated_catalog import ROOT, SOURCE_COMMIT, SOURCE_DATE_EPOCH


def test_build_works_with_a_failing_git_first_on_path(tmp_path: Path) -> None:
    stub_dir = tmp_path / "bin"
    stub_dir.mkdir()
    marker = tmp_path / "git-was-called"
    stub = stub_dir / "git"
    stub.write_text(f"#!/bin/sh\ntouch '{marker}'\nexit 1\n")
    stub.chmod(0o755)
    output = tmp_path / "out"

    subprocess.run(
        [
            sys.executable,
            str(ROOT / "tools/build-catalog-index"),
            "--source-commit",
            SOURCE_COMMIT,
            "--source-date-epoch",
            str(SOURCE_DATE_EPOCH),
            "--output-dir",
            str(output),
        ],
        cwd=ROOT,
        check=True,
        stdout=subprocess.DEVNULL,
        env={**os.environ, "PATH": f"{stub_dir}{os.pathsep}{os.environ['PATH']}"},
    )

    assert not marker.exists()
    index = json.loads((output / "catalog-index.json").read_text())
    assert index["source_commit"] == SOURCE_COMMIT
    assert index["updated_at"] == "1970-01-01T00:00:00Z"
