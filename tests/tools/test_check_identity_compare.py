"""An added identity, image or build-key comparison outside the identity module fails."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest
from added_diff import added_diff
from repo_root import ROOT

STATUS = "src/recipe_tooling/qualification/status.py"


def run(root: Path, files: dict[str, str]) -> subprocess.CompletedProcess[str]:
    for name, text in files.items():
        (root / name).parent.mkdir(parents=True, exist_ok=True)
        (root / name).write_text(text)
    diff = root / "added.diff"
    diff.write_text(added_diff(files))
    return subprocess.run(
        [sys.executable, str(ROOT / "tools/check-identity-compare"),
         "--root", str(root), "--diff", str(diff)],
        capture_output=True, text=True, check=False,
    )  # fmt: skip


@pytest.mark.parametrize(
    "line",
    [
        "same = record.identity == current\n",
        "changed = previous.runtime_image != current.runtime_image\n",
        "moved = pin.build_key != plan_key\n",
    ],
)
def test_an_added_comparison_outside_the_identity_module_exits_one(
    tmp_path: Path, line: str
) -> None:
    result = run(tmp_path, {STATUS: line})
    assert result.returncode == 1
    assert f"{STATUS}:1:" in result.stdout


@pytest.mark.parametrize(
    "name",
    ["src/recipe_tooling/qualification/identity.py", "spark_sweep/qualification.py"],
)
def test_the_identity_module_and_the_sweep_are_out_of_scope(
    tmp_path: Path, name: str
) -> None:
    assert run(tmp_path, {name: "same = record.identity == current\n"}).returncode == 0


def test_digest_comparisons_and_presence_checks_pass(tmp_path: Path) -> None:
    source = (
        "a = image_digest(previous) != image_digest(current)\n"
        "b = previous.runtime_image is None\n"
    )
    assert run(tmp_path, {STATUS: source}).returncode == 0
