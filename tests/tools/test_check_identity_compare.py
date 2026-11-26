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
        'moved = previous.get("build_key") == key\n',
        'moved = row["runtime_image"] != image\n',
        "same = content_identity(accepted) == content_identity(recipe)\n",
        "known = {r.identity.model_dump_json() for r in records}\n",
        "groups.setdefault(record.identity, []).append(record)\n",
        "latest = {r.identity: r for r in records}\n",
    ],
)
def test_an_added_comparison_outside_the_identity_module_exits_one(
    tmp_path: Path, line: str
) -> None:
    result = run(tmp_path, {STATUS: line})
    assert result.returncode == 1
    assert f"{STATUS}:1:" in result.stdout


def test_the_sweep_is_in_scope_and_the_identity_module_owns_the_comparison(
    tmp_path: Path,
) -> None:
    line = "same = record.identity == current\n"
    assert run(tmp_path, {"spark_sweep/sweep_schedule.py": line}).returncode == 1
    owner = "src/recipe_tooling/qualification/identity.py"
    assert run(tmp_path, {owner: line}).returncode == 0


def test_digest_comparisons_and_presence_checks_pass(tmp_path: Path) -> None:
    source = (
        "a = image_digest(previous) != image_digest(current)\n"
        "b = previous.runtime_image is None\n"
        "c = execution_key(previous.identity) == execution_key(current)\n"
    )
    assert run(tmp_path, {STATUS: source}).returncode == 0
