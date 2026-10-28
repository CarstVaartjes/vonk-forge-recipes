"""New adapter Dockerfile lines may not put image-private state on the read-only image."""

from __future__ import annotations

import importlib.machinery
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOADER = importlib.machinery.SourceFileLoader(
    "check_writable_state_env", str(ROOT / "tools/check-writable-state-env")
)
SPEC = importlib.util.spec_from_loader(LOADER.name, LOADER)
assert SPEC is not None
guard = importlib.util.module_from_spec(SPEC)
sys.modules[LOADER.name] = guard
LOADER.exec_module(guard)

BUILT = "adapters/x/demo/Dockerfile"
EXISTING = "FROM base\nENV OLD_STATE_PATH=/state\n"


def _git(root: Path, *args: str) -> None:
    subprocess.run(
        ["git", "-C", str(root), "-c", "user.name=t", "-c", "user.email=t@t", *args],
        check=True,
        capture_output=True,
    )


def _repo(tmp_path: Path) -> Path:
    (tmp_path / "tests/fixtures").mkdir(parents=True)
    fixture = ROOT / guard.FIXTURE
    (tmp_path / guard.FIXTURE).write_text(fixture.read_text())
    (tmp_path / "recipes").mkdir()
    (tmp_path / "recipes/demo.json").write_text(
        json.dumps({"execution": {"build": {"dockerfile": BUILT}}})
    )
    (tmp_path / "adapters/x/demo").mkdir(parents=True)
    for name in (BUILT, "adapters/x/demo/Dockerfile.upstream"):
        (tmp_path / name).write_text(EXISTING)
    _git(tmp_path, "init", "-q", "-b", "main")
    _git(tmp_path, "add", "-A")
    _git(tmp_path, "commit", "-q", "-m", "base")
    return tmp_path


def _change(root: Path, name: str, text: str) -> list[str]:
    (root / name).write_text(text)
    _git(root, "add", "-A")
    _git(root, "commit", "-q", "-m", "change")
    return guard.violations(root, "HEAD~1")


def test_flags_added_image_state_path(tmp_path: Path) -> None:
    found = _change(
        _repo(tmp_path), BUILT, EXISTING + "ENV STATE_PATH=/state \\\n    X=1\n"
    )
    assert len(found) == 1 and "STATE_PATH=/state" in found[0]


def test_flags_the_mimo_dockerfile_lines_as_added(tmp_path: Path) -> None:
    # The MiMo adapter's Dockerfile before it moved state under /outputs/cache.
    mimo = (
        "FROM base\nENV MODEL_PATH=/models/target \\\n    STATE_PATH=/state \\\n"
        "    HOME=/state \\\n    XDG_CACHE_HOME=/outputs/cache \\\n"
        "    TRITON_CACHE_DIR=/outputs/cache/triton\nWORKDIR /state\n"
    )
    found = _change(_repo(tmp_path), BUILT, mimo)
    assert len(found) == 1
    assert found[0].startswith(f"{BUILT}:3: STATE_PATH=/state")


def test_ignores_existing_lines_unbuilt_files_and_platform_names(
    tmp_path: Path,
) -> None:
    root = _repo(tmp_path)
    assert not _change(root, BUILT, EXISTING + "ENV HOME=/state TMPDIR=/state\n")
    assert not _change(
        root, "adapters/x/demo/Dockerfile.upstream", EXISTING + "ENV A_DIR=/opt/a\n"
    )


def test_accepts_writable_read_only_and_earlier_stage(tmp_path: Path) -> None:
    root = _repo(tmp_path)
    text = (
        "FROM a AS build\nENV BUILD_DIR=/src\nFROM base\n"
        "ENV STATE_PATH=/outputs/cache/state SCRATCH_DIR=/tmp/s MODEL_PATH=/models/t\n"
    )
    assert not _change(root, BUILT, text)
