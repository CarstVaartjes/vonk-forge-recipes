"""A changed file over 1 MiB fails with the declare command; small and unlisted pass."""

from __future__ import annotations

import importlib.machinery
import importlib.util
import sys
from pathlib import Path

from repo_root import ROOT

LOADER = importlib.machinery.SourceFileLoader(
    "check_added_file_size", str(ROOT / "tools/check-added-file-size")
)
SPEC = importlib.util.spec_from_loader(LOADER.name, LOADER)
assert SPEC is not None
guard = importlib.util.module_from_spec(SPEC)
sys.modules[LOADER.name] = guard
LOADER.exec_module(guard)


def test_small_changed_file_passes(tmp_path: Path) -> None:
    (tmp_path / "small.bin").write_bytes(b"x" * guard.LIMIT)
    assert guard.oversized(["small.bin"], tmp_path) == []


def test_large_changed_file_fails_naming_the_command(tmp_path: Path) -> None:
    (tmp_path / "big.bin").write_bytes(b"x" * (guard.LIMIT + 1))
    [error] = guard.oversized(["big.bin"], tmp_path)
    assert "tools/adapter-sources add big.bin" in error


def test_unlisted_large_file_and_deleted_path_are_not_checked(tmp_path: Path) -> None:
    (tmp_path / "old.bin").write_bytes(b"x" * (guard.LIMIT + 1))
    assert guard.oversized(["gone.bin"], tmp_path) == []
