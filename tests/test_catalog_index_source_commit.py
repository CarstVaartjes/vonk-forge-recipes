"""The generated-index check must not depend on the recorded source commit.

A merge rewrites the commit a contribution was built from, so the SHA recorded
in `catalog-index.json` is normally unreachable: it is not an ancestor of the
current checkout and, once the contribution branch is gone, git refuses to fetch
it. Requiring ancestry made the gate fail on every change merged after a
publication, and requiring the object to be fetchable failed in CI for the same
reason. Package bytes depend only on the working tree, so the recorded value is
metadata and all the check needs is a well-formed SHA.
"""

from __future__ import annotations

import json
import runpy
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
TOOL = runpy.run_path(str(ROOT / "tools/build-catalog-index"))
# A well-formed SHA that no repository contains.
UNREACHABLE_COMMIT = "0123456789abcdef0123456789abcdef01234567"


@contextmanager
def _recorded_commit_lookup(index: Path) -> Iterator[None]:
    """Point the tool's recorded-commit lookup at a throwaway index file."""

    # runpy returns a copy of the module namespace, so the function under test
    # still resolves OUTPUT through its own __globals__.
    namespace = TOOL["_recorded_source_commit"].__globals__
    original = dict(namespace)
    try:
        namespace.update(OUTPUT=index)
        yield
    finally:
        namespace.clear()
        namespace.update(original)


def test_unreachable_recorded_source_commit_is_accepted(tmp_path: Path) -> None:
    index = tmp_path / "catalog-index.json"
    index.write_text(
        json.dumps({"schema_version": 2, "source_commit": UNREACHABLE_COMMIT}),
        encoding="utf-8",
    )
    with _recorded_commit_lookup(index):
        assert TOOL["_recorded_source_commit"]() == UNREACHABLE_COMMIT


def test_recorded_source_commit_must_be_a_full_sha(tmp_path: Path) -> None:
    index = tmp_path / "catalog-index.json"
    index.write_text(
        json.dumps({"schema_version": 2, "source_commit": "short"}),
        encoding="utf-8",
    )
    with (
        _recorded_commit_lookup(index),
        pytest.raises(SystemExit, match="does not record a full source commit"),
    ):
        TOOL["_recorded_source_commit"]()


def test_missing_recorded_source_commit_is_rejected(tmp_path: Path) -> None:
    index = tmp_path / "catalog-index.json"
    index.write_text(json.dumps({"schema_version": 2}), encoding="utf-8")
    with (
        _recorded_commit_lookup(index),
        pytest.raises(SystemExit, match="does not record a full source commit"),
    ):
        TOOL["_recorded_source_commit"]()
