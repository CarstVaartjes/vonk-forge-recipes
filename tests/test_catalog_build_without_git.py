"""The catalog build takes its provenance as arguments and never runs git."""

from __future__ import annotations

import json

from tests.generated_catalog import GENERATED, GIT_MARKER, SOURCE_COMMIT


def test_the_session_build_ran_with_a_failing_git_first_on_path() -> None:
    # tests/generated_catalog.py builds every session's catalog with a git stub
    # that fails and leaves this marker, so that build is the proof.
    assert not (GENERATED / GIT_MARKER).exists()
    index = json.loads((GENERATED / "catalog-index.json").read_text())
    assert index["source_commit"] == SOURCE_COMMIT
    assert index["updated_at"] == "1970-01-01T00:00:00Z"
