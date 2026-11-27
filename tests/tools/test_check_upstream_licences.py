from __future__ import annotations

import json
from pathlib import Path

from recipe_tooling import upstream_licences as guard

COMMIT = "a" * 40
ADAPTER = "llm/creator-single"


def _repo(
    tmp_path: Path, entries: dict[str, object], repo: str = "creator/recipe"
) -> Path:
    adapter = tmp_path / "adapters" / ADAPTER
    adapter.mkdir(parents=True)
    (adapter / "Dockerfile").write_text("FROM scratch\n")
    (adapter / "start.sh").write_text("echo creator\n")  # vendored
    (adapter / "wrapper").write_text("echo ours\n")
    manifest = {"source": {"repo": repo, "commit": COMMIT}, "ours": ["wrapper"]}
    (adapter / "vendored-upstream.json").write_text(json.dumps(manifest))
    wrapper_only = tmp_path / "adapters" / "llm" / "wrapper-only"
    wrapper_only.mkdir(parents=True)
    (wrapper_only / "Dockerfile").write_text("FROM scratch\n")
    (wrapper_only / "wrapper").write_text("echo ours\n")
    (wrapper_only / "vendored-upstream.json").write_text(
        json.dumps(
            {"source": {"repo": "other/kit", "commit": COMMIT}, "ours": ["wrapper"]}
        )
    )
    (tmp_path / guard.LICENCES).write_text(json.dumps({"upstreams": entries}))
    return tmp_path


def test_an_added_adapter_needs_an_entry_and_a_wrapper_does_not(tmp_path: Path) -> None:
    problems, notes = guard.check(
        _repo(tmp_path, {}), added={ADAPTER, "llm/wrapper-only"}
    )
    assert len(problems) == 1
    assert problems[0].startswith("creator/recipe:")
    assert notes == []


def test_a_refresh_that_moves_an_existing_adapter_upstream_is_not_blocked(
    tmp_path: Path,
) -> None:
    entry = {
        "license": "unlicensed-upstream",
        "request": "https://github.com/creator/recipe/issues/1",
    }
    root = _repo(tmp_path, {"creator/recipe": entry}, repo="creator/renamed-recipe")
    problems, notes = guard.check(root, added=set())
    assert problems == []
    assert len(notes) == 1


def test_a_missing_text_fails_only_for_an_added_adapter(tmp_path: Path) -> None:
    text = "licenses/creator/recipe/LICENSE"
    root = _repo(tmp_path, {"creator/recipe": {"license": "MIT", "texts": [text]}})
    problems, notes = guard.check(root, added=set())
    assert problems == []
    assert len(notes) == 1
    assert text in notes[0]
    problems, _ = guard.check(root, added={ADAPTER})
    assert len(problems) == 1
    (root / text).parent.mkdir(parents=True)
    (root / text).write_text("MIT License\n")
    assert guard.check(root, added={ADAPTER}) == ([], [])
