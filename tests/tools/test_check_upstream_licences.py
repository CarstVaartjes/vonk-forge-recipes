from __future__ import annotations

import json
from pathlib import Path

from recipe_tooling import upstream_licences as guard

COMMIT = "a" * 40


def _repo(tmp_path: Path, entries: dict[str, object]) -> Path:
    adapter = tmp_path / "adapters" / "llm" / "creator-single"
    adapter.mkdir(parents=True)
    (adapter / "Dockerfile").write_text("FROM scratch\n")
    (adapter / "start.sh").write_text("echo creator\n")  # vendored
    (adapter / "wrapper").write_text("echo ours\n")
    manifest = {
        "source": {"repo": "creator/recipe", "commit": COMMIT},
        "ours": ["wrapper"],
    }
    (adapter / "vendored-upstream.json").write_text(json.dumps(manifest))
    kit_only = tmp_path / "adapters" / "llm" / "wrapper-only"
    kit_only.mkdir(parents=True)
    (kit_only / "Dockerfile").write_text("FROM scratch\n")
    (kit_only / "wrapper").write_text("echo ours\n")
    (kit_only / "vendored-upstream.json").write_text(
        json.dumps(
            {"source": {"repo": "other/kit", "commit": COMMIT}, "ours": ["wrapper"]}
        )
    )
    (tmp_path / guard.LICENCES).write_text(json.dumps({"upstreams": entries}))
    return tmp_path


def test_a_vendored_upstream_needs_an_entry_and_a_wrapper_does_not(
    tmp_path: Path,
) -> None:
    problems = guard.problems(_repo(tmp_path, {}))
    assert len(problems) == 1
    assert problems[0].startswith("creator/recipe:")


def test_licence_text_or_request_satisfies_the_guard(tmp_path: Path) -> None:
    text = "licenses/creator/recipe/LICENSE"
    root = _repo(tmp_path, {"creator/recipe": {"license": "MIT", "texts": [text]}})
    missing = guard.problems(root)
    assert len(missing) == 1
    assert text in missing[0]
    (root / text).parent.mkdir(parents=True)
    (root / text).write_text("MIT License\n")
    assert guard.problems(root) == []
    request = {
        "license": "unlicensed-upstream",
        "request": "https://github.com/creator/recipe/issues/1",
    }
    (root / guard.LICENCES).write_text(
        json.dumps({"upstreams": {"creator/recipe": request}})
    )
    assert guard.problems(root) == []
