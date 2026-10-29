"""The catalog index carries each document's display and leaves documents as authored."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

import pytest
from generated_catalog import GENERATED
from vonk_forge_contracts import (
    ModelDefinition,
    ModelDisplay,
    RecipeDefinition,
    RecipeDisplay,
    document_sha256,
    require_display_of,
)

ROOT = Path(__file__).resolve().parents[1]
INDEX = GENERATED / "catalog-index.json"


def _index() -> dict[str, Any]:
    return json.loads(INDEX.read_text(encoding="utf-8"))


def test_every_recipe_entry_carries_its_display() -> None:
    entries = _index()["recipes"]
    assert entries
    for entry in entries:
        require_display_of(
            RecipeDisplay.model_validate(entry["display"]),
            RecipeDefinition.model_validate(entry["document"]).identity,
        )


def test_every_catalog_entity_carries_its_display() -> None:
    entries = _index()["catalog_entities"]
    assert entries
    for entry in entries:
        require_display_of(
            ModelDisplay.model_validate(entry["display"]),
            ModelDefinition.model_validate(entry["document"]).identity,
        )


def _git(*arguments: str) -> str | None:
    result = subprocess.run(
        ["git", *arguments], cwd=ROOT, capture_output=True, text=True, check=False
    )
    return result.stdout if result.returncode == 0 else None


def test_embedding_display_changes_no_document_or_digest() -> None:
    """Every unchanged document is indexed exactly as the merge base held it."""

    base = _git("merge-base", "HEAD", "origin/main")
    if base is None:
        pytest.skip("origin/main is not available to take a merge base")
    changed = _git("diff", "--name-only", base.strip(), "--", "recipes", "models")
    assert changed is not None
    edited = set(changed.split())
    compared = 0
    index = _index()
    for entry in index["recipes"] + index["catalog_entities"]:
        path = entry["source_path"]
        if path in edited:
            continue
        shown = _git("show", f"{base.strip()}:{path}")
        assert shown is not None, path
        original = json.loads(shown)
        assert entry["document"] == original, path
        assert entry["content_sha256"] == document_sha256(original), path
        compared += 1
    assert compared
