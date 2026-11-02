"""The catalog index carries each document's display, read from its display file."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

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


def test_every_entry_embeds_its_document_as_authored() -> None:
    """The index carries each file's document unchanged, hashed as written."""

    entries = _index()["recipes"] + _index()["catalog_entities"]
    assert entries
    for entry in entries:
        original = json.loads((ROOT / entry["source_path"]).read_text(encoding="utf-8"))
        assert entry["document"] == original, entry["source_path"]
        assert entry["content_sha256"] == document_sha256(original), entry[
            "source_path"
        ]
