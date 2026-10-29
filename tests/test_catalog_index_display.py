"""The catalog index carries each document's display and leaves documents as authored."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from generated_catalog import GENERATED
from vonk_forge_contracts import (
    document_sha256,
    read_model_display,
    read_recipe_display,
)

ROOT = Path(__file__).resolve().parents[1]

MAX_INDEX_BYTES = 12 * 1024 * 1024
INDEX = GENERATED / "catalog-index.json"


def _index() -> dict[str, Any]:
    return json.loads(INDEX.read_text(encoding="utf-8"))


def test_every_recipe_entry_carries_its_display() -> None:
    entries = _index()["recipes"]
    assert entries
    for entry in entries:
        document = entry["document"]
        display = read_recipe_display(entry["display"])
        assert (display.identity.publisher, display.identity.slug) == (
            document["identity"]["publisher"],
            document["identity"]["slug"],
        )


def test_every_catalog_entity_carries_its_display() -> None:
    entries = _index()["catalog_entities"]
    assert entries
    for entry in entries:
        document = entry["document"]
        display = read_model_display(entry["display"])
        assert (display.identity.publisher, display.identity.slug) == (
            document["identity"]["publisher"],
            document["identity"]["slug"],
        )


def test_embedding_display_changes_no_document_or_digest() -> None:
    for entry in _index()["recipes"] + _index()["catalog_entities"]:
        source = json.loads((ROOT / entry["source_path"]).read_text(encoding="utf-8"))
        assert entry["document"] == source
        assert entry["content_sha256"] == document_sha256(source)


def test_the_index_stays_well_inside_its_size_cap() -> None:
    size = INDEX.stat().st_size
    print(f"catalog-index.json is {size} bytes (cap {MAX_INDEX_BYTES})")
    assert size < MAX_INDEX_BYTES - 1024 * 1024
