"""The future vector holds an unknown word at every tolerant site, and the reader survives it."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
from generated_catalog import GENERATED
from pydantic import ValidationError
from repo_root import ROOT

from recipe_tooling.catalog_index import MAX_INDEX_BYTES, rendered_vector
from recipe_tooling.future_vector import (
    KIND,
    SLUG,
    FutureVector,
    build_future_vector,
    missing_fields,
    words_in,
)
from vonk_forge_contracts import read_catalog_index
from vonk_forge_contracts.catalog_reader import (
    CatalogSection,
    FaultReason,
    SkippedEntry,
    UnrecognisedWordReport,
    read_catalog,
)

SCHEMA_PATH = (
    ROOT / "contracts/src/vonk_forge_contracts/schema/catalog-index-v2.schema.json"
)
SCHEMA: dict[str, Any] = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def vector(built_catalog: tuple[dict[str, Any], Path]) -> FutureVector:
    return build_future_vector(built_catalog[0], SCHEMA)


def _enum_blocks(node: object, seen: frozenset[str] = frozenset()) -> bool:
    """Whether a schema subtree holds a closed set of words (through ``$ref``)."""
    if isinstance(node, dict):
        if len(node.get("enum", ())) > 1:
            return True
        ref = node.get("$ref")
        if isinstance(ref, str):
            name = ref.rsplit("/", 1)[-1]
            return name not in seen and _enum_blocks(
                SCHEMA["components"]["schemas"][name], seen | {name}
            )
        return any(_enum_blocks(item, seen) for item in node.values())
    if isinstance(node, list):
        return any(_enum_blocks(item, seen) for item in node)
    return False


def test_the_vector_holds_the_unknown_word_at_every_tolerant_site(
    vector: FutureVector,
) -> None:
    assert vector.tolerant
    assert words_in(SCHEMA, vector.document) == set(vector.tolerant)


def test_every_entry_block_with_words_in_the_schema_has_a_tolerant_site(
    vector: FutureVector,
) -> None:
    schemas = SCHEMA["components"]["schemas"]
    for section, entry in (
        ("catalog_entities", "CatalogModelEntry"),
        ("recipes", "CatalogRecipeEntry"),
    ):
        blocks = [
            name
            for name, node in schemas[entry]["properties"].items()
            if name != "document" and _enum_blocks(node)
        ]
        assert blocks
        for block in blocks:
            assert any(
                site.startswith(f"{section}/{block}/") for site in vector.tolerant
            ), f"{section}.{block}: the walk found no word site the schema has"


def test_every_closed_object_carries_an_unknown_field(vector: FutureVector) -> None:
    assert missing_fields(SCHEMA, vector.document) == []


def test_the_reader_keeps_everything_but_the_withheld_entry(
    vector: FutureVector,
) -> None:
    read = read_catalog(vector.document)
    skipped = [p for p in read.problems if isinstance(p, SkippedEntry)]
    assert [(p.section, p.entry, p.reason) for p in skipped] == [
        (CatalogSection.RECIPES, vector.withheld, FaultReason.UNKNOWN_WORD)
    ]
    assert all(
        isinstance(p, SkippedEntry | UnrecognisedWordReport) for p in read.problems
    )
    assert len(read.index.recipes) == len(vector.document["recipes"]) - 1
    assert SLUG in vector.withheld
    assert KIND not in json.dumps(
        [entry.document.model_dump(mode="json") for entry in read.index.recipes]
    )


def test_the_producers_strict_reader_still_refuses_the_vector(
    vector: FutureVector,
) -> None:
    with pytest.raises(ValidationError):
        read_catalog_index(vector.document)


def test_the_vector_fits_the_index_size_limit(vector: FutureVector) -> None:
    assert len(rendered_vector(vector.document)) <= MAX_INDEX_BYTES


def test_the_tool_writes_the_same_bytes(vector: FutureVector, tmp_path: Path) -> None:
    out = tmp_path / "future-canary-index.json"
    done = subprocess.run(
        [
            sys.executable,
            str(ROOT / "tools/build-future-vector"),
            "--index",
            str(GENERATED / "catalog-index.json"),
            "--schema",
            str(SCHEMA_PATH),
            "--out",
            str(out),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert done.returncode == 0, done.stderr
    assert out.read_bytes() == rendered_vector(vector.document)
