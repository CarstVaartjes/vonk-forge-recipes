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
    MODE_FIELD,
    MODE_SLUG,
    SLUG,
    FutureVector,
    build_future_vector,
    missing_fields,
    words_in,
)
from recipe_tooling.qualification.api_checks import check_recipe_api
from vonk_forge_contracts import UnrecognisedApiBlock, read_catalog_index
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
EXAMPLES = ROOT / "contracts/src/vonk_forge_contracts/examples"


@pytest.fixture(scope="module")
def vector(built_catalog: tuple[dict[str, Any], Path]) -> FutureVector:
    return build_future_vector(built_catalog[0], SCHEMA, EXAMPLES)


SCHEMAS = SCHEMA["components"]["schemas"]


def _words(node: dict[str, Any]) -> bool:
    """Whether a property is a closed word or a list of them (through ``$ref``/unions)."""
    if "$ref" in node:
        return _words(SCHEMAS[node["$ref"].rsplit("/", 1)[-1]])
    if "enum" in node:
        return True
    if node.get("type") == "array":
        return _words(node.get("items", {}))
    return any(_words(item) for item in node.get("anyOf", node.get("oneOf", ())))


def _kinds(node: dict[str, Any], seen: frozenset[str]) -> set[str]:
    """The property names holding words anywhere under ``node`` (through ``$ref``)."""
    if "$ref" in node:
        name = node["$ref"].rsplit("/", 1)[-1]
        return set() if name in seen else _kinds(SCHEMAS[name], seen | {name})
    found: set[str] = set()
    for key, item in node.get("properties", {}).items():
        if _words(item):
            found.add(key)
        found |= _kinds(item, seen)
    for item in (
        *node.get("anyOf", node.get("oneOf", ())),
        *([node["items"]] if "items" in node else []),
        *(
            [node["additionalProperties"]]
            if isinstance(node.get("additionalProperties"), dict)
            else []
        ),
    ):
        found |= _kinds(item, seen)
    return found


def test_the_vector_holds_the_unknown_word_at_every_site(vector: FutureVector) -> None:
    assert vector.sites
    assert words_in(SCHEMA, vector.document) == set(vector.sites)


def test_every_word_kind_in_the_schema_has_a_site_in_the_vector(
    vector: FutureVector,
) -> None:
    wanted = {
        kind
        for entry in ("CatalogModelEntry", "CatalogRecipeEntry")
        for name, node in SCHEMAS[entry]["properties"].items()
        if name != "document"
        for kind in ({name} if _words(node) else set()) | _kinds(node, frozenset())
    }
    held = {
        [part for part in site.split("/") if part != "[]" and not part.startswith("<")][
            -1
        ]
        for site in vector.sites
    }
    assert wanted
    assert wanted <= held, f"no site for {sorted(wanted - held)}"


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
            "--examples",
            str(EXAMPLES),
            "--out",
            str(out),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert done.returncode == 0, done.stderr
    assert out.read_bytes() == rendered_vector(vector.document)


def test_every_recipe_entry_the_reader_keeps_is_consistent_with_its_api(
    vector: FutureVector,
) -> None:
    checked = 0
    for entry in read_catalog(vector.document).index.recipes:
        if entry.api is not None and not isinstance(
            entry.api.api, UnrecognisedApiBlock
        ):
            check_recipe_api(entry.document, entry.api)
            checked += 1
    assert checked


def test_the_reader_keeps_the_entry_whose_api_mode_is_unknown(
    vector: FutureVector,
) -> None:
    read = read_catalog(vector.document)
    blocks = {
        entry.document.identity.slug: entry.api.api
        for entry in read.index.recipes
        if entry.api is not None
    }
    assert isinstance(blocks[MODE_SLUG], UnrecognisedApiBlock)
    assert [
        slug
        for slug, block in blocks.items()
        if isinstance(block, UnrecognisedApiBlock)
    ] == [MODE_SLUG]
    assert any(
        isinstance(problem, UnrecognisedWordReport)
        and problem.field == MODE_FIELD
        and problem.entries == (f"vonk-forge/{MODE_SLUG}",)
        for problem in read.problems
    )
