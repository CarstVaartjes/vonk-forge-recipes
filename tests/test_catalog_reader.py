"""An unknown value never breaks the catalog; only integrity edges refuse it."""

from __future__ import annotations

import copy
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError
from vonk_forge_contracts import (
    CatalogIndexRefused,
    CatalogModelEntry,
    UnrecognisedWord,
    published_index_json,
    read_catalog,
)
from vonk_forge_contracts.catalog_reader import (
    CatalogSection,
    RefusalReason,
    SkippedEntry,
    SkipReason,
    UnrecognisedWordReport,
)

Index = dict[str, Any]


def _name(entry: dict) -> str:
    identity = entry["document"]["identity"]
    return f"{identity['publisher']}/{identity['slug']}"


def _names(entries: tuple[Any, ...]) -> list[str]:
    return [
        f"{e.document.identity.publisher}/{e.document.identity.slug}" for e in entries
    ]


@pytest.fixture
def index(built_catalog: tuple[dict, Path]) -> Index:
    return copy.deepcopy(built_catalog[0])


def test_the_built_index_reads_whole_and_publishes_the_same_bytes(
    built_catalog: tuple[dict, Path],
) -> None:
    read = read_catalog(built_catalog[0])
    assert read.problems == ()
    assert published_index_json(read.index) == built_catalog[0]


def test_unknown_words_keep_their_entries_and_one_malformed_entry_is_skipped(
    index: Index,
) -> None:
    model, recipe, broken = (
        index["catalog_entities"][0],
        index["recipes"][0],
        index["recipes"][1],
    )
    model["display"]["capabilities"].append("telepathy")
    recipe["qualification"]["status"] = "Proven twice"
    broken["package"]["expected_bytes"] = "12345"  # a string is not a size

    read = read_catalog(index)

    assert sorted(read.problems, key=repr) == sorted(
        [
            UnrecognisedWordReport(
                section=CatalogSection.MODELS,
                field="display.capabilities",
                word="telepathy",
                entries=(_name(model),),
            ),
            UnrecognisedWordReport(
                section=CatalogSection.RECIPES,
                field="qualification.status",
                word="Proven twice",
                entries=(_name(recipe),),
            ),
            SkippedEntry(
                section=CatalogSection.RECIPES,
                entry=_name(broken),
                field="package.expected_bytes",
                reason=SkipReason.WRONG_TYPE,
            ),
        ],
        key=repr,
    )
    assert _names(read.index.catalog_entities) == [
        _name(e) for e in index["catalog_entities"]
    ]
    assert _names(read.index.recipes) == [
        _name(e) for e in index["recipes"] if e is not broken
    ]
    capability = read.index.catalog_entities[0].display.capabilities[-1]
    assert isinstance(capability, UnrecognisedWord) and capability == "telepathy"
    # The raw word survives a re-publication of what was read.
    recipes = published_index_json(read.index)["recipes"]
    assert isinstance(recipes, list)
    assert recipes[0]["qualification"]["status"] == "Proven twice"


def test_a_mutable_image_tag_skips_only_its_entry(index: Index) -> None:
    recipe = index["recipes"][2]
    recipe["prebuilt_image"] = {
        "reference": "ghcr.io/example/runtime:latest",
        "build_key": "0" * 64,
    }
    read = read_catalog(index)
    assert read.problems == (
        SkippedEntry(
            section=CatalogSection.RECIPES,
            entry=_name(recipe),
            field="prebuilt_image.reference",
            reason=SkipReason.INVALID_VALUE,
        ),
    )


def test_a_producer_still_refuses_an_unknown_word(index: Index) -> None:
    entry = index["catalog_entities"][0]
    entry["display"]["capabilities"].append("telepathy")
    with pytest.raises(ValidationError):
        CatalogModelEntry.model_validate(entry)


def _wrong_major(index: Index) -> None:
    index["contract_version"] = "5.0.0"


def _document_changed(index: Index) -> None:
    index["recipes"][0]["document"]["release"]["version"] = "99.0.0"


def _package_names_another_recipe(index: Index) -> None:
    index["recipes"][0]["package"]["recipe_content_sha256"] = "0" * 64


def _month_thirteen(index: Index) -> None:
    index["updated_at"] = "2026-13-01T00:00:00Z"


@pytest.mark.parametrize(
    ("damage", "reason"),
    [
        (_wrong_major, RefusalReason.SCHEMA_UNSUPPORTED),
        (_document_changed, RefusalReason.DIGEST_MISMATCH),
        (_package_names_another_recipe, RefusalReason.DIGEST_MISMATCH),
        (_month_thirteen, RefusalReason.ENVELOPE_INVALID),
    ],
)
def test_integrity_edges_refuse_the_whole_index(
    index: Index, damage: Callable[[Index], None], reason: RefusalReason
) -> None:
    damage(index)
    with pytest.raises(CatalogIndexRefused) as refused:
        read_catalog(index)
    assert refused.value.reason is reason
