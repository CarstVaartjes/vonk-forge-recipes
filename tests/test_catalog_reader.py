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
    CatalogBlock,
    CatalogSection,
    DroppedBlock,
    FaultReason,
    RefusalReason,
    SkippedEntry,
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
                reason=FaultReason.WRONG_TYPE,
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
    display = read.index.catalog_entities[0].display
    assert display is not None
    capability = display.capabilities[-1]
    assert isinstance(capability, UnrecognisedWord) and capability == "telepathy"
    # The raw word survives a re-publication of what was read.
    recipes = published_index_json(read.index)["recipes"]
    assert isinstance(recipes, list)
    assert recipes[0]["qualification"]["status"] == "Proven twice"


def _mutable_image_tag(recipe: dict) -> None:
    recipe["prebuilt_image"] = {
        "reference": "ghcr.io/example/runtime:latest",
        "build_key": "0" * 64,
    }


def _unknown_api_mode(recipe: dict) -> None:
    recipe["api"]["api"]["mode"] = "realtime"


def _display_title_not_text(recipe: dict) -> None:
    recipe["display"]["title"] = 5


@pytest.mark.parametrize(
    ("damage", "block", "field", "reason"),
    [
        (
            _mutable_image_tag,
            CatalogBlock.PREBUILT_IMAGE,
            "prebuilt_image.reference",
            FaultReason.INVALID_VALUE,
        ),
        (_unknown_api_mode, CatalogBlock.API, "api.api", FaultReason.UNKNOWN_WORD),
        (
            _display_title_not_text,
            CatalogBlock.DISPLAY,
            "display.title",
            FaultReason.WRONG_TYPE,
        ),
    ],
)
def test_an_unreadable_optional_block_is_dropped_and_its_recipe_stays(
    index: Index,
    damage: Callable[[dict], None],
    block: CatalogBlock,
    field: str,
    reason: FaultReason,
) -> None:
    recipe = next(r for r in index["recipes"] if r.get("api") is not None)
    damage(recipe)
    read = read_catalog(index)
    assert read.problems == (
        DroppedBlock(
            section=CatalogSection.RECIPES,
            entry=_name(recipe),
            block=block,
            field=field,
            reason=reason,
        ),
    )
    assert _names(read.index.recipes) == [_name(r) for r in index["recipes"]]
    (kept,) = [e for e in read.index.recipes if _names((e,)) == [_name(recipe)]]
    assert getattr(kept, block.value) is None


def test_a_producer_still_refuses_an_unknown_word(index: Index) -> None:
    entry = index["catalog_entities"][0]
    entry["display"]["capabilities"].append("telepathy")
    with pytest.raises(ValidationError):
        CatalogModelEntry.model_validate(entry)
    # Not even an UnrecognisedWord a reader kept, from a Python dump.
    (kept,) = read_catalog(index).index.catalog_entities[:1]
    with pytest.raises(ValidationError):
        CatalogModelEntry.model_validate(kept.model_dump())


def test_a_new_word_in_a_full_list_is_kept_without_dropping_the_block(
    index: Index,
) -> None:
    recipe = next(
        r
        for r in index["recipes"]
        if r.get("api") is not None and len(r["api"]["api"].get("endpoints", ())) == 4
    )
    recipe["api"]["api"]["endpoints"].append("/v1/realtime")
    read = read_catalog(index)
    assert read.problems == (
        UnrecognisedWordReport(
            section=CatalogSection.RECIPES,
            field="api.api.endpoints",
            word="/v1/realtime",
            entries=(_name(recipe),),
        ),
    )


def test_a_wire_null_display_is_reported_as_a_missing_block(index: Index) -> None:
    model = index["catalog_entities"][0]
    model["display"] = None
    read = read_catalog(index)
    assert read.problems == (
        DroppedBlock(
            section=CatalogSection.MODELS,
            entry=_name(model),
            block=CatalogBlock.DISPLAY,
            field="display",
            reason=FaultReason.MISSING,
        ),
    )
    assert read.index.catalog_entities[0].display is None


def test_a_duplicate_identity_skips_both_entries(index: Index) -> None:
    recipe = index["recipes"][0]
    index["recipes"].append(copy.deepcopy(recipe))
    read = read_catalog(index)
    duplicate = SkippedEntry(
        section=CatalogSection.RECIPES,
        entry=_name(recipe),
        field="document.identity",
        reason=FaultReason.DUPLICATE,
    )
    assert read.problems == (duplicate, duplicate)
    assert _name(recipe) not in _names(read.index.recipes)
    assert len(read.index.recipes) == len(index["recipes"]) - 2


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
