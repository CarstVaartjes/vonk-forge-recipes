"""Display is its own contract; 2.x derives it from the document, unchanged."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from documents import model_display, recipe_display
from pydantic import ValidationError
from vonk_forge_contracts import (
    ModelDefinition,
    ModelDisplay,
    RecipeDefinition,
    RecipeDisplay,
    model_display_of,
    published_json,
    read_model_display,
    read_recipe_display,
    recipe_display_of,
)

ROOT = Path(__file__).resolve().parents[1]
INVALID = ROOT / "contracts/src/vonk_forge_contracts/examples/invalid"
RECIPES = sorted((ROOT / "recipes").glob("*.json"))
MODELS = sorted((ROOT / "models").glob("*.json"))


def _json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def test_the_library_is_the_size_the_brief_assumes() -> None:
    assert len(RECIPES) >= 355
    assert len(MODELS) >= 280


def test_every_committed_document_round_trips_through_its_contract() -> None:
    for path in RECIPES:
        document = _json(path)
        assert published_json(RecipeDefinition.model_validate(document)) == document, (
            path.name
        )
    for path in MODELS:
        document = _json(path)
        assert published_json(ModelDefinition.model_validate(document)) == document, (
            path.name
        )


def test_every_document_has_a_valid_display_that_round_trips() -> None:
    for path in RECIPES:
        display = recipe_display(path.stem)
        assert display == recipe_display_of(
            RecipeDefinition.model_validate(_json(path))
        )
        assert read_recipe_display(published_json(display)) == display
    for path in MODELS:
        display_model = model_display(path.stem)
        assert read_model_display(published_json(display_model)) == display_model


def test_display_fields_are_exactly_the_moved_set() -> None:
    assert set(RecipeDisplay.model_fields) == {
        "kind",
        "identity",
        "title",
        "description",
        "tags",
        "alignment",
    }
    assert set(ModelDisplay.model_fields) == {
        "kind",
        "identity",
        "family_title",
        "model_title",
        "description",
        "tags",
    }


def test_display_carries_the_documents_text_unchanged() -> None:
    document = _json(
        ROOT / "contracts/src/vonk_forge_contracts/examples/model-definition.json"
    )
    display = model_display_of(ModelDefinition.model_validate(document))
    assert display.family_title == document["identity"]["family"]["title"]
    assert display.model_title == document["identity"]["model"]["title"]
    assert display.description == document["metadata"]["description"]
    assert display.tags == document["metadata"]["tags"]


def _check_pair(pair: dict[str, Any]) -> None:
    document = RecipeDefinition.model_validate(pair["document"])
    display = read_recipe_display(pair["display"])
    if (display.identity.publisher, display.identity.slug) != (
        document.identity.publisher,
        document.identity.slug,
    ):
        raise ValueError("display identity differs from its document")


def refuse(contract: str, document: dict[str, Any]) -> None:
    """Run one invalid vector through the contract it names."""

    # Producers validate strictly; readers ignore unknown fields on purpose.
    if contract == "RecipeDisplay":
        RecipeDisplay.model_validate(document)
    elif contract == "ModelDisplay":
        ModelDisplay.model_validate(document)
    elif contract == "RecipeDisplayOfDocument":
        _check_pair(document)
    else:
        raise AssertionError(f"unknown contract {contract!r}")


def invalid_vectors() -> list[Path]:
    return sorted(
        path
        for path in INVALID.glob("*.json")
        if not path.name.endswith(".expect.json")
    )


def test_there_are_invalid_vectors_for_each_named_failure() -> None:
    assert {path.stem for path in invalid_vectors()} == {
        "model-display-duplicate-tag",
        "recipe-display-identity-mismatch",
        "recipe-display-kind-recipe",
        "recipe-display-with-runtime",
    }


@pytest.mark.parametrize("path", invalid_vectors(), ids=lambda path: path.stem)
def test_every_invalid_vector_is_refused_with_its_named_error(path: Path) -> None:
    expect = _json(path.with_suffix(".expect.json"))
    with pytest.raises((ValidationError, ValueError)) as caught:
        refuse(expect["contract"], _json(path))
    assert expect["error_contains"] in str(caught.value)
