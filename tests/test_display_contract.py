"""Display is its own contract: its files, shared vectors and strict refusals."""

from __future__ import annotations

import json
import re
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
    published_json,
    read_model,
    read_model_display,
    read_recipe,
    read_recipe_display,
    require_display_of,
)

ROOT = Path(__file__).resolve().parents[1]
INVALID = ROOT / "contracts/src/vonk_forge_contracts/examples/invalid"
RECIPES = sorted((ROOT / "recipes").glob("*.json"))
MODELS = sorted((ROOT / "models").glob("*.json"))


def _json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def test_the_library_is_not_empty() -> None:
    assert RECIPES and MODELS


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
        assert display.identity.slug == path.stem
        assert read_recipe_display(published_json(display)) == display
    for path in MODELS:
        display_model = model_display(path.stem)
        assert read_model_display(published_json(display_model)) == display_model


def _check_pair(pair: dict[str, Any]) -> None:
    display = pair["display"]
    if display["kind"] == "model-display":
        require_display_of(
            ModelDisplay.model_validate(display),
            ModelDefinition.model_validate(pair["document"]).identity,
        )
    else:
        require_display_of(
            RecipeDisplay.model_validate(display),
            RecipeDefinition.model_validate(pair["document"]).identity,
        )


def refuse(contract: str, document: dict[str, Any]) -> None:
    """Run one invalid vector through the contract it names."""

    # Producers validate strictly; readers ignore unknown fields on purpose.
    if contract == "RecipeDefinition":
        RecipeDefinition.model_validate(document)
    elif contract == "ModelDefinition":
        ModelDefinition.model_validate(document)
    elif contract == "RecipeDisplay":
        RecipeDisplay.model_validate(document)
    elif contract == "ModelDisplay":
        ModelDisplay.model_validate(document)
    elif contract == "require_display_of":
        _check_pair(document)
    else:
        raise AssertionError(f"unknown contract {contract!r}")


def invalid_vectors() -> list[Path]:
    return sorted(
        path
        for path in INVALID.glob("*.json")
        if not path.name.endswith(".expect.json")
        and not path.stem.startswith("recipe-api-")
    )


def test_there_are_invalid_vectors_for_each_named_failure() -> None:
    assert {path.stem for path in invalid_vectors()} == {
        "model-display-bad-tag",
        "model-display-bad-variant-kind",
        "model-display-duplicate-tag",
        "model-display-identity-mismatch",
        "model-with-display-facts",
        "model-with-identity-family",
        "recipe-display-bad-alignment",
        "recipe-display-empty-title",
        "recipe-display-identity-extra-key",
        "recipe-display-identity-mismatch",
        "recipe-display-kind-recipe",
        "recipe-display-title-too-long",
        "recipe-display-with-runtime",
        "recipe-serving-multipart-for-chat",
        "recipe-serving-speech-input-not-text",
        "recipe-serving-speech-not-wav",
        "recipe-serving-speech-voice-not-text",
        "recipe-serving-transcription-over-json",
        "recipe-serving-transcription-without-assertion",
        "recipe-with-metadata",
    }


@pytest.mark.parametrize("path", invalid_vectors(), ids=lambda path: path.stem)
def test_every_invalid_vector_is_refused_with_its_named_error(path: Path) -> None:
    expect = _json(path.with_suffix(".expect.json"))
    with pytest.raises((ValidationError, ValueError)) as caught:
        refuse(expect["contract"], _json(path))
    assert expect["error_contains"] in str(caught.value)


def test_producer_only_vectors_are_accepted_by_the_lenient_readers() -> None:
    marked = [
        path
        for path in invalid_vectors()
        if _json(path.with_suffix(".expect.json")).get("producer_only")
    ]
    assert {path.stem for path in marked} == {
        "model-with-display-facts",
        "model-with-identity-family",
        "recipe-display-identity-extra-key",
        "recipe-display-with-runtime",
        "recipe-with-metadata",
    }
    for path in marked:
        document = _json(path)
        if document["kind"] == "recipe-display":
            read_recipe_display(document)
        elif document["kind"] == "recipe":
            read_recipe(document)
        else:
            read_model(document)


# The status badge owns verification status. Display text and tags describe what a
# recipe is and does; they never say it is pending, untested, a candidate or
# awaiting physical acceptance, qualification or retesting.
_STATUS_WORDING = (
    re.compile(
        r"\b(physical|physically|vonk|hardware|fleet|operator)\b[^.;]{0,70}"
        r"\b(acceptance|qualification|requalification|retesting|canary|verification|execution|qualified)\b"
        r"[^.;]{0,80}\b(pending|required|unverified|unqualified|untested|proven)\b"
        r"|\bnot physically (qualified|accepted)\b|\bremains? to be proven\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(Executable|Installable) (Candidate|Pending)\b|^Candidate\b"
        r"|HARDWARE-UNTESTED|[Uu]ntested on hardware|sweep is on hold"
        r"|sweep skips|installable but untested|not measured fleet acceptance"
        r"|remains? untested here"
    ),
)
_STATUS_TAGS = {
    "candidate",
    "physical-acceptance-required",
    "pending",
    "build-unvalidated",
}


def _states_status(sentence: str) -> bool:
    return any(pattern.search(sentence) for pattern in _STATUS_WORDING)


def test_status_guard_ignores_technical_wording() -> None:
    for sentence in (
        "On the physical Spark, MTP acceptance remains about 70 percent at K=3.",
        "Pending upstream PR 1234, the recipe applies the parser patch locally.",
        "The creator notes that the 1M-token context remains untested upstream.",
    ):
        assert not _states_status(sentence), sentence
    for sentence in (
        "Not physically qualified on this fleet.",
        "Hardware fit remains to be proven.",
        "Physical Spark acceptance is pending.",
        "Executable Candidate recipe for X.",
    ):
        assert _states_status(sentence), sentence


def test_display_text_and_tags_never_state_verification_status() -> None:
    offenders = []
    for path in sorted((ROOT / "display/recipes").glob("*.json")):
        display = recipe_display(path.stem)
        offenders += [
            f"{path.stem}: tag {tag}" for tag in display.tags if tag in _STATUS_TAGS
        ]
        offenders += [
            f"{path.stem}: {sentence}"
            for sentence in re.split(r"(?<=\.)\s+", display.description)
            if _states_status(sentence)
        ]
    assert not offenders, "\n".join(offenders)
