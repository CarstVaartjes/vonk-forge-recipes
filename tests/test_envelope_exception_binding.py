"""An envelope exception is bound to one exact recipe content, never to a slug."""

import pytest
from catalog_documents import catalog_models
from vonk_forge_contracts import RecipeDefinition, document_sha256
from warning_asserts import assert_warns

from qualification import documents, memory_need
from qualification.memory_need import load_inputs, memory_estimate_warnings

SLUG = "deepseek-v4-flash-0731-sparkinfer-single"
INPUTS = load_inputs()


@pytest.fixture(autouse=True)
def unrecorded(monkeypatch: pytest.MonkeyPatch) -> None:
    # Judge the exception alone, even once the evidence records this digest.
    monkeypatch.setattr(memory_need, "unchanged_recipe", lambda *_: False)


def authored() -> tuple[RecipeDefinition, str]:
    # tools/check-recipe-memory-need runs the exact authored digest in CI.
    return documents.recipe(SLUG), memory_need.envelope_exceptions()[
        SLUG
    ].content_sha256


def check(recipe: RecipeDefinition, digest: str) -> list[str]:
    return memory_estimate_warnings(
        recipe, catalog_models(), INPUTS, document_digest=digest, touched=True
    )


def test_listed_exact_content_passes() -> None:
    recipe, digest = authored()
    assert check(recipe, digest) == []


def test_same_slug_at_another_utilization_is_refused() -> None:
    recipe, _ = authored()
    for item in recipe.runtime.arguments:
        if item.name == "gpu-memory-utilization":
            item.value = "0.84"
    edited = document_sha256(recipe.model_dump(mode="json"))
    assert_warns(
        check(recipe, edited),
        "gpu-memory-utilization",
    )


def test_unlisted_slug_gets_the_envelope_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recipe, digest = authored()
    monkeypatch.setattr(memory_need, "envelope_exceptions", dict)
    assert_warns(
        check(recipe, digest),
        "peak understates vLLM budget",
    )
