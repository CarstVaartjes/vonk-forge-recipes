"""An envelope exception is bound to one exact recipe content, never to a slug."""

import pytest
from catalog_documents import catalog_models
from warning_asserts import assert_not_warns, assert_warns, declare_peak

from recipe_tooling.qualification import documents
from recipe_tooling.qualification.memory_need import evidence as mn_evidence
from recipe_tooling.qualification.memory_need import records as mn_records
from recipe_tooling.qualification.memory_need import warnings as mn_warnings
from recipe_tooling.qualification.memory_need.evidence import load_inputs
from recipe_tooling.qualification.memory_need.warnings import memory_estimate_warnings
from vonk_forge_contracts import RecipeDefinition, document_sha256

SLUG = "deepseek-v4-flash-0731-sparkinfer-single"
INPUTS = load_inputs()


@pytest.fixture(autouse=True)
def unrecorded(monkeypatch: pytest.MonkeyPatch) -> None:
    # Judge the exception alone, even once the evidence records this digest.
    monkeypatch.setattr(mn_evidence, "unchanged_recipe", lambda *_: False)


def authored(monkeypatch: pytest.MonkeyPatch) -> tuple[RecipeDefinition, str]:
    """A synthetic declaration of the real recipe's shape, with its own exception."""
    recipe = documents.recipe(SLUG)
    declare_peak(recipe, 100_000_000_000)
    digest = document_sha256(recipe.model_dump(mode="json"))
    entry = mn_records.EnvelopeException(
        content_sha256=digest,
        reason="synthetic",
        owner_decision="synthetic",
        date="2026-11-02",
    )
    monkeypatch.setattr(mn_warnings, "envelope_exceptions", lambda: {SLUG: entry})
    return recipe, digest


def check(recipe: RecipeDefinition, digest: str) -> list[str]:
    return memory_estimate_warnings(
        recipe, catalog_models(), INPUTS, document_digest=digest, touched=True
    )


def test_listed_exact_content_passes(monkeypatch: pytest.MonkeyPatch) -> None:
    recipe, digest = authored(monkeypatch)
    assert_not_warns(check(recipe, digest), "peak understates vLLM budget")


def test_same_slug_at_another_utilization_is_refused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recipe, _ = authored(monkeypatch)
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
    recipe, digest = authored(monkeypatch)
    monkeypatch.setattr(mn_warnings, "envelope_exceptions", dict)
    assert_warns(
        check(recipe, digest),
        "peak understates vLLM budget",
    )
