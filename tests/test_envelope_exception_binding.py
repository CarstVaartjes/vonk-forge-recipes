"""An envelope exception is bound to one exact recipe content, never to a slug."""

import json
from pathlib import Path

import pytest
from catalog_documents import catalog_models
from vonk_forge_contracts import RecipeDefinition, document_sha256

from qualification import memory_need
from qualification.memory_need import load_inputs, require_memory_need

ROOT = Path(__file__).resolve().parents[1]
SLUG = "deepseek-v4-flash-0731-sparkinfer-single"
INPUTS = load_inputs()


def authored() -> tuple[RecipeDefinition, str]:
    document = json.loads((ROOT / "recipes" / f"{SLUG}.json").read_text())
    return RecipeDefinition.model_validate(document), document_sha256(document)


def check(recipe: RecipeDefinition, digest: str) -> None:
    require_memory_need(
        recipe, catalog_models(), INPUTS, document_digest=digest, touched=True
    )


def test_listed_exact_content_passes() -> None:
    recipe, digest = authored()
    assert memory_need.envelope_exceptions()[SLUG].content_sha256 == digest
    check(recipe, digest)


def test_same_slug_at_another_utilization_is_refused() -> None:
    recipe, _ = authored()
    for item in recipe.runtime.arguments:
        if item.name == "gpu-memory-utilization":
            item.value = "0.84"
    edited = recipe.model_dump(mode="json", exclude_none=False)
    with pytest.raises(ValueError, match="gpu-memory-utilization"):
        check(recipe, document_sha256(edited))


def test_unlisted_slug_gets_the_envelope_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recipe, digest = authored()
    monkeypatch.setattr(memory_need, "envelope_exceptions", dict)
    with pytest.raises(ValueError, match="peak understates vLLM budget"):
        check(recipe, digest)
