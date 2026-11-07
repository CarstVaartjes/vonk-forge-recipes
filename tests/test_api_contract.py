"""Recipe API facts are their own contract: files, shared vectors, checks, guard."""

from __future__ import annotations

import json
import runpy
import shutil
from pathlib import Path
from typing import Any

import pytest
from documents import model_display, recipe, recipe_api
from jsonschema import Draft202012Validator
from pydantic import ValidationError
from vonk_forge_contracts import (
    RecipeApi,
    RecipeApiError,
    RecipeDefinition,
    check_api_models,
    check_recipe_api,
    published_json,
    read_recipe_api,
    recipe_api_json_schema,
)
from vonk_forge_contracts.api import ApiTokenArgument
from vonk_forge_contracts.api_checks import INPUT_MODALITY_CAPABILITIES
from vonk_forge_contracts.resolver import ContractResolutionError, validate_recipe_apis

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "contracts/src/vonk_forge_contracts/examples"
INVALID = EXAMPLES / "invalid"
APIS = sorted((ROOT / "api/recipes").glob("*.json"))
GUARD = runpy.run_path(str(ROOT / "tools/check-recipe-api"))


def _json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def test_the_backfill_proves_each_shape() -> None:
    assert {path.stem for path in APIS} >= {
        "glm-5-3-flash-exl3-dflash2-tensorfold-mia-dual",
        "deepseek-v4-1-flash-1017-mia-tensorfold-dual",
        "qwen3-8-27b-exl3-dflash2-mia-single",
        "llama-3-1-8b-instruct-fp8-nvidia-trtllm-single",
    }


@pytest.mark.parametrize("path", APIS, ids=lambda path: path.stem)
def test_every_api_document_round_trips_and_agrees_with_its_recipe(path: Path) -> None:
    api = recipe_api(path.stem)
    assert published_json(api) == _json(path)
    assert read_recipe_api(published_json(api)) == api
    Draft202012Validator(recipe_api_json_schema()).validate(_json(path))
    document = recipe(path.stem)
    check_recipe_api(document, api)
    check_api_models(
        document, api, [model_display(s.model.slug) for s in document.models]
    )


def test_the_api_document_leaves_the_recipe_untouched() -> None:
    """The recipe has no API field; its digest cannot depend on the sibling."""

    assert "api" not in RecipeDefinition.model_fields


def test_an_orphan_api_document_is_refused() -> None:
    slug = "qwen3-8-27b-exl3-dflash2-mia-single"
    api = recipe_api(slug)
    with pytest.raises(ContractResolutionError, match="no recipe"):
        validate_recipe_apis(
            [recipe("llama-3-1-8b-instruct-fp8-nvidia-trtllm-single")], {slug: api}
        )
    validate_recipe_apis([recipe(slug)], {slug: api})
    validate_recipe_apis([recipe(slug)], {})


def test_option_vectors_resolve_the_block_in_force() -> None:
    vectors = _json(EXAMPLES / "api-vectors/option-choices.json")
    options = RecipeDefinition.model_validate(_json(EXAMPLES / vectors["recipe"]))
    api = RecipeApi.model_validate(_json(EXAMPLES / vectors["api"]))
    assert vectors["vectors"]
    for vector in vectors["vectors"]:
        applied = api.for_choices(options, vector["choices"])
        assert applied.model_dump(mode="json") == vector["api"], vector["choices"]


def test_an_option_choice_replaces_the_whole_block() -> None:
    document = recipe("glm-5-3-flash-exl3-dflash2-tensorfold-mia-dual")
    api = recipe_api(document.identity.slug)
    default = api.for_choices(document, {})
    text_only = api.for_choices(document, {"images_video": "text-only"})
    assert default.input_modalities.image and not default.input_modalities.video
    assert not text_only.input_modalities.image
    assert api.for_choices(document, {"drafter": "mtp"}) == default


def _refuse(contract: str, document: dict[str, Any]) -> None:
    if contract == "RecipeApi":
        RecipeApi.model_validate(document)
    elif contract == "check_recipe_api":
        api = RecipeApi.model_validate(document["api"])
        check_recipe_api(RecipeDefinition.model_validate(document["recipe"]), api)
    else:
        raise AssertionError(f"unknown contract {contract!r}")


API_VECTORS = sorted(
    path
    for path in INVALID.glob("recipe-api-*.json")
    if not path.name.endswith(".expect.json")
)


def test_there_are_invalid_vectors_for_each_named_failure() -> None:
    assert {path.stem for path in API_VECTORS} == {
        "recipe-api-context-argument-missing",
        "recipe-api-context-unbound",
        "recipe-api-default-output-names-missing-argument",
        "recipe-api-effort-default-not-a-level",
        "recipe-api-identity-mismatch",
        "recipe-api-image-with-language-model-only",
        "recipe-api-missing-field",
        "recipe-api-two-options-carry-blocks",
    }


@pytest.mark.parametrize("path", API_VECTORS, ids=lambda path: path.stem)
def test_every_invalid_api_vector_is_refused_with_its_named_error(path: Path) -> None:
    expect = _json(path.with_suffix(".expect.json"))
    with pytest.raises((ValidationError, RecipeApiError, ValueError)) as caught:
        _refuse(expect["contract"], _json(path))
    assert expect["error_contains"] in str(caught.value)


def test_a_block_with_an_unknown_mode_or_extra_field_is_refused() -> None:
    document = _json(EXAMPLES / "recipe-api-source-build.json")
    with pytest.raises(ValidationError):
        RecipeApi.model_validate(
            {**document, "api": {**document["api"], "mode": "rerank"}}
        )
    with pytest.raises(ValidationError):
        RecipeApi.model_validate({**document, "api": {**document["api"], "extra": 1}})
    read_recipe_api({**document, "extra": 1})


def _library(tmp_path: Path, *slugs: str) -> Path:
    for folder in ("recipes", "api/recipes", "models", "display/models"):
        (tmp_path / folder).mkdir(parents=True, exist_ok=True)
    for folder in ("models", "display/models"):
        for path in (ROOT / folder).glob("*.json"):
            shutil.copy(path, tmp_path / folder)
    for slug in slugs:
        shutil.copy(ROOT / f"recipes/{slug}.json", tmp_path / "recipes")
        shutil.copy(ROOT / f"api/recipes/{slug}.json", tmp_path / "api/recipes")
        (tmp_path / "display/recipes").mkdir(parents=True, exist_ok=True)
        shutil.copy(ROOT / f"display/recipes/{slug}.json", tmp_path / "display/recipes")
    return tmp_path


def test_the_guard_passes_the_backfilled_recipes(tmp_path: Path) -> None:
    slugs = [path.stem for path in APIS]
    assert GUARD["problems"](_library(tmp_path, *slugs), slugs) == []


def test_the_guard_requires_a_document_on_a_touched_openai_recipe(
    tmp_path: Path,
) -> None:
    slug = "qwen3-8-27b-exl3-dflash2-mia-single"
    root = _library(tmp_path, slug)
    (root / f"api/recipes/{slug}.json").unlink()
    assert GUARD["problems"](root, [slug]) == [f"{slug}: needs api/recipes/{slug}.json"]


def test_the_guard_refuses_a_modality_no_model_declares(tmp_path: Path) -> None:
    slug = "qwen3-8-27b-exl3-dflash2-mia-single"
    root = _library(tmp_path, slug)
    path = root / f"api/recipes/{slug}.json"
    document = _json(path)
    document["api"]["input_modalities"]["audio"] = True
    path.write_text(json.dumps(document), encoding="utf-8")
    found = GUARD["problems"](root, [slug])
    assert len(found) == 1 and "audio input needs a Model" in found[0]


def test_the_guard_refuses_an_image_the_model_does_not_declare(tmp_path: Path) -> None:
    slug = "llama-3-1-8b-instruct-fp8-nvidia-trtllm-single"
    root = _library(tmp_path, slug)
    path = root / f"api/recipes/{slug}.json"
    document = _json(path)
    document["api"]["input_modalities"]["image"] = True
    path.write_text(json.dumps(document), encoding="utf-8")
    found = GUARD["problems"](root, [slug])
    assert len(found) == 1 and "image input needs a Model" in found[0]


def test_the_guard_checks_an_option_choice_against_its_own_runtime(
    tmp_path: Path,
) -> None:
    slug = "glm-5-3-flash-exl3-dflash2-tensorfold-mia-dual"
    root = _library(tmp_path, slug)
    path = root / f"api/recipes/{slug}.json"
    document = _json(path)
    document["option_choices"][0]["api"]["max_input_tokens"] = 2**30
    path.write_text(json.dumps(document), encoding="utf-8")
    found = GUARD["problems"](root, [slug])
    assert len(found) == 1 and "option images_video=text-only" in found[0]
    assert "max_input_tokens exceeds context_tokens" in found[0]


def test_the_catalog_index_carries_each_recipes_api_or_null() -> None:
    from generated_catalog import GENERATED

    entries = _json(GENERATED / "catalog-index.json")["recipes"]
    with_api = {
        entry["document"]["identity"]["slug"]: entry["api"]
        for entry in entries
        if entry["api"] is not None
    }
    assert set(with_api) == {path.stem for path in APIS}
    for entry in entries:
        assert "api" in entry
        if entry["api"] is not None:
            assert entry["api"] == _json(
                ROOT / "api/recipes" / f"{entry['document']['identity']['slug']}.json"
            )
            assert entry["content_sha256"] == _json_digest(entry["source_path"])


def _json_digest(source_path: str) -> str:
    from vonk_forge_contracts import document_sha256

    return document_sha256(_json(ROOT / source_path))


def _chat_api(slug: str, **changes: Any) -> RecipeApi:
    """A complete chat document for ``slug`` with tools and reasoning declared."""

    document = _json(EXAMPLES / "recipe-api-source-build.json")
    context = recipe(slug).settings.model_dump()["context_tokens"]["value"]
    block = {
        **document["api"],
        "max_input_tokens": context,
        "tools": {"kind": "supported", "tool_choice": ["auto"], "parallel_calls": True},
        "reasoning": {"kind": "always-on", "effort": None},
        **changes,
    }
    identity = {"publisher": "vonk-forge", "slug": slug}
    return RecipeApi.model_validate(
        {**document, "identity": identity, "api": block, "option_choices": []}
    )


def test_the_guard_refuses_tools_and_reasoning_a_trtllm_launch_does_not_enable() -> (
    None
):
    slug = "llama-3-1-8b-instruct-fp8-nvidia-trtllm-single"
    found = GUARD["parser_problems"](ROOT, slug, recipe(slug), _chat_api(slug))
    assert len(found) == 1
    assert "tools need a tool-call parser" in found[0]
    assert "reasoning needs a reasoning parser" in found[0]
    tools_only = _chat_api(slug, reasoning={"kind": "none"})
    found = GUARD["parser_problems"](ROOT, slug, recipe(slug), tools_only)
    assert "reasoning" not in found[0] and "tools need" in found[0]


@pytest.mark.parametrize(
    "slug",
    [
        "ling-3-0-flash-dspark-sglang-single",
        "qwen3-8-27b-nvfp4-dspark-sglang-single",
        "mimo-v2-6-flash-rl-sglang-mia-dual",
        "deepseek-v4-flash-0731-mia-sparkinfer-single",
        "qwen3-8-flash-next-nvfp4-r0b0tlab-sglang-single",
    ],
)
def test_parsers_set_by_wrappers_and_kit_flags_count_as_served(slug: str) -> None:
    """The guard asks ``serving_capabilities``, which reads adapters and kit flags."""

    assert GUARD["parser_problems"](ROOT, slug, recipe(slug), _chat_api(slug)) == []


def test_a_missing_context_argument_needs_an_explicit_input_limit() -> None:
    slug = "qwen3-8-flash-next-nvfp4-r0b0tlab-sglang-single"
    declared = _chat_api(slug)
    check_recipe_api(recipe(slug), declared)
    bare = recipe(slug).model_copy(
        update={
            "runtime": recipe(slug).runtime.model_copy(
                update={
                    "arguments": [
                        a
                        for a in recipe(slug).runtime.arguments
                        if a.name != "context-length"
                    ]
                }
            )
        }
    )
    check_recipe_api(bare, declared)
    undeclared = _chat_api(slug, max_input_tokens=None)
    with pytest.raises(RecipeApiError, match="no context argument"):
        check_recipe_api(bare, undeclared)


def test_default_output_tokens_names_a_runtime_argument() -> None:
    slug = "glm-5-3-flash-exl3-dflash2-tensorfold-mia-dual"
    document = recipe(slug)
    api = recipe_api(slug)
    assert api.api.default_output_tokens.model_dump() == {"argument": "max-tokens"}  # type: ignore[union-attr]
    check_recipe_api(document, api)
    other = api.model_copy(
        update={
            "api": api.api.model_copy(
                update={"default_output_tokens": ApiTokenArgument(argument="nope")}
            )
        }
    )
    with pytest.raises(RecipeApiError, match="--nope, which has no integer value"):
        check_recipe_api(document, other)
    # A setting-bound argument resolves through the recipe's settings.
    bound = api.api.model_copy(
        update={"default_output_tokens": ApiTokenArgument(argument="context")}
    )
    check_recipe_api(document, api.model_copy(update={"api": bound}))


def test_the_synthetic_decision_example_is_checked_like_any_recipe() -> None:
    document = RecipeDefinition.model_validate(_json(EXAMPLES / "recipe-decision.json"))
    api = RecipeApi.model_validate(_json(EXAMPLES / "recipe-api-decision.json"))
    assert api.api.mode == "decision"
    check_recipe_api(document, api)


def test_image_input_has_one_capability_name() -> None:
    assert INPUT_MODALITY_CAPABILITIES["image"] == "image-understanding"


@pytest.mark.parametrize(
    ("modality", "capability"), sorted(INPUT_MODALITY_CAPABILITIES.items())
)
def test_an_accepted_input_modality_needs_its_understanding_capability(
    modality: str, capability: str
) -> None:
    """A Model that lists the modality without the capability does not satisfy it."""

    slug = "qwen3-8-27b-mia-tensorfold-single"
    document = recipe(slug)
    raw = _json(ROOT / "api" / "recipes" / f"{slug}.json")
    raw["api"]["input_modalities"][modality] = True
    api = RecipeApi.model_validate(raw)
    shown = [model_display(s.model.slug) for s in document.models]
    complete = [
        d.model_copy(
            update={
                "modalities": sorted({*d.modalities, modality}),
                "capabilities": sorted({*d.capabilities, capability}),
            }
        )
        for d in shown
    ]
    check_api_models(document, api, complete)
    listed_only = [
        d.model_copy(
            update={
                "modalities": sorted({*d.modalities, modality}),
                "capabilities": [c for c in d.capabilities if c != capability],
            }
        )
        for d in shown
    ]
    with pytest.raises(
        RecipeApiError,
        match=f"{modality} input needs a Model with the {modality} modality "
        f"and the {capability} capability",
    ):
        check_api_models(document, api, listed_only)


def test_changed_slugs_keeps_only_recipe_and_api_documents() -> None:
    assert GUARD["changed_slugs"](
        [
            "recipes/added.json",
            "api/recipes/api-edited.json",
            "recipes/edited.json",
            "api/recipes/edited.json",
            "notes.txt",
            "recipes/readme.md",
            "adapters/x/recipes/nested.json",
            "display/recipes/other.json",
        ]
    ) == ["added", "api-edited", "edited"]


def test_exllamav3_context_arguments_name_per_request_context_only() -> None:
    from vonk_forge_contracts.api_checks import CONTEXT_ARGUMENTS

    row = CONTEXT_ARGUMENTS["exllamav3"]
    # TabbyAPI's cache-size is the shared KV pool, not a request's context.
    assert "cache-size" not in row
    # The memory estimator reads row[0]; Mia's cache_size is its context.
    assert row[0] == "cache_size"
