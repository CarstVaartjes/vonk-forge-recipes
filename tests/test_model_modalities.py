"""The Model capability guard: modalities, understanding and served routes agree."""

from __future__ import annotations

import runpy
import shutil
from pathlib import Path

from vonk_forge_contracts import ChatApi, DecisionApi
from vonk_forge_contracts.api import ApiReasoningNone, ApiToolsNone

from qualification.documents import has_recipe_api, recipe, recipe_api

ROOT = Path(__file__).resolve().parents[1]
GUARD = runpy.run_path(str(ROOT / "tools/check-model-modalities"))


def test_understanding_capabilities_and_input_modalities_agree() -> None:
    check = GUARD["disagreements"]
    assert check(["chat", "image-understanding"], ["image", "text"]) == []
    assert len(check(["chat", "image-understanding"], ["text"])) == 1
    assert len(check(["chat"], ["text", "video"])) == 1
    # A d3 decision Model has no text-generation capability and is still covered.
    assert len(check(["decision", "video-understanding"], ["image", "text"])) == 2


def test_generators_are_not_checked_and_only_model_files_are_named() -> None:
    assert GUARD["disagreements"](["video-generation"], ["video"]) == []
    assert GUARD["changed_slugs"](
        ["display/models/a.json", "models/b.json", "display/models/x/c.json"]
    ) == ["a"]


def test_a_text_generating_model_is_judged_not_skipped() -> None:
    refused = GUARD["disagreements"](
        ["text-generation", "image-understanding"], ["text"]
    )
    assert len(refused) == 1
    code = (["code-generation", "video-understanding"], ["text"])
    assert len(GUARD["disagreements"](*code)) == 1


def test_roles_come_from_the_contract_not_from_name_suffixes() -> None:
    from vonk_forge_contracts import CapabilityName, CapabilityRole

    assert CapabilityName("text-to-speech").role is CapabilityRole.GENERATOR
    assert CapabilityName("image-upscaling").role is CapabilityRole.GENERATOR
    assert CapabilityName("code-generation").role is CapabilityRole.OTHER
    assert CapabilityName("video-understanding").role is CapabilityRole.UNDERSTANDING
    assert {c for c in CapabilityName if c.role is CapabilityRole.UNDERSTANDING} == set(
        GUARD["INPUT_MODALITY_CAPABILITIES"].values()
    )


def test_every_capability_has_an_explicit_role() -> None:
    from vonk_forge_contracts import CapabilityName
    from vonk_forge_contracts.capability import CAPABILITY_ROLES

    assert set(CAPABILITY_ROLES) == set(CapabilityName)


def test_a_generator_with_text_plus_media_modalities_is_not_refused() -> None:
    check = GUARD["disagreements"]
    assert check(["image-upscaling"], ["image", "text"]) == []
    assert check(["voice-cloning"], ["audio", "text"]) == []
    assert check(["voice-design", "audio-generation"], ["audio", "text"]) == []


def _api_recipes() -> list[str]:
    return [
        p.stem
        for p in sorted((ROOT / "recipes").glob("*.json"))
        if has_recipe_api(p.stem, ROOT)
    ]


def _chat_recipe() -> str:
    """A catalog recipe whose base route is chat with reasoning and tools, and an option."""

    return next(
        slug
        for slug in _api_recipes()
        if isinstance(block := recipe_api(slug, ROOT).api, ChatApi)
        and not isinstance(block.reasoning, ApiReasoningNone)
        and not isinstance(block.tools, ApiToolsNone)
        and recipe(slug, ROOT).options
    )


def _with_blocks(api, base: dict, choices: list[dict]):
    """The API document with its base block and option blocks replaced."""

    return type(api).model_validate(
        api.model_dump(mode="json") | {"api": base, "option_choices": choices}
    )


def test_a_chat_route_requires_chat_text_reasoning_and_tool_use() -> None:
    slug = _chat_recipe()
    document, api = recipe(slug, ROOT), recipe_api(slug, ROOT)
    base = api.api.model_dump(mode="json")
    route = GUARD["route_capabilities"]
    assert route(document, _with_blocks(api, base, [])) == {
        "chat",
        "text-generation",
        "reasoning",
        "tool-use",
    }
    off = base | {"reasoning": {"kind": "none"}, "tools": {"kind": "none"}}
    assert route(document, _with_blocks(api, off, [])) == {"chat", "text-generation"}


def test_a_completions_only_route_requires_text_generation_not_chat() -> None:
    slug = _chat_recipe()
    document, api = recipe(slug, ROOT), recipe_api(slug, ROOT)
    base = api.api.model_dump(mode="json")
    off = base | {
        "reasoning": {"kind": "none"},
        "tools": {"kind": "none"},
        "endpoints": ["/v1/completions"],
    }
    route = GUARD["route_capabilities"]
    assert route(document, _with_blocks(api, off, [])) == {"text-generation"}
    for endpoints in (
        ["/v1/completions", "/v1/chat/completions"],
        ["/v1/responses"],
        ["/v1/messages"],
    ):
        block = off | {"endpoints": endpoints}
        assert route(document, _with_blocks(api, block, [])) == {
            "chat",
            "text-generation",
        }


def test_one_option_variant_that_serves_tools_requires_tool_use() -> None:
    slug = _chat_recipe()
    document, api = recipe(slug, ROOT), recipe_api(slug, ROOT)
    option = document.options[0]
    other = next(c.value for c in option.choices if not c.default)
    base = api.api.model_dump(mode="json")
    off = base | {"reasoning": {"kind": "none"}, "tools": {"kind": "none"}}
    choice = {
        "option": option.name,
        "value": other,
        "api": off | {"tools": base["tools"]},
    }
    assert GUARD["route_capabilities"](document, _with_blocks(api, off, [choice])) == {
        "chat",
        "text-generation",
        "tool-use",
    }


def test_a_decision_route_requires_no_chat() -> None:
    slug = next(
        s for s in _api_recipes() if isinstance(recipe_api(s, ROOT).api, DecisionApi)
    )
    assert (
        GUARD["route_capabilities"](recipe(slug, ROOT), recipe_api(slug, ROOT)) == set()
    )


def test_missing_names_each_absent_capability_and_its_recipes() -> None:
    missing = GUARD["missing"]
    required = {"chat": ["r1"], "tool-use": ["r1", "r2"]}
    assert missing(["chat", "tool-use", "text-generation"], required) == []
    assert missing(["chat"], required) == [
        "needs capability 'tool-use': served by r1, r2"
    ]
    assert len(missing([], required)) == 2
    assert missing([], {}) == []


def test_only_the_served_model_of_a_recipe_with_an_api_document_is_required(
    tmp_path: Path,
) -> None:
    multi = next(
        s
        for s in _api_recipes()
        if isinstance(recipe_api(s, ROOT).api, ChatApi)
        and len(recipe(s, ROOT).models) > 1
    )
    without = next(
        p.stem
        for p in sorted((ROOT / "recipes").glob("*.json"))
        if not (ROOT / "api/recipes" / p.name).is_file()
    )
    (tmp_path / "api/recipes").mkdir(parents=True)
    (tmp_path / "recipes").mkdir()
    for slug in (multi, without):
        shutil.copy(ROOT / "recipes" / f"{slug}.json", tmp_path / "recipes")
    shutil.copy(ROOT / "api/recipes" / f"{multi}.json", tmp_path / "api/recipes")

    document = recipe(multi, ROOT)
    served = document.models[0].model.slug
    required = GUARD["served_requirements"](tmp_path)
    assert set(required) == {served}
    assert required[served]["chat"] == [multi]
    assert required[served]["text-generation"] == [multi]


def test_a_changed_recipe_or_api_document_names_its_served_model() -> None:
    slug = _chat_recipe()
    served = recipe(slug, ROOT).models[0].model.slug
    changed = GUARD["changed_slugs"]
    assert changed([f"recipes/{slug}.json"]) == [served]
    assert changed([f"api/recipes/{slug}.json", "display/models/a.json"]) == sorted(
        {served, "a"}
    )
    assert changed(["recipes/no-such-recipe.json", f"recipes/{slug}.md"]) == []
