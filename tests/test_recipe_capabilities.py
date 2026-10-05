"""A recipe claims, and the sweep tests, only what its engine configuration serves.

vLLM, SGLang and TensorRT-LLM parse tool calls and reasoning only when the
launch asks them to. A recipe that follows a playbook or kit which passes no
parser must not tag ``tool-use`` or run tool smoke cases: vLLM answers HTTP 400
and the others return the raw ``<tool_call>`` text in ``content``. The same
holds for image cases and a recipe that is not multimodal.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from qualification.capabilities import (
    ROOT,
    UNPARSED_ARITHMETIC,
    Capabilities,
    generic_case_problems,
    serving_capabilities,
)
from qualification.case_assertions import invalid_case_assertions
from qualification.definitions_loader import load_definitions

DEFINITIONS = load_definitions()
MODELS = {
    document["identity"]["slug"]: document
    for document in (
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted((ROOT / "models").glob("*.json"))
    )
}


def _recipe(slug: str) -> dict[str, Any]:
    return json.loads((ROOT / "recipes" / f"{slug}.json").read_text(encoding="utf-8"))


SERVICE_SLUGS = sorted(key.split("/", 1)[1] for key in DEFINITIONS["service_recipes"])


@pytest.mark.parametrize("slug", SERVICE_SLUGS)
def test_a_recipes_smoke_cases_and_tags_match_what_its_engine_serves(slug: str) -> None:
    recipe = _recipe(slug)
    caps = serving_capabilities(recipe, slug, MODELS)
    cases = DEFINITIONS["service_recipes"][f"vonk-forge/{slug}"]["smoke_cases"]
    assert generic_case_problems(slug, recipe, cases, caps) == []


def test_every_smoke_case_names_a_reviewed_template() -> None:
    templates = DEFINITIONS["service_case_templates"]
    for key, service in DEFINITIONS["service_recipes"].items():
        assert set(service["smoke_cases"]) <= set(templates), key


def _caps(**changes: bool) -> Capabilities:
    values = {"tools": False, "reasoning_parser": False, "vision": False}
    values.update(changes)
    return Capabilities(flags_from="test", **values)


def _vllm(*arguments: tuple[str, Any], tags: tuple[str, ...] = ()) -> dict[str, Any]:
    return {
        "metadata": {"tags": list(tags)},
        "models": [],
        "runtime": {
            "engine": "vllm",
            "arguments": [{"name": n, "value": v} for n, v in arguments],
        },
    }


def test_vllm_tools_need_both_the_parser_and_auto_tool_choice() -> None:
    both = _vllm(("tool-call-parser", "hermes"), ("enable-auto-tool-choice", True))
    parser_only = _vllm(("tool-call-parser", "hermes"))
    assert serving_capabilities(both, "x", {}).tools
    assert not serving_capabilities(parser_only, "x", {}).tools
    assert not serving_capabilities(_vllm(), "x", {}).tools


def test_sglang_and_trtllm_use_their_own_flag_spelling() -> None:
    sglang = _vllm(("tool-call-parser", "qwen"), ("reasoning-parser", "qwen3"))
    sglang["runtime"]["engine"] = "sglang"
    trtllm = _vllm(("tool_parser", "qwen3"), ("reasoning_parser", "qwen3"))
    trtllm["runtime"]["engine"] = "tensorrt-llm"
    for recipe in (sglang, trtllm):
        caps = serving_capabilities(recipe, "x", {})
        assert caps.tools and caps.reasoning_parser


def test_a_tool_case_without_a_parser_is_reported() -> None:
    recipe = _vllm()
    problems = generic_case_problems(
        "x", recipe, ["M0", "A323", "T_PRODUCT"], serving_capabilities(recipe, "x", {})
    )
    assert any("tool cases" in problem for problem in problems)


def test_the_tool_use_tag_must_match_the_parser() -> None:
    recipe = _vllm(tags=("tool-use",))
    problems = generic_case_problems(
        "x", recipe, ["M0", "A323"], serving_capabilities(recipe, "x", {})
    )
    assert any("tag tool-use" in problem for problem in problems)


def test_a_reasoning_model_without_a_parser_uses_the_unparsed_arithmetic_case() -> None:
    recipe = _vllm(tags=("reasoning",))
    caps = serving_capabilities(recipe, "x", {})
    assert generic_case_problems("x", recipe, ["M0", "A323"], caps)
    assert not generic_case_problems("x", recipe, ["M0", "A323_UNPARSED"], caps)
    parsed = _vllm(("reasoning-parser", "qwen3"), tags=("reasoning",))
    parsed_caps = serving_capabilities(parsed, "x", {})
    assert not generic_case_problems("x", parsed, ["M0", "A323"], parsed_caps)
    assert generic_case_problems("x", parsed, ["M0", "A323_UNPARSED"], parsed_caps)


def test_an_image_case_needs_a_multimodal_recipe() -> None:
    recipe = _vllm()
    assert generic_case_problems(
        "x", recipe, ["M0", "V7"], serving_capabilities(recipe, "x", {})
    )
    assert not generic_case_problems("x", recipe, ["M0", "V7"], _caps(vision=True))


def test_unparsed_reasoning_is_valid_content_not_a_functional_failure() -> None:
    templates = DEFINITIONS["service_case_templates"]
    for strict, relaxed in UNPARSED_ARITHMETIC.items():
        functional = json.dumps(templates[relaxed]["assertions"])
        assert "<think>" not in functional and "</think>" not in functional
        assert "finish_reason" not in functional
        quality = json.dumps(templates[relaxed]["quality_assertions"])
        assert "<think>" in quality
        assert "<think>" in json.dumps(templates[strict]["assertions"])
    assert invalid_case_assertions(DEFINITIONS) == []


def test_every_kit_flag_entry_cites_its_source_and_names_a_recipe() -> None:
    entries = json.loads(
        Path(ROOT / "qualification" / "kit-engine-flags.json").read_text("utf-8")
    )
    for slug, entry in entries.items():
        assert (ROOT / "recipes" / f"{slug}.json").exists(), slug
        assert entry["source"] and (
            entry.get("tool_parser") or entry.get("reasoning_parser")
        )
