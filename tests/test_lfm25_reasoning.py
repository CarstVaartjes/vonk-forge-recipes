"""LFM2.5 always opens a thinking turn; the service must separate it from content."""

from __future__ import annotations

import json
from pathlib import Path

from documents import recipe_display

from qualification.capabilities import generic_case_problems, serving_capabilities
from qualification.definitions_loader import load_definitions

ROOT = Path(__file__).resolve().parents[1]
SLUG = "lfm2-5-24b-a2b-bf16-vllm-single"


def test_lfm25_serves_parsed_reasoning_and_keeps_strict_smoke() -> None:
    recipe = json.loads((ROOT / "recipes" / f"{SLUG}.json").read_text())
    cases = load_definitions()["service_recipes"][f"vonk-forge/{SLUG}"]["smoke_cases"]
    display = recipe_display(SLUG)
    caps = serving_capabilities(recipe, SLUG, {}, display)
    assert caps.reasoning_parser
    arguments = {
        arg["name"]: arg.get("value") for arg in recipe["runtime"]["arguments"]
    }
    # The pinned template opens <think>; qwen3 starts in reasoning mode and
    # consumes </think> even though the opening token is in the prompt.
    assert arguments["reasoning-parser"] == "qwen3"
    assert arguments["tool-call-parser"] == "lfm2"
    assert "reasoning" in display.tags
    assert "A323" in cases
    assert generic_case_problems(SLUG, recipe, cases, caps, display) == []

    # Removing the parser must trigger the shared capability guard, rather than
    # letting an always-thinking model pass strict content checks unparsed.
    recipe["runtime"]["arguments"] = [
        arg
        for arg in recipe["runtime"]["arguments"]
        if arg["name"] != "reasoning-parser"
    ]
    assert generic_case_problems(
        SLUG, recipe, cases, serving_capabilities(recipe, SLUG, {}, display), display
    )
    assert all(option["name"] != "thinking" for option in recipe.get("options", []))
