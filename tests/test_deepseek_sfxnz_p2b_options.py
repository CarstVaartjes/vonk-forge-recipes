"""Verify SFXNZ's user-selectable P2B modes bind the existing compiled kernels."""

from __future__ import annotations

import json
from pathlib import Path

from vonk_forge_contracts import read_recipe

ROOT = Path(__file__).resolve().parents[1]
RECIPE = ROOT / "recipes/deepseek-v4-1-flash-exl3-sfxnz-vllm-dual.json"


def test_p2b_choices_resolve_to_compiled_mode_environment_on_each_runtime() -> None:
    recipe = read_recipe(json.loads(RECIPE.read_text(encoding="utf-8")))
    option = next(option for option in recipe.options if option.name == "p2b_moe")
    assert {choice.value for choice in option.choices} == {
        "dataflow",
        "cooperative",
        "off",
    }
    assert (
        next(choice.value for choice in option.choices if choice.default) == "dataflow"
    )

    expected = {"dataflow": "2", "cooperative": "1", "off": "0"}
    for choice, mode in expected.items():
        resolved = recipe.with_option_choices({"p2b_moe": choice})
        values = {entry.name: entry.value for entry in resolved.runtime.environment}
        assert values["DSV41_P2B_COOP"] == mode
        assert resolved.runtime.engine == recipe.runtime.engine
        assert resolved.runtime.arguments == recipe.runtime.arguments
