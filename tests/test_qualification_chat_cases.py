"""The shared chat smoke cases test the answer, not a model's reasoning budget.

Thinking is chosen per request. A thinking-default model spends a small token
limit on reasoning and the answer is cut off, so the generic chat cases switch
thinking off in the request and leave room to pass if a model ignores it.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]
TEMPLATES: dict[str, Any] = json.loads(
    (ROOT / "qualification" / "shared.json").read_text(encoding="utf-8")
)["service_case_templates"]

GENERIC_CHAT_CASES = ("A391", "A323", "T_REPORT", "T_PRODUCT", "V_RED", "V7")

# Top-level request fields every serving engine accepts. TensorRT-LLM's server
# rejects any other top-level field (its request model forbids extras), so the
# thinking switch rides in ``chat_template_kwargs``, which vLLM, SGLang,
# llama.cpp, TensorRT-LLM, ds4 and TensorFold take or skip.
ACCEPTED_TOP_LEVEL = {
    "model",
    "messages",
    "temperature",
    "max_tokens",
    "stream",
    "seed",
    "tools",
    "tool_choice",
    "chat_template_kwargs",
}


@pytest.mark.parametrize("case", GENERIC_CHAT_CASES)
def test_a_generic_chat_case_turns_thinking_off_per_request(case: str) -> None:
    body = TEMPLATES[case]["body"]
    assert body["chat_template_kwargs"] == {
        "enable_thinking": False,
        "thinking": False,
    }
    assert set(body) <= ACCEPTED_TOP_LEVEL


@pytest.mark.parametrize("case", GENERIC_CHAT_CASES)
def test_a_generic_chat_case_can_still_pass_when_a_model_ignores_the_switch(
    case: str,
) -> None:
    assert TEMPLATES[case]["body"]["max_tokens"] >= 256


@pytest.mark.parametrize("case", ["A391", "A323", "V_RED", "V7"])
def test_the_expected_answer_stays_strictly_matched_as_quality(case: str) -> None:
    regexes = [
        a
        for a in TEMPLATES[case]["quality_assertions"]
        if a["kind"] == "path.regex" and a["path"] == "choices.0.message.content"
    ]
    assert len(regexes) == 1 and regexes[0]["value"].startswith("^")


@pytest.mark.parametrize("case", ["A391", "A323"])
def test_reasoning_tags_still_fail_the_case(case: str) -> None:
    forbidden = [
        value
        for a in TEMPLATES[case]["assertions"]
        if a["kind"] == "raw.not-contains"
        for value in a["values"]
    ]
    assert "<think>" in forbidden and "</think>" in forbidden


def test_the_cases_that_choose_their_own_thinking_are_left_alone() -> None:
    assert TEMPLATES["NEM_REASON"]["body"]["chat_template_kwargs"] == {
        "enable_thinking": True
    }
    assert TEMPLATES["NEM_A391_OFF"]["body"]["chat_template_kwargs"] == {
        "enable_thinking": False
    }
    assert "chat_template_kwargs" not in TEMPLATES["UI_ACTION"]["body"]


def test_the_sweep_sends_the_switch_with_the_rendered_case() -> None:
    from sweep_fakes import FakeRecipe, make_definitions

    recipe = FakeRecipe("glm")
    cases = make_definitions([recipe]).service_cases(recipe.key, "glm-alias")
    chat = {case["id"]: case for case in cases if case["method"] == "POST"}
    assert chat["A391"]["body"]["model"] == "glm-alias"
    assert chat["A391"]["body"]["chat_template_kwargs"]["enable_thinking"] is False
    assert chat["A391"]["body"]["max_tokens"] >= 256
