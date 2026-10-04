"""Shared service cases: functional assertions gate, quality assertions inform."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from qualification.case_assertions import (
    invalid_case_assertions,
    require_valid_case_assertions,
)
from qualification.definitions_loader import load_definitions
from spark_sweep.smoke import check_assertions, check_quality

ROOT = Path(__file__).resolve().parents[1]
TEMPLATES: dict[str, Any] = json.loads(
    (ROOT / "qualification" / "shared.json").read_text(encoding="utf-8")
)["service_case_templates"]


def _chat(content: Any, **message: Any) -> dict[str, Any]:
    return {
        "model": "alias",
        "choices": [
            {
                "message": {"content": content, **message},
                "finish_reason": "stop",
            }
        ],
    }


def _tool_call(name: str, arguments: str) -> dict[str, Any]:
    return {
        "model": "alias",
        "choices": [
            {
                "message": {
                    "content": None,
                    "tool_calls": [
                        {"function": {"name": name, "arguments": arguments}}
                    ],
                },
                "finish_reason": "tool_calls",
            }
        ],
    }


def _functional(case: str, response: dict[str, Any]) -> None:
    template = json.loads(
        json.dumps(TEMPLATES[case]["assertions"]).replace("$ALIAS", "alias")
    )
    check_assertions(response, json.dumps(response), template)


def _quality(case: str, response: dict[str, Any]) -> list[dict[str, Any]]:
    return check_quality(
        case, response, json.dumps(response), TEMPLATES[case]["quality_assertions"]
    )


# (case, a wrong but well-formed reply, the right reply)
ANSWERS = [
    ("A391", _chat("305"), _chat("391")),
    ("A323", _chat("324"), _chat("323")),
    ("V_RED", _chat("blue"), _chat("red.")),
    ("V7", _chat("1"), _chat("7")),
    (
        "T_REPORT",
        _tool_call("report_temperature", '{"city": "Rotterdam"}'),
        _tool_call("report_temperature", '{"city": "Amsterdam"}'),
    ),
    (
        "T_PRODUCT",
        _tool_call("record_product", '{"left": 17, "right": 19, "product": 320}'),
        _tool_call("record_product", '{"product": 323, "right": 19, "left": 17}'),
    ),
]


@pytest.mark.parametrize(("case", "wrong", "right"), ANSWERS)
def test_a_wrong_answer_passes_the_functional_check_and_misses_quality(
    case: str, wrong: dict[str, Any], right: dict[str, Any]
) -> None:
    _functional(case, wrong)
    [miss] = _quality(case, wrong)
    assert miss["ok"] is False and miss["case"] == case and miss["got"]
    _functional(case, right)
    [hit] = _quality(case, right)
    assert hit["ok"] is True and hit["expected"]


@pytest.mark.parametrize(
    ("case", "broken"),
    [
        ("A391", _chat("The answer is 391")),
        ("A391", _chat("")),
        ("A391", _chat(None)),
        ("A391", _chat("391", reasoning_content="<think>x</think>")),
        ("A323", _chat("323 <|im_end|>")),
        ("V_RED", _chat("it is red and green")),
        ("V7", _chat("seven")),
        ("T_REPORT", _tool_call("report_temperature", "{}")),
        ("T_REPORT", _tool_call("other_tool", '{"city": "Amsterdam"}')),
        ("T_REPORT", _tool_call("report_temperature", "not json")),
        ("T_PRODUCT", _tool_call("record_product", '{"left": 17}')),
        (
            "T_PRODUCT",
            _tool_call("record_product", '{"left": "a", "right": 1, "product": 2}'),
        ),
    ],
)
def test_a_malformed_reply_fails_the_functional_check(
    case: str, broken: dict[str, Any]
) -> None:
    with pytest.raises(Exception, match="failed|contains|missing"):
        _functional(case, broken)


def test_every_shared_case_is_valid_and_quality_names_its_expectation() -> None:
    require_valid_case_assertions(load_definitions())
    assert any(t.get("quality_assertions") for t in TEMPLATES.values())


@pytest.mark.parametrize(
    ("template", "problem"),
    [
        ({"assertions": []}, "needs functional assertions"),
        (
            {
                "assertions": [{"kind": "path.nonempty", "path": "x"}],
                "quality_assertions": [
                    {"kind": "path.equals", "path": "x", "value": 1}
                ],
            },
            "must say what it expected",
        ),
        (
            {
                "assertions": [{"kind": "path.nonempty", "path": "x"}],
                "quality_assertions": [
                    {"kind": "raw.not-contains", "values": ["<think>"], "expected": "x"}
                ],
            },
            "functional, not quality",
        ),
        (
            {
                "assertions": [{"kind": "path.nonempty", "path": "x", "expected": "x"}],
            },
            "unknown fields",
        ),
        (
            {
                "assertions": [{"kind": "path.regex", "path": "x", "value": "("}],
            },
            "does not compile",
        ),
        (
            {"assertions": [{"kind": "path.shrug", "path": "x"}]},
            "unsupported kind",
        ),
    ],
)
def test_the_split_is_enforced(template: dict[str, Any], problem: str) -> None:
    problems = invalid_case_assertions({"service_case_templates": {"X": template}})
    assert any(problem in p for p in problems)
    with pytest.raises(ValueError):
        require_valid_case_assertions({"service_case_templates": {"X": template}})
