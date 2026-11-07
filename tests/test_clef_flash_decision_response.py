"""Clef-Flash's native SystemOne answer becomes a valid platform decision result."""

from __future__ import annotations

import importlib.util
from pathlib import Path

from vonk_forge_contracts.systemone import (
    SystemOneRequest,
    SystemOneResponse,
    validate_decision_result,
)

ADAPTER = Path(__file__).resolve().parents[1] / "adapters/llm/clef-flash-transformers"


def _load():
    spec = importlib.util.spec_from_file_location(
        "clef_decision_response", ADAPTER / "decision_response.py"
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


REQUEST = {
    "model": "clef-flash",
    "state": "Our checkout started returning errors and orders are blocked.",
    "questions": {
        "department": {
            "type": "choice",
            "criteria": {"billing": "Payments", "technical": "Outages"},
        },
        "urgency": {"type": "score", "criteria": ["Can wait", "This week", "Today"]},
        "outage": {"type": "noul", "instructions": "Is a service down?"},
    },
}
# The shape and four-decimal rounding of joint_schema_model.systemone().
NATIVE = {
    "model": "clef-flash",
    "answers": {
        "department": {
            "type": "choice",
            "choice": "technical",
            "confidence": 0.9552,
            "probabilities": {"billing": 0.0448, "technical": 0.9552},
        },
        "urgency": {
            "type": "score",
            "score": 1.7968,
            "confidence": 0.8663,
            "legend": {"0": "Can wait", "1": "This week", "2": "Today"},
            "probabilities": {"0": 0.0695, "1": 0.0642, "2": 0.8663},
        },
        "outage": {"type": "noul", "noul": 0.8347},
    },
    "usage": {"input_tokens": 300, "output_tokens": 0},
}


def test_native_answer_validates_against_the_platform_contract() -> None:
    result = _load().platform_response(REQUEST, NATIVE)
    validate_decision_result(
        SystemOneRequest.model_validate(REQUEST),
        SystemOneResponse.model_validate(result),
    )
    assert result["object"] == "structured_decision"
    assert result["answers"]["outage"]["probabilities"].keys() == {"yes", "no"}
    assert result["answers"]["department"]["choice"] == "technical"
    assert result["usage"] == NATIVE["usage"]


def test_request_id_is_the_response_id() -> None:
    result = _load().platform_response({**REQUEST, "request_id": "abc"}, NATIVE)
    assert result["id"] == "abc"
