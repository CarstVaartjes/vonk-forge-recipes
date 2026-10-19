"""Decision responses must preserve their request's options and probabilities."""

from __future__ import annotations

import copy
import json

import pytest
from pydantic import ValidationError
from vonk_forge_contracts import RecipeDefinition
from vonk_forge_contracts.systemone import (
    SystemOneRequest,
    SystemOneResponse,
    validate_decision_result,
)


def decision_exchange() -> tuple[SystemOneRequest, SystemOneResponse]:
    request = SystemOneRequest.model_validate_json(
        json.dumps(
            {
                "model": "d3-edge",
                "state": {"text": "Write Python code", "optional": None},
                "questions": {
                    "kind": {
                        "type": "choice",
                        "criteria": {"code": "Code", "chat": None},
                    },
                    "reasoning": {"type": "noul"},
                    "difficulty": {"type": "score", "criteria": ["easy", "hard"]},
                },
            }
        )
    )
    response = SystemOneResponse.model_validate_json(
        json.dumps(
            {
                "id": "decision-example",
                "object": "structured_decision",
                "created": 1,
                "model": "d3-edge",
                "answers": {
                    "kind": {
                        "type": "choice",
                        "choice": "code",
                        "probabilities": {"code": 0.8, "chat": 0.2},
                        "confidence": 0.8,
                    },
                    "reasoning": {
                        "type": "noul",
                        "noul": 0.6,
                        "probabilities": {"yes": 0.6, "no": 0.4},
                        "confidence": 0.6,
                    },
                    "difficulty": {
                        "type": "score",
                        "score": 0.3,
                        "legend": {"0": "easy", "1": "hard"},
                        "probabilities": {"0": 0.7, "1": 0.3},
                        "confidence": 0.7,
                    },
                },
                "usage": {"input_tokens": 30, "output_tokens": 0},
                "diagnostics": {
                    name: {"label_mass": 1.0, "argmax_is_label": True}
                    for name in request.questions
                },
                "engine_extension": {"value": None},
            }
        )
    )
    return request, response


def test_decision_roundtrip_preserves_typed_questions_and_json_extensions() -> None:
    request, response = decision_exchange()
    validate_decision_result(request, response)
    decoded = SystemOneRequest.model_validate_json(
        request.model_dump_json(exclude_unset=True)
    )
    assert decoded.state == {"text": "Write Python code", "optional": None}
    assert decoded.questions["kind"].criteria == {"code": "Code", "chat": None}
    assert SystemOneResponse.model_validate_json(response.model_dump_json()) == response
    assert response.model_extra == {"engine_extension": {"value": None}}


@pytest.mark.parametrize(
    "mutation",
    ["missing", "options", "sum", "choice", "noul", "score", "confidence", "type"],
)
def test_decision_result_rejects_inconsistent_answers(mutation: str) -> None:
    request, response = decision_exchange()
    data = copy.deepcopy(response.model_dump())
    if mutation == "missing":
        del data["answers"]["reasoning"]
    elif mutation == "options":
        data["answers"]["kind"]["probabilities"] = {"wrong": 0.8, "chat": 0.2}
    elif mutation == "sum":
        data["answers"]["kind"]["probabilities"]["code"] = 0.7
    elif mutation == "choice":
        data["answers"]["kind"]["choice"] = "chat"
    elif mutation == "type":
        data["answers"]["kind"]["type"] = "noul"
    else:
        question = {"noul": "reasoning", "score": "difficulty", "confidence": "kind"}[
            mutation
        ]
        data["answers"][question][mutation] = 0.1
    with pytest.raises(ValueError):
        validate_decision_result(
            request, SystemOneResponse.model_validate_json(json.dumps(data))
        )


@pytest.mark.parametrize("value", [-0.1, 1.1, float("nan"), float("inf")])
def test_decision_result_rejects_invalid_probability_values(value: float) -> None:
    _, response = decision_exchange()
    data = response.model_dump()
    data["answers"]["kind"]["probabilities"]["code"] = value
    with pytest.raises(ValidationError):
        SystemOneResponse.model_validate_json(json.dumps(data))


def test_recipe_roundtrip_preserves_decision_settings_and_native_probe() -> None:
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    document = json.loads(
        (
            root
            / "contracts/src/vonk_forge_contracts/examples/recipe-source-build.json"
        ).read_text()
    )
    document["settings"]["kind"] = "decision"
    request, _ = decision_exchange()
    check = {
        "name": "decision",
        "kind": "openai.systemone",
        "request": {
            "transport": "http",
            "method": "POST",
            "path": "/v1/systemone",
            "body": request.model_dump(exclude_unset=True),
        },
        "assertions": ["decision.probabilities"],
    }
    document["validation"]["serving"]["checks"] = [check]
    recipe = RecipeDefinition.model_validate_json(json.dumps(document))
    assert RecipeDefinition.model_validate_json(recipe.model_dump_json()) == recipe
    document["validation"]["serving"]["checks"][0]["request"]["path"] = (
        "/v1/chat/completions"
    )
    with pytest.raises(ValidationError, match="systemone"):
        RecipeDefinition.model_validate_json(json.dumps(document))
