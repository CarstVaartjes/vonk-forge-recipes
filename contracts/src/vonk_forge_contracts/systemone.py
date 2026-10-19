"""Typed System One HTTP payloads and decision-result validation.

Based on vLLM e68efc685ab1e6f1eda004ced7157e3ca411da2f. The endpoint
retains engine-owned extensions and JSON state, including meaningful nulls.
"""

from __future__ import annotations

import math
from typing import Annotated, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    StrictBool,
    StrictInt,
    StrictStr,
)


class _Payload(BaseModel):
    model_config = ConfigDict(extra="allow", strict=True, allow_inf_nan=False)


Probability = Annotated[float, Field(ge=0, le=1)]


class SystemOneQuestion(_Payload):
    type: StrictStr
    instructions: JsonValue = ""
    criteria: JsonValue = None


class SystemOneRequest(_Payload):
    model: StrictStr | None = None
    state: JsonValue
    questions: dict[StrictStr, SystemOneQuestion] = Field(min_length=1)
    instructions: StrictStr | None = None
    images: list[StrictStr] | None = None
    audio: list[StrictStr] | None = None
    videos: list[StrictStr] | None = None
    chat_template_kwargs: dict[StrictStr, JsonValue] | None = None
    cache_salt: StrictStr | None = None
    priority: StrictInt = Field(default=0, ge=-(2**63), le=2**63 - 1)
    request_id: StrictStr | None = None


class SystemOneAnswer(_Payload):
    type: StrictStr
    probabilities: dict[StrictStr, Probability] = Field(min_length=1)
    confidence: Probability
    choice: StrictStr | None = None
    noul: Probability | None = None
    score: float | None = None
    legend: dict[StrictStr, StrictStr] | None = None


class SystemOneUsage(_Payload):
    input_tokens: Annotated[StrictInt, Field(ge=0)]
    output_tokens: Annotated[StrictInt, Field(ge=0)]


class SystemOneDiagnostics(_Payload):
    label_mass: Probability
    argmax_is_label: StrictBool


class SystemOneResponse(_Payload):
    id: StrictStr
    object: Literal["structured_decision"]
    created: StrictInt
    model: StrictStr
    answers: dict[StrictStr, SystemOneAnswer]
    usage: SystemOneUsage
    diagnostics: dict[StrictStr, SystemOneDiagnostics]


def validate_decision_result(
    request: SystemOneRequest, response: SystemOneResponse
) -> None:
    """Reject missing questions, invalid distributions and inconsistent results."""
    if response.answers.keys() != request.questions.keys():
        raise ValueError("decision response must answer every requested question")
    if response.diagnostics.keys() != request.questions.keys():
        raise ValueError("decision response must diagnose every requested question")
    for name, question in request.questions.items():
        answer = response.answers[name]
        if answer.type != question.type:
            raise ValueError(f"decision {name}: answer type differs from the question")
        probabilities = answer.probabilities
        if not math.isclose(sum(probabilities.values()), 1, rel_tol=1e-5, abs_tol=1e-5):
            raise ValueError(f"decision {name}: probabilities must sum to one")
        if question.type == "choice":
            if (
                not isinstance(question.criteria, dict)
                or probabilities.keys() != question.criteria.keys()
            ):
                raise ValueError(
                    f"decision {name}: option probabilities differ from the question"
                )
            if answer.choice not in probabilities or probabilities[
                answer.choice
            ] != max(probabilities.values()):
                raise ValueError(
                    f"decision {name}: choice must select a most probable option"
                )
        elif question.type == "noul":
            if (
                probabilities.keys() != {"yes", "no"}
                or answer.noul is None
                or not math.isclose(answer.noul, probabilities["yes"], abs_tol=1e-6)
            ):
                raise ValueError(
                    f"decision {name}: noul must equal the yes probability"
                )
        elif question.type == "score":
            if not isinstance(question.criteria, list):
                raise ValueError(f"decision {name}: score criteria must be a list")
            legend = {str(i): str(level) for i, level in enumerate(question.criteria)}
            if answer.legend != legend or probabilities.keys() != legend.keys():
                raise ValueError(
                    f"decision {name}: score levels differ from the question"
                )
            expected = sum(i * probabilities[str(i)] for i in range(len(legend)))
            if answer.score is None or not math.isclose(
                answer.score, expected, abs_tol=1e-5
            ):
                raise ValueError(
                    f"decision {name}: score differs from its expected level"
                )
        confidence = max(probabilities.values()) * response.diagnostics[name].label_mass
        if not math.isclose(answer.confidence, confidence, abs_tol=1e-5):
            raise ValueError(f"decision {name}: confidence differs from its label mass")
