# SPDX-License-Identifier: Apache-2.0
"""Complete Clef-Flash's native SystemOne answer into the platform's decision response.

``joint_schema_model.systemone`` returns ``model``, ``answers`` and ``usage``.
The platform's decision contract (``vonk_forge_contracts.systemone``) also needs
``id``, ``object``, ``created``, per-question ``diagnostics`` and, on every
answer, ``probabilities`` and ``confidence``. Everything native is kept
(``choice``, ``noul``, ``score``, ``legend``, ``usage``); what is added follows
from the native numbers:

* The head scores only the options the question allows, so the probability
  mass on the labels is 1.0 and the argmax is always a label by construction.
* ``noul`` is the head's probability of "true"; ``probabilities`` is
  ``{yes: p, no: 1 - p}``.
* The native probabilities are rounded to four decimals, so each question's
  distribution is renormalised to sum to one, and ``score`` and ``confidence``
  are recomputed from it (the contract checks them to 1e-5).
"""

from __future__ import annotations

import time
import uuid
from typing import Any


def _normalised(probabilities: dict[str, float]) -> dict[str, float]:
    total = sum(probabilities.values())
    if total <= 0:
        raise ValueError("decision answer has no probability mass")
    return {key: value / total for key, value in probabilities.items()}


def platform_response(request: dict[str, Any], native: dict[str, Any]) -> dict[str, Any]:
    answers: dict[str, Any] = {}
    diagnostics: dict[str, Any] = {}
    for name, answer in native["answers"].items():
        completed = dict(answer)
        if answer["type"] == "noul":
            yes = min(max(float(answer["noul"]), 0.0), 1.0)
            probabilities = {"yes": yes, "no": 1.0 - yes}
            completed["noul"] = yes
        else:
            probabilities = _normalised(answer["probabilities"])
        if answer["type"] == "score":
            completed["score"] = sum(
                index * probabilities[str(index)] for index in range(len(probabilities))
            )
        completed["probabilities"] = probabilities
        completed["confidence"] = max(probabilities.values())
        answers[name] = completed
        diagnostics[name] = {"label_mass": 1.0, "argmax_is_label": True}
    return {
        "id": str(request.get("request_id") or f"decision-{uuid.uuid4().hex}"),
        "object": "structured_decision",
        "created": int(time.time()),
        "model": native["model"],
        "answers": answers,
        "usage": native["usage"],
        "diagnostics": diagnostics,
    }
