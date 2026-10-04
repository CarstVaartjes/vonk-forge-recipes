"""Service smoke cases separate what must work from what is only information.

A case's ``assertions`` are functional and gate a recipe: the server answers, the
reply is well formed, nothing leaked. Its optional ``quality_assertions`` name
the exact answer the model is expected to give. Answer quality, like speed,
belongs to the model and the recipe's creator, so a miss is reported as a
quality note and never fails a recipe. Every quality assertion says what it
``expected`` in words, so the note is readable without the pattern.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any

KINDS = frozenset(
    {
        "array.path-count-equals",
        "path.count",
        "path.empty",
        "path.equals",
        "path.json-equals",
        "path.lte",
        "path.nonempty",
        "path.regex",
        "raw.not-contains",
    }
)
_FIELDS = frozenset({"kind", "path", "value", "values", "item_path", "count"})


def _assertion_problems(case: str, tier: str, assertion: object) -> list[str]:
    where = f"{case} {tier} assertion"
    if not isinstance(assertion, Mapping):
        return [f"{where} must be an object"]
    problems: list[str] = []
    kind = assertion.get("kind")
    if kind not in KINDS:
        problems.append(f"{where} has an unsupported kind {kind!r}")
    allowed = _FIELDS | ({"expected"} if tier == "quality" else set())
    extra = sorted(set(assertion) - allowed)
    if extra:
        problems.append(f"{where} has unknown fields {extra}")
    if kind == "path.regex":
        try:
            re.compile(str(assertion.get("value")))
        except re.error:
            problems.append(f"{where} has a regex that does not compile")
    if tier == "quality":
        if kind == "raw.not-contains":
            problems.append(f"{case}: a leaked tag is functional, not quality")
        if not isinstance(assertion.get("expected"), str) or not assertion["expected"]:
            problems.append(f"{where} must say what it expected")
    return problems


def invalid_case_assertions(definitions: Mapping[str, Any]) -> list[str]:
    """Every way a shared service case breaks the functional/quality split."""
    problems: list[str] = []
    for case, template in definitions.get("service_case_templates", {}).items():
        functional = template.get("assertions")
        if not isinstance(functional, list) or not functional:
            problems.append(f"{case} needs functional assertions")
            functional = []
        quality = template.get("quality_assertions", [])
        if not isinstance(quality, list):
            problems.append(f"{case} quality_assertions must be a list")
            quality = []
        for tier, items in (("functional", functional), ("quality", quality)):
            for assertion in items:
                problems += _assertion_problems(case, tier, assertion)
    return problems


def require_valid_case_assertions(definitions: Mapping[str, Any]) -> None:
    problems = invalid_case_assertions(definitions)
    if problems:
        raise ValueError("service cases are invalid: " + "; ".join(problems))
