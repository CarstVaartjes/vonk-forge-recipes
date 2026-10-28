"""The sweep's trigger enum and the helper's host-memory-guard triggers are one vocabulary."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import pytest
from vonk_forge_contracts import QualificationCauseCode
from vonk_forge_contracts.sweep import (
    SweepFailureClass,
    WorkloadExitFacts,
    WorkloadMemoryTrigger,
)

from spark_sweep import policy

# Vendored verbatim from vonk-forge (the one owner of the trigger words: the
# helper's Trigger::ALL, asserted against this vector by the helper's own tests).
VECTOR = Path(__file__).parent / "fixtures" / "host-memory-guard-summary-v1.json"
VECTOR_SOURCE = (
    "https://github.com/CarstVaartjes/vonk-forge/blob/27d09b02cc0f757b3522c317fa94bffe220bcb9d/"
    "agent_protocol/src/vonk_agent_protocol/vectors/host-memory-guard-summary-v1.json"
)
VECTOR_SHA256 = "4e38a0396f466a06e49cf9cdb7cad4d2d8762740a1f1f7f08392acf71b5248ce"


def _worst_case() -> dict[str, str]:
    summaries = json.loads(VECTOR.read_text(encoding="utf-8"))["worst_case"]
    return {re.findall(r"trigger=([a-z_]+)", s)[0]: s for s in summaries}


def test_vendored_vector_is_the_published_bytes() -> None:
    assert hashlib.sha256(VECTOR.read_bytes()).hexdigest() == VECTOR_SHA256, (
        f"re-vendor from {VECTOR_SOURCE} and update the sha"
    )


def test_trigger_enum_and_helper_vector_hold_the_same_words() -> None:
    vector = set(_worst_case())
    assert vector, "the vector lists no trigger"
    for word in vector:
        assert WorkloadMemoryTrigger(word)  # a helper word the sweep cannot parse
    assert {t.value for t in WorkloadMemoryTrigger} == vector


@pytest.mark.parametrize("word", sorted(_worst_case()))
def test_every_helper_kill_summary_parses_and_is_not_a_recipe_failure(
    word: str,
) -> None:
    summary = _worst_case()[word]
    facts = WorkloadExitFacts.model_validate(
        dict(re.findall(r"(?:^|\s)(exit_cause|trigger)=([a-z_]+)(?=\s|$)", summary))
    )
    assert facts.trigger == word
    failure = policy.classify_application_failure(
        "start", "application.failed", summary
    )
    if word == WorkloadMemoryTrigger.NVIDIA_ALLOCATION_FAILURE:
        assert failure.code == QualificationCauseCode.RUNTIME_OOM
    else:
        assert failure.klass == SweepFailureClass.PLATFORM_EXIT
        assert failure.code not in policy.RECIPE_FAILURE_CODES
