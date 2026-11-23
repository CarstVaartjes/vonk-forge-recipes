"""The sweep's WorkloadExitCause enum and the helper's ExitCause tokens are one vocabulary."""

from __future__ import annotations

import hashlib
import json

import pytest
from repo_root import FIXTURES

from spark_sweep import policy
from spark_sweep.sweep_contract import SweepFailureClass, WorkloadExitCause
from vonk_forge_contracts import QualificationCauseCode

# Vendored verbatim from vonk-forge (the one owner of the exit-cause tokens:
# its agent_protocol tests prove the vector equals ExitCause::token()).
VECTOR = FIXTURES / "exit-causes-v1.json"
VECTOR_SOURCE = (
    "https://github.com/CarstVaartjes/vonk-forge/blob/02f7a66b4273e1ed803ef03ef6b9fe9215af12dd/"
    "agent_protocol/src/vonk_agent_protocol/vectors/exit-causes-v1.json"
)
VECTOR_SHA256 = "bab73138c2a519af755221ce99f3679914ac17007461157a88e6487bbbcd8bb3"

RECIPE_EXIT = {
    "bad_arguments",
    "python_exception",
    "missing_file",
    "unsupported_gpu_arch",
    "image_arch_mismatch",
}


def _tokens() -> set[str]:
    return set(json.loads(VECTOR.read_text(encoding="utf-8"))["tokens"].values())


def test_vendored_vector_is_the_published_bytes() -> None:
    assert hashlib.sha256(VECTOR.read_bytes()).hexdigest() == VECTOR_SHA256, (
        f"re-vendor from {VECTOR_SOURCE} and update the sha"
    )


def test_exit_cause_enum_and_helper_vector_hold_the_same_tokens() -> None:
    assert {c.value for c in WorkloadExitCause} == _tokens()


@pytest.mark.parametrize("token", sorted(_tokens()))
def test_every_helper_exit_cause_is_classified_by_the_sweep(token: str) -> None:
    failure = policy.classify_application_failure(
        "start", "runtime_process_exited", f"exit_cause={token}"
    )
    assert failure.evidence["exit_cause"] == token  # parsed as a fact, not dropped
    if token in RECIPE_EXIT:
        assert failure.code == QualificationCauseCode.RUNTIME_EXIT
    elif token == "oom_killed":
        assert failure.code == QualificationCauseCode.RUNTIME_OOM
    else:
        assert failure.klass == SweepFailureClass.PLATFORM_EXIT
        assert failure.code not in policy.RECIPE_FAILURE_CODES
