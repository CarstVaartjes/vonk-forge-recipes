"""A hardware pass records the memory it measured, bound to its content identity."""

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError
from vonk_forge_contracts import (
    QualificationBlame,
    QualificationCauseCode,
    QualificationEvidenceRecord,
    QualificationIdentity,
    QualificationMeasuredMemory,
    QualificationOutcome,
    QualificationPlatform,
    QualificationRoleMemory,
    QualificationSummary,
    RecipeQualificationEvidence,
)

NOW = datetime(2026, 11, 4, tzinfo=UTC)
MEASURED = QualificationMeasuredMemory(
    budget_bytes=96_000_000_000,
    roles=(
        QualificationRoleMemory(role="entrypoint", steady_bytes=101_000_000_000),
        QualificationRoleMemory(role="worker", steady_bytes=99_000_000_000),
    ),
)


def passed(
    memory: QualificationMeasuredMemory | None = None,
) -> QualificationEvidenceRecord:
    return QualificationEvidenceRecord(
        identity=QualificationIdentity(content_sha256="a" * 64, required_sparks=2),
        outcome=QualificationOutcome.PASSED,
        cause_code=QualificationCauseCode.HARDWARE_PASS,
        cause_summary=QualificationSummary.PASSED,
        tested_at=NOW,
        platform=QualificationPlatform(spark_count=2),
        memory=memory,
    )


def test_a_pass_round_trips_its_measured_memory() -> None:
    record = passed(memory=MEASURED)
    evidence = RecipeQualificationEvidence(recipe_key="vonk-forge/x", records=(record,))
    again = RecipeQualificationEvidence.model_validate_json(
        evidence.model_dump_json(exclude_none=True)
    )
    assert again.records[0].memory == MEASURED


def test_a_record_without_a_measurement_keeps_its_bytes() -> None:
    """Existing evidence is "not measured": no migration, the same published bytes."""
    record = passed()
    assert record.memory is None
    assert "memory" not in record.model_dump(mode="json", exclude_none=True)


def test_only_a_pass_records_measured_memory() -> None:
    with pytest.raises(ValidationError, match="only a hardware pass"):
        QualificationEvidenceRecord(
            identity=QualificationIdentity(content_sha256="a" * 64),
            outcome=QualificationOutcome.FAILED,
            blame=QualificationBlame.RECIPE,
            cause_code=QualificationCauseCode.RUNTIME_OOM,
            cause_summary=QualificationSummary.RUNTIME_OOM,
            tested_at=NOW,
            platform=QualificationPlatform(),
            memory=MEASURED,
        )


@pytest.mark.parametrize(
    "roles",
    [
        (),
        ("worker", "entrypoint"),
        ("entrypoint", "entrypoint"),
    ],
)
def test_measured_roles_are_nonempty_sorted_and_unique(roles: tuple[str, ...]) -> None:
    with pytest.raises(ValidationError):
        QualificationMeasuredMemory(
            roles=tuple(
                QualificationRoleMemory(role=name, steady_bytes=1) for name in roles
            )
        )


@pytest.mark.parametrize("field", ["node_id", "hostname", "used_bytes"])
def test_a_measurement_carries_no_infrastructure_or_unnamed_fields(field: str) -> None:
    payload = passed(memory=MEASURED).model_dump(mode="json")
    payload["memory"]["roles"][0][field] = "spk_private"
    with pytest.raises(ValidationError):
        QualificationEvidenceRecord.model_validate(payload)
