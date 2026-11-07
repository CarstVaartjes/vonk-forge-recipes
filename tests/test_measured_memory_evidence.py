"""Measured memory is its own evidence table, keyed by (content_sha256, role)."""

from pathlib import Path

import pytest
from pydantic import ValidationError
from vonk_forge_contracts import (
    QualificationMeasuredRoleMemory,
    RecipeQualificationEvidence,
)

from qualification import documents

ROOT = Path(__file__).resolve().parents[1]


def row(
    content: str = "a", role: str = "entrypoint"
) -> QualificationMeasuredRoleMemory:
    return QualificationMeasuredRoleMemory(
        content_sha256=content * 64,
        role=role,
        steady_bytes=101_000_000_000,
        budget_bytes=96_000_000_000,
    )


def test_the_table_round_trips() -> None:
    evidence = RecipeQualificationEvidence(
        recipe_key="vonk-forge/x",
        measured_memory=(row("a"), row("a", "worker"), row("b")),
    )
    again = RecipeQualificationEvidence.model_validate_json(
        evidence.model_dump_json(exclude_none=True)
    )
    assert again == evidence


def test_every_committed_evidence_file_keeps_its_bytes() -> None:
    """No migration: an empty table is left out, so today's files are unchanged."""
    paths = sorted((ROOT / "qualification/evidence").rglob("*.json"))
    assert paths
    for path in paths:
        text = path.read_text()
        evidence = RecipeQualificationEvidence.model_validate_json(text)
        assert evidence.model_dump_json(indent=2, exclude_none=True) + "\n" == text


@pytest.mark.parametrize(
    "rows",
    [
        (row("b"), row("a")),
        (row("a", "worker"), row("a", "entrypoint")),
        (row("a"), row("a")),
    ],
)
def test_rows_are_sorted_and_unique_by_content_and_role(rows) -> None:
    with pytest.raises(ValidationError, match="sorted and unique"):
        RecipeQualificationEvidence(recipe_key="vonk-forge/x", measured_memory=rows)


@pytest.mark.parametrize(
    "field,value",
    [
        ("node_id", "spk_private"),
        ("hostname", "spark-1"),
        ("role", "Not A Role"),
        ("content_sha256", "rev-1"),
        ("steady_bytes", -1),
    ],
)
def test_a_row_carries_no_infrastructure_and_only_valid_values(field, value) -> None:
    payload = row().model_dump(mode="json") | {field: value}
    with pytest.raises(ValidationError):
        QualificationMeasuredRoleMemory.model_validate(payload)


def test_no_job_recipe_has_a_measured_row() -> None:
    """A job's post-run sample is idle, not a peak: steady rows are serving rows."""
    for path in sorted((ROOT / "qualification/evidence").rglob("*.json")):
        evidence = RecipeQualificationEvidence.model_validate_json(path.read_text())
        if evidence.measured_memory:
            assert documents.recipe(path.stem).settings.kind != "job", path.name
