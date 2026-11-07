"""The catalog carries each role's measured steady peak as the sibling key ``measured_memory``."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from generated_catalog import GENERATED
from pydantic import ValidationError
from vonk_forge_contracts import (
    PublishedMeasuredMemory,
    PublishedQualificationStatusView,
    PublishedRoleMemory,
    QualificationBlame,
    QualificationCauseCode,
    QualificationMeasuredRoleMemory,
    QualificationOutcome,
    QualificationSummary,
    RecipeDefinition,
    RecipeQualificationEvidence,
    document_sha256,
)

from qualification import documents
from qualification.memory_need import (
    ADMISSION_MAXIMUM_PEAK_BYTES,
    OS_FLOOR_BYTES,
    holds_within,
    measured_peak_bytes,
    memory_variants,
    published_measured_peaks,
    vllm_budget_bytes,
)

ROOT = Path(__file__).resolve().parents[1]
VECTOR = json.loads(
    (
        ROOT
        / "contracts/src/vonk_forge_contracts/examples/measured-memory-vectors"
        / "catalog-measured-memory-v1.json"
    ).read_text()
)
# One pool-less vLLM recipe with launch variants (budgets differ by option choice).
VLLM_DUAL = "qwen3-6-35b-a3b-fp8-eugr-vllm-dual"
GB = 1_000_000_000


def evidence_of(slug: str) -> RecipeQualificationEvidence:
    return RecipeQualificationEvidence.model_validate_json(
        (ROOT / "qualification/evidence/vonk-forge" / f"{slug}.json").read_text()
    )


def digest_of(slug: str) -> str:
    return document_sha256(json.loads((ROOT / "recipes" / f"{slug}.json").read_text()))


def measured(slug: str, *rows: tuple[str, int, int | None]):
    """The recipe's real hardware proof with exactly these measured rows."""
    digest = digest_of(slug)
    return evidence_of(slug).model_copy(
        update={
            "measured_memory": tuple(
                QualificationMeasuredRoleMemory(
                    content_sha256=digest,
                    role=role,
                    steady_bytes=used,
                    budget_bytes=budget,
                )
                for role, used, budget in sorted(rows)
            )
        }
    )


def published(slug: str, evidence: RecipeQualificationEvidence, recipe=None):
    return published_measured_peaks(
        recipe or documents.recipe(slug), digest_of(slug), evidence
    )


@pytest.mark.parametrize("used", [0, 1, OS_FLOOR_BYTES - 1, OS_FLOOR_BYTES, 10**11])
def test_the_conversion_is_the_one_boundary(used: int) -> None:
    assert measured_peak_bytes(used) == max(0, used - OS_FLOOR_BYTES)
    for declared in (0, measured_peak_bytes(used) - 1, measured_peak_bytes(used) + 1):
        assert holds_within(declared, used) == (measured_peak_bytes(used) <= declared)


def test_the_vector_pins_the_shared_constants() -> None:
    assert VECTOR["os_floor_bytes"] == OS_FLOOR_BYTES
    assert VECTOR["admission_maximum_peak_bytes"] == ADMISSION_MAXIMUM_PEAK_BYTES


@pytest.mark.parametrize("case", VECTOR["cases"], ids=lambda case: case["name"])
def test_the_vector_is_what_the_builder_publishes(case: dict[str, Any]) -> None:
    slug = case["recipe"]
    document = documents.recipe(slug).model_dump(mode="json", by_alias=True)
    for role in document["topology"]["roles"]:
        role["resources"]["memory"]["peak_bytes"] = case["declared_peak_bytes"][
            role["name"]
        ]
    recipe = RecipeDefinition.model_validate(document)
    rows = tuple(
        (row["role"], row["steady_bytes"], row["budget_bytes"])
        for row in case["evidence_rows"]
    )
    peaks = published(slug, measured(slug, *rows), recipe)
    expected = case["measured_memory"]
    assert (
        PublishedMeasuredMemory(roles=peaks).model_dump(mode="json") if peaks else None
    ) == expected
    # The Controller admits on peak_bytes: measured, or declared when over the maximum.
    for item in expected["roles"] if expected else ():
        assert item["peak_bytes"] == (
            case["declared_peak_bytes"][item["role"]]
            if item["exceeds_admission_maximum"]
            else item["measured_peak_bytes"]
        )


def test_no_rows_publish_nothing() -> None:
    slug = "deepseek-v4-1-flash-exl3-sfxnz-vllm-dual"
    assert published(slug, measured(slug)) == ()


def test_a_row_of_another_content_is_ignored() -> None:
    slug = "deepseek-v4-1-flash-exl3-sfxnz-vllm-dual"
    other = measured(slug, ("entrypoint", 130 * GB, None))
    other = other.model_copy(
        update={
            "measured_memory": tuple(
                row.model_copy(update={"content_sha256": "f" * 64})
                for row in other.measured_memory
            )
        }
    )
    assert published(slug, other) == ()


def test_content_that_is_not_hardware_proven_publishes_nothing() -> None:
    slug = "deepseek-v4-1-flash-exl3-sfxnz-vllm-dual"
    evidence = measured(slug, ("entrypoint", 130 * GB, None))
    assert published(slug, evidence)
    unproven = evidence.model_copy(update={"records": ()})
    assert published(slug, unproven) == ()


def test_a_later_recipe_failure_withdraws_the_value() -> None:
    slug = "deepseek-v4-1-flash-exl3-sfxnz-vllm-dual"
    evidence = measured(slug, ("entrypoint", 130 * GB, None))
    digest = digest_of(slug)
    failures = tuple(
        record.model_copy(
            update={
                "outcome": QualificationOutcome.FAILED,
                "blame": QualificationBlame.RECIPE,
                "cause_code": QualificationCauseCode.RECIPE_FAILURE,
                "cause_summary": QualificationSummary.RECIPE_FAILED,
                "tested_at": record.tested_at.replace(year=record.tested_at.year + 1),
            }
        )
        for record in evidence.records
        if record.identity.content_sha256 == digest
        and record.outcome == QualificationOutcome.PASSED
    )
    failed = evidence.model_copy(update={"records": (*evidence.records, *failures)})
    assert published(slug, failed) == ()


def test_vllm_variants_within_the_measured_budget_publish_a_lower_value() -> None:
    recipe = documents.recipe(VLLM_DUAL)
    budgets = [vllm_budget_bytes(variant) for variant in memory_variants(recipe)]
    assert len(budgets) > 1 and None not in budgets
    used = 60 * GB
    covered = max(b for b in budgets if b is not None)
    assert covered > 0
    [entry] = published(VLLM_DUAL, measured(VLLM_DUAL, ("entrypoint", used, covered)))[
        :1
    ]
    assert entry.measured_peak_bytes == used - OS_FLOOR_BYTES
    assert (
        entry.measured_peak_bytes < recipe.topology.roles[0].resources.memory.peak_bytes
    )
    # One variant above the budget that ran: not every launch is covered.
    short = min(b for b in budgets if b is not None) - 1
    assert published(VLLM_DUAL, measured(VLLM_DUAL, ("entrypoint", used, short))) == ()


def test_over_the_admission_maximum_keeps_the_declared_peak_and_flags_it() -> None:
    slug = "deepseek-v4-flash-0731-mia-sparkinfer-single"
    recipe = documents.recipe(slug)
    declared = recipe.topology.roles[0].resources.memory.peak_bytes
    used = ADMISSION_MAXIMUM_PEAK_BYTES + OS_FLOOR_BYTES + 1
    [entry] = published(slug, measured(slug, ("entrypoint", used, None)))
    assert entry.exceeds_admission_maximum
    assert entry.measured_peak_bytes == ADMISSION_MAXIMUM_PEAK_BYTES + 1
    assert entry.peak_bytes == declared
    at_maximum = measured(slug, ("entrypoint", used - 1, None))
    [entry] = published(slug, at_maximum)
    assert not entry.exceeds_admission_maximum
    assert entry.peak_bytes == entry.measured_peak_bytes == ADMISSION_MAXIMUM_PEAK_BYTES


def test_only_an_over_maximum_role_may_admit_below_its_measurement() -> None:
    with pytest.raises(ValidationError, match="over-maximum"):
        PublishedRoleMemory(role="entrypoint", peak_bytes=1, measured_peak_bytes=2)
    PublishedRoleMemory(
        role="entrypoint",
        peak_bytes=1,
        measured_peak_bytes=2,
        exceeds_admission_maximum=True,
    )


def test_roles_are_sorted_and_unique() -> None:
    def role(name: str) -> PublishedRoleMemory:
        return PublishedRoleMemory(role=name, peak_bytes=1, measured_peak_bytes=1)

    for roles in (
        (role("worker"), role("entrypoint")),
        (role("worker"), role("worker")),
    ):
        with pytest.raises(ValidationError, match="sorted and unique"):
            PublishedMeasuredMemory(roles=roles)
    with pytest.raises(ValidationError):
        PublishedMeasuredMemory(roles=())


def test_the_catalog_carries_the_sibling_key_exactly_where_rows_publish() -> None:
    entries = json.loads((GENERATED / "catalog-index.json").read_text())["recipes"]
    keyed = 0
    for entry in entries:
        document = entry["document"]
        slug = document["identity"]["slug"]
        key = f"{document['identity']['publisher']}/{slug}"
        expected = published_measured_peaks(
            RecipeDefinition.model_validate(document),
            entry["content_sha256"],
            RecipeQualificationEvidence.model_validate_json(
                (ROOT / "qualification/evidence" / f"{key}.json").read_text()
            )
            if (ROOT / "qualification/evidence" / f"{key}.json").is_file()
            else RecipeQualificationEvidence(recipe_key=key),
        )
        if expected:
            keyed += 1
            assert (
                PublishedMeasuredMemory.model_validate(entry["measured_memory"]).roles
                == expected
            ), slug
        else:
            assert "measured_memory" not in entry, slug
    assert keyed


def test_the_status_view_older_controllers_read_never_carries_the_field() -> None:
    """Older Controllers reject unknown status fields; the field is a sibling key."""
    assert "measured_memory" not in PublishedQualificationStatusView.model_fields
    entries = json.loads((GENERATED / "catalog-index.json").read_text())["recipes"]
    for entry in entries:
        assert set(entry["qualification"]) <= set(
            PublishedQualificationStatusView.model_fields
        )
