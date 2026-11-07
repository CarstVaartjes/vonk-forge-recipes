"""The catalog carries each role's measured steady peak as the sibling key ``measured_memory``."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from generated_catalog import GENERATED
from pydantic import ValidationError
from test_admission_maximum_cap import record
from vonk_forge_contracts import (
    PublishedMeasuredMemory,
    PublishedQualificationStatusView,
    PublishedRoleMemory,
    QualificationMeasuredRoleMemory,
    RecipeDefinition,
    RecipeQualificationEvidence,
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
# A pool-less vLLM recipe; the test gives it launch variants (budgets differ by option choice).
VLLM_DUAL = "qwen3-6-35b-a3b-fp8-eugr-vllm-dual"
GB = 1_000_000_000


def catalog_entries() -> list[dict[str, Any]]:
    catalog = json.loads((GENERATED / "catalog-index.json").read_text())
    return catalog["recipes"]


def digest_of(slug: str) -> str:
    return documents.recipe_digest(slug)


def measured(slug: str, *rows: tuple[str, int, int | None]):
    """Synthetic hardware proof of the recipe's content with exactly these measured rows."""
    digest = digest_of(slug)
    return RecipeQualificationEvidence(
        recipe_key=f"vonk-forge/{slug}",
        records=(record(documents.recipe(slug), digest),),
        measured_memory=tuple(
            QualificationMeasuredRoleMemory(
                content_sha256=digest,
                role=role,
                steady_bytes=used,
                budget_bytes=budget,
            )
            for role, used, budget in sorted(rows)
        ),
    )


def with_launch_variants(
    payload: dict[str, Any], utilizations: tuple[str, ...] = ()
) -> dict[str, Any]:
    """One launch (no options), or one option whose choices set these utilizations.

    The rule depends on the launch variants, so a test states them instead of
    reading whatever options the creator's recipe has today.
    """
    payload["options"] = (
        [
            {
                "name": "utilization",
                "label": "Utilization",
                "help": "Synthetic launch variants for the test.",
                "choices": [
                    {
                        "value": f"u{index}",
                        "label": f"u{index}",
                        "help": "Synthetic launch variant.",
                        "default": index == 0,
                        "args": [{"name": "gpu-memory-utilization", "value": value}],
                    }
                    for index, value in enumerate(utilizations)
                ],
            }
        ]
        if utilizations
        else []
    )
    return payload


def declaring(
    slug: str, peak_bytes: int, utilizations: tuple[str, ...] = ()
) -> RecipeDefinition:
    """The recipe with a synthetic declaration and launch variants: the rule is judged on those."""
    document = documents.recipe(slug).model_dump(mode="json", by_alias=True)
    for role in document["topology"]["roles"]:
        role["resources"]["memory"]["peak_bytes"] = peak_bytes
    return RecipeDefinition.model_validate(with_launch_variants(document, utilizations))


def published(slug: str, evidence: RecipeQualificationEvidence, recipe=None):
    return published_measured_peaks(
        recipe or documents.recipe(slug), digest_of(slug), evidence
    )


@pytest.mark.parametrize("used", [0, 1, OS_FLOOR_BYTES - 1, OS_FLOOR_BYTES, 10**11])
def test_the_conversion_is_the_one_boundary(used: int) -> None:
    assert measured_peak_bytes(used) == max(0, used - OS_FLOOR_BYTES)
    for declared in (0, measured_peak_bytes(used) - 1, measured_peak_bytes(used) + 1):
        assert holds_within(declared, used) == (measured_peak_bytes(used) <= declared)


def test_the_vector_pins_the_admission_maximum() -> None:
    assert VECTOR["admission_maximum_peak_bytes"] == ADMISSION_MAXIMUM_PEAK_BYTES


@pytest.mark.parametrize("case", VECTOR["cases"], ids=lambda case: case["name"])
def test_the_vector_is_what_the_builder_publishes(case: dict[str, Any]) -> None:
    slug = case["recipe"]
    document = documents.recipe(slug).model_dump(mode="json", by_alias=True)
    for role in document["topology"]["roles"]:
        role["resources"]["memory"]["peak_bytes"] = case["declared_peak_bytes"][
            role["name"]
        ]
    recipe = RecipeDefinition.model_validate(
        with_launch_variants(
            document, ("0.80", "0.90") if case["has_launch_options"] else ()
        )
    )
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
    recipe = declaring(slug, 100 * GB)
    other = measured(slug, ("entrypoint", 130 * GB, None))
    other = other.model_copy(
        update={
            "measured_memory": tuple(
                row.model_copy(update={"content_sha256": "f" * 64})
                for row in other.measured_memory
            )
        }
    )
    assert published(slug, other, recipe) == ()


def test_content_that_is_not_hardware_proven_publishes_nothing() -> None:
    slug = "deepseek-v4-1-flash-exl3-sfxnz-vllm-dual"
    recipe = declaring(slug, 100 * GB)
    evidence = measured(slug, ("entrypoint", 130 * GB, None))
    assert published(slug, evidence, recipe)
    unproven = evidence.model_copy(update={"records": ()})
    assert published(slug, unproven, recipe) == ()


def test_a_later_recipe_failure_withdraws_the_value() -> None:
    slug = "deepseek-v4-1-flash-exl3-sfxnz-vllm-dual"
    recipe = declaring(slug, 100 * GB)
    evidence = measured(slug, ("entrypoint", 130 * GB, None))
    assert published(slug, evidence, recipe)
    failure = record(documents.recipe(slug), digest_of(slug), failed=True, day=1)
    failed = evidence.model_copy(update={"records": (*evidence.records, failure)})
    assert published(slug, failed, recipe) == ()


def test_vllm_variants_within_the_measured_budget_publish_a_lower_value() -> None:
    recipe = declaring(VLLM_DUAL, 110 * GB, ("0.80", "0.90"))
    budgets = [vllm_budget_bytes(variant) for variant in memory_variants(recipe)]
    assert len(budgets) > 1 and None not in budgets
    used = 60 * GB
    covered = max(b for b in budgets if b is not None)
    [entry, *_] = published(
        VLLM_DUAL, measured(VLLM_DUAL, ("entrypoint", used, covered)), recipe
    )
    assert entry.measured_peak_bytes == used - OS_FLOOR_BYTES
    assert entry.peak_bytes == entry.measured_peak_bytes < 110 * GB
    # One variant above the budget that ran: not every launch is covered.
    short = min(b for b in budgets if b is not None) - 1
    assert (
        published(VLLM_DUAL, measured(VLLM_DUAL, ("entrypoint", used, short)), recipe)
        == ()
    )


def test_over_the_admission_maximum_keeps_the_declared_peak_and_flags_it() -> None:
    slug = "deepseek-v4-flash-0731-mia-sparkinfer-single"
    declared = 120 * GB
    recipe = declaring(slug, declared)
    used = ADMISSION_MAXIMUM_PEAK_BYTES + OS_FLOOR_BYTES + 1
    [entry] = published(slug, measured(slug, ("entrypoint", used, None)), recipe)
    assert entry.exceeds_admission_maximum
    assert entry.measured_peak_bytes == ADMISSION_MAXIMUM_PEAK_BYTES + 1
    assert entry.peak_bytes == declared
    at_maximum = measured(slug, ("entrypoint", used - 1, None))
    [entry] = published(slug, at_maximum, recipe)
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
    entries = catalog_entries()
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
    entries = catalog_entries()
    for entry in entries:
        assert set(entry["qualification"]) <= set(
            PublishedQualificationStatusView.model_fields
        )
