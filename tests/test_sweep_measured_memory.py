"""A sweep pass writes what it measured into its evidence; nothing else is guessed."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
from sweep_fakes import GB, FakeModel, FakeRecipe, Gateway, make_sweep
from vonk_forge_contracts import (
    QualificationMeasuredMemory,
    QualificationRoleMemory,
    RecipeDefinition,
)

from qualification.memory_need import TOTAL_BYTES, vllm_budget_bytes
from spark_sweep import memory
from spark_sweep.qualification import from_history

pytest_plugins = ["sweep_bounds"]

ROOT = Path(__file__).resolve().parents[1]
DOCUMENT = json.loads(
    (ROOT / "recipes/agents-a1-nvfp4-r0b0tlab-vllm-single.json").read_text()
)
RECIPE = RecipeDefinition.model_validate(DOCUMENT)
BUDGET = vllm_budget_bytes(RECIPE)
ALIAS = "served"


def observed(*nodes: tuple[str, int, bool]) -> dict[str, Any]:
    return {
        "declared_bytes": 1,
        "nodes": [
            {"node_id": node, "used_bytes": used, "alone": alone}
            for node, used, alone in nodes
        ],
    }


def application(*placements: tuple[str, str], choices: dict[str, str] | None = None):
    return {
        "progress": {
            "intended_profile": {
                "assignments": [
                    {"alias": "other", "nodes": []},
                    {
                        "alias": ALIAS,
                        "nodes": [
                            {"node_id": node, "role": role} for node, role in placements
                        ],
                        "option_choices": choices or {},
                    },
                ]
            }
        }
    }


def test_the_budget_is_the_launchs_own_utilization_share() -> None:
    assert BUDGET is not None
    assert BUDGET == pytest.approx(0.70 * TOTAL_BYTES, abs=1)


def test_each_role_keeps_the_largest_use_of_the_sparks_it_had_to_itself() -> None:
    measured = memory.measurement(
        observed(("n1", 90 * GB, True), ("n2", 80 * GB, True), ("n3", 99 * GB, False)),
        application(("n1", "worker"), ("n2", "worker"), ("n3", "entrypoint")),
        ALIAS,
        RECIPE,
    )
    assert measured == QualificationMeasuredMemory(
        budget_bytes=BUDGET,
        roles=(QualificationRoleMemory(role="worker", steady_bytes=90 * GB),),
    )


@pytest.mark.parametrize(
    "sample,placed,recipe",
    [
        (None, application(("n1", "entrypoint")), RECIPE),
        (observed(("n1", GB, False)), application(("n1", "entrypoint")), RECIPE),
        (observed(("n1", GB, True)), application(("n9", "entrypoint")), RECIPE),
        (observed(("n1", GB, True)), {"progress": {}}, RECIPE),
        (observed(("n1", GB, True)), application(("n1", "entrypoint")), None),
        (
            observed(("n1", GB, True)),
            application(("n1", "entrypoint"), choices={"no-such-option": "x"}),
            RECIPE,
        ),
    ],
)
def test_what_cannot_be_told_apart_is_not_measured(sample, placed, recipe) -> None:
    assert memory.measurement(sample, placed, ALIAS, recipe) is None


def test_only_the_document_of_the_content_that_ran_gives_the_budget() -> None:
    detail = {"identity": {"content_sha256": "a" * 64}, "document": DOCUMENT}
    assert memory.ran_recipe(detail, "a" * 64) == RECIPE
    assert memory.ran_recipe(detail, "b" * 64) is None
    assert memory.ran_recipe({"error": "not found"}, "a" * 64) is None


def row(**extra: Any) -> dict[str, Any]:
    return {
        "recipe": "vonk-forge/example",
        "step": "smoke",
        "status": "passed",
        "recorded_at": datetime(2026, 11, 4, tzinfo=UTC).isoformat(),
        "content_sha256": "a" * 64,
        "node_count": 1,
        **extra,
    }


MEASURED = QualificationMeasuredMemory(
    budget_bytes=BUDGET,
    roles=(QualificationRoleMemory(role="entrypoint", steady_bytes=97 * GB),),
)


def test_a_pass_row_carries_its_measurement_into_the_record() -> None:
    record = from_history(row(measured_memory=MEASURED.model_dump(mode="json")))
    assert record is not None
    assert record.memory == MEASURED


def test_rows_without_a_measurement_publish_the_same_record() -> None:
    """Old rows are "not measured": republishing them never adds a duplicate record."""
    record = from_history(row())
    assert record is not None and record.memory is None
    assert "memory" not in record.model_dump(mode="json", exclude_none=True)


@pytest.mark.parametrize(
    "extra",
    [
        {"measured_memory": {"roles": []}},
        {"measured_memory": "unreadable"},
        {"status": "failed", "measured_memory": MEASURED.model_dump(mode="json")},
    ],
)
def test_an_unreadable_or_failed_measurement_never_costs_the_verdict(extra) -> None:
    record = from_history(row(**extra))
    assert record is not None
    assert record.memory is None


@pytest.fixture
def gateway():
    gateway = Gateway()
    yield gateway
    gateway.stop()


@pytest.mark.usefixtures("bounded_clock")
@pytest.mark.parametrize("shown", [True, False])
def test_a_sweep_pass_records_the_measured_memory(
    tmp_path: Path, gateway: Gateway, shown: bool
) -> None:
    sweep, _, _ = make_sweep(
        tmp_path,
        [
            FakeRecipe(
                "a",
                memory=100 * GB,
                observed_memory=93 * GB,
                detail=DOCUMENT if shown else None,
            )
        ],
        [FakeModel("m1")],
        gateway=gateway,
    )
    sweep.run()
    assert sweep.state.recipes["vonk-forge/a"]["status"] == "passed"
    [smoke] = [
        json.loads(line)
        for line in (tmp_path / "results.jsonl").read_text().splitlines()
        if json.loads(line).get("step") == "smoke"
    ]
    if not shown:
        assert "measured_memory" not in smoke
        return
    assert smoke["measured_memory"] == {
        "budget_bytes": BUDGET,
        # What the Spark held (the fake's 4 GB OS share included), never the declaration.
        "roles": [{"role": "entrypoint", "steady_bytes": 97 * GB}],
    }
