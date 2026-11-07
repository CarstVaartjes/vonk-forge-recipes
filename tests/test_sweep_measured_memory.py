"""A sweep pass writes what it measured into its evidence table; nothing is guessed."""

from __future__ import annotations

import json
import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
from sweep_fakes import GB, FakeModel, FakeRecipe, Gateway, make_sweep
from vonk_forge_contracts import (
    QualificationMeasuredRoleMemory,
    RecipeQualificationEvidence,
)

from qualification import documents
from qualification.documents import ROOT
from qualification.memory_need import TOTAL_BYTES, vllm_budget_bytes
from spark_sweep import memory
from spark_sweep.qualification import history_evidence, merge_evidence

pytest_plugins = ["sweep_bounds"]

SINGLE = documents.recipe("agents-a1-nvfp4-r0b0tlab-vllm-single")
# As the Controller's technical detail shows it: the canonical definition.
DOCUMENT = SINGLE.model_dump(mode="json", by_alias=True)
DUAL = documents.recipe("qwen3-6-35b-a3b-fp8-eugr-vllm-dual")
ALIAS = "served"
RAN = "a" * 64


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
                    {"nodes": []},  # an assignment without an alias is not this run
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


def row(role: str, used: int, budget: int | None, content: str = RAN):
    return QualificationMeasuredRoleMemory(
        content_sha256=content, role=role, steady_bytes=used, budget_bytes=budget
    )


def test_the_budget_is_the_launchs_own_utilization_share() -> None:
    budget = vllm_budget_bytes(SINGLE)
    assert budget == pytest.approx(0.70 * TOTAL_BYTES, abs=1)


def test_each_declared_role_keeps_the_largest_use_of_sparks_it_had_alone() -> None:
    budget = vllm_budget_bytes(DUAL)
    rows = memory.measurement(
        observed(
            ("n1", 90 * GB, True),
            ("n2", 80 * GB, True),
            ("n3", 99 * GB, False),
            ("n4", 70 * GB, True),
        ),
        application(
            ("n1", "worker"),
            ("n2", "worker"),
            ("n3", "entrypoint"),
            ("n4", "unresolved"),
        ),
        ALIAS,
        DUAL,
        RAN,
    )
    assert rows == (row("worker", 90 * GB, budget),)


def test_the_budget_follows_the_option_choices_that_ran() -> None:
    choices = {"speculation": "native-mtp"}
    [measured] = memory.measurement(
        observed(("n1", GB, True)),
        application(("n1", "entrypoint"), choices=choices),
        ALIAS,
        DUAL,
        RAN,
    )
    assert measured.budget_bytes == vllm_budget_bytes(DUAL.with_option_choices(choices))


@pytest.mark.parametrize(
    "sample,placed,recipe",
    [
        (None, application(("n1", "entrypoint")), SINGLE),
        (observed(("n1", GB, False)), application(("n1", "entrypoint")), SINGLE),
        (observed(("n1", GB, True)), application(("n9", "entrypoint")), SINGLE),
        (observed(("n1", GB, True)), {"progress": {}}, SINGLE),
        (observed(("n1", GB, True)), application(("n1", "entrypoint")), None),
        (
            observed(("n1", GB, True)),
            application(("n1", "entrypoint"), choices={"no-such-option": "x"}),
            SINGLE,
        ),
    ],
)
def test_what_cannot_be_told_apart_is_not_measured(sample, placed, recipe) -> None:
    assert memory.measurement(sample, placed, ALIAS, recipe, RAN) == ()


@pytest.mark.parametrize("error", [ValueError, TypeError, ArithmeticError])
def test_a_budget_that_cannot_be_computed_is_not_measured(
    monkeypatch: pytest.MonkeyPatch, error: type[Exception]
) -> None:
    """E.g. a utilization bound to a knob: argument_value raises; the pass stands."""

    def broken(_recipe: object) -> int:
        raise error("no declared value")

    monkeypatch.setattr(memory, "vllm_budget_bytes", broken)
    sample, placed = observed(("n1", GB, True)), application(("n1", "entrypoint"))
    assert memory.measurement(sample, placed, ALIAS, SINGLE, RAN) == ()


def test_only_the_document_of_the_content_that_ran_gives_the_budget() -> None:
    detail = {"identity": {"content_sha256": RAN}, "document": DOCUMENT}
    assert memory.ran_recipe(detail, RAN) == SINGLE
    assert memory.ran_recipe(detail, "b" * 64) is None
    assert memory.ran_recipe({"error": "not found"}, RAN) is None


def history(**extra: Any) -> dict[str, Any]:
    return {
        "recipe": "vonk-forge/example",
        "step": "smoke",
        "status": "passed",
        "recorded_at": datetime(2026, 11, 4, tzinfo=UTC).isoformat(),
        "content_sha256": RAN,
        "node_count": 1,
        **extra,
    }


MEASURED = row("entrypoint", 97 * GB, 91 * GB)


def test_a_pass_row_puts_its_measurement_in_the_table() -> None:
    evidence = history_evidence(
        history(measured_memory=[MEASURED.model_dump(mode="json")])
    )
    assert evidence is not None
    assert evidence.measured_memory == (MEASURED,)
    assert len(evidence.records) == 1


def test_rows_without_a_measurement_publish_the_same_evidence() -> None:
    """Old rows are "not measured": republishing them never changes a file."""
    evidence = history_evidence(history())
    assert evidence is not None and evidence.measured_memory == ()
    assert "measured_memory" not in evidence.model_dump(mode="json", exclude_none=True)


def test_a_failed_row_measures_nothing() -> None:
    evidence = history_evidence(
        history(status="failed", measured_memory=[MEASURED.model_dump(mode="json")])
    )
    assert evidence is not None and evidence.measured_memory == ()


def test_the_same_content_and_role_merge_to_the_conservative_side() -> None:
    def table(*rows: QualificationMeasuredRoleMemory) -> RecipeQualificationEvidence:
        return RecipeQualificationEvidence(
            recipe_key="vonk-forge/example", measured_memory=rows
        )

    other = row("entrypoint", GB, None, content="b" * 64)
    merged = merge_evidence(
        table(row("entrypoint", 90 * GB, 80 * GB)),
        table(row("entrypoint", 95 * GB, 85 * GB), other),
    )
    assert merged.measured_memory == (row("entrypoint", 95 * GB, 80 * GB), other)
    unbudgeted = merge_evidence(
        merged, table(row("entrypoint", GB, None), row("worker", GB, 1))
    )
    assert unbudgeted.measured_memory == (
        row("entrypoint", 95 * GB, None),
        row("worker", GB, 1),
        other,
    )


@pytest.fixture
def gateway():
    gateway = Gateway()
    yield gateway
    gateway.stop()


def smoke_row(tmp_path: Path) -> dict[str, Any]:
    [smoke] = [
        json.loads(line)
        for line in (tmp_path / "results.jsonl").read_text().splitlines()
        if json.loads(line).get("step") == "smoke"
    ]
    return smoke


@pytest.mark.usefixtures("bounded_clock")
@pytest.mark.parametrize("shown", [True, False])
def test_a_sweep_pass_records_the_measured_memory(
    tmp_path: Path, gateway: Gateway, shown: bool
) -> None:
    fake = FakeRecipe(
        "a",
        memory=100 * GB,
        observed_memory=93 * GB,
        content=RAN,
        detail=DOCUMENT if shown else None,
    )
    sweep, _, _ = make_sweep(tmp_path, [fake], [FakeModel("m1")], gateway=gateway)
    sweep.run()
    assert sweep.state.recipes["vonk-forge/a"]["status"] == "passed"
    smoke = smoke_row(tmp_path)
    if not shown:
        assert "measured_memory" not in smoke
        return
    assert smoke["measured_memory"] == [
        {
            "content_sha256": fake.content,
            "role": "entrypoint",
            # What the Spark held (the fake's 4 GB OS share included), never the declaration.
            "steady_bytes": 97 * GB,
            "budget_bytes": vllm_budget_bytes(SINGLE),
        }
    ]


@pytest.mark.usefixtures("bounded_clock")
def test_a_measurement_that_raises_never_costs_the_pass(
    tmp_path: Path, gateway: Gateway, monkeypatch: pytest.MonkeyPatch
) -> None:
    def broken(_recipe: object) -> int:
        raise ValueError("gpu-memory-utilization has no declared value")

    monkeypatch.setattr(memory, "vllm_budget_bytes", broken)
    fake = FakeRecipe("a", observed_memory=GB, content=RAN, detail=DOCUMENT)
    sweep, _, _ = make_sweep(tmp_path, [fake], [FakeModel("m1")], gateway=gateway)
    sweep.run()
    assert sweep.state.recipes["vonk-forge/a"]["status"] == "passed"
    smoke = smoke_row(tmp_path)
    assert smoke["status"] == "passed" and "measured_memory" not in smoke


def test_a_job_recipe_is_never_measured_its_sample_is_taken_after_the_job() -> None:
    job = documents.recipe("ltx-2-19b-distilled-diffusers-single")
    assert job.settings.kind == "job"
    assert (
        memory.measurement(
            observed(("n1", GB, True)),
            application(("n1", "entrypoint")),
            ALIAS,
            job,
            RAN,
        )
        == ()
    )


def test_a_job_pass_row_replayed_through_backfill_adds_no_measured_rows(
    tmp_path: Path,
) -> None:
    from spark_sweep.qualification import backfill, read_evidence

    (tmp_path / "recipes").mkdir()
    for slug in (
        "ltx-2-19b-distilled-diffusers-single",
        "agents-a1-nvfp4-r0b0tlab-vllm-single",
    ):
        shutil.copy(ROOT / "recipes" / f"{slug}.json", tmp_path / "recipes")
    log = tmp_path / "results.jsonl"
    keys = {
        "job": "vonk-forge/ltx-2-19b-distilled-diffusers-single",
        "serving": "vonk-forge/agents-a1-nvfp4-r0b0tlab-vllm-single",
    }
    log.write_text(
        "\n".join(
            json.dumps(
                history(
                    recipe=key,
                    measured_memory=[
                        row("entrypoint", 5 * GB, None).model_dump(mode="json")
                    ],
                )
            )
            for key in keys.values()
        )
    )
    assert backfill([log], tmp_path).imported == 2
    assert read_evidence(tmp_path, keys["job"]).measured_memory == ()
    assert len(read_evidence(tmp_path, keys["serving"]).measured_memory) == 1
