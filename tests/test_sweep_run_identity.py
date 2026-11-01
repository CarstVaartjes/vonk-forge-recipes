"""Evidence binds to the bytes that ran: a run of another revision proves nothing."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from sweep_fakes import FakeModel, FakeRecipe, make_sweep

pytest_plugins = ["sweep_bounds"]
pytestmark = pytest.mark.usefixtures("bounded_clock")

from spark_sweep import policy
from spark_sweep.catalog import fetch_fleet


def _smoke_with_run_revision(tmp_path: Path, ran: str | None):
    fake = FakeRecipe(
        "ran-other",
        digests=("1" * 64,),
        content="c" * 64,
        local="cached",
        adapter="readiness",
    )
    sweep, fleet, clock = make_sweep(
        tmp_path, [fake], [FakeModel("1" * 64, local="cached")]
    )
    sweep.refresh_catalog()
    accepted = sweep.recipes[fake.key]
    sweep._open_slot(
        policy.Placement(accepted, ("spk_a",)), "accepted-request", clock.now()
    )
    slot = sweep.state.slots[fake.key]
    slot.update(phase="smoking", ready_at=clock.now(), run_id="the-run")
    run = {
        "alias": slot["alias"],
        "recipe": fake.key,
        "run_id": "the-run",
        "node_ids": list(slot["node_ids"]),
        "ready": True,
    }
    if ran is not None:
        run["recipe_revision_id"] = ran
    fleet.runs.append(run)
    sweep.fleet = fetch_fleet(sweep.vk, sweep.cfg.default_spark_memory)
    sweep._submit_smoke(fake.key, slot)
    sweep._advance_smoking(fake.key, slot)
    rows = [
        json.loads(x) for x in (tmp_path / "results.jsonl").read_text().splitlines()
    ]
    return sweep, fake, rows


def test_a_run_of_another_revision_is_platform_blamed_and_no_evidence(
    tmp_path: Path,
) -> None:
    sweep, fake, rows = _smoke_with_run_revision(tmp_path, "rev-reverted")
    assert not [r for r in rows if r["step"] == "smoke"]
    row = next(r for r in rows if r["step"] == "identity")
    assert row["blame"] == "platform"
    assert row["code"] == "platform.revision_mismatch"
    assert row["evidence"]["ran_revisions"] == ["rev-reverted"]
    assert sweep.state.entry(fake.key)["status"] == "pending"
    assert not sweep.qualification_evidence.get(fake.key)


@pytest.mark.parametrize("ran", [None, "same"])
def test_a_matching_or_unreported_run_records_as_before(
    tmp_path: Path, ran: str | None
) -> None:
    probe = FakeRecipe("ran-other", digests=("1" * 64,), content="c" * 64)
    revision = f"rev-{probe.content}" if ran else None
    sweep, fake, rows = _smoke_with_run_revision(tmp_path, revision)
    assert [r["status"] for r in rows if r["step"] == "smoke"] == ["passed"]
    assert not [r for r in rows if r["step"] == "identity"]
    assert sweep.qualification_evidence[fake.key].records
