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


def _world(tmp_path: Path, ran: str | None, *, ready: bool = True):
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
    run = {
        "alias": slot["alias"],
        "recipe": fake.key,
        "run_id": "the-run",
        "node_ids": list(slot["node_ids"]),
        "ready": ready,
    }
    if ran is not None:
        run["recipe_revision_id"] = ran
    fleet.runs.append(run)
    return sweep, fleet, clock, fake, slot


def _observe(sweep, fake, slot) -> None:
    sweep.fleet = fetch_fleet(sweep.vk, sweep.cfg.default_spark_memory)
    sweep._advance_loading(fake.key, slot, sweep.clock.now())


def _rows(tmp_path: Path) -> list[dict]:
    return [
        json.loads(x) for x in (tmp_path / "results.jsonl").read_text().splitlines()
    ]


def _smoke(sweep, fake, slot) -> None:
    slot.update(phase="smoking", run_id="the-run", ready_at=sweep.clock.now())
    sweep._submit_smoke(fake.key, slot)
    sweep._advance_smoking(fake.key, slot)


def test_a_mismatching_run_waits_paced_and_is_never_evidence(tmp_path: Path) -> None:
    sweep, _, clock, fake, slot = _world(tmp_path, "rev-reverted")
    _observe(sweep, fake, slot)
    row = next(r for r in _rows(tmp_path) if r["step"] == "identity")
    assert (row["status"], row["blame"]) == ("waiting", "platform")
    assert row["code"] == policy.REVISION_MISMATCH
    entry = sweep.state.entry(fake.key)
    assert entry["status"] == "pending"
    first = entry["not_before"] - clock.now()
    # The same mismatch again backs off further instead of reloading every retry delay.
    slot2 = dict(slot)
    slot2.pop("ran_revision")
    sweep.state.slots[fake.key] = slot2
    slot2["phase"] = "loading"
    _observe(sweep, fake, slot2)
    assert entry["revision_mismatches"] == 2
    assert entry["not_before"] - clock.now() > first
    assert not [r for r in _rows(tmp_path) if r["step"] == "smoke"]
    assert not sweep.qualification_evidence.get(fake.key)


def test_a_run_that_crashed_after_ready_with_a_wrong_revision_is_no_evidence(
    tmp_path: Path,
) -> None:
    sweep, fleet, _, fake, slot = _world(tmp_path, None)
    slot["ran_revision"] = "rev-reverted"  # captured while the run was visible
    fleet.runs.clear()  # the Controller dropped the crashed run
    _smoke(sweep, fake, slot)
    smoke = [r for r in _rows(tmp_path) if r["step"] == "smoke"]
    assert smoke
    assert not sweep.qualification_evidence.get(fake.key)


def test_a_load_failure_without_a_captured_revision_is_no_evidence(
    tmp_path: Path,
) -> None:
    sweep, _, _, fake, _ = _world(tmp_path, "rev-reverted", ready=False)
    sweep._record_failure(
        fake.key,
        policy.classify("start", "runtime_process_exited", "exit_cause=oom_killed"),
        None,
    )
    assert [r for r in _rows(tmp_path) if r["step"] == "smoke"]
    assert not sweep.qualification_evidence.get(fake.key)


def test_after_a_matching_revision_is_seen_the_lane_records_normally(
    tmp_path: Path,
) -> None:
    sweep, _, _, fake, slot = _world(tmp_path, None)
    entry = sweep.state.entry(fake.key)
    entry["revision_mismatches"] = 3
    _observe(sweep, fake, slot)
    assert slot["ran_revision"] == f"rev-{fake.content}"
    assert "revision_mismatches" not in entry
    _smoke(sweep, fake, slot)
    assert [r["status"] for r in _rows(tmp_path) if r["step"] == "smoke"] == ["passed"]
    assert sweep.qualification_evidence[fake.key].records
