"""A Controller retry cannot conceal a copy that stopped moving."""

from pathlib import Path

import pytest
from sweep_fakes import FakeModel, FakeRecipe, make_sweep

from spark_sweep.lifecycle import LifecycleState
from spark_sweep.verdict import Progress


@pytest.mark.parametrize("state", [LifecycleState.RUNNING, LifecycleState.BACKOFF])
def test_copy_retry_ages_by_real_progress_across_restart(
    tmp_path: Path, state: LifecycleState
) -> None:
    sweep, _, clock = make_sweep(tmp_path, [FakeRecipe("copy")], [FakeModel("m1")])
    key = "owned-copy"
    slot = {
        "request_key": key,
        "phase": "loading",
        "alias": "copy",
        "node_ids": [],
        "started_at": clock.now(),
        "deadline": clock.now() + 100,
    }
    load = {
        "copying": True,
        "state": state,
        "signal": {"signature": [123, "target-copy"], "waiting_on": "controller retry"},
    }
    sweep.state.data["loads"][key] = load
    assert sweep._observe_load(slot, clock.now()) is Progress.PROGRESSING
    clock.sleep(sweep.cfg.copy_stall_seconds + 1)
    # The persisted watch survives a resumed sweep; neither retry timing nor
    # resetting last_tick is evidence that another byte was transferred.
    sweep.state.slots["vonk-forge/copy"] = slot
    sweep.state.save()
    resumed, _, _ = make_sweep(tmp_path, [], [])
    restored = resumed.state.slots["vonk-forge/copy"]
    assert resumed._observe_load(restored, clock.now()) is Progress.STALLED
    load = resumed.state.data["loads"][key]
    load["signal"]["signature"][0] += 1
    assert resumed._observe_load(restored, clock.now()) is Progress.PROGRESSING


def test_stalled_copy_retry_releases_and_requeues_instead_of_holding_lane(
    tmp_path: Path,
) -> None:
    import json

    stuck = FakeRecipe("copy", copy_seconds=2000, copy_stalls=True)
    sweep, fleet, clock = make_sweep(tmp_path, [stuck], [FakeModel("m1")])
    original = sweep.vk.runner

    def retrying(argv, timeout):
        code, out, err = original(argv, timeout)
        document = json.loads(out)
        if (
            isinstance(document, dict)
            and document.get("progress", {}).get("child_progress", {}).get("phase")
            == "target-copy"
        ):
            document["next_attempt_at"] = "2026-10-20T15:49:17Z"
        return code, json.dumps(document), err

    sweep.vk.runner = retrying
    for _ in range(200):
        sweep.tick()
        if any(c[:2] == ("profile", "cancel") for _, c in fleet.calls):
            break
        clock.sleep(sweep.cfg.poll_seconds)
    else:
        raise AssertionError("the retried copy held its lane beyond the stall window")
    rows = [
        json.loads(line)
        for line in (tmp_path / "results.jsonl").read_text().splitlines()
    ]
    assert any(row.get("failure_class") == "copy-stalled" for row in rows)
    assert sweep.state.entry(stuck.key)["status"] != LifecycleState.FAILED


def test_copy_status_shows_the_enforced_inactivity_window(tmp_path: Path) -> None:
    from spark_sweep.report import build_status

    job = FakeRecipe("copy", copy_seconds=3000)
    sweep, _, clock = make_sweep(tmp_path, [job], [FakeModel("m1")])
    sweep.refresh_catalog()
    for _ in range(100):
        sweep.tick()
        slot = sweep.state.slots.get(job.key)
        if slot is not None and slot.get("copying"):
            break
        clock.sleep(sweep.cfg.poll_seconds)
    else:
        raise AssertionError("never observed the copy")
    clock.sleep(123)
    lanes = [
        item
        for items in build_status(sweep)["lanes"].values()
        for item in items
        if item["recipe"] == job.key
    ]
    assert lanes
    assert all(lane["limit_s"] == sweep.cfg.copy_stall_seconds for lane in lanes)
    assert all(lane["elapsed_s"] == 123 for lane in lanes)
