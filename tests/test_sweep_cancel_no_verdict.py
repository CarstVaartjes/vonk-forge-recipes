"""A cancel is never a recipe verdict; the sweep never cancels an application a live lane rides.

Recorded incident (2026-10-13, hunyuanocr-1-5-vllm-dflash-single on spark-2297): the
fastcontext load record, whose lane had passed, followed its superseded application into
the hunyuanocr load's own application. When the last lane that record had adopted ended,
the sweep released it and cancelled that application under the live hunyuanocr lane,
which was waiting on the Controller's retry of its copy. The Controller ended the cancel
with the copy child's effect unknown and the sweep deferred the recipe on its own cancel.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from sweep_fakes import FakeModel, FakeRecipe, Gateway, make_sweep

INCIDENT = (
    Path(__file__).parent
    / "fixtures"
    / "sweep-verdicts"
    / "hunyuanocr-cancelled-copy-retry.json"
)


def _incident() -> dict[str, Any]:
    return json.loads(INCIDENT.read_text())


def _loading(tmp_path: Path):
    sweep, fleet, clock = make_sweep(
        tmp_path,
        [FakeRecipe("a", local="cached", copy_seconds=9000)],
        [FakeModel("m1", local="cached")],
    )
    sweep.preflight()
    sweep.refresh_catalog()
    sweep.queue = list(sweep.recipes)
    sweep.cached_models = sweep._cached_models()
    sweep.schedule(clock.now())
    live = next(iter(sweep.state.data["loads"].values()))
    assert live["app_id"]
    return sweep, fleet, clock, live


def _cancels(fleet) -> list[tuple]:
    return [c for _, c in fleet.calls if c[:2] == ("profile", "cancel")]


def test_a_finished_load_never_cancels_the_application_a_live_lane_rides(
    tmp_path: Path,
) -> None:
    sweep, fleet, _clock, live = _loading(tmp_path)
    # A load whose lanes ended, bound to the live load's application by a followed
    # supersession: releasing it forgets the record and cancels nothing.
    sweep.state.data["loads"]["finished"] = {
        "request_key": "finished",
        "app_id": live["app_id"],
        "state": "queued",
        "seq": 0,
        "adopted_requests": [],
        "adopted_cleanup_requests": [],
    }
    sweep.release_orphan_load()
    assert "finished" not in sweep.state.data["loads"]
    assert live["request_key"] in sweep.state.data["loads"]
    assert not _cancels(fleet)
    assert fleet.apps[live["app_id"]]["state"] != "cancelled"


def test_a_load_whose_lanes_ended_alone_is_still_cancelled(tmp_path: Path) -> None:
    sweep, fleet, _clock, live = _loading(tmp_path)
    for slot in sweep.state.slots.values():
        slot["phase"] = "finished"
    sweep.release_orphan_load()
    assert [c for c in _cancels(fleet) if live["app_id"] in c]


def test_the_recorded_cancel_requeues_the_recipe_without_a_verdict(
    tmp_path: Path,
) -> None:
    sweep, _fleet, clock, live = _loading(tmp_path)
    key = next(iter(sweep.state.slots))
    attempts = sweep.state.entry(key).get("attempts", 0)
    recorded = {**_incident(), "id": live["app_id"]}
    real = sweep.vk.runner

    def runner(argv, timeout):
        args = list(argv)
        if "progress" in args and live["app_id"] in args:
            return 0, json.dumps(recorded), ""
        return real(argv, timeout)

    sweep.vk.runner = runner
    sweep.advance_slots(clock.now())
    sweep.advance_slots(clock.now())
    entry = sweep.state.entry(key)
    assert entry["status"] == "pending"
    assert entry.get("attempts", 0) == attempts
    assert entry["not_before"] > clock.now()
    rows = [r for r in sweep.log.entries() if r["recipe"] == key]
    assert not [r for r in rows if r["status"] in ("deferred", "failed")]
    requeue = next(r for r in rows if r["step"] == "requeue")
    assert "cause operator" in requeue["reason"]
    assert "effect unknown" in requeue["evidence"]["status_reason"]


def test_a_request_a_newer_load_adopted_is_not_cancelled(tmp_path: Path) -> None:
    sweep, fleet, _clock, live = _loading(tmp_path)
    for slot in sweep.state.slots.values():
        slot["phase"] = "finished"
    # A newer profile load adopted this request: its intent owns the workload.
    sweep.state.data["loads"]["newer"] = {
        "request_key": "newer",
        "app_id": "other-app",
        "state": "queued",
        "seq": 1,
        "adopted_requests": [live["request_key"]],
        "adopted_cleanup_requests": [],
    }
    sweep._release_orphan_load(live)
    assert not [c for c in _cancels(fleet) if live["app_id"] in c]
    assert live["request_key"] not in sweep.state.data["loads"]


def _adopted_pair(tmp_path: Path, gateway, *extra: FakeRecipe):
    """Two lanes in one application: the whole-profile load adopted the first lane's load."""
    from dataclasses import replace

    bad = FakeRecipe("bad", ("m1",), local="cached", copy_seconds=0, load_seconds=9000)
    other = FakeRecipe("good", ("m2",), local="cached", load_seconds=9000)
    recipes = [bad, other, *extra]
    sweep, fleet, clock = make_sweep(
        tmp_path,
        recipes,
        [FakeModel("m1", local="cached"), FakeModel("m2", local="cached")],
        gateway=gateway,
        only=("bad",),
    )
    sweep.preflight()
    sweep.refresh_catalog()
    sweep._tick()
    clock.sleep(20)
    sweep._tick()
    sweep.cfg = replace(sweep.cfg, only=())
    sweep.refresh_catalog()
    clock.sleep(20)
    sweep._tick()
    old = sweep.state.slots[bad.key]
    loads = sweep.state.data["loads"]
    (older,) = [x for x in loads.values() if not x.get("adopted_requests")]
    (newer,) = [x for x in loads.values() if x.get("adopted_requests")]
    assert newer["adopted_requests"] == [older["request_key"]]
    return sweep, fleet, clock, older, newer, bad, other, old


def test_a_cancel_the_controller_refuses_is_logged_with_its_cause(
    tmp_path: Path, gateway
) -> None:
    sweep, fleet, _clock, older, newer, *_ = _adopted_pair(tmp_path, gateway)
    # A stale record: the sweep does not know the newer load adopted this request.
    newer["adopted_requests"] = []
    for slot in sweep.state.slots.values():
        slot["phase"] = "finished"
    sweep._release_orphan_load(older)
    assert [c for c in _cancels(fleet) if older["app_id"] in c]
    assert fleet.apps[older["app_id"]]["state"] == "running"  # refused: not cancelled
    messages = [e["message"] for e in sweep.state.data["events"]]
    assert any(
        f"cancel of {older['app_id']} failed" in m and "newer selected request" in m
        for m in messages
    )


def test_a_skipped_cancel_does_not_block_a_later_overlapping_load(
    tmp_path: Path, gateway
) -> None:
    third = FakeRecipe("third", ("m1",), local="cached", load_seconds=9000)
    sweep, fleet, clock, older, _newer, bad, other, _ = _adopted_pair(
        tmp_path, gateway, third
    )
    (adopter,) = [k for k in sweep.state.slots if k != bad.key]
    later = next(k for k in (other.key, third.key) if k != adopter)
    sweep.state.slots[bad.key]["phase"] = "finished"
    sweep.release_orphan_load()
    assert not [c for c in _cancels(fleet) if older["app_id"] in c]
    # The next profile load leaves the adopted workload out and overlaps its Spark:
    # it is admitted, and the Controller supersedes what the sweep did not cancel.
    sweep.enqueue_finished_cleanup()
    sweep.queue = [later]
    sweep.schedule(clock.now())
    assert sweep.state.slots[later]["phase"] == "loading"
    assert fleet.apps[older["app_id"]]["state"] == "superseded"


@pytest.fixture
def gateway():
    gateway = Gateway()
    yield gateway
    gateway.stop()
