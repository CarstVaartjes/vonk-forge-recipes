"""Stalled work the Controller reports (``progress.stalled``) is a platform problem.

The lane following a stalled application keeps its receipt and emits an ``observe``
row. Independent lanes continue. The field is additive: absent means no stall.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from sweep_fakes import FakeFleet, FakeModel, FakeRecipe, Gateway, make_sweep
from vonk_forge_contracts.sweep import SweepWaitReason

from spark_sweep import verdict
from spark_sweep.policy import TimeoutPolicy
from spark_sweep.report import write_status

HOUR = 3600.0
ALARM = {
    "code": verdict.STALLED_CODE,
    "severity": "error",
    "detail": "No progress across 5 observations since 2026-10-14T00:00:00+00:00: "
    "profile.retry_conflict",
}


@pytest.fixture
def gateway():
    gateway = Gateway()
    yield gateway
    gateway.stop()


def _sweep(tmp_path: Path, gateway: Gateway) -> tuple[Any, FakeFleet, Any, FakeRecipe]:
    held = FakeRecipe("big", blocked=True, blocked_for=2 * HOUR)
    sweep, fleet, clock = make_sweep(
        tmp_path,
        [held],
        [FakeModel("m1")],
        gateway=gateway,
        timeouts=TimeoutPolicy(
            first_start_seconds=600, floor_seconds=600, cap_seconds=600
        ),
        poll_seconds=60.0,
    )
    return sweep, fleet, clock, held


def _drive(sweep: Any, clock: Any, until: Any, limit: int = 2000) -> None:
    for _ in range(limit):
        sweep.tick()
        if until():
            return
        clock.sleep(sweep.cfg.poll_seconds)
    raise AssertionError("the scenario did not settle")


def _loading(sweep: Any, key: str) -> bool:
    slot = sweep.state.slots.get(key)
    return bool(slot and slot["phase"] == "loading" and slot.get("watch"))


def _followed(sweep: Any, fleet: FakeFleet) -> dict[str, Any]:
    (load,) = sweep.state.data["loads"].values()
    return fleet.apps[load["app_id"]]


def _followed_live(sweep: Any, fleet: FakeFleet) -> dict[str, Any]:
    """The application of the load a live lane rides (released loads linger)."""
    keys = {
        s["request_key"] for s in sweep.state.slots.values() if s["phase"] != "finished"
    }
    (key,) = keys
    return fleet.apps[sweep.state.data["loads"][key]["app_id"]]


def _rows(tmp_path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in (tmp_path / "results.jsonl").read_text().splitlines()
    ]


def _calls(fleet: FakeFleet, *verb: str) -> int:
    return sum(1 for _, args in fleet.calls if args[: len(verb)] == verb)


def test_a_stalled_application_releases_only_its_lane_at_budget(
    tmp_path: Path, gateway: Gateway
) -> None:
    sweep, fleet, clock, held = _sweep(tmp_path, gateway)
    _drive(sweep, clock, lambda: _loading(sweep, held.key))
    app = _followed(sweep, fleet)
    app["blockers"] = [*app.get("blockers", []), ALARM]
    fleet.stalled_applications = [
        {
            "application_id": app["id"],
            "profile_id": "p",
            "state": app["state"],
            "detail": "profile.retry_conflict",
            "repeats": 5,
            "first_seen_at": "2026-10-14T00:00:00+00:00",
        }
    ]
    cancels = _calls(fleet, "profile", "cancel")
    sweep.tick()

    row = _rows(tmp_path)[-1]
    assert (row["disposition"], row["code"]) == ("observe", verdict.STALLED_CODE)
    assert row["failure_class"] == verdict.PLATFORM_STALLED
    assert sweep.state.status(held.key) == "pending"
    assert sweep.state.entry(held.key).get("attempts", 0) == 0
    assert not sweep.paused
    assert held.key in sweep.state.slots
    write_status(sweep)
    status = (tmp_path / "status.md").read_text()
    assert f"Controller reports stalled work: {app['id']}" in status
    assert "PAUSED:" not in status

    # The platform alarm cannot hold a local lane past its budget.
    _drive(sweep, clock, lambda: held.key not in sweep.state.slots)
    assert not sweep.paused
    assert sweep.state.entry(held.key).get("attempts", 0) == 0
    assert (
        sweep.state.entry(held.key)["wait_deferral"]["wait"]["reason"]
        == SweepWaitReason.CONTROLLER_STALLED
    )
    assert _calls(fleet, "profile", "cancel") == cancels

    # The Controller stops reporting it: the sweep resumes by itself.
    app["blockers"] = [b for b in app["blockers"] if b is not ALARM]
    fleet.stalled_applications = []
    clock.sleep(sweep.cfg.poll_seconds)
    sweep.tick()
    assert not sweep.paused
    _drive(sweep, clock, lambda: sweep.state.status(held.key) == "passed")
    # The original application recovers without cancellation.
    assert _calls(fleet, "profile", "cancel", app["id"]) == 0


def test_repeated_stalls_never_spend_an_attempt(
    tmp_path: Path, gateway: Gateway
) -> None:
    sweep, fleet, clock, held = _sweep(tmp_path, gateway)
    for _ in range(5):
        _drive(sweep, clock, lambda: _loading(sweep, held.key))
        app = _followed_live(sweep, fleet)
        app["blockers"] = [*app.get("blockers", []), ALARM]
        sweep.tick()
        assert not sweep.paused
        app["blockers"] = [b for b in app["blockers"] if b is not ALARM]
        clock.sleep(sweep.cfg.poll_seconds)
        sweep.tick()
        # Each fresh stall episode is visible without replacing the receipt.
    entry = sweep.state.entry(held.key)
    assert entry.get("attempts", 0) == 0
    assert entry["status"] != "deferred"
    stalls = [r for r in _rows(tmp_path) if r.get("code") == verdict.STALLED_CODE]
    assert len(stalls) == 5


def test_an_owner_pause_holds_the_re_plan_of_a_preparing_recipe(
    tmp_path: Path, gateway: Gateway
) -> None:
    recipes = [
        FakeRecipe(
            "bad",
            ("m1",),
            local="cached",
            blocked=True,
            blocker_code="profile.preparation_unavailable",
            blocked_for=3000,
        ),
        FakeRecipe("good1", ("m1",), local="cached"),
    ]
    sweep, fleet, clock = make_sweep(
        tmp_path,
        recipes,
        [FakeModel("m1", local="cached")],
        gateway=gateway,
        timeouts=TimeoutPolicy(
            first_start_seconds=600, floor_seconds=600, cap_seconds=600
        ),
        blocked_release_seconds=60,
    )
    _drive(
        sweep,
        clock,
        lambda: any(s.get("blocked_since") for s in sweep.state.slots.values()),
    )
    fleet.owner_load(2, duration=3000)
    loads, removes = (
        _calls(fleet, "profile", "load"),
        _calls(fleet, "profile", "remove"),
    )
    for _ in range(20):
        clock.sleep(sweep.cfg.poll_seconds)
        sweep.tick()
    assert sweep.paused
    assert _calls(fleet, "profile", "load") == loads
    assert _calls(fleet, "profile", "remove") == removes


def test_the_platform_list_alerts_without_holding_the_fleet(
    tmp_path: Path, gateway: Gateway
) -> None:
    sweep, fleet, clock, held = _sweep(tmp_path, gateway)
    fleet.stalled_applications = [
        {"application_id": "app_other", "detail": "DistributionUnknown manifest"}
    ]
    sweep.tick()
    assert sweep.stall_reason == (
        "Controller reports stalled work: app_other DistributionUnknown manifest"
    )
    assert not sweep.paused
    fleet.stalled_applications = []
    _drive(sweep, clock, lambda: sweep.state.status(held.key) == "passed")


def test_without_the_field_nothing_changes(tmp_path: Path, gateway: Gateway) -> None:
    sweep, fleet, clock, held = _sweep(tmp_path, gateway)
    assert fleet.stalled_applications is None  # an older Controller
    _drive(sweep, clock, lambda: sweep.state.status(held.key) == "passed")
    assert not any(r.get("code") == "platform.stalled" for r in _rows(tmp_path))
    assert not sweep.paused


def test_stalled_detail_reads_the_alarm_and_never_waits_on_it() -> None:
    document = {
        "state": "running",
        "blockers": [
            {"code": "profile.retry_conflict", "severity": "warning"},
            ALARM,
        ],
    }
    assert verdict.stalled_detail(document) == ALARM["detail"]
    assert verdict.application_signal(document).waiting_on is None
    assert verdict.stalled_detail({"state": "running", "blockers": []}) is None
    assert verdict.stalled_detail({"state": "running"}) is None
