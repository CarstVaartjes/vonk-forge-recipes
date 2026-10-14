"""Stalled work the Controller reports (``progress.stalled``) is a platform problem.

The lane following a stalled application is released with an ``observe`` row (code
``platform.stalled``), nothing is cancelled, and the sweep admits no new load until the
Controller reports no stalled work. The field is additive: absent means no stall.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from sweep_fakes import FakeFleet, FakeModel, FakeRecipe, Gateway, make_sweep

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


def _rows(tmp_path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in (tmp_path / "results.jsonl").read_text().splitlines()
    ]


def _calls(fleet: FakeFleet, *verb: str) -> int:
    return sum(1 for _, args in fleet.calls if args[: len(verb)] == verb)


def test_a_stalled_application_releases_the_lane_and_pauses_until_it_clears(
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
    loads, cancels = (
        _calls(fleet, "profile", "load"),
        _calls(fleet, "profile", "cancel"),
    )
    sweep.tick()

    row = _rows(tmp_path)[-1]
    assert (row["disposition"], row["code"]) == ("observe", "platform.stalled")
    assert row["failure_class"] == verdict.PLATFORM_STALLED
    assert sweep.state.status(held.key) == "pending"
    assert sweep.paused and "Controller reports stalled work" in sweep.pause_reason
    write_status(sweep)
    status = (tmp_path / "status.md").read_text()
    assert f"PAUSED: Controller reports stalled work: {app['id']}" in status

    # Stalled for half an hour: no new load, and the sweep cancels nothing.
    for _ in range(30):
        clock.sleep(sweep.cfg.poll_seconds)
        sweep.tick()
    assert sweep.paused
    assert _calls(fleet, "profile", "load") == loads
    assert _calls(fleet, "profile", "cancel") == cancels

    # The Controller stops reporting it: the sweep resumes by itself.
    app["blockers"] = [b for b in app["blockers"] if b is not ALARM]
    fleet.stalled_applications = []
    clock.sleep(sweep.cfg.poll_seconds)
    sweep.tick()
    assert not sweep.paused
    _drive(sweep, clock, lambda: sweep.state.status(held.key) == "passed")
    # The recovered application is forgotten, never cancelled by the sweep.
    assert _calls(fleet, "profile", "cancel", app["id"]) == 0


def test_only_the_platform_list_still_holds_the_pause(
    tmp_path: Path, gateway: Gateway
) -> None:
    sweep, fleet, clock, held = _sweep(tmp_path, gateway)
    fleet.stalled_applications = [
        {"application_id": "app_other", "detail": "DistributionUnknown manifest"}
    ]
    sweep.tick()
    assert sweep.pause_reason == (
        "Controller reports stalled work: app_other DistributionUnknown manifest"
    )
    assert _calls(fleet, "profile", "load") == 0
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
