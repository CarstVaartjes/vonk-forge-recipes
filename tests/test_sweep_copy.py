"""Copying bytes to a Spark is not load time; a copy that stops is; a killed sweep adopts its load."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from sweep_fakes import FakeModel, FakeRecipe, Gateway, make_sweep

from spark_sweep import policy
from spark_sweep.policy import TimeoutPolicy


@pytest.fixture
def gateway():
    gateway = Gateway()
    yield gateway
    gateway.stop()


def _entry(sweep, slug: str = "big") -> dict:
    return sweep.state.recipes[f"vonk-forge/{slug}"]


def _timeouts(seconds: float = 600.0) -> TimeoutPolicy:
    return TimeoutPolicy(
        first_start_seconds=seconds, floor_seconds=seconds, cap_seconds=seconds
    )


# -- reading the progress document ----------------------------------------------------------


def test_copying_is_recognised_from_the_child_phase_or_the_operation_phase() -> None:
    copying = policy.distribution(
        {
            "progress": {
                "child_progress": {
                    "phase": "target-copy",
                    "bytes": 5,
                    "operation": {"completed_bytes": 9, "last_progress_at": "t1"},
                }
            }
        }
    )
    assert (
        copying.copying
        and copying.phase == "target-copy"
        and copying.signature == (9, "t1")
    )
    by_operation = policy.distribution(
        {
            "progress": {
                "child_progress": {
                    "phase": "prepare",
                    "operation": {"phase": "copying", "completed_bytes": 4},
                }
            }
        }
    )
    assert by_operation.copying
    for phase in ("runtime-install", "start", "final-verify", "container-build"):
        moving = policy.distribution(
            {
                "progress": {
                    "child_progress": {
                        "phase": phase,
                        "operation": {"phase": "prepare"},
                    }
                }
            }
        )
        assert not moving.copying and moving.signature is None, phase
    assert not policy.distribution({}).copying


# -- the load timeout does not run while copying -----------------------------------------------


def test_a_healthy_copy_longer_than_the_load_timeout_is_not_failed(
    tmp_path: Path, gateway: Gateway
) -> None:
    big = FakeRecipe("big", copy_seconds=3000, load_seconds=200)
    sweep, _, _ = make_sweep(
        tmp_path, [big], [FakeModel("m1")], gateway=gateway, timeouts=_timeouts(600)
    )
    assert sweep.run() == 0
    assert _entry(sweep)["status"] == "passed"
    assert _entry(sweep)["timings"]["load_s"] >= 3200  # it did take that long


def test_learned_timings_exclude_the_copy(tmp_path: Path, gateway: Gateway) -> None:
    big = FakeRecipe("big", copy_seconds=3000, load_seconds=200)
    sweep, _, _ = make_sweep(
        tmp_path, [big], [FakeModel("m1")], gateway=gateway, timeouts=_timeouts(600)
    )
    sweep.run()
    ((_, seconds),) = sweep.state.data["learned"]["vllm"]
    assert 200 <= seconds < 400  # install and start only, not the 3000 s of copying


def test_the_limit_still_applies_once_the_bytes_are_in_place(
    tmp_path: Path, gateway: Gateway
) -> None:
    slow = FakeRecipe("big", copy_seconds=1000, load_seconds=10**6)
    sweep, fleet, _ = make_sweep(
        tmp_path, [slow], [FakeModel("m1")], gateway=gateway, timeouts=_timeouts(600)
    )
    sweep.run()
    entry = _entry(sweep)
    assert (entry["status"], entry["phase"], entry["failure_class"]) == (
        "failed",
        "timeout",
        "timeout",
    )
    assert (
        "1000s of copying not counted" in entry["error"]
        or "of copying not counted" in entry["error"]
    )
    started = min(
        t
        for t, (p, c) in zip(fleet.call_times, fleet.calls, strict=True)
        if p == 10 and c[:3] == ("profile", "load", "--yes")
    )
    assert (
        entry["finished_at"] - started >= 1000 + 600
    )  # the copy first, then the 600 s limit


def test_a_copy_that_stops_moving_fails_after_the_stall_window_and_is_retried_once(
    tmp_path: Path, gateway: Gateway
) -> None:
    stuck = FakeRecipe("big", copy_seconds=10**5, copy_stalls=True)
    sweep, _, _ = make_sweep(
        tmp_path,
        [stuck],
        [FakeModel("m1")],
        gateway=gateway,
        timeouts=_timeouts(600),
        copy_stall_seconds=600,
    )
    sweep.run()
    entry = _entry(sweep)
    assert (entry["status"], entry["failure_class"], entry["attempts"]) == (
        "failed",
        "copy-stalled",
        2,
    )
    assert "no progress" in entry["error"] and "target-copy" in entry["error"]
    assert policy.platform_side(
        "copy-stalled"
    )  # a platform fix may cure it: requeued on a release change


def test_an_application_the_controller_holds_back_is_admission_stalled_not_a_load_timeout(
    tmp_path: Path, gateway: Gateway
) -> None:
    held = FakeRecipe("big", blocked=True)
    sweep, _, _ = make_sweep(
        tmp_path,
        [held],
        [FakeModel("m1")],
        gateway=gateway,
        timeouts=_timeouts(600),
        blocked_seconds=300,
    )
    sweep.run()
    entry = _entry(sweep)
    assert (entry["status"], entry["failure_class"]) == ("failed", "admission-stalled")
    assert "held this application back" in entry["error"]
    assert "resident_usage_unknown" in entry["error"]
    assert policy.platform_side(
        "admission-stalled"
    )  # a platform fix may cure it: requeued on a release change


def test_blocked_seconds_are_not_load_time() -> None:
    document = {
        "progress": {
            "blockers": [
                {
                    "code": "run-switch.phase-retry",
                    "detail": "run.capacity_busy",
                    "severity": "warning",
                },
                {
                    "code": "run-switch.inventory-stale",
                    "detail": "old",
                    "severity": "error",
                },
                {
                    "code": "run-switch.something",
                    "detail": "brief hold",
                    "severity": "warning",
                },
            ]
        }
    }
    assert [b["code"] for b in policy.admission_blockers(document)] == [
        "run-switch.phase-retry",
        "run-switch.inventory-stale",
    ]
    assert policy.admission_blockers({"progress": {"blockers": []}}) == []
    assert policy.admission_blockers({}) == []


def test_a_copy_that_keeps_moving_is_never_stalled(
    tmp_path: Path, gateway: Gateway
) -> None:
    sweep, _, _ = make_sweep(
        tmp_path,
        [FakeRecipe("big", copy_seconds=5000)],
        [FakeModel("m1")],
        gateway=gateway,
        timeouts=_timeouts(600),
        copy_stall_seconds=300,
    )
    sweep.run()
    assert _entry(sweep)["status"] == "passed"


def test_the_status_page_shows_a_lane_that_is_copying(
    tmp_path: Path, gateway: Gateway
) -> None:
    sweep, _, clock = make_sweep(
        tmp_path,
        [FakeRecipe("big", copy_seconds=800)],
        [FakeModel("m1")],
        gateway=gateway,
        status_seconds=0,
    )
    pages: list[str] = []
    clock.hooks.append(
        lambda _t: (
            pages.append((tmp_path / "status.md").read_text())
            if (tmp_path / "status.md").exists()
            else None
        )
    )
    sweep.run()
    assert any("loading (copying)" in page for page in pages)


# -- results from the old rules ------------------------------------------------------------------


def test_a_load_timeout_recorded_under_the_old_rules_is_retried_once(
    tmp_path: Path, gateway: Gateway
) -> None:
    recipe = FakeRecipe("big", copy_seconds=1000, load_seconds=120)
    sweep, fleet, _ = make_sweep(
        tmp_path, [recipe], [FakeModel("m1")], gateway=gateway, timeouts=_timeouts(600)
    )
    sweep.run()
    assert _entry(sweep)["status"] == "passed"
    # As the old sweep left it: failed on a wall-clock timeout during the copy, no rules recorded.
    entry = _entry(sweep)
    entry.update(
        status="failed",
        phase="timeout",
        failure_class="timeout",
        error="not serving after 3600s",
    )
    entry.pop("rules", None)
    sweep.state.save()
    again, _, _ = make_sweep(
        tmp_path, [recipe], [FakeModel("m1")], fleet=fleet, timeouts=_timeouts(600)
    )
    assert again.run() == 0
    assert _entry(again)["status"] == "passed"
    requeue = [
        json.loads(x)
        for x in (tmp_path / "results.jsonl").read_text().splitlines()
        if '"requeue"' in x
    ]
    assert [x["reason"] for x in requeue] == [
        "load-timeout rules changed: copying no longer counts"
    ]


def test_a_timeout_under_the_new_rules_is_not_requeued_again(
    tmp_path: Path, gateway: Gateway
) -> None:
    slow = FakeRecipe("big", copy_seconds=0, load_seconds=10**6)
    sweep, fleet, _ = make_sweep(
        tmp_path, [slow], [FakeModel("m1")], gateway=gateway, timeouts=_timeouts(300)
    )
    sweep.run()
    assert _entry(sweep)["rules"] == policy.LOAD_RULES
    again, _, _ = make_sweep(
        tmp_path, [slow], [FakeModel("m1")], fleet=fleet, timeouts=_timeouts(300)
    )
    again.run()
    assert not [
        x
        for x in (tmp_path / "results.jsonl").read_text().splitlines()
        if '"requeue"' in x
    ]


# -- killed mid-copy, restarted ----------------------------------------------------------------------


def test_a_sweep_killed_mid_copy_adopts_its_load_on_restart_and_passes(
    tmp_path: Path, gateway: Gateway
) -> None:
    big = FakeRecipe("big", copy_seconds=3000, load_seconds=200)
    sweep, fleet, clock = make_sweep(
        tmp_path, [big], [FakeModel("m1")], gateway=gateway, timeouts=_timeouts(600)
    )
    start = clock.now()

    def sigkill(
        now: float,
    ) -> None:  # no cleanup runs: the state file is whatever the last pass saved
        if now - start >= 700:
            raise RuntimeError("SIGKILL")

    clock.hooks.append(sigkill)
    with pytest.raises(RuntimeError):
        sweep.run()
    saved = json.loads((tmp_path / "state.json").read_text())
    assert (
        saved["slots"]["vonk-forge/big"]["phase"] == "loading"
        and saved["load"]["app_id"]
    )
    application = saved["load"]["app_id"]
    assert (
        fleet.apps[application]["state"] == "running"
    )  # still copying on the Controller

    clock.hooks.clear()
    clock.t += 1800  # the sweep is down for half an hour, longer than its load timeout
    calls_before = len(fleet.calls)
    again, _, _ = make_sweep(
        tmp_path, [big], [FakeModel("m1")], fleet=fleet, timeouts=_timeouts(600)
    )
    assert again.run() == 0
    assert _entry(again)["status"] == "passed" and _entry(again)["attempts"] == 1
    later = [c for _, c in fleet.calls[calls_before:]]
    assert not [
        c for c in later if c[:2] == ("profile", "cancel")
    ]  # it did not cancel the load
    assert not [
        c for c in later if c[:2] == ("profile", "add")
    ]  # nor place the recipe again
    loads = [c for c in later if c[:3] == ("profile", "load", "--yes")]
    assert len(loads) == 1  # only the final stop; no clearing load, no second placement
    assert any(
        "adopted an in-flight load of vonk-forge/big" in e["message"]
        for e in again.state.data["events"]
    )
    assert fleet.apps[application]["state"] == "succeeded"
