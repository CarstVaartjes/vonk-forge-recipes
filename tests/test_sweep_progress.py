"""A load is judged by its progress, not by the clock; a load the sweep gives up is released first."""

from __future__ import annotations

from pathlib import Path

import pytest
from sweep_fakes import GB, FakeModel, FakeRecipe, Gateway, make_sweep

pytest_plugins = ["sweep_bounds"]
pytestmark = pytest.mark.usefixtures("bounded_clock")

from spark_sweep.policy import TimeoutPolicy

WINDOW = 600.0


@pytest.fixture
def gateway():
    gateway = Gateway()
    yield gateway
    gateway.stop()


def _entry(sweep, slug: str) -> dict:
    return sweep.state.recipes[f"vonk-forge/{slug}"]


def _timeouts() -> TimeoutPolicy:
    return TimeoutPolicy(
        first_start_seconds=WINDOW, floor_seconds=WINDOW, cap_seconds=WINDOW
    )


def _loads(fleet) -> list[float]:
    return [
        t
        for t, (profile, call) in zip(fleet.call_times, fleet.calls, strict=True)
        if profile == 10 and call[:3] == ("profile", "load", "--yes")
    ]


def test_a_load_that_keeps_progressing_past_the_stall_window_is_not_given_up(
    tmp_path: Path, gateway: Gateway
) -> None:
    slow = FakeRecipe("slow", load_seconds=4 * WINDOW, start_progress=True)
    sweep, fleet, _ = make_sweep(
        tmp_path, [slow], [FakeModel("m1")], gateway=gateway, timeouts=_timeouts()
    )
    assert sweep.run() == 0
    assert _entry(sweep, "slow")["status"] == "passed"
    assert not any(c[:2] == ("profile", "cancel") for _, c in fleet.calls)


def test_a_load_without_progress_for_the_stall_window_is_given_up(
    tmp_path: Path, gateway: Gateway
) -> None:
    stuck = FakeRecipe("stuck", load_seconds=10**6)
    sweep, fleet, _ = make_sweep(
        tmp_path, [stuck], [FakeModel("m1")], gateway=gateway, timeouts=_timeouts()
    )
    sweep.run()
    entry = _entry(sweep, "stuck")
    assert (entry["status"], entry["failure_class"]) == ("deferred", "timeout")
    assert 2 * WINDOW > entry["finished_at"] - _loads(fleet)[0] >= WINDOW
    assert any(c[:2] == ("profile", "cancel") for _, c in fleet.calls)


def test_a_given_up_lane_is_not_reused_until_its_workload_is_gone(
    tmp_path: Path, gateway: Gateway
) -> None:
    orphan = 400.0
    stuck = FakeRecipe(
        "a-stuck", load_seconds=10**6, memory=90 * GB, orphan_seconds=orphan
    )
    good = FakeRecipe("b-good", load_seconds=60, memory=90 * GB)
    sweep, fleet, _ = make_sweep(
        tmp_path,
        [stuck, good],
        [FakeModel("m1")],
        gateway=gateway,
        timeouts=_timeouts(),
    )
    fleet.sparks = fleet.sparks[:1]  # one lane: the next load needs the held Spark
    sweep.run()
    gave_up = _entry(sweep, "a-stuck")["finished_at"]
    assert _entry(sweep, "b-good")["status"] == "passed"
    assert _loads(fleet)[1] >= gave_up + orphan


def test_progress_is_what_the_controller_shows_moving_not_when_it_was_asked() -> None:
    from spark_sweep import policy

    def application(**progress):
        return {"state": "running", "progress": progress}

    base = application(
        child_progress={"phase": "start", "operation": {"observed_at": "t1"}}
    )
    later = application(
        child_progress={"phase": "start", "operation": {"observed_at": "t2"}}
    )
    moved = application(
        child_progress={"phase": "start", "operation": {"completed_items": 3}}
    )
    assert policy.activity(base) == policy.activity(later)
    assert policy.activity(base) != policy.activity(moved)
    assert policy.platform_wait(application(admission_pending=True))
    assert policy.platform_wait({"state": "queued"})
    assert policy.platform_wait({"state": "backoff"})
    assert not policy.platform_wait(base)


def test_retries_and_timestamps_are_not_progress_but_bytes_are() -> None:
    from spark_sweep import policy

    def application(attempt, **operation):
        return {
            "state": "backoff" if attempt % 2 else "running",
            "updated_at": f"t{attempt}",
            "status_text": f"workload preparation is temporarily unavailable ({attempt})",
            "progress": {
                "attempt": attempt,
                "admission_attempt": attempt,
                "retry_due_at": f"t{attempt}",
                "current_label": f"retry {attempt}",
                "child_progress": {
                    "phase": "start",
                    "operation": {"last_progress_at": "t0", **operation},
                },
            },
        }

    stuck = {policy.activity(application(n)) for n in range(1, 35)}
    assert len(stuck) == 1  # 34 retries: the stall window runs out
    moving = {policy.activity(application(n, completed_bytes=n)) for n in range(1, 4)}
    assert len(moving) == 3  # bytes advance: the clock keeps restarting


def test_controller_last_progress_time_counts_as_progress() -> None:
    from spark_sweep import policy

    def application(stamp):
        return {
            "progress": {"child_progress": {"operation": {"last_progress_at": stamp}}}
        }

    assert policy.activity(application("t1")) != policy.activity(application("t2"))
