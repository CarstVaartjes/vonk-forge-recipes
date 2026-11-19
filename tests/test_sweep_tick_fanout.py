"""A pass reads its independent observations at once, once each, and sleeps to its poll deadline."""

from __future__ import annotations

import threading
from pathlib import Path

import pytest
from sweep_fakes import FakeModel, FakeRecipe, make_sweep

from spark_sweep.fanout import gather
from spark_sweep.run import DaemonExecutor

pytest_plugins = ["sweep_bounds"]
pytestmark = pytest.mark.usefixtures("bounded_clock")


def _command(argv) -> tuple[str, ...]:
    words = [a for a in argv[1:] if a not in ("--json", "--no-input")]
    while words and words[0] == "--profile":
        words = words[2:]
    return tuple(words[:2])


def _sweep(tmp_path: Path):
    sweep, fleet, clock = make_sweep(
        tmp_path, [FakeRecipe("a", ("m1",))], [FakeModel("m1")]
    )
    sweep.state.slots["vonk-forge/a"] = {
        "phase": "finished",
        "alias": "sweep-a",
    }  # a fleet read is due
    return sweep, fleet, clock


def test_the_reads_of_a_pass_overlap(tmp_path: Path) -> None:
    sweep, _, _ = _sweep(tmp_path)
    inner = sweep.vk.runner
    # 3 owner profiles + platform + fleet: all five must be in flight together.
    barrier = threading.Barrier(5, timeout=5)

    def runner(argv, timeout):
        if _command(argv) in {("profile", "progress"), ("platform",), ("fleet",)}:
            barrier.wait()
        return inner(argv, timeout)

    sweep.vk.runner = runner
    sweep._observe_concurrently()  # a serial pass would time the barrier out
    assert not barrier.broken
    assert set(sweep._reads) == {"platform", "fleet"}
    assert set(sweep.guard.reads) == {1, 2, 3}


def test_a_tick_reads_each_observation_once_and_errors_reach_their_consumers(
    tmp_path: Path,
) -> None:
    sweep, _, _ = _sweep(tmp_path)
    inner = sweep.vk.runner
    seen: list[tuple[str, ...]] = []

    def runner(argv, timeout):
        seen.append(_command(argv))
        if _command(argv) == ("fleet",):
            return 1, "", "boom"
        if _command(argv) == ("profile", "progress") and "2" in argv:
            return 1, "", "boom"
        return inner(argv, timeout)

    sweep.vk.runner = runner
    sweep.tick()
    # advance_slots took the prefetched (failed) fleet read: blind, no second read of its own.
    assert sweep.blind
    assert seen.count(("platform",)) == 1
    # The unreadable owner profile reached guard.check as an unreadable profile.
    assert "owner-profile-2" in sweep.state.data["infra"]
    assert seen.count(("profile", "progress")) >= 3
    assert (
        not sweep._reads and not sweep.guard.reads
    )  # nothing leaks into the next pass
    assert "tick" not in sweep.state.data["infra"]  # the pass ran to its end


def test_a_read_error_surfaces_at_its_consumer() -> None:
    def boom() -> None:
        raise RuntimeError("down")

    done = gather({"ok": lambda: 1, "bad": boom}, DaemonExecutor())
    assert done["ok"].result() == 1
    with pytest.raises(RuntimeError):
        done["bad"].result()


@pytest.mark.parametrize(
    ("work", "sleep_for"), [(4.0, 6.0), (15.0, 0.0), (-30.0, 10.0)]
)
def test_the_loop_sleeps_to_the_poll_deadline(
    tmp_path: Path, work: float, sleep_for: float
) -> None:
    sweep, _, clock = make_sweep(
        tmp_path, [FakeRecipe("a", ("m1",))], [FakeModel("m1")]
    )
    slept: list[float] = []
    real_sleep = clock.sleep

    def sleep(seconds: float) -> None:
        slept.append(seconds)
        real_sleep(seconds)

    def tick() -> None:
        clock.t += (
            work  # a pass inside, past, or (clock stepped back) before the 10 s poll
        )

    clock.sleep = sleep
    sweep.tick = tick
    sweep.done = lambda: len(slept) >= 3
    sweep.cfg.watch_seconds = 0
    sweep.run()
    assert slept[:3] == [sleep_for] * 3


def test_the_lane_progress_reads_of_a_poll_overlap(tmp_path: Path) -> None:
    sweep, _, _ = _sweep(tmp_path)
    sweep.state.data["loads"] = {
        key: {"request_key": key, "app_id": f"{key}-app"} for key in ("l1", "l2", "l3")
    }
    inner = sweep.vk.runner
    barrier = threading.Barrier(3, timeout=5)

    def runner(argv, timeout):
        # The first read of each lane names its application; the by-key retry does not.
        if _command(argv) == ("profile", "progress") and "--application" in argv:
            barrier.wait()  # releases only when every lane's first read is in flight
        return inner(argv, timeout)

    sweep.vk.runner = runner
    sweep.poll_application()


def test_the_startup_client_checks_overlap(tmp_path: Path) -> None:
    sweep, _, _ = _sweep(tmp_path)
    inner = sweep.vk.runner
    barrier = threading.Barrier(3, timeout=5)

    def runner(argv, timeout):
        if _command(argv) in {("--version",), ("platform",), ("update",)}:
            barrier.wait()
        return inner(argv, timeout)

    sweep.vk.runner = runner
    sweep.check_client()  # a serial check would time the barrier out


def test_a_lane_whose_read_fails_leaves_the_others_observed(tmp_path: Path) -> None:
    sweep, fleet, _ = _sweep(tmp_path)
    sweep.state.data["loads"] = {
        key: {"request_key": key, "app_id": f"{key}-app"} for key in ("a", "b")
    }
    fleet.observations.extend(
        [
            (
                ("profile", "progress", "--application", "a-app"),
                2,
                {"error_type": "control_api", "code": "http.503", "http_status": 503},
            ),
            (
                ("profile", "progress", "--application", "b-app"),
                0,
                {"id": "b-app", "state": "succeeded"},
            ),
        ]
    )
    assert not sweep.poll_application()
    assert sweep.state.data["apps"]["b"]["state"] == "succeeded"
    assert "b" not in sweep.state.data["loads"]
    assert "a" in sweep.state.data["loads"] and sweep.unobserved_loads == {"a"}


def test_the_download_progress_reads_of_a_poll_overlap(tmp_path: Path) -> None:
    sweep, _, _ = _sweep(tmp_path)
    sweep.state.downloads.update(
        {
            f"vonk-forge/d{n}": {"request_key": f"k{n}", "operation_id": f"op{n}"}
            for n in range(3)
        }
    )
    inner = sweep.vk.runner
    barrier = threading.Barrier(3, timeout=5)

    def runner(argv, timeout):
        if _command(argv) == ("recipe", "progress"):
            barrier.wait()
        return inner(argv, timeout)

    sweep.vk.runner = runner
    sweep.prefetcher._poll()  # serial reads would time the barrier out


def test_gather_runs_at_most_limit_reads_together() -> None:
    barrier = threading.Barrier(2, timeout=5)
    lock = threading.Lock()
    running = peak = 0

    def read() -> None:
        nonlocal running, peak
        with lock:
            running += 1
            peak = max(peak, running)
        barrier.wait()  # two reads must be in flight together to pass
        with lock:
            running -= 1

    outcomes = gather({str(n): read for n in range(4)}, DaemonExecutor(), limit=2)
    assert all(o.error is None for o in outcomes.values())
    assert peak == 2
