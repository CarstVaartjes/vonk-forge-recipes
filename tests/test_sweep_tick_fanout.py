"""A pass costs its slowest independent read, and the loop sleeps to its poll deadline."""

from __future__ import annotations

import threading
import time
from pathlib import Path

import pytest
from sweep_fakes import FakeModel, FakeRecipe, make_sweep

from spark_sweep.fanout import gather

pytest_plugins = ["sweep_bounds"]
pytestmark = pytest.mark.usefixtures("bounded_clock")

DELAY = 0.3


def _slow_reads(sweep, fleet):
    """The fake Controller answers each read after DELAY; one lock keeps the fake safe."""
    lock = threading.Lock()
    inner = fleet.runner

    def runner(argv, timeout):
        time.sleep(DELAY)
        with lock:
            return inner(argv, timeout)

    sweep.vk.runner = runner


def _observe(tmp_path: Path, workers: int) -> tuple[float, set[str]]:
    sweep, fleet, _ = make_sweep(
        tmp_path / str(workers),
        [FakeRecipe("a", ("m1",))],
        [FakeModel("m1")],
        read_workers=workers,
    )
    (tmp_path / str(workers)).mkdir(exist_ok=True)
    _slow_reads(sweep, fleet)
    started = time.monotonic()
    sweep._observe_concurrently()
    return time.monotonic() - started, set(sweep._reads) | {
        f"owner:{n}" for n in sweep.guard.reads
    }


def test_the_reads_of_a_pass_cost_the_slowest_not_the_sum(tmp_path: Path) -> None:
    sequential, names = _observe(tmp_path, 1)
    concurrent, same = _observe(tmp_path, 4)
    assert names == same == {"platform", "owner:1", "owner:2", "owner:3"}
    assert sequential >= 4 * DELAY
    assert concurrent < 2 * DELAY


def test_a_read_error_surfaces_at_its_consumer() -> None:
    def boom() -> None:
        raise RuntimeError("down")

    done = gather({"ok": lambda: 1, "bad": boom}, 4)
    assert done["ok"].result() == 1
    with pytest.raises(RuntimeError):
        done["bad"].result()


@pytest.mark.parametrize(("work", "sleep_for"), [(4.0, 6.0), (15.0, 0.0)])
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
        clock.t += work  # a pass inside (or past) the 10 s poll

    clock.sleep = sleep
    sweep.tick = tick
    sweep.done = lambda: len(slept) >= 3
    sweep.cfg.watch_seconds = 0
    sweep.run()
    assert slept[:3] == [sleep_for] * 3
