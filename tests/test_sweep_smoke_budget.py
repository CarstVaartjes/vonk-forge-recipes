"""A smoking job lane shows the budget its smoke enforces, and only its own wait.

Recorded on the hardware sweep (2026-10-16): Step1X job lanes showed the 20-minute
load window as their limit while the job smoke followed its job for the job's
``timeout_seconds`` plus the grace, and kept showing the load's last wait
("controller retry") for the whole smoke.
"""

from __future__ import annotations

from concurrent.futures import Future
from pathlib import Path

import pytest
from sweep_fakes import FakeModel, FakeRecipe, make_sweep

from spark_sweep.report import build_status
from spark_sweep.smoke_job import job_budget_seconds

pytest_plugins = ["sweep_bounds"]
pytestmark = pytest.mark.usefixtures("bounded_clock")


class _Following:
    """An executor whose job smokes are still following their jobs."""

    def submit(self, *_args, **_kwargs) -> Future:
        return Future()


def test_a_smoking_job_lane_shows_its_smokes_own_budget_and_wait(
    tmp_path: Path,
) -> None:
    job = FakeRecipe("ocr", adapter="artifact-job")
    sweep, _fleet, clock = make_sweep(
        tmp_path, [job], [FakeModel("m1")], job_recipes_loadable=True
    )
    sweep.executor = _Following()
    sweep.refresh_catalog()
    for _ in range(200):
        sweep.tick()
        slot = sweep.state.slots.get(job.key)
        if slot is not None and slot["phase"] == "smoking":
            break
        if slot is not None:
            # The load's last observation was a wait on the Controller's retry.
            slot.setdefault("watch", {})["waiting_on"] = "controller retry"
        clock.sleep(sweep.cfg.poll_seconds)
    else:
        raise AssertionError("the job lane never reached its smoke")

    clock.sleep(120)
    (lane,) = [
        item
        for items in build_status(sweep)["lanes"].values()
        for item in items
        if item["recipe"] == job.key
    ]
    budget = job_budget_seconds(sweep.defs.artifact(job.key))
    assert lane["phase"] == "smoking"
    assert lane["limit_s"] == round(budget)
    assert lane["elapsed_s"] == 120
    assert lane["waiting_on"] is None
