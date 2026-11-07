"""A test load is a cold start: an identical running run is cleared first, never kept and renamed.

vonk-forge #2112 makes a profile load keep a run whose execution identity matches an
assignment and only rename it. A kept run has no startup peak and no real load, so the
sweep clears it (its existing clearing load) before loading the recipe it tests.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from sweep_fakes import GB, FakeModel, FakeRecipe, Gateway, make_sweep

pytest_plugins = ["sweep_bounds"]
pytestmark = pytest.mark.usefixtures("bounded_clock")

OWNER_GLM = {
    "alias": "owner-glm",
    "recipe": "vonk-forge/glm",
    "run_id": "run-owner",
    "node_ids": ["spk_a"],
    "ready": True,
}


@pytest.fixture
def gateway():
    gateway = Gateway()
    yield gateway
    gateway.stop()


def test_an_identical_running_run_is_cleared_and_the_recipe_starts_cold(
    tmp_path: Path, gateway: Gateway
) -> None:
    recipes = [FakeRecipe("glm", local="cached", memory=20 * GB)]
    sweep, fleet, clock = make_sweep(
        tmp_path, recipes, [FakeModel("m1", local="cached")], gateway=gateway
    )
    # The sweep already took the fleet over; the owner then brought the same recipe up.
    sweep.state.data["took_over"] = True
    fleet.runs.append(dict(OWNER_GLM))
    renamed: list[str] = []
    clock.hooks.append(
        lambda _now: renamed.extend(
            r["alias"]
            for r in fleet.runs
            if r["run_id"] == "run-owner" and r["alias"] != "owner-glm"
        )
    )

    assert sweep.run() == 0

    assert sweep.state.recipes["vonk-forge/glm"]["status"] == "passed"
    assert not any(r["run_id"] == "run-owner" for r in fleet.runs)
    assert any("review would keep a running" in e for e in _events(sweep))
    assert not renamed  # the owner's run was never kept under the sweep's name


def test_a_run_that_outlives_the_clearing_is_waited_for_not_kept(
    tmp_path: Path, gateway: Gateway
) -> None:
    recipes = [FakeRecipe("glm", local="cached", memory=20 * GB)]
    sweep, fleet, clock = make_sweep(
        tmp_path, recipes, [FakeModel("m1", local="cached")], gateway=gateway
    )
    sweep.state.data["took_over"] = True
    fleet.runs.append(dict(OWNER_GLM))
    fleet.cleanup_stop_seconds["owner-glm"] = 10**9  # the stop never completes
    clock.hooks.append(lambda _now: None)
    sweep.preflight()
    sweep.refresh_catalog()
    for _ in range(6):
        sweep.schedule(clock.now())
        clock.sleep(sweep.cfg.defer_delay + 1)

    assert sweep.state.recipes["vonk-forge/glm"]["status"] != "passed"
    assert not sweep.state.slots
    assert any(r["run_id"] == "run-owner" for r in fleet.runs)


def _events(sweep) -> list[str]:
    return [str(e) for e in sweep.state.data.get("events", [])]
