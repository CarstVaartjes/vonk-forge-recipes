"""A test load is a cold start: an identical running run is cleared first, never kept and renamed.

vonk-forge #2112 makes a profile load keep a run whose execution identity matches an
assignment and only rename it. A kept run has no startup peak and no real load, so the
sweep clears it (its existing clearing load) before loading the recipe it tests.
"""

from __future__ import annotations

import json
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

    _tick_until_done(sweep, clock)

    assert sweep.state.recipes["vonk-forge/glm"]["status"] == "passed"
    assert not any(r["run_id"] == "run-owner" for r in fleet.runs)
    assert any("review would keep a running" in e for e in _events(sweep))
    assert not renamed  # the owner's run was never kept under the sweep's name


def _rows(tmp_path: Path) -> list[dict]:
    path = tmp_path / "results.jsonl"
    lines = path.read_text(encoding="utf-8").splitlines() if path.exists() else []
    return [json.loads(line) for line in lines]


def test_a_run_that_outlives_the_clearing_is_waited_for_and_others_proceed(
    tmp_path: Path, gateway: Gateway
) -> None:
    recipes = [
        FakeRecipe("glm", local="cached", memory=20 * GB),
        FakeRecipe("other", local="cached", memory=20 * GB),
    ]
    sweep, fleet, clock = make_sweep(
        tmp_path,
        recipes,
        [FakeModel("m1", local="cached")],
        gateway=gateway,
        only=("glm", "other"),
    )
    # The fleet was taken over and the owner's run was cleared once: the clearing was
    # submitted for exactly this run (decided for its run id), and the run is still there.
    sweep.state.data.update(
        took_over=True, cleared_for=["run-owner"], cleared_by="submitted-clearing"
    )
    fleet.runs.append(dict(OWNER_GLM))
    sweep.preflight()
    sweep.refresh_catalog()

    for _ in range(40):  # run()'s own ticks, until the lanes and the wait are in place
        sweep.tick()
        if "vonk-forge/other" in sweep.state.slots:
            break
        clock.sleep(sweep.cfg.poll_seconds)

    rows = {r["recipe"]: r for r in _rows(tmp_path)}
    waited = rows["vonk-forge/glm"]
    assert waited["disposition"] == "wait"
    assert "outlived a clearing: owner-glm" in waited["reason"]
    glm = sweep.state.recipes["vonk-forge/glm"]
    assert glm.get("attempts", 0) == 0
    assert glm["defer_until"] > clock.now()
    assert "vonk-forge/glm" not in sweep.state.slots
    assert "vonk-forge/other" in sweep.state.slots  # an independent recipe is admitted
    assert not any(r["alias"] == "glm" for r in fleet.runs)  # never kept under its name


def test_a_later_run_of_the_same_recipe_is_a_new_holder_and_is_cleared_again(
    tmp_path: Path, gateway: Gateway
) -> None:
    recipes = [FakeRecipe("glm", local="cached", memory=20 * GB)]
    sweep, fleet, clock = make_sweep(
        tmp_path, recipes, [FakeModel("m1", local="cached")], gateway=gateway
    )
    # An earlier clearing was for another run; this run (a new id) was never cleared.
    sweep.state.data.update(
        took_over=True, cleared_for=["run-earlier"], cleared_by="old-clearing"
    )
    fleet.runs.append(dict(OWNER_GLM))
    sweep.preflight()
    sweep.refresh_catalog()

    for _ in range(3):
        sweep.tick()
        clock.sleep(sweep.cfg.poll_seconds)

    assert any("review would keep" in e for e in _events(sweep))
    assert sweep.state.data["cleared_for"] == ["run-owner"]
    assert not any("outlived" in str(r.get("reason")) for r in _rows(tmp_path))


def test_another_recipes_run_on_the_lanes_sparks_does_not_trigger_the_clearing(
    tmp_path: Path, gateway: Gateway
) -> None:
    recipes = [
        FakeRecipe("glm", local="cached", memory=20 * GB),
        FakeRecipe("other", local="cached", memory=20 * GB),
    ]
    sweep, fleet, clock = make_sweep(
        tmp_path, recipes, [FakeModel("m1", local="cached")], gateway=gateway
    )
    sweep.state.data["took_over"] = True
    fleet.runs.append(dict(OWNER_GLM, recipe="vonk-forge/other", alias="owner-x"))

    _tick_until_done(sweep, clock)

    assert not any("review would keep" in e for e in _events(sweep))
    assert {e["status"] for e in sweep.state.recipes.values()} == {"passed"}


def _tick_until_done(sweep, clock) -> None:
    """run()'s own ticks without its startup, so the test's took_over setup stands."""
    sweep.preflight()
    sweep.refresh_catalog()
    for _ in range(200):
        sweep.tick()
        if sweep.done():
            return
        clock.sleep(sweep.cfg.poll_seconds)


def _events(sweep) -> list[str]:
    return [str(e) for e in sweep.state.data.get("events", [])]
