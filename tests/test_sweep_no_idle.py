"""The sweep never idles while ready work fits a free lane.

On hardware a dual's admission stalled; its lane failed and cancelled the application detached,
but the Controller kept holding it, so the sweep's load record stayed `running` forever and no
lane was scheduled again for over an hour.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
from sweep_fakes import FakeModel, FakeRecipe, Gateway, make_sweep

pytest_plugins = ["sweep_bounds"]
pytestmark = pytest.mark.usefixtures("bounded_clock")


@pytest.fixture
def gateway():
    gateway = Gateway()
    yield gateway
    gateway.stop()


def _orphan(sweep) -> None:
    sweep.state.data["loads"]["stale-request"] = {
        "request_key": "stale-request",
        "app_id": "app-stale",
        "seq": 0,
        "submitted_at": 0.0,
        "state": "running",
    }


def test_a_load_whose_lanes_ended_does_not_block_scheduling(
    tmp_path: Path, gateway: Gateway
) -> None:
    recipes = [FakeRecipe("a", ("m1",)), FakeRecipe("b", ("m2",))]
    sweep, fleet, _ = make_sweep(
        tmp_path, recipes, [FakeModel("m1"), FakeModel("m2")], gateway=gateway
    )
    _orphan(sweep)
    assert sweep.run() == 0
    assert {e["status"] for e in sweep.state.recipes.values()} == {"passed"}
    assert sweep.state.data["loads"] == {}
    cancels = [c for _, c in fleet.calls if c[:2] == ("profile", "cancel")]
    assert any("app-stale" in c for c in cancels)  # ours, unsettled: released


def test_a_load_with_a_live_lane_is_kept(tmp_path: Path, gateway: Gateway) -> None:
    sweep, _, _ = make_sweep(
        tmp_path, [FakeRecipe("a", ("m1",))], [FakeModel("m1")], gateway=gateway
    )
    _orphan(sweep)
    sweep.state.slots["a"] = {
        "request_key": "stale-request",
        "phase": "loading",
        "node_ids": [],
    }
    sweep.release_orphan_load()
    assert sweep.state.data["loads"]


def test_idle_with_ready_work_raises_an_alert(tmp_path: Path, gateway: Gateway) -> None:
    sweep, _, _ = make_sweep(
        tmp_path, [FakeRecipe("a", ("m1",))], [FakeModel("m1")], gateway=gateway
    )
    sweep.sparks = lambda: [SimpleNamespace(id="spk_a")]  # type: ignore[method-assign]
    sweep.ready_candidates = lambda: [object()]  # type: ignore[method-assign]
    now = sweep.clock.now()
    sweep.check_idle(now)
    assert not sweep.state.data.get("alerts")
    sweep.check_idle(now + sweep.cfg.idle_alert_seconds + 1)
    assert sweep.state.data["alerts"]
    assert "IDLE WITH READY WORK" in sweep.state.data["alerts"][0]
    sweep.ready_candidates = list  # type: ignore[method-assign]
    sweep.check_idle(now + 2 * sweep.cfg.idle_alert_seconds)
    assert sweep.state.data["alerts"] == []


def test_incremental_profile_fake_preserves_and_waits_for_exact_running_child(
    tmp_path: Path, gateway: Gateway
) -> None:
    slow = FakeRecipe("slow", load_seconds=600)
    fast = FakeRecipe("fast", load_seconds=60)
    _, fleet, clock = make_sweep(
        tmp_path, [slow, fast], [FakeModel("m1")], gateway=gateway
    )
    assignment = {
        "assignment_name": "slow",
        "recipe_selector": slow.key,
        "spark_ids": ["spark-a"],
        "desired_state": "running",
    }
    first_profile = {"assignments": [assignment]}
    _, first = fleet._load(10, first_profile, ["--yes", "--request-key", "first"])
    original = fleet.apps[first["id"]]
    child = original["assign"][0]
    run_id = fleet.runs[0]["run_id"]
    clock.sleep(20)
    second_profile = {
        "assignments": [
            assignment,
            {
                "assignment_name": "fast",
                "recipe_selector": fast.key,
                "spark_ids": ["spark-b"],
                "desired_state": "running",
            },
        ]
    }
    _, second = fleet._load(10, second_profile, ["--yes", "--request-key", "second"])
    incremental = fleet.apps[second["id"]]
    assert incremental["assign"][0] is child
    assert original["state"] == "running"
    assert fleet.runs[0]["run_id"] == run_id
    clock.sleep(60)
    fleet._tick()
    assert original["state"] == incremental["state"] == "running"
    assert not child["resolved"]
    clock.sleep(520)
    fleet._tick()
    assert original["state"] == incremental["state"] == "succeeded"
    assert fleet.runs[0]["run_id"] == run_id


def test_free_lane_starts_next_recipe_while_other_lane_keeps_loading(
    tmp_path: Path, gateway: Gateway
) -> None:
    recipes = [
        FakeRecipe("a-slow", ("m1",), local="cached", load_seconds=1200),
        FakeRecipe("b-fast", ("m2",), local="cached", load_seconds=60),
        FakeRecipe("c-next", ("m3",), local="cached", load_seconds=60),
    ]
    sweep, fleet, clock = make_sweep(
        tmp_path,
        recipes,
        [FakeModel(f"m{i}", local="cached") for i in range(1, 4)],
        gateway=gateway,
    )
    original_run = []

    def observe(_now: float) -> None:
        running = [run for run in fleet.runs if run["recipe"] == recipes[0].key]
        if running:
            if not original_run:
                original_run.append(running[0]["run_id"])
            assert running[0]["run_id"] == original_run[0]

    clock.hooks.append(observe)
    assert sweep.run() == 0
    assert {entry["status"] for entry in sweep.state.recipes.values()} == {"passed"}
    slow_child = next(
        item
        for app in fleet.apps.values()
        for item in app.get("assign", [])
        if item["alias"] == "a-slow"
    )
    next_started = min(
        app["created"]
        for app in fleet.apps.values()
        if any(item["alias"] == "c-next" for item in app.get("assign", []))
    )
    assert next_started < slow_child["at"], (
        "the free lane must run work before the slow lane finishes"
    )
    assert original_run
