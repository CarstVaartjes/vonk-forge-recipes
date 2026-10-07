"""Offline workloads and unknown clearing bookkeeping cannot hold free online lanes."""

from pathlib import Path

import pytest
from sweep_fakes import FakeModel, FakeRecipe, Gateway, make_sweep

from spark_sweep.run import MAX_TAKEOVER_ATTEMPTS, TAKEOVER_RETRY_SECONDS

pytest_plugins = ["sweep_bounds"]
pytestmark = pytest.mark.usefixtures("bounded_clock")


@pytest.mark.parametrize("offline", [True, False])
def test_bounded_clearing_admits_ready_online_work(
    tmp_path: Path, offline: bool
) -> None:
    gateway = Gateway()
    try:
        recipes = [FakeRecipe(slug, local="cached") for slug in ("first", "next")]
        sweep, fleet, clock = make_sweep(
            tmp_path, recipes, [FakeModel("m1", local="cached")], gateway=gateway
        )
        fleet.runs.append(
            {
                "alias": "external",
                "recipe": "external",
                "run_id": "run-external",
                "node_ids": ["spk_b"],
                "ready": True,
            }
        )
        if offline:
            fleet.offline_nodes.add("spk_b")
        else:
            fleet.cleanup_projection = False
        started = clock.now()
        alerts = []
        clock.hooks.append(
            lambda _now: alerts.extend(sweep.state.data.get("alerts", []))
        )
        assert sweep.run() == 0
        assert {entry["status"] for entry in sweep.state.recipes.values()} == {"passed"}
        assert sweep.state.data["takeover_budget"]["attempts"] == MAX_TAKEOVER_ATTEMPTS
        assert sweep.state.data["takeover_fallback"]
        assert not alerts
        placements = [
            item for app in fleet.apps.values() for item in app.get("assign", [])
        ]
        assert {item["alias"] for item in placements} >= {"first", "next"}
        if offline:
            assert all(
                run["node_ids"] == ["spk_a"]
                for run in fleet.runs
                if run["recipe"] in {recipe.key for recipe in recipes}
            )
            assert any(run["run_id"] == "run-external" for run in fleet.runs)
        first_started = min(
            app["created"] for app in fleet.apps.values() if app.get("assign")
        )
        assert first_started - started <= 60
        if offline:
            # No empty sweep load is submitted to clear the offline workload.
            assert not any(
                item["kind"] == "takeover" for item in sweep.state.data["own_loads"]
            )
    finally:
        gateway.stop()


def test_clearing_status_names_offline_blocker_and_budget_survives_restart(
    tmp_path: Path,
) -> None:
    recipe = FakeRecipe("ready", local="cached")
    sweep, fleet, clock = make_sweep(
        tmp_path, [recipe], [FakeModel("m1", local="cached")]
    )
    fleet.offline_nodes.add("spk_b")
    fleet.runs.append(
        {
            "alias": "external",
            "recipe": "external",
            "run_id": "external",
            "node_ids": ["spk_b"],
            "ready": True,
        }
    )
    sweep.preflight()
    sweep.refresh_catalog()
    assert not sweep._take_over(clock.now())
    sweep.state.save()
    restarted, _, _ = make_sweep(
        tmp_path, [recipe], [FakeModel("m1", local="cached")], fleet=fleet
    )
    restarted.preflight()
    restarted.refresh_catalog()
    assert restarted.state.data["takeover_budget"]["attempts"] == 1
    assert not restarted._take_over(clock.now())
    assert restarted.state.data["takeover_budget"]["attempts"] == 1
    restarted.cached_models = restarted._cached_models()
    restarted.queue = [recipe.key]
    restarted.check_idle(clock.now())
    restarted.check_idle(clock.now() + restarted.cfg.idle_alert_seconds + 1)
    assert "clearing waits on offline spark-b" in restarted.state.data["alerts"][0]
    for _ in range(MAX_TAKEOVER_ATTEMPTS - 1):
        clock.sleep(TAKEOVER_RETRY_SECONDS)
        restarted._take_over(clock.now())
    assert restarted._take_over(clock.now())
    assert [lane.id for lane in restarted._bins()] == ["spk_a"]


def test_clearing_budget_ceiling_only_falls() -> None:
    assert 1 <= MAX_TAKEOVER_ATTEMPTS <= 3
    assert 0 < TAKEOVER_RETRY_SECONDS <= 30
