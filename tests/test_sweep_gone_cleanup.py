"""Gone cleanup bookkeeping releases admission; unavailable reads never invent absence."""

from pathlib import Path

import pytest
from sweep_fakes import FakeModel, FakeRecipe, make_sweep
from vonk_forge_contracts.sweep import CleanupEnd, CleanupEndReason

from spark_sweep import policy

pytest_plugins = ["sweep_bounds"]
pytestmark = pytest.mark.usefixtures("bounded_clock")


def scenario(tmp_path: Path):
    recipe = FakeRecipe("ready", local="cached")
    sweep, fleet, clock = make_sweep(
        tmp_path, [recipe], [FakeModel("m1", local="cached")]
    )
    sweep.preflight()
    sweep.refresh_catalog()
    sweep.state.data["cleanup_loads"]["old-request"] = {
        "request_key": "old-request",
        "app_id": "gone-load",
    }
    record = sweep.state.data["cleanup_loads"]["old-request"]
    sweep.state.data["took_over"] = True
    sweep.note_infra("clearing", "old missing cleanup")
    return sweep, fleet, clock, recipe, record


def absent(fleet):
    fleet.observations.append(
        (
            ("profile", "progress"),
            2,
            {"error_type": "control_api", "code": "controller.not_found"},
        )
    )


def admit(sweep, clock, recipe):
    assert sweep._cleanup_nodes() == set()
    assert sweep._take_over(clock.now())
    sweep._apply([policy.Placement(sweep.recipes[recipe.key], ("spk_a",))], clock.now())
    assert recipe.key in sweep.state.slots


@pytest.mark.parametrize("app_id", [None, "gone-load"])
def test_gone_cleanup_ends_and_admits_after_restart(tmp_path: Path, app_id) -> None:
    sweep, fleet, clock, recipe, record = scenario(tmp_path)
    record["app_id"] = app_id
    absent(fleet)
    sweep.observe_cleanup()
    assert (
        CleanupEnd.model_validate(record["end"]).reason
        == CleanupEndReason.TARGET_ABSENT
    )
    assert "clearing" not in sweep.state.data["infra"]
    assert "terminal_receipt" not in record  # no synthetic Controller success
    restarted, _, _ = make_sweep(
        tmp_path, [recipe], [FakeModel("m1", local="cached")], fleet=fleet
    )
    restarted.preflight()
    restarted.refresh_catalog()
    admit(restarted, clock, recipe)


def test_present_run_keeps_cleanup_until_observed_gone(tmp_path: Path) -> None:
    sweep, fleet, clock, recipe, record = scenario(tmp_path)
    fleet.runs.append(
        {
            "alias": "old",
            "recipe": "external",
            "run_id": "old-run",
            "node_ids": ["spk_a"],
            "ready": True,
        }
    )
    absent(fleet)
    sweep.observe_cleanup()
    assert "end" not in record
    assert sweep._cleanup_nodes() == {"spk_a", "spk_b"}
    fleet.runs.clear()
    clock.sleep(30)
    absent(fleet)
    sweep.observe_cleanup()
    assert (
        CleanupEnd.model_validate(record["end"]).reason
        == CleanupEndReason.TARGET_ABSENT
    )
    admit(sweep, clock, recipe)


@pytest.mark.parametrize("observation", [None, {}, {"nodes": []}])
def test_unavailable_fleet_reobserves_with_capped_backoff(
    tmp_path: Path, observation
) -> None:
    sweep, fleet, clock, recipe, record = scenario(tmp_path)
    for attempt in range(1, 9):
        absent(fleet)
        fleet.observations.append((("fleet",), 0, observation))
        sweep.observe_cleanup()
        assert "end" not in record
        budget = record["observation_budget"]
        assert budget["attempts"] == attempt
        delay = budget["next_check"] - clock.now()
        assert 0 < delay <= 600
        assert sweep.state.data["infra"]["clearing"]["until"] == budget["next_check"]
        assert sweep.state.data["infra"]["clearing"]["count"] == attempt + 1
        calls = len(fleet.calls)
        sweep.observe_cleanup()
        assert len(fleet.calls) == calls
        clock.sleep(delay)
    absent(fleet)
    sweep.observe_cleanup()
    assert "observation_budget" not in record
    admit(sweep, clock, recipe)


def test_transport_error_with_empty_fleet_is_not_absence(tmp_path: Path) -> None:
    sweep, fleet, clock, recipe, record = scenario(tmp_path)
    fleet.observations.append((("profile", "progress"), 124, {}))
    sweep.observe_cleanup()
    assert "end" not in record
    clock.sleep(30)
    absent(fleet)
    sweep.observe_cleanup()
    admit(sweep, clock, recipe)


def test_absent_receipt_with_exact_present_run_keeps_original_clear(
    tmp_path: Path,
) -> None:
    from vonk_forge_contracts.sweep import ReviewedCleanupStop

    sweep, fleet, clock, recipe, record = scenario(tmp_path)
    record["reviewed_stops"] = [
        ReviewedCleanupStop(run_id="old-run", node_ids=["spk_a"]).model_dump_json()
    ]
    fleet.runs.append(
        {
            "alias": "old",
            "recipe": "external",
            "run_id": "old-run",
            "node_ids": ["spk_a"],
            "ready": True,
        }
    )
    absent(fleet)
    sweep.observe_cleanup()
    assert "end" not in record
    assert record["app_id"] == "gone-load"
    fleet.runs.clear()
    clock.sleep(30)
    absent(fleet)
    sweep.observe_cleanup()
    admit(sweep, clock, recipe)


def test_offline_fleet_keeps_observing_until_online(tmp_path: Path) -> None:
    sweep, fleet, clock, recipe, record = scenario(tmp_path)
    fleet.offline_nodes.add("spk_a")
    absent(fleet)
    sweep.observe_cleanup()
    assert "end" not in record
    fleet.offline_nodes.clear()
    clock.sleep(30)
    absent(fleet)
    sweep.observe_cleanup()
    admit(sweep, clock, recipe)


def test_gone_clear_releases_existing_backoff_and_schedules_immediately(
    tmp_path: Path,
) -> None:
    sweep, fleet, clock, recipe, record = scenario(tmp_path)
    assert sweep.backoff_until > clock.now()
    absent(fleet)
    sweep.observe_cleanup()
    assert "end" in record
    assert sweep.backoff_until <= clock.now()
    sweep.cached_models = {"m1"}
    sweep.queue = [recipe.key]
    sweep.schedule(clock.now())
    assert recipe.key in sweep.state.slots


def test_present_cleanup_still_observes_stop_receipt_and_admits_fresh_load(
    tmp_path: Path,
) -> None:
    from test_sweep_effect_cleanup import scenario as present_scenario

    sweep, _fleet, clock, recipe, _key, record, _app = present_scenario(
        tmp_path, dual=True
    )
    clock.sleep(3600)
    sweep.observe_cleanup()
    assert "end" not in record
    assert record["projection_known"]
    admit(sweep, clock, recipe)


def test_cleanup_contract_module_is_registered() -> None:
    from pydantic import TypeAdapter

    root = Path(__file__).resolve().parents[1]
    registered = TypeAdapter(list[str]).validate_json(
        (root / "tools/python-model-registry.json").read_text()
    )
    assert CleanupEnd.__module__ in registered
    assert CleanupEnd.model_json_schema()["$defs"]["CleanupEndReason"]["enum"] == [
        CleanupEndReason.TARGET_ABSENT
    ]


def test_unrelated_offline_node_does_not_hold_absent_exact_target(
    tmp_path: Path,
) -> None:
    from vonk_forge_contracts.sweep import ReviewedCleanupStop

    sweep, fleet, clock, recipe, record = scenario(tmp_path)
    record["reviewed_stops"] = [
        ReviewedCleanupStop(run_id="old-run", node_ids=["spk_a"]).model_dump_json()
    ]
    fleet.offline_nodes.add("spk_b")
    absent(fleet)
    sweep.observe_cleanup()
    assert (
        CleanupEnd.model_validate(record["end"]).reason
        == CleanupEndReason.TARGET_ABSENT
    )
    admit(sweep, clock, recipe)
