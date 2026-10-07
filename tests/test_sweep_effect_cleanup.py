"""Exact child receipts free disjoint lanes; root timers never replace intent."""

from __future__ import annotations

import copy
from pathlib import Path

import pytest
from sweep_fakes import FakeModel, FakeRecipe, make_sweep

from spark_sweep import cleanup, policy

pytest_plugins = ["sweep_bounds"]
pytestmark = pytest.mark.usefixtures("bounded_clock")


def run(alias: str, nodes: list[str]) -> dict:
    return {
        "alias": alias,
        "recipe": "external",
        "run_id": f"run-{alias}",
        "node_ids": nodes,
        "ready": True,
    }


def scenario(tmp_path: Path, *, dual: bool = False):
    recipe = FakeRecipe("ready", local="cached", load_seconds=10)
    sweep, fleet, clock = make_sweep(
        tmp_path, [recipe], [FakeModel("m1", local="cached")]
    )
    if dual:
        fleet.runs.append(run("gang", ["spk_a", "spk_b"]))
        fleet.cleanup_stop_seconds["gang"] = 3600
    else:
        fleet.runs.extend([run("healthy", ["spk_a"]), run("offline", ["spk_b"])])
        # Recover an older whole-fleet clearing accepted while both nodes were online.
        fleet.cleanup_stop_seconds["offline"] = 3600
    sweep.preflight()
    sweep.refresh_catalog()
    assert sweep._take_over(clock.now())
    key, original = next(iter(sweep.state.data["cleanup_loads"].items()))
    app = fleet.apps[original["app_id"]]
    if not dual:
        fleet.offline_nodes.add("spk_b")
        sweep._take_over(clock.now())
    return sweep, fleet, clock, recipe, key, original, app


def mutations(fleet) -> list[tuple]:
    return [
        command
        for profile, command in fleet.calls
        if profile == 10
        and command[:2] == ("profile", "load")
        and "--review" not in command
    ]


def apply_healthy(sweep, clock, recipe) -> None:
    sweep._apply([policy.Placement(sweep.recipes[recipe.key], ("spk_a",))], clock.now())


def test_healthy_lane_does_not_require_offline_cleanup_adoption(
    tmp_path: Path,
) -> None:
    sweep, fleet, clock, recipe, _key, original, app = scenario(tmp_path)
    assert sweep._cleanup_nodes() == set()
    stopped, pending = original["effects"]
    assert cleanup.stopped(stopped)
    assert pending["state"] == "pending" and pending["target_id"] == "run-offline"
    apply_healthy(sweep, clock, recipe)
    slot = sweep.state.slots[recipe.key]
    current = sweep.state.data["loads"][slot["request_key"]]
    assert current["adopted_cleanup_requests"] == []
    links = fleet.reviewed_cleanup_effects[-1]
    assert any(cleanup.adopted({"effects": {"adopted": links}}, pending) for _ in [0])
    assert app["state"] == "running"
    assert fleet.apps[current["app_id"]]["state"] == "running"
    assert any(
        row["application_id"] == app["id"]
        and row["operation_id"] == pending["operation_id"]
        for row in fleet.apps[current["app_id"]]["effects"]
    )
    assert any(
        command[:2] == ("profile", "load") and "--review" in command
        for _, command in fleet.calls
    )


def test_missing_offline_cleanup_adoption_admits_fresh_load_without_blaming_recipe(
    tmp_path: Path,
) -> None:
    sweep, fleet, clock, recipe, _key, _original, _app = scenario(tmp_path)
    before = len(mutations(fleet))
    fleet.cleanup_adoption = False
    apply_healthy(sweep, clock, recipe)
    assert len(mutations(fleet)) == before + 1
    assert recipe.key in sweep.state.slots
    assert sweep.state.status(recipe.key) != "failed"
    assert sweep._cleanup_nodes() == set()


def test_fresh_capacity_review_can_refuse_a_locally_cached_free_lane(
    tmp_path: Path,
) -> None:
    sweep, fleet, clock, recipe, _key, original, _app = scenario(tmp_path)
    before = len(mutations(fleet))
    # Cached local placement facts are stale; the actual new Controller review owns admission.
    fleet.recipes[recipe.key].memory = 200 * 2**30
    apply_healthy(sweep, clock, recipe)
    assert len(mutations(fleet)) == before
    assert recipe.key not in sweep.state.slots
    assert original["effects"][1]["state"] == "pending"
    assert fleet.reviewed_cleanup_effects[-1]


def test_offline_cleanup_does_not_hold_finished_lane_observer(
    tmp_path: Path,
) -> None:
    sweep, _fleet, clock, recipe, _key, _original, app = scenario(tmp_path)
    apply_healthy(sweep, clock, recipe)
    slot = sweep.state.slots[recipe.key]
    request = slot["request_key"]
    current = sweep.state.data["loads"][request]
    slot["phase"] = "finished"
    sweep._release_orphan_load(current)
    assert request not in sweep.state.data["loads"]
    assert current["adopted_cleanup_requests"] == []
    assert app["state"] == "running"
    apply_healthy(sweep, clock, recipe)
    assert recipe.key in sweep.state.slots


def test_pending_root_keeps_exact_child_identity_beyond_old_wait_budget_and_restart(
    tmp_path: Path,
) -> None:
    sweep, fleet, clock, recipe, key, original, app = scenario(tmp_path)
    previous = copy.deepcopy(original["effects"])
    before = len(mutations(fleet))
    for _ in range(4):
        clock.sleep(300)
        sweep.observe_cleanup()
        assert sweep._take_over(clock.now())
    assert len(mutations(fleet)) == before
    assert original["effects"] == previous
    assert original["request_key"] == key and original["app_id"] == app["id"]
    assert not any(command[:2] == ("profile", "cancel") for _, command in fleet.calls)
    sweep.state.save()
    restarted, _, _ = make_sweep(
        tmp_path, [recipe], [FakeModel("m1", local="cached")], fleet=fleet
    )
    restarted.preflight()
    restarted.observe_cleanup()
    assert restarted._take_over(clock.now())
    assert restarted.state.data["cleanup_loads"][key]["effects"] == previous
    assert len(mutations(fleet)) == before


@pytest.mark.parametrize(
    "corruption",
    ["missing-projection", "run-mismatch", "scope-mismatch", "receipt-mismatch"],
)
def test_unknown_or_mismatched_full_gang_stop_never_frees_one_member(
    tmp_path: Path, corruption: str
) -> None:
    sweep, fleet, _clock, _recipe, _key, original, app = scenario(tmp_path, dual=True)
    row = app["effects"][0]
    if corruption == "missing-projection":
        fleet.cleanup_projection = False
    elif corruption == "run-mismatch":
        row["target_id"] = "different-run"
    elif corruption == "scope-mismatch":
        row["stop_effect"]["node_ids"] = ["spk_a"]
    else:
        row["state"] = "succeeded"
        row["result"] = {
            "run_switch_operation_id": row["operation_id"],
            "run_switch": {
                "phase_results": [{"phase": "stop", "run_id": "different-run"}]
            },
        }
    sweep.observe_cleanup()
    assert sweep._cleanup_nodes() == {"spk_a", "spk_b"}
    assert not cleanup.stopped(original["effects"][0])


def test_gang_claim_occupies_full_topology_until_same_exact_stop_receipt(
    tmp_path: Path,
) -> None:
    sweep, _fleet, clock, _recipe, _key, original, _app = scenario(tmp_path, dual=True)
    assert sweep._cleanup_nodes() == {"spk_a", "spk_b"}
    clock.sleep(1800)
    sweep.observe_cleanup()
    assert sweep._cleanup_nodes() == {"spk_a", "spk_b"}
    clock.sleep(1800)
    sweep.observe_cleanup()
    assert cleanup.stopped(original["effects"][0])
    assert sweep._cleanup_nodes() == set()


def test_terminal_cleanup_may_reenter_fresh_admission_but_changed_topology_may_not(
    tmp_path: Path,
) -> None:
    sweep, fleet, clock, _recipe, _key, original, app = scenario(tmp_path, dual=True)
    app["state"] = "cancelled"
    app["effects"][0]["state"] = "cancelled"
    sweep.observe_cleanup()
    fleet.runs[0]["node_ids"] = ["spk_a"]
    before = len(mutations(fleet))
    assert sweep._take_over(clock.now()) is False
    assert len(mutations(fleet)) == before
    assert sweep._cleanup_nodes() == {"spk_a", "spk_b"}
    fleet.runs[0]["node_ids"] = ["spk_a", "spk_b"]
    clock.sleep(30)
    fleet.cleanup_stop_seconds["gang"] = 0
    assert sweep._take_over(clock.now())
    assert len(mutations(fleet)) == before + 1
    assert (
        original["effects"][0]["state"] == "cancelled"
    )  # historical observation survives.
    assert sweep._cleanup_nodes() == set()


def test_older_successful_stop_receipt_cannot_release_newer_stop_intent(
    tmp_path: Path,
) -> None:
    sweep, _fleet, _clock, _recipe, _key, original, _app = scenario(tmp_path)
    older = original["effects"][0]
    assert cleanup.stopped(older)
    newer = copy.deepcopy(older)
    newer.update(
        effect_id="new-owner:queue:0:stop:run-healthy",
        application_id="new-owner",
        workload_intent_ordinal=older["workload_intent_ordinal"] + 1,
        request_key="new-request",
        operation_id="new-operation",
        state="pending",
        result=None,
    )
    sweep.state.data["cleanup_loads"]["new-request"] = {
        "request_key": "new-request",
        "app_id": "new-owner",
        "projection_known": True,
        "state": "running",
        "effects": [newer],
    }
    assert sweep._cleanup_nodes() == {"spk_a"}
