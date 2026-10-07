"""No silent or unbounded waits: ownership of leftovers, bounded clearing, fresh status, watchdog."""

from __future__ import annotations

import itertools
import json
from pathlib import Path

import pytest
from sweep_fakes import FakeModel, FakeRecipe, Gateway, make_sweep

from spark_sweep import policy
from spark_sweep.run import TICK_WATCHDOG_SECONDS, TickStuck


@pytest.fixture
def gateway():
    gateway = Gateway()
    yield gateway
    gateway.stop()


def _run(alias: str) -> dict:
    return {
        "alias": alias,
        "recipe": "x",
        "run_id": f"run-{alias}",
        "node_ids": ["spk_a"],
        "ready": True,
    }


def test_a_released_lanes_workload_is_ours_not_foreign(
    tmp_path: Path, gateway: Gateway
) -> None:
    sweep, fleet, _ = make_sweep(
        tmp_path, [FakeRecipe("a")], [FakeModel("m1")], gateway=gateway
    )
    sweep.refresh_catalog()
    sweep.state.data["own_aliases"].append(
        "qwen-dual"
    )  # a lane that passed and was released
    fleet.runs += [_run("qwen-dual"), _run("owner-glm")]
    assert sweep._fleet_runs() == (["owner-glm"], ["qwen-dual"])

    fleet.runs[:] = [_run("qwen-dual")]
    assert sweep._foreign_runs() is False


def test_a_review_blocked_by_our_own_finished_workload_clears_it_as_ours(
    tmp_path: Path, gateway: Gateway
) -> None:
    sweep, fleet, clock = make_sweep(
        tmp_path, [FakeRecipe("a")], [FakeModel("m1")], gateway=gateway
    )
    sweep.refresh_catalog()
    sweep.state.data["own_aliases"].append("qwen-dual")
    sweep.state.data["took_over"] = True
    fleet.runs.append(_run("qwen-dual"))
    recipe = next(iter(sweep.recipes.values()))
    placement = policy.Placement(recipe, ("spk_a",))
    sweep._blocked({}, {}, [placement], clock.now())
    messages = [e["message"] for e in sweep.state.data["events"]]
    assert any("our own finished workload qwen-dual: clearing" in m for m in messages)
    assert not any("not ours" in m for m in messages)
    assert sweep.state.data["took_over"] is False  # the next pass clears and retries
    assert sweep.state.status(recipe.key) != "failed"


def test_pending_cleanup_does_not_replace_intent_after_an_aggregate_wait_budget(
    tmp_path: Path,
) -> None:
    sweep, fleet, clock = make_sweep(tmp_path, [FakeRecipe("a")], [FakeModel("m1")])
    fleet.runs.append(_run("stubborn"))
    fleet.offline_nodes.add("spk_a")
    sweep.preflight()
    sweep.refresh_catalog()
    start = clock.now()
    assert sweep._take_over(start)
    assert clock.now() == start  # Acceptance/observation returns to the main loop.
    key, original = next(iter(sweep.state.data["cleanup_loads"].items()))
    identity = [
        (row["effect_id"], row["operation_id"], row["request_key"])
        for row in original["effects"]
    ]
    for _ in range(4):
        clock.sleep(300)
        sweep.observe_cleanup()
        assert sweep._take_over(clock.now())
    assert original["request_key"] == key
    assert [
        (row["effect_id"], row["operation_id"], row["request_key"])
        for row in original["effects"]
    ] == identity
    assert (
        len(
            [
                command
                for profile, command in fleet.calls
                if profile == sweep.cfg.sweep_profile
                and command[:2] == ("profile", "load")
                and "--review" not in command
            ]
        )
        == 1
    )
    assert not any(command[:2] == ("profile", "cancel") for _, command in fleet.calls)
    assert sweep._cleanup_nodes() == {"spk_a"}
    assert sweep.state.status("vonk-forge/a") != "failed"


def test_status_and_state_stay_fresh_while_original_cleanup_is_pending(
    tmp_path: Path,
) -> None:
    sweep, fleet, clock = make_sweep(
        tmp_path, [FakeRecipe("a")], [FakeModel("m1")], only=("unselected",)
    )
    fleet.runs.append(_run("stubborn"))
    fleet.offline_nodes.add("spk_a")
    sweep.preflight()
    sweep.refresh_catalog()
    assert sweep._take_over(clock.now())
    key, original = next(iter(sweep.state.data["cleanup_loads"].items()))
    seen: list[float] = []
    for _ in range(4):
        clock.sleep(300)
        sweep.tick()
        saved = json.loads((tmp_path / "state.json").read_text())
        seen.append(saved["updated_at"])
        assert saved["cleanup_loads"][key]["app_id"] == original["app_id"]
        assert saved["cleanup_loads"][key]["effects"][0]["state"] == "pending"
        assert (tmp_path / "status.json").exists()
        assert (tmp_path / "status.md").exists()
    assert len(set(seen)) == 4
    assert all(a < b for a, b in itertools.pairwise(seen))
    assert not any(command[:2] == ("profile", "cancel") for _, command in fleet.calls)


def test_the_watchdog_abandons_a_stuck_pass_and_the_sweep_continues(
    tmp_path: Path, gateway: Gateway
) -> None:
    sweep, _, clock = make_sweep(
        tmp_path, [FakeRecipe("a")], [FakeModel("m1")], gateway=gateway
    )

    def stuck() -> None:
        started = clock.now()
        while True:
            sweep.waiting("something that never happens", started)
            clock.sleep(10)

    sweep._tick = stuck  # type: ignore[method-assign]
    sweep.tick()  # returns instead of hanging
    assert "watchdog" in sweep.state.data["infra"]
    assert any(
        "watchdog: abandoned" in e["message"] for e in sweep.state.data["events"]
    )
    assert "waiting" not in sweep.state.data
    with pytest.raises(TickStuck):
        sweep._tick_started = clock.now() - TICK_WATCHDOG_SECONDS - 1
        sweep.waiting("x", clock.now())
