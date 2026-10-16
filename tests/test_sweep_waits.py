"""No silent waits: ownership of leftovers, bounded clearing, fresh status; a wait never expires."""

from __future__ import annotations

import itertools
import json
from pathlib import Path

import pytest
from sweep_fakes import FakeModel, FakeRecipe, Gateway, make_sweep

from spark_sweep import policy


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
    fleet.cleanup_stop_seconds["stubborn"] = 3600
    sweep.preflight()
    sweep.refresh_catalog()
    start = clock.now()
    assert sweep._take_over(start)
    assert clock.now() == start  # Acceptance/observation returns to the main loop.
    fleet.offline_nodes.add("spk_a")
    sweep._take_over(clock.now())
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
    assert sweep._cleanup_nodes() == set()
    assert sweep.state.status("vonk-forge/a") != "failed"


def test_status_and_state_stay_fresh_while_original_cleanup_is_pending(
    tmp_path: Path,
) -> None:
    sweep, fleet, clock = make_sweep(
        tmp_path, [FakeRecipe("a")], [FakeModel("m1")], only=("unselected",)
    )
    fleet.runs.append(_run("stubborn"))
    fleet.cleanup_stop_seconds["stubborn"] = 3600
    sweep.preflight()
    sweep.refresh_catalog()
    assert sweep._take_over(clock.now())
    fleet.offline_nodes.add("spk_a")
    sweep._take_over(clock.now())
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


def test_a_wait_is_shown_on_every_poll_and_never_expires(
    tmp_path: Path, gateway: Gateway
) -> None:
    sweep, _, clock = make_sweep(
        tmp_path, [FakeRecipe("a")], [FakeModel("m1")], gateway=gateway
    )
    started = clock.now()
    for _ in range(30):  # 30 hours: no watchdog turns the wait into an abandoned pass
        clock.t += 3600
        sweep.waiting("the Controller's admission", started)
        assert sweep.state.data["waiting"]["at"] == clock.now()
    assert "watchdog" not in sweep.state.data["infra"]
    assert any(
        "waiting for the Controller's admission" in e["message"]
        for e in sweep.state.data["events"]
    )
    sweep.done_waiting()
    assert "waiting" not in sweep.state.data


def test_a_missing_build_beside_another_lanes_disk_blocker_is_a_wait_not_a_verdict(
    tmp_path: Path, gateway: Gateway
) -> None:
    sweep, _, clock = make_sweep(
        tmp_path,
        [FakeRecipe("a"), FakeRecipe("b")],
        [FakeModel("m1")],
        gateway=gateway,
    )
    sweep.refresh_catalog()
    lane_a, lane_b = (
        policy.Placement(r, ("spk_a",)) for r in list(sweep.recipes.values())[:2]
    )
    aliases = {sweep._alias(p.recipe): p for p in (lane_a, lane_b)}
    document = {
        "admission_decisions": [
            {
                "alias": sweep._alias(lane_a.recipe),
                "allowed": False,
                "blockers": [{"code": "run-switch.recipe-build-unavailable"}],
            },
            {
                "alias": sweep._alias(lane_b.recipe),
                "allowed": False,
                "blockers": [{"code": "run-switch.insufficient-disk"}],
            },
        ]
    }
    sweep._blocked(document, aliases, [lane_a, lane_b], clock.now())
    entry_a = sweep.state.entry(lane_a.recipe.key)
    assert sweep.state.status(lane_a.recipe.key) != "failed"
    assert entry_a["defer_until"] > clock.now()
    assert sweep.state.entry(lane_b.recipe.key)["attempts"] == 1  # as before: recorded
    assert entry_a.get("attempts", 0) == 0


def test_a_blocked_review_keeps_the_controllers_document_on_every_row(
    tmp_path: Path, gateway: Gateway
) -> None:
    # On hardware a review came back not allowed with no blocker the sweep could
    # read: rows said only "the review was not allowed" and the Controller's
    # answer was gone, so a Spark idled with no cause to look at.
    sweep, _, clock = make_sweep(
        tmp_path, [FakeRecipe("a")], [FakeModel("m1")], gateway=gateway
    )
    sweep.refresh_catalog()
    recipe = next(iter(sweep.recipes.values()))
    sweep.state.slots["other"] = {
        "alias": "other",
        "phase": "smoking",
        "node_ids": ["spk_b"],
        "request_key": "k",
    }
    document = {"allowed": False, "unexplained": "the Controller's own words"}
    sweep._blocked(document, {}, [policy.Placement(recipe, ("spk_a",))], clock.now())
    rows = [
        json.loads(line)
        for line in (tmp_path / "results.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    (row,) = [r for r in rows if r.get("step") == "review"]
    assert row["status"] == "waiting"
    assert row["controller_document"]["unexplained"] == "the Controller's own words"
