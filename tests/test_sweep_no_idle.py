"""The sweep never idles while ready work fits a free lane.

On hardware a dual's admission stalled; its lane failed and cancelled the application detached,
but the Controller kept holding it, so the sweep's load record stayed `running` forever and no
lane was scheduled again for over an hour.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

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
    _, review = fleet._load(10, second_profile, ["--review"])
    adopted = review["effects"]["adopted"]
    assert len(adopted) == 1
    assert adopted[0]["application_id"] == original["id"]
    assert adopted[0]["plan_digest"] == original["plan_digest"]
    assert adopted[0]["assignment_ids"] == child["assignment_ids"]
    assert adopted[0]["node_ids"] == fleet.runs[0]["node_ids"]
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


@pytest.mark.parametrize("adoption_bound", [True, False])
def test_free_lane_starts_only_with_bound_unchanged_effects(
    tmp_path: Path, gateway: Gateway, adoption_bound: bool
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
    actual_load = fleet._load

    def load(n: int, data: dict[str, Any], a: list[str]) -> tuple[int, Any]:
        code, document = actual_load(n, data, a)
        if "--review" in a and not adoption_bound:
            document["effects"].pop("adopted", None)
        return code, document

    fleet._load = load

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
    if adoption_bound:
        assert next_started < slow_child["at"], (
            "a bound unchanged child permits free-lane work"
        )
    else:
        assert next_started >= slow_child["at"], (
            "an unbound child must not be cancelled by overlap"
        )
        assert "unchanged loading effects are not bound for adoption" in str(
            sweep.state.data["events"]
        )
    assert original_run


@pytest.mark.parametrize("previous_lane", ["finished", "loading"])
def test_cached_lane_starts_before_blocked_preparation_maintenance(
    tmp_path: Path,
    gateway: Gateway,
    monkeypatch: pytest.MonkeyPatch,
    previous_lane: str,
) -> None:
    from spark_sweep.vonkctl import VonkctlError

    recipes = [
        FakeRecipe("slow", ("m1",), local="cached", load_seconds=1200),
        FakeRecipe("ready", ("m2",), local="cached", load_seconds=60),
    ]
    sweep, fleet, clock = make_sweep(
        tmp_path,
        recipes,
        [FakeModel("m1", local="cached"), FakeModel("m2", local="cached")],
        gateway=gateway,
    )
    sweep.preflight()
    sweep.start_takeover()
    sweep.refresh_catalog()
    original_request = None
    original_application = None
    if previous_lane == "loading":
        sweep.cfg.only = ("slow",)
        sweep.tick()
        original_request = sweep.state.slots[recipes[0].key]["request_key"]
        original_application = sweep.state.data["loads"][original_request]["app_id"]
        sweep.cfg.only = ()
    else:
        sweep.state.slots["retired"] = {
            "request_key": "old-request",
            "phase": "finished",
            "alias": "already-absent",
            "node_ids": ["spk_a"],
            "started_at": clock.now(),
        }
    tracked = {
        "request_key": "00000000-0000-4000-8000-000000000001",
        "operation_id": "00000000-0000-4000-8000-000000000002",
        "state": "queued",
    }
    sweep.state.downloads[recipes[1].key] = dict(tracked)
    before = clock.now()
    observed = []

    def blocked_maintenance(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
        observed.append([run["recipe"] for run in fleet.runs])
        clock.sleep(240)
        raise VonkctlError(("recipe", "progress"), None, "controller.transport_timeout")

    monkeypatch.setattr(sweep.prefetcher, "tick", blocked_maintenance)
    sweep.tick()
    assert observed and recipes[1].key in observed[0]
    slot = sweep.state.slots[recipes[1].key]
    request = slot["request_key"]
    load = sweep.state.data["loads"][request]
    assert fleet.apps[load["app_id"]]["created"] == before
    assert fleet.apps[load["app_id"]]["request_key"] == request
    assert sweep.state.downloads[recipes[1].key] == tracked
    assert "retired" not in sweep.state.slots
    assert sweep.state.data["cleanup"] == {}
    if original_request is not None:
        assert original_application is not None
        assert sweep.state.slots[recipes[0].key]["request_key"] == original_request
        assert (
            sweep.state.data["loads"][original_request]["app_id"]
            == original_application
        )
        assert fleet.apps[original_application]["state"] == "running"


def test_cached_lane_starts_before_blocked_cleanup_and_reused_alias_is_retained(
    tmp_path: Path, gateway: Gateway, monkeypatch: pytest.MonkeyPatch
) -> None:
    from collections.abc import Sequence

    from spark_sweep.vonkctl import VonkctlTimeout

    recipe = FakeRecipe("ready", local="cached", load_seconds=60)
    sweep, fleet, clock = make_sweep(
        tmp_path, [recipe], [FakeModel("m1", local="cached")], gateway=gateway
    )
    sweep.preflight()
    sweep.start_takeover()
    sweep.refresh_catalog()
    for key, alias in (("old-reused", "ready"), ("old-unused", "old-unused")):
        sweep.state.slots[key] = {
            "request_key": "old-request",
            "phase": "finished",
            "alias": alias,
            "node_ids": [],
            "started_at": clock.now(),
        }
    original_runner = sweep.vk.runner
    observed = []

    def cleanup_wait(argv: Sequence[str], timeout: float) -> tuple[int, str, str]:
        if "profile" in argv:
            command = list(argv)[list(argv).index("profile") :]
            if command[:2] == ["profile", "remove"]:
                observed.append(command[2])
                if command[2] == "old-unused":
                    # The receipt and released occupancy are on disk before this wait.
                    import json

                    saved = json.loads(sweep.state.path.read_text())
                    slot = saved["slots"][recipe.key]
                    request = slot["request_key"]
                    assert saved["loads"][request]["app_id"] in fleet.apps
                    assert "old-reused" not in saved["slots"]
                    clock.sleep(240)
                    raise VonkctlTimeout(argv, None, "controller.transport_timeout")
        return original_runner(argv, timeout)

    monkeypatch.setattr(sweep.vk, "runner", cleanup_wait)
    before = clock.now()
    sweep.tick()
    assert observed == ["old-unused"]  # never remove the reused live assignment
    slot = sweep.state.slots[recipe.key]
    load = sweep.state.data["loads"][slot["request_key"]]
    assert fleet.apps[load["app_id"]]["created"] == before
    assert sweep.state.data["cleanup"][slot["alias"]] == {
        "attempts": 0,
        "next_check": 0.0,
    }
    assert "old-unused" not in sweep.state.data["cleanup"]
