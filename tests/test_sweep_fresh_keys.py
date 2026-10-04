"""Clearing, stop and restore loads use a fresh request key per attempt and are verified.

The Controller answers a repeated request key with the application it already made. A
takeover with a fixed key therefore returned yesterday's finished application, stopped
nothing, and the sweep looped on "clearing" while the owner's workload kept running.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from sweep_fakes import FakeModel, FakeRecipe, Gateway, make_sweep

OWNER = {
    "alias": "owner-glm",
    "recipe": "x",
    "run_id": "run-owner",
    "node_ids": ["spk_a", "spk_b"],
    "ready": True,
}


@pytest.fixture
def gateway():
    gateway = Gateway()
    yield gateway
    gateway.stop()


def _load_keys(fleet, profile: int) -> list[str]:
    return [
        c[c.index("--request-key") + 1]
        for p, c in fleet.calls
        if p == profile and c[:3] == ("profile", "load", "--yes")
    ]


def test_a_second_takeover_in_the_same_state_directory_really_clears_the_fleet(
    tmp_path: Path, gateway: Gateway
) -> None:
    recipes = [FakeRecipe("a", ("m1",))]
    first, fleet, _ = make_sweep(tmp_path, recipes, [FakeModel("m1")], gateway=gateway)
    fleet.runs.append(dict(OWNER))
    first.run()
    assert first.state.recipes["vonk-forge/a"]["status"] == "passed"
    # Yesterday's takeover succeeded. Today the owner's workload is back and there is new work.
    fleet.runs.append(dict(OWNER))
    fleet.recipes["vonk-forge/b"] = FakeRecipe("b", ("m2",))
    fleet.models["m2"] = FakeModel("m2")
    again, _, _ = make_sweep(
        tmp_path,
        [*recipes, fleet.recipes["vonk-forge/b"]],
        [FakeModel("m1"), FakeModel("m2")],
        fleet=fleet,
    )
    again.state.data["took_over"] = False
    assert again.run() == 0
    assert again.state.recipes["vonk-forge/b"]["status"] == "passed"
    assert all(r["alias"] != "owner-glm" for r in fleet.runs)
    keys = _load_keys(fleet, 10)
    assert len(keys) == len(set(keys))  # no key was ever sent twice
    assert [x["kind"] for x in again.state.data["own_loads"]].count("takeover") == 2


def test_every_attempt_has_its_own_key_recorded_in_own_loads(
    tmp_path: Path, gateway: Gateway
) -> None:
    sweep, _, _ = make_sweep(
        tmp_path, [FakeRecipe("a")], [FakeModel("m1")], gateway=gateway
    )
    keys = {
        sweep._own_key(kind, profile)
        for kind in ("takeover", "stop")
        for profile in (2, 10)
        for _ in range(3)
    }
    assert len(keys) == 12 and sweep.state.data["own_seq"] == 12


def test_an_old_application_returned_for_a_new_request_is_stale_and_retried(
    tmp_path: Path, gateway: Gateway
) -> None:
    sweep, fleet, _ = make_sweep(
        tmp_path, [FakeRecipe("a")], [FakeModel("m1")], gateway=gateway
    )
    fleet.runs.append(dict(OWNER))
    sweep.state.data["own_loads"].append(
        {
            "request_key": "old",
            "kind": "takeover",
            "profile": 10,
            "application_id": "app-0001",
        }
    )
    fleet.apps["app-0001"] = {
        "id": "app-0001",
        "profile": 10,
        "created": 0,
        "assign": [],
        "state": "succeeded",
        "request_key": "old",
    }
    fleet.profiles[10] = {
        "revision": 1,
        "assignments": [],
        "latest": "app-0001",
        "labels": {"purpose": "hardware-sweep"},
    }
    fleet.stale_loads = (
        2  # the Controller answers the first two attempts with the old application
    )
    assert sweep.run() == 0
    assert sweep.state.recipes["vonk-forge/a"]["status"] == "passed"
    assert (
        sum(
            "returned an old application" in e["message"]
            for e in sweep.state.data["events"]
        )
        == 2
    )
    assert all(r["alias"] != "owner-glm" for r in fleet.runs)


def test_a_controller_that_never_starts_a_new_application_is_an_error_not_a_loop(
    tmp_path: Path, gateway: Gateway
) -> None:
    sweep, fleet, _ = make_sweep(
        tmp_path, [FakeRecipe("a")], [FakeModel("m1")], gateway=gateway
    )
    fleet.runs.append(dict(OWNER))
    fleet.apps["app-0001"] = {
        "id": "app-0001",
        "profile": 10,
        "created": 0,
        "assign": [],
        "state": "succeeded",
        "request_key": "old",
    }
    fleet.profiles[10] = {
        "revision": 1,
        "assignments": [],
        "latest": "app-0001",
        "labels": {"purpose": "hardware-sweep"},
    }
    sweep.state.data["own_loads"].append(
        {
            "request_key": "old",
            "kind": "takeover",
            "profile": 10,
            "application_id": "app-0001",
        }
    )
    fleet.stale_loads = 99
    with pytest.raises(RuntimeError, match="not new; nothing was started"):
        sweep.run()
    assert len(_load_keys(fleet, 10)) == 3  # three attempts, each with its own key


def test_a_new_application_that_leaves_the_sparks_busy_is_retried(
    tmp_path: Path, gateway: Gateway
) -> None:
    sweep, fleet, _ = make_sweep(
        tmp_path, [FakeRecipe("a")], [FakeModel("m1")], gateway=gateway
    )
    fleet.runs.append(dict(OWNER))
    fleet.ignore_clearing = (
        1  # the first clearing load "succeeds" but the owner's workload keeps running
    )
    assert sweep.run() == 0
    assert sweep.state.recipes["vonk-forge/a"]["status"] == "passed"
    assert any(
        "still run something after clearing load 1" in e["message"]
        for e in sweep.state.data["events"]
    )
    assert [x["kind"] for x in sweep.state.data["own_loads"]].count("takeover") == 2


def test_sparks_that_never_go_idle_raise_instead_of_testing_on_a_busy_fleet(
    tmp_path: Path, gateway: Gateway
) -> None:
    sweep, fleet, _ = make_sweep(
        tmp_path, [FakeRecipe("a")], [FakeModel("m1")], gateway=gateway
    )
    fleet.runs.append(dict(OWNER))
    fleet.ignore_clearing = 99
    with pytest.raises(
        RuntimeError, match=r"still run workloads after 3 clearing loads: owner-glm"
    ):
        sweep.run()
    assert not [
        c for p, c in fleet.calls if p == 10 and c[:2] == ("profile", "add")
    ]  # nothing was placed


def test_restores_after_resumed_runs_never_reuse_a_key(
    tmp_path: Path, gateway: Gateway
) -> None:
    recipes = [FakeRecipe("a", load_seconds=900)]
    fleet = None
    for round_ in range(2):
        sweep, fleet, clock = make_sweep(
            tmp_path,
            recipes,
            [FakeModel("m1")],
            gateway=gateway if fleet is None else None,
            fleet=fleet,
            restore_owner=2,
        )
        start, fired = clock.now(), []

        def hook(now: float, start: float = start, fired: list = fired) -> None:
            if now - start >= 200 and not fired:
                fired.append(True)
                raise KeyboardInterrupt

        clock.hooks[:] = [hook]
        assert sweep.run() == 130
    restores = [
        x
        for x in json.loads((tmp_path / "state.json").read_text())["own_loads"]
        if x["kind"] == "restore-owner"
    ]
    assert (
        len(restores) == 2 and restores[0]["request_key"] != restores[1]["request_key"]
    )
    sent = _load_keys(fleet, 2)
    assert (
        len(sent) == len(set(sent)) == 2
    )  # the second restore was not answered with the first one
