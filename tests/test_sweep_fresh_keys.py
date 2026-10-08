"""New authorized intents use fresh keys; uncertain owned requests keep theirs."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from sweep_fakes import FakeModel, FakeRecipe, Gateway, make_sweep

pytest_plugins = ["sweep_bounds"]
pytestmark = pytest.mark.usefixtures("bounded_clock")

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


def test_ended_owned_application_keeps_audit_identity_and_admits_new_loads(
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
    sweep.preflight()
    sweep.refresh_catalog()
    sweep.start_takeover()
    for _ in range(6):
        sweep.tick()
        sweep.clock.sleep(300)
    original = sweep.state.data["cleanup_loads"]["old"]
    assert original["app_id"] == "app-0001"
    assert original["observation_unknown"]
    assert original["terminal_receipt"]["id"] == "app-0001"
    assert _load_keys(fleet, 10) and "old" not in _load_keys(fleet, 10)
    assert len(sweep.state.data["own_loads"]) > 1
    assert not any(r["alias"] == "owner-glm" for r in fleet.runs)
    assert sweep.state.recipes["vonk-forge/a"]["status"] != "failed"
    assert any(item["kind"] == "placement" for item in sweep.state.data["own_loads"])


def test_foreign_acceptance_reconciles_only_the_original_owned_request(
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
    fleet.stale_loads = 1
    sweep.preflight()
    sweep.refresh_catalog()
    with pytest.raises(RuntimeError, match="acceptance does not match"):
        sweep.start_takeover()
    original = sweep.state.data["own_loads"][-1]
    request = original["request_key"]
    assert original.get("application_id") is None
    assert _load_keys(fleet, 10) == [request]
    sweep.clock.sleep(1800)
    for malformed in (
        {"code": "controller.not_found"},
        {"error_type": "arguments", "code": "controller.not_found"},
    ):
        fleet.observations.append((("profile", "progress"), 2, malformed))
        sweep.observe_cleanup()
        assert _load_keys(fleet, 10) == [request]
        assert original.get("application_id") is None
    # A canonical not-found receipt and unchanged allowed review authorize only
    # reentry with the same request, never a timer-generated replacement key.
    sweep.observe_cleanup()
    sweep.observe_cleanup()
    assert _load_keys(fleet, 10) == [request, request]
    assert len(sweep.state.data["own_loads"]) == 1
    assert sweep.state.data["cleanup_loads"][request]["projection_known"]
    assert not sweep._cleanup_nodes()
    assert not [c for _, c in fleet.calls if c[:2] == ("profile", "cancel")]


def test_empty_aggregate_success_reobserves_fleet_and_admits_new_clear(
    tmp_path: Path, gateway: Gateway
) -> None:
    sweep, fleet, _ = make_sweep(
        tmp_path, [FakeRecipe("a")], [FakeModel("m1")], gateway=gateway
    )
    fleet.runs.append(dict(OWNER))
    fleet.ignore_clearing = (
        1  # the first clearing load "succeeds" but the owner's workload keeps running
    )
    sweep.preflight()
    sweep.refresh_catalog()
    sweep.start_takeover()
    original = dict(sweep.state.data["own_loads"][-1])
    for _ in range(6):
        sweep.tick()
        sweep.clock.sleep(300)
    assert sweep.state.data["own_loads"][0] == original
    assert len(sweep.state.data["own_loads"]) > 1
    assert len(set(_load_keys(fleet, 10))) == len(_load_keys(fleet, 10))
    assert not sweep._cleanup_nodes()
    assert sweep.state.recipes["vonk-forge/a"]["status"] != "failed"
    assert any(item["kind"] == "placement" for item in sweep.state.data["own_loads"])
    assert not any(r["alias"] == "owner-glm" for r in fleet.runs)


def test_unknown_stop_projection_keeps_claims_without_testing_on_busy_fleet(
    tmp_path: Path, gateway: Gateway
) -> None:
    sweep, fleet, _ = make_sweep(
        tmp_path, [FakeRecipe("a")], [FakeModel("m1")], gateway=gateway
    )
    fleet.runs.append(dict(OWNER))
    fleet.cleanup_projection = False
    fleet.cleanup_stop_seconds["owner-glm"] = 10**6
    sweep.refresh_catalog()
    assert sweep._take_over(sweep.clock.now()) is False
    original = dict(sweep.state.data["own_loads"][-1])
    sweep.clock.sleep(1800)
    sweep.observe_cleanup()
    assert sweep.state.data["own_loads"] == [original]
    assert sweep._cleanup_nodes() == {"spk_a", "spk_b"}
    assert not [c for p, c in fleet.calls if p == 10 and c[:2] == ("profile", "add")]


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
