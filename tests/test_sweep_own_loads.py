"""The sweep's own loads (restore, clearing, stop) are never mistaken for an owner's load."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

import pytest
from sweep_fakes import FakeModel, FakeRecipe, Gateway, make_sweep

from spark_sweep.owner import OwnerGuard
from spark_sweep.state import State
from spark_sweep.vonkctl import Vonkctl


@pytest.fixture
def gateway():
    gateway = Gateway()
    yield gateway
    gateway.stop()


def _interrupt_at(clock, seconds: float) -> None:
    start, fired = clock.now(), []

    def hook(now: float) -> None:
        if now - start >= seconds and not fired:
            fired.append(True)
            raise KeyboardInterrupt

    clock.hooks.append(hook)


def test_a_restart_after_an_interrupted_restore_does_not_hold_for_the_owner(
    tmp_path: Path, gateway: Gateway
) -> None:
    recipes = [FakeRecipe("a", load_seconds=900)]
    first, fleet, clock = make_sweep(
        tmp_path,
        recipes,
        [FakeModel("m1")],
        gateway=gateway,
        restore_owner=2,
        owner_hold_seconds=1800,
    )
    _interrupt_at(clock, 200)
    assert first.run() == 130  # the interrupt restored the owner's profile 2
    saved = json.loads((tmp_path / "state.json").read_text())
    restores = [x for x in saved["own_loads"] if x["kind"] == "restore-owner"]
    assert (
        len(restores) == 1
        and restores[0]["profile"] == 2
        and restores[0]["application_id"]
    )

    clock.hooks.clear()
    fleet.recipes["vonk-forge/a"].load_seconds = 120
    second, _, _ = make_sweep(
        tmp_path, recipes, [FakeModel("m1")], fleet=fleet, owner_hold_seconds=1800
    )
    start = clock.now()
    reasons: list[str] = []
    clock.hooks.append(lambda _t: reasons.append(second.owner_status.reason))
    assert second.run() == 0
    assert second.state.recipes["vonk-forge/a"]["status"] == "passed"
    assert not any(reasons), "the sweep never paused for its own restore"
    assert second.state.data["owner"]["hold_until"] == 0
    assert clock.now() - start < 1800  # no 30 minute hold
    assert not any(
        "owner" in e["message"]
        for e in second.state.data["events"]
        if "preempt" in e["message"] or "requeued" in e["message"]
    )


def test_it_waits_for_its_own_restore_to_settle_without_calling_it_an_owner_load(
    tmp_path: Path, gateway: Gateway
) -> None:
    owner = FakeRecipe("owner-glm", ("mo",), load_seconds=300)
    recipes = [FakeRecipe("a", load_seconds=120), owner]
    models = [FakeModel("m1"), FakeModel("mo", local="cached")]
    owner.local = "cached"
    first, fleet, clock = make_sweep(
        tmp_path,
        recipes,
        models,
        gateway=gateway,
        restore_owner=2,
        owner_hold_seconds=1800,
    )
    fleet.profiles[2] = {
        "revision": 1,
        "assignments": [
            {
                "recipe_selector": owner.key,
                "spark_ids": ["spark-a", "spark-b"],
                "assignment_name": "owner-glm",
                "desired_state": "running",
            }
        ],
        "latest": None,
    }
    _interrupt_at(clock, 200)
    assert first.run() == 130  # restoring profile 2 takes 300 s to come up

    clock.hooks.clear()
    second, _, _ = make_sweep(
        tmp_path, recipes, models, fleet=fleet, owner_hold_seconds=1800
    )
    start = clock.now()
    reasons: list[str] = []
    clock.hooks.append(lambda _t: reasons.append(second.owner_status.reason))
    assert second.run() == 0
    waited = [r for r in reasons if r]
    assert waited and all(
        "our own load" in r for r in waited
    )  # it waited, and said whose load it was
    assert not any("holding after an owner load" in r for r in reasons)
    assert second.state.data["owner"]["hold_until"] == 0
    assert clock.now() - start < 1800


def test_a_load_the_sweep_started_is_recognised_by_request_key_even_without_its_id(
    tmp_path: Path,
) -> None:
    state = State.load(tmp_path / "state.json")
    state.data["own_loads"].append(
        {"request_key": "our-key", "kind": "restore-owner", "profile": 2}
    )  # a crash before the reply
    state.data["owner"]["baseline"]["2"] = {"id": "old", "state": "succeeded"}
    answers = {"id": "new-app", "state": "succeeded", "request_key": "our-key"}

    def run(argv, timeout):
        number = argv[argv.index("--profile") + 1]
        return (
            (0, json.dumps(answers), "")
            if number == "2"
            else (
                2,
                json.dumps(
                    {
                        "error_type": "control_api",
                        "code": "controller.not_found",
                        "detail": "no application",
                    }
                ),
                "",
            )
        )

    guard = OwnerGuard(Vonkctl("vonkctl", runner=run), state, hold_seconds=1800)
    status = guard.check(100.0)
    assert (
        not status.paused
        and not status.new_activity
        and state.data["owner"]["hold_until"] == 0
    )
    # The same shape of application with someone else's key is an owner load.
    answers["id"], answers["request_key"] = "other-app", "someone-elses"
    status = guard.check(200.0)
    assert status.new_activity and state.data["owner"]["hold_until"] == 200.0 + 1800


def test_every_load_the_sweep_submits_is_remembered(
    tmp_path: Path, gateway: Gateway
) -> None:
    sweep, fleet, _ = make_sweep(
        tmp_path, [FakeRecipe("a")], [FakeModel("m1")], gateway=gateway
    )
    fleet.runs.append(
        {
            "alias": "owner-glm",
            "recipe": "x",
            "run_id": "r",
            "node_ids": ["spk_a", "spk_b"],
            "ready": True,
        }
    )
    sweep.run()
    kinds = [x["kind"] for x in sweep.state.data["own_loads"]]
    assert kinds == ["takeover", "placement", "stop"]
    keys = [
        c[c.index("--request-key") + 1]
        for _, c in fleet.calls
        if c[:3] == ("profile", "load", "--yes")
    ]
    assert keys == [x["request_key"] for x in sweep.state.data["own_loads"]]
    assert all(x.get("application_id") for x in sweep.state.data["own_loads"])


@pytest.mark.parametrize(
    ("document", "exit_code", "paused"),
    [
        ({"error_type": "control_api", "code": "controller.not_found"}, 2, False),
        ({"code": "controller.not_found"}, 2, True),
        ({"error_type": "arguments", "code": "controller.not_found"}, 2, True),
        ({"error_type": "control_api", "code": "controller.forbidden"}, 2, True),
        ({"error_type": "control_api", "code": "controller.unauthorized"}, 2, True),
        ({"error_type": "control_api", "code": "controller.not_found"}, 0, True),
        ({"error_type": "control_api", "code": "not_found"}, 2, True),
        (None, 2, True),
    ],
)
def test_owner_absence_requires_native_declared_rejection(
    tmp_path: Path, document: object, exit_code: int, paused: bool
) -> None:
    state = State.load(tmp_path / "state.json")

    def run(argv: Sequence[str], timeout: float) -> tuple[int, str, str]:
        return exit_code, json.dumps(document), ""

    guard = OwnerGuard(
        Vonkctl("vonkctl", runner=run), state, hold_seconds=1800, profiles=[1]
    )
    status = guard.check(100.0)
    assert status.paused is paused
    assert not status.new_activity
    assert not state.data["own_loads"]
    assert state.data["owner"]["hold_until"] == 0
    if paused:
        assert state.data["owner"]["baseline"] == {}
        assert "ownership is unreadable" in status.reason
    else:
        assert state.data["owner"]["baseline"]["1"] == {"id": "", "state": ""}
