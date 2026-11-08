"""A Controller on another control contract: one update, one retry; otherwise an alert and a wait."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from sweep_fakes import FakeClock, FakeFleet, FakeModel, FakeRecipe, Gateway, make_sweep

pytest_plugins = ["sweep_bounds"]

from spark_sweep.contract_recovery import (
    RECHECK_SECONDS,
    ContractRecovery,
    ContractState,
)
from spark_sweep.verdict import Disposition, reply_disposition
from spark_sweep.vonkctl import CLIENT_CONTRACT_MISMATCH, Vonkctl

LIBRARY = ("recipe", "library", "--limit", "1")
DETAIL = "this vonkctl and the Controller use different control contracts"


@pytest.fixture
def gateway():
    gateway = Gateway()
    yield gateway
    gateway.stop()


def _applies(fleet: FakeFleet) -> list[tuple[str, ...]]:
    return [c for _, c in fleet.calls if c[:1] == ("update",) and "--apply" in c]


def _client(fleet: FakeFleet) -> tuple[Vonkctl, list[str], list[str]]:
    alerts: list[str] = []
    events: list[str] = []
    vk = Vonkctl("vonkctl", runner=fleet.runner)
    vk.recovery = ContractRecovery(
        fleet.clock.now,
        alerts.append,
        lambda: events.append("clear"),
        lambda: events.append("updated"),
    )
    return vk, alerts, events


def _fleet() -> FakeFleet:
    return FakeFleet(
        FakeClock(), [FakeRecipe("a")], [FakeModel("m1")], "http://127.0.0.1:1"
    )


def test_the_typed_code_is_a_wait_never_a_recipe_verdict() -> None:
    fleet = _fleet()
    fleet.faults = [(LIBRARY, CLIENT_CONTRACT_MISMATCH, DETAIL)]
    reply = Vonkctl("vonkctl", runner=fleet.runner).run(*LIBRARY)
    assert reply.is_contract_mismatch
    assert reply_disposition(reply) is Disposition.WAIT


def test_a_mismatch_updates_once_and_retries_the_failed_call() -> None:
    fleet = _fleet()
    vk, alerts, events = _client(fleet)
    fleet.accepted_version = "1.1"
    fleet.faults = [(LIBRARY, CLIENT_CONTRACT_MISMATCH, DETAIL)]
    reply = vk.run(*LIBRARY)
    assert reply.ok, reply
    assert len(_applies(fleet)) == 1 and fleet.client_build["version"] == "1.1"
    assert [c for _, c in fleet.calls].count(LIBRARY) == 2  # failed, then retried
    assert alerts == [] and events == ["clear", "updated"]
    assert vk.recovery.state is ContractState.UPDATED  # type: ignore[union-attr]


def test_a_second_mismatch_after_the_update_does_not_apply_again() -> None:
    fleet = _fleet()
    vk, alerts, _ = _client(fleet)
    fleet.accepted_version = "1.1"
    fleet.faults = [(LIBRARY, CLIENT_CONTRACT_MISMATCH, DETAIL)] * 2
    reply = vk.run(*LIBRARY)  # mismatch, update, retry: mismatch again
    assert reply.is_contract_mismatch
    fleet.faults = [(LIBRARY, CLIENT_CONTRACT_MISMATCH, DETAIL)]
    assert vk.run(*LIBRARY).is_contract_mismatch
    assert len(_applies(fleet)) == 1
    assert len(alerts) == 1 and "client.contract_mismatch" in alerts[0]


def test_no_update_available_alerts_waits_and_never_applies() -> None:
    fleet = _fleet()
    vk, alerts, events = _client(fleet)  # the client is not behind: it is newer
    fleet.faults = [(LIBRARY, CLIENT_CONTRACT_MISMATCH, DETAIL)] * 3
    for _ in range(3):
        assert vk.run(*LIBRARY).is_contract_mismatch
    assert _applies(fleet) == []
    reads = [c for _, c in fleet.calls if c[:1] == ("update",)]
    assert len(reads) == 1  # the stuck client does not ask again within the interval
    assert len(alerts) == 1 and "no vonkctl update is available" in alerts[0]
    assert vk.recovery.state is ContractState.STUCK  # type: ignore[union-attr]
    # Whoever fixes the client ends the alert: the next good call clears it.
    assert vk.run("--version").ok  # another command succeeding proves nothing
    assert vk.recovery.state is ContractState.STUCK  # type: ignore[union-attr]
    assert vk.run(*LIBRARY).ok
    assert events == ["clear"] and vk.recovery.state is ContractState.ALIGNED  # type: ignore[union-attr]


def test_a_stuck_client_rechecks_and_applies_an_update_that_appears_once() -> None:
    fleet = _fleet()
    vk, alerts, _ = _client(fleet)
    fleet.faults = [(LIBRARY, CLIENT_CONTRACT_MISMATCH, DETAIL)]
    assert vk.run(*LIBRARY).is_contract_mismatch and _applies(fleet) == []
    fleet.clock.t += RECHECK_SECONDS + 1
    fleet.accepted_version = "1.1"  # the publication caught up
    fleet.faults = [(LIBRARY, CLIENT_CONTRACT_MISMATCH, DETAIL)]
    assert vk.run(*LIBRARY).ok
    assert len(_applies(fleet)) == 1 and len(alerts) == 1


def test_a_failed_apply_alerts_and_the_next_call_does_not_apply_again() -> None:
    fleet = _fleet()
    vk, alerts, _ = _client(fleet)
    fleet.accepted_version = "1.1"
    fleet.faults = [
        (LIBRARY, CLIENT_CONTRACT_MISMATCH, DETAIL),
        (("update", "--apply"), "controller.transport_timeout", "no network"),
    ]
    assert vk.run(*LIBRARY).is_contract_mismatch
    assert len(alerts) == 1 and "did not apply" in alerts[0]
    fleet.clock.t += RECHECK_SECONDS + 1
    fleet.faults = [(LIBRARY, CLIENT_CONTRACT_MISMATCH, DETAIL)]
    assert vk.run(*LIBRARY).is_contract_mismatch
    assert len(_applies(fleet)) == 1  # the one apply of this process is spent


@pytest.mark.usefixtures("bounded_clock")
def test_the_sweep_alerts_and_waits_on_a_mismatch_without_failing_a_recipe(
    tmp_path: Path, gateway: Gateway
) -> None:
    sweep, fleet, clock = make_sweep(
        tmp_path, [FakeRecipe("a")], [FakeModel("m1")], gateway=gateway
    )
    fleet.faults = [
        (("profile", "load", "--review"), CLIENT_CONTRACT_MISMATCH, DETAIL)
    ] * 2
    seen: list[dict] = []
    clock.hooks.append(
        lambda _t: seen.append(json.loads(json.dumps(sweep.state.data["infra"])))
    )
    assert sweep.run() == 0
    entry = sweep.state.recipes["vonk-forge/a"]
    assert entry["status"] == "passed" and entry["attempts"] == 1
    assert _applies(fleet) == []
    assert any("client-contract" in infra for infra in seen), "the alert was raised"
    alert = next(i["client-contract"] for i in seen if "client-contract" in i)
    assert alert["escalate_at"] is not None
    assert "client-contract" not in sweep.state.data["infra"]  # cleared once it healed
