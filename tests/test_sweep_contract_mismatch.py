"""A Controller on another control contract: one client check, one retry; otherwise an alert."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from sweep_fakes import (
    CONTRACT_MISMATCH_FIXTURE,
    FakeModel,
    FakeRecipe,
    Gateway,
    make_sweep,
)

pytest_plugins = ["sweep_bounds"]

from spark_sweep.policy import CLIENT_CONTRACT
from spark_sweep.verdict import Disposition, reply_disposition
from spark_sweep.vonkctl import CLIENT_CONTRACT_MISMATCH

LIBRARY = ("recipe", "library", "--limit", "1")
FAULT = (LIBRARY, CLIENT_CONTRACT_MISMATCH, "")


@pytest.fixture
def gateway():
    gateway = Gateway()
    yield gateway
    gateway.stop()


def _applies(fleet) -> list[tuple[str, ...]]:
    return [c for _, c in fleet.calls if c[:1] == ("update",) and "--apply" in c]


def _sweep(tmp_path: Path, gateway: Gateway, **overrides):
    sweep, fleet, clock = make_sweep(
        tmp_path, [FakeRecipe("a")], [FakeModel("m1")], gateway=gateway, **overrides
    )
    sweep.check_client()  # startup: aligned
    return sweep, fleet, clock


def test_the_shared_vector_is_the_documented_mismatch(
    tmp_path: Path, gateway: Gateway
) -> None:
    document = json.loads(CONTRACT_MISMATCH_FIXTURE.read_text())
    assert document["decision"] == "exit" and document["retryable"] is False
    assert document["recovery_actions"] == ["vonkctl update --apply"]
    sweep, fleet, _ = _sweep(tmp_path, gateway)
    fleet.faults = [FAULT]
    sweep.vk.recovery = None  # the raw reply
    reply = sweep.vk.run(*LIBRARY)
    assert reply.is_contract_mismatch
    assert reply_disposition(reply) is Disposition.WAIT


def test_a_mismatch_runs_the_client_check_updates_and_retries_once(
    tmp_path: Path, gateway: Gateway
) -> None:
    sweep, fleet, _ = _sweep(tmp_path, gateway)
    fleet.accepted_version = "1.1"  # the Controller deploy
    fleet.faults = [FAULT]
    assert sweep.vk.run(*LIBRARY).ok
    assert len(_applies(fleet)) == 1 and fleet.client_build["version"] == "1.1"
    assert [c for _, c in fleet.calls].count(LIBRARY) == 2  # failed, then retried
    assert sweep.state.data["client"]["version"] == "1.1"
    assert CLIENT_CONTRACT not in sweep.state.data["infra"]


def test_two_deploys_in_one_process_both_heal(tmp_path: Path, gateway: Gateway) -> None:
    sweep, fleet, _ = _sweep(tmp_path, gateway)
    for version in ("1.1", "1.2"):
        fleet.accepted_version = version
        fleet.faults = [FAULT]
        assert sweep.vk.run(*LIBRARY).ok
        assert fleet.client_build["version"] == version
    assert len(_applies(fleet)) == 2
    assert CLIENT_CONTRACT not in sweep.state.data["infra"]


def test_no_update_available_alerts_and_does_not_apply(
    tmp_path: Path, gateway: Gateway
) -> None:
    sweep, fleet, _ = _sweep(tmp_path, gateway)  # the client is newer, not behind
    fleet.faults = [FAULT]
    assert sweep.vk.run(*LIBRARY).is_contract_mismatch
    assert _applies(fleet) == []
    incident = sweep.state.data["infra"][CLIENT_CONTRACT]
    assert incident["escalate_at"] is not None
    # Any successful command ends the alert, not only the failed one.
    assert sweep.vk.run("--version").ok
    assert CLIENT_CONTRACT not in sweep.state.data["infra"]


def test_allowed_version_skew_never_applies_and_alerts(
    tmp_path: Path, gateway: Gateway
) -> None:
    sweep, fleet, _ = _sweep(tmp_path, gateway, allow_version_skew=True)
    fleet.accepted_version = "1.1"
    fleet.faults = [FAULT]
    assert sweep.vk.run(*LIBRARY).is_contract_mismatch
    assert _applies(fleet) == []
    assert CLIENT_CONTRACT in sweep.state.data["infra"]


def test_a_mismatch_after_the_update_alerts_without_a_second_retry(
    tmp_path: Path, gateway: Gateway
) -> None:
    sweep, fleet, _ = _sweep(tmp_path, gateway)
    fleet.accepted_version = "1.1"
    fleet.faults = [FAULT, FAULT]
    assert sweep.vk.run(*LIBRARY).is_contract_mismatch
    assert len(_applies(fleet)) == 1
    assert [c for _, c in fleet.calls].count(LIBRARY) == 2
    assert CLIENT_CONTRACT in sweep.state.data["infra"]


def test_the_typed_mismatch_does_not_run_the_untyped_skew_diagnosis(
    tmp_path: Path, gateway: Gateway
) -> None:
    sweep, fleet, _ = _sweep(tmp_path, gateway)
    fleet.faults = [FAULT]
    reply = sweep.vk.run(*LIBRARY)
    before = [c for _, c in fleet.calls].count(("update",))
    sweep._blind(reply.error_text, reply)
    assert [c for _, c in fleet.calls].count(("update",)) == before
    assert "vonkctl may be older" not in sweep.state.data["infra"]["observe"]["message"]
    # The older client's untyped protocol_invalid still diagnoses.
    sweep._skew_checked_at = float("-inf")
    sweep._blind("controller.protocol_invalid: does not match the OpenAPI schema")
    assert [c for _, c in fleet.calls].count(("update",)) == before + 1


@pytest.mark.usefixtures("bounded_clock")
def test_the_sweep_run_updates_and_passes_the_recipe(
    tmp_path: Path, gateway: Gateway
) -> None:
    sweep, fleet, clock = make_sweep(
        tmp_path, [FakeRecipe("a")], [FakeModel("m1")], gateway=gateway
    )
    fleet.faults = [(("profile", "load", "--review"), CLIENT_CONTRACT_MISMATCH, "")]
    clock.hooks.append(lambda _t: setattr(fleet, "accepted_version", "1.1"))
    assert sweep.run() == 0
    entry = sweep.state.recipes["vonk-forge/a"]
    assert entry["status"] == "passed" and entry["attempts"] == 1
    assert CLIENT_CONTRACT not in sweep.state.data["infra"]
    assert len(_applies(fleet)) == 1 and fleet.client_build["version"] == "1.1"
