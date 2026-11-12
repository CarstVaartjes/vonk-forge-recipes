"""A Controller on another control contract: the main loop checks the client once; callers wait.

The shared vector ``sweep-verdicts/cli-contract-mismatch.json`` is the document vonkctl prints
(vonk-forge f1b95dc3, tests/cluster_profiles/test_control_contract_mismatch.py,
``_library_failure``: a library answer whose ``next_cursor`` is an object, while the contracts
differ). It is validated here against the error schema vonk-forge publishes
(``vonkctl-error-v2.schema.json``, schemas/vonkctl/error-v2.schema.json at the same commit). The
schema excerpt in ``detail`` is abbreviated to ``[...]``. vonk-forge should publish such error
documents as generated vectors; until then this one is hand-recorded.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path

import jsonschema
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
from spark_sweep.run import INFRA_BASE
from spark_sweep.verdict import Disposition, reply_disposition
from spark_sweep.vonkctl import CLIENT_CONTRACT_MISMATCH

LIBRARY = ("recipe", "library", "--limit", "1")
FAULT = (LIBRARY, CLIENT_CONTRACT_MISMATCH, "")
SCHEMA = Path(__file__).parent / "fixtures/vonkctl-error-v2.schema.json"


@pytest.fixture
def gateway():
    gateway = Gateway()
    yield gateway
    gateway.stop()


def _count(fleet, *prefix: str) -> int:
    return sum(1 for _, c in fleet.calls if c[: len(prefix)] == prefix)


def _applies(fleet) -> int:
    return sum(1 for _, c in fleet.calls if c[:1] == ("update",) and "--apply" in c)


def _sweep(tmp_path: Path, gateway: Gateway, **overrides):
    sweep, fleet, clock = make_sweep(
        tmp_path, [FakeRecipe("a")], [FakeModel("m1")], gateway=gateway, **overrides
    )
    sweep.check_client()  # startup: aligned
    return sweep, fleet, clock


def _pass(sweep, clock) -> None:
    """One main-loop pass of the release check, a poll later than the previous one."""
    clock.t += INFRA_BASE + 1
    sweep.check_release(clock.now())


def test_the_shared_vector_is_the_published_error_document() -> None:
    document = json.loads(CONTRACT_MISMATCH_FIXTURE.read_text())
    jsonschema.Draft202012Validator(json.loads(SCHEMA.read_text())).validate(document)
    assert document["code"] == CLIENT_CONTRACT_MISMATCH
    assert document["decision"] == "exit" and document["retryable"] is False
    assert document["recovery_actions"] == ["vonkctl update --apply"]
    assert (
        "OpenAPI" in document["detail"]
    )  # the untyped diagnosis would have fired on it


def test_the_typed_mismatch_is_a_wait_never_a_recipe_verdict(
    tmp_path: Path, gateway: Gateway
) -> None:
    sweep, fleet, _ = _sweep(tmp_path, gateway)
    fleet.faults = [FAULT]
    reply = sweep.vk.run(*LIBRARY)
    assert reply.is_contract_mismatch
    assert reply_disposition(reply) is Disposition.WAIT


def test_a_call_on_a_worker_thread_only_flags_the_mismatch(
    tmp_path: Path, gateway: Gateway
) -> None:
    sweep, fleet, _ = _sweep(tmp_path, gateway)
    fleet.accepted_version = "1.1"
    fleet.faults = [FAULT]
    infra_before = json.dumps(sweep.state.data["infra"])
    calls_before = len(fleet.calls)
    worker = threading.Thread(target=lambda: sweep.vk.run(*LIBRARY))
    worker.start()
    worker.join()
    assert json.dumps(sweep.state.data["infra"]) == infra_before
    assert [c for _, c in fleet.calls[calls_before:]] == [
        LIBRARY
    ]  # no check, no update
    assert _applies(fleet) == 0


def test_the_main_loop_updates_once_and_the_next_pass_succeeds(
    tmp_path: Path, gateway: Gateway
) -> None:
    sweep, fleet, clock = _sweep(tmp_path, gateway)
    fleet.accepted_version = "1.1"
    for _ in range(5):  # several calls meet it before the main loop looks
        fleet.faults = [FAULT]
        assert sweep.vk.run(*LIBRARY).is_contract_mismatch
    _pass(sweep, clock)
    assert _applies(fleet) == 1 and fleet.client_build["version"] == "1.1"
    assert sweep.state.data["client"]["version"] == "1.1"
    assert sweep.vk.run(*LIBRARY).ok  # the failed operation retries on its next pass


def test_two_deploys_in_one_process_both_heal(tmp_path: Path, gateway: Gateway) -> None:
    sweep, fleet, clock = _sweep(tmp_path, gateway)
    for version in ("1.1", "1.2"):
        fleet.accepted_version = version
        fleet.faults = [FAULT]
        assert sweep.vk.run(*LIBRARY).is_contract_mismatch
        _pass(sweep, clock)
        assert sweep.vk.run(*LIBRARY).ok and fleet.client_build["version"] == version
        _pass(sweep, clock)
    assert _applies(fleet) == 2
    assert CLIENT_CONTRACT not in sweep.state.data["infra"]


def test_no_update_available_alerts_with_bounded_reads_and_only_the_failed_command_heals(
    tmp_path: Path, gateway: Gateway
) -> None:
    sweep, fleet, clock = _sweep(tmp_path, gateway)  # the client is newer, not behind
    reads = _count(fleet, "update")
    for _ in range(10):
        fleet.faults = [FAULT]
        assert sweep.vk.run(*LIBRARY).is_contract_mismatch
    clock.t += INFRA_BASE + 1
    sweep.check_release(clock.now())
    sweep.check_release(clock.now())  # nothing new flagged: no second check
    assert _count(fleet, "update") == reads + 1 and _applies(fleet) == 0
    assert sweep.state.data["infra"][CLIENT_CONTRACT]["escalate_at"] is not None
    # Still stuck: a flag inside the gap waits for it, so reads stay bounded per pass.
    fleet.faults = [FAULT]
    sweep.vk.run(*LIBRARY)
    sweep.check_release(clock.now() + 1)
    assert _count(fleet, "update") == reads + 1
    # Another command succeeding is no evidence: the mismatch is per response shape.
    assert sweep.vk.run("--version").ok and sweep.vk.run("update").ok
    assert sweep.vk.run("platform").ok
    _pass(sweep, clock)
    assert CLIENT_CONTRACT in sweep.state.data["infra"]
    # Nor does a Controller whose contract fingerprint still differs from the client's.
    fleet.api_contract_sha256 = "e" * 64
    clock.t += sweep.cfg.release_seconds
    sweep.check_release(clock.now())
    assert CLIENT_CONTRACT in sweep.state.data["infra"]


def test_a_matching_contract_fingerprint_clears_the_alert_without_the_rare_command(
    tmp_path: Path, gateway: Gateway
) -> None:
    sweep, fleet, clock = _sweep(tmp_path, gateway)
    fleet.faults = [FAULT]
    assert sweep.vk.run(*LIBRARY).is_contract_mismatch
    _pass(sweep, clock)
    assert CLIENT_CONTRACT in sweep.state.data["infra"]
    # The rare command never runs again; the periodic client check sees equal fingerprints.
    clock.t += sweep.cfg.release_seconds
    sweep.check_release(clock.now())
    assert CLIENT_CONTRACT not in sweep.state.data["infra"]
    # The command that met the mismatch succeeding clears it as before.
    fleet.faults = [FAULT]
    sweep.vk.run(*LIBRARY)
    _pass(sweep, clock)
    assert CLIENT_CONTRACT in sweep.state.data["infra"]
    assert sweep.vk.run(*LIBRARY).ok
    sweep.check_release(clock.now())
    assert CLIENT_CONTRACT not in sweep.state.data["infra"]


def test_allowed_version_skew_never_applies_and_alerts(
    tmp_path: Path, gateway: Gateway
) -> None:
    sweep, fleet, clock = _sweep(tmp_path, gateway, allow_version_skew=True)
    fleet.accepted_version = "1.1"
    fleet.faults = [FAULT]
    assert sweep.vk.run(*LIBRARY).is_contract_mismatch
    _pass(sweep, clock)
    assert _applies(fleet) == 0
    assert CLIENT_CONTRACT in sweep.state.data["infra"]


def test_the_typed_mismatch_does_not_run_the_untyped_skew_diagnosis(
    tmp_path: Path, gateway: Gateway
) -> None:
    sweep, fleet, _ = _sweep(tmp_path, gateway)
    fleet.faults = [FAULT]
    reply = sweep.vk.run(*LIBRARY)
    before = _count(fleet, "update")
    sweep._blind(reply.error_text, reply)
    assert _count(fleet, "update") == before
    assert "vonkctl may be older" not in sweep.state.data["infra"]["observe"]["message"]
    # The older client's untyped protocol_invalid still diagnoses.
    sweep._skew_checked_at = float("-inf")
    sweep._blind("controller.protocol_invalid: does not match the OpenAPI schema")
    assert _count(fleet, "update") == before + 1


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
    assert _applies(fleet) == 1 and fleet.client_build["version"] == "1.1"
