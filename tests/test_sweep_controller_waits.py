"""Replay the 2026-10-23 waits at the Controller document/scheduler boundary."""

from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import replace
from datetime import datetime
from pathlib import Path

import pytest
from sweep_fakes import FakeModel, FakeRecipe, make_sweep
from vonk_forge_contracts.sweep import SweepWaitReason

from spark_sweep import policy, verdict
from spark_sweep.identity import AcceptedRequest, ControllerWaitHold, content_identity
from spark_sweep.lifecycle import LifecycleState
from spark_sweep.report import build_status, write_status
from spark_sweep.state import State

FIXTURE = (
    Path(__file__).parent / "fixtures/sweep-verdicts/controller-waits-2026-10-23.json"
)
RECORDS = json.loads(FIXTURE.read_text())
COPY = next(row for row in RECORDS if row["code"] == "copy.stalled")
RECEIPT = RECORDS[0]
OBSERVATION = next(
    row
    for row in RECORDS
    if row["controller_document"]["wait"]["reason"] == "observation-unavailable"
)


@pytest.mark.parametrize("row", RECORDS, ids=lambda row: row["recorded_at"])
def test_exact_records_preserve_the_declared_wait_and_deadline(row) -> None:
    document = row["controller_document"]
    signal = verdict.application_signal(document)
    assert signal.wait is not None
    assert signal.wait.reason is SweepWaitReason.CONTROLLER_WAIT
    assert signal.wait.controller is not None
    assert signal.wait.controller.reason == document["wait"]["reason"]
    assert signal.wait.controller.identity == (
        document["wait"]["reason"],
        document["wait"]["blockers"][0]["operation"]["kind"],
    )
    assert document["wait"]["blockers"][0]["detail"] in signal.waiting_on
    now = datetime.fromisoformat(row["recorded_at"]).timestamp()
    watch = {}
    assert (
        verdict.progress_verdict(signal, watch, window=3600, now=now)
        is verdict.Progress.WAITING
    )
    declared = document["wait"].get("dependency_deadline")
    deadline = (
        min(now + 3600, datetime.fromisoformat(declared).timestamp())
        if declared
        else now + 3600
    )
    assert watch["wait_deadline"] == deadline
    # Heartbeats/reasons/restarts cannot silently extend the declared deadline.
    watch = json.loads(json.dumps(watch))
    assert (
        verdict.progress_verdict(signal, watch, window=7200, now=deadline + 1)
        is verdict.Progress.WAIT_EXPIRED
    )
    assert watch["wait_deadline"] == deadline


def test_child_target_copy_is_not_copy_progress() -> None:
    document = deepcopy(COPY["controller_document"])
    assert not policy.distribution(document).copying
    child = document["progress"]["child_progress"]
    child["operation"]["completed_bytes"] = 10
    assert policy.distribution(document, previous_bytes=9).copying
    assert not policy.distribution(document, previous_bytes=10).copying
    child["operation"]["phase"] = "target-copy"
    assert policy.distribution(document, previous_bytes=10).copying


def prepare(tmp_path, monkeypatch, *, second=False, record=COPY):
    # Every record is a bound (admitted) application whose child waits.
    slow = {"digests": ("1" * 64,), "local": "cached", "load_seconds": 10**6}
    recipes = [FakeRecipe("a", content="a" * 64, **slow)]
    if second:
        recipes.append(FakeRecipe("b", content="b" * 64, **slow))
    sweep, fleet, clock = make_sweep(
        tmp_path, recipes, [FakeModel("1" * 64, local="cached")]
    )
    clock.t = datetime.fromisoformat(record["recorded_at"]).timestamp()
    sweep.preflight()
    sweep.refresh_catalog()
    sweep.queue = list(sweep.recipes)
    sweep.cached_models = sweep._cached_models()
    sweep.schedule(clock.now())
    original = fleet._app_doc

    def held_document(app):
        document = original(app)
        document.update(deepcopy(record["controller_document"]))
        document["id"] = app["id"]
        return document

    monkeypatch.setattr(fleet, "_app_doc", held_document)
    sweep.poll_application()
    return sweep, fleet, clock, original


def expire(sweep, clock):
    for key, slot in list(sweep.state.slots.items()):
        assert sweep._observe_load(slot, clock.now()) is verdict.Progress.WAITING
        # Advance beyond the smaller local/Controller budget, then consume it.
        deadline = slot["watch"]["wait_deadline"]
        clock.t = max(clock.now(), deadline + 1)
        sweep._advance_loading(key, slot, clock.now())
        assert slot["phase"] == "finished"
    sweep.enqueue_finished_cleanup()


def test_exact_copy_record_waits_in_the_loading_consumer(tmp_path, monkeypatch) -> None:
    sweep, _, clock, _ = prepare(tmp_path, monkeypatch)
    slot = next(iter(sweep.state.slots.values()))
    assert sweep._observe_load(slot, clock.now()) is verdict.Progress.WAITING
    clock.sleep(603)
    sweep._advance_loading("vonk-forge/a", slot, clock.now())
    assert slot["phase"] == "loading"
    assert not slot["copying"]
    assert sweep.state.entry("vonk-forge/a").get("attempts", 0) == 0
    assert "observation-unavailable" in slot["watch"]["waiting_on"]


def test_declared_wait_wins_even_over_a_real_copy_phase(tmp_path, monkeypatch) -> None:
    record = deepcopy(COPY)
    record["controller_document"]["progress"]["child_progress"]["operation"][
        "phase"
    ] = "target-copy"
    sweep, _, clock, _ = prepare(tmp_path, monkeypatch, record=record)
    slot = next(iter(sweep.state.slots.values()))
    assert sweep._observe_load(slot, clock.now()) is verdict.Progress.WAITING
    clock.sleep(603)
    assert sweep._observe_load(slot, clock.now()) is verdict.Progress.WAITING


@pytest.mark.parametrize("record", [COPY, OBSERVATION, RECEIPT])
def test_expired_wait_reobserves_original_receipt_without_identical_loads(
    tmp_path, monkeypatch, record
) -> None:
    sweep, fleet, clock, original = prepare(tmp_path, monkeypatch, record=record)
    expire(sweep, clock)
    before = sum(c[:3] == ("profile", "load", "--yes") for _, c in fleet.calls)
    # Persistence and repeated deadline expiry do not create a retry loop.
    sweep.state.save()
    sweep.state = State.load(tmp_path / "state.json", clock.now)
    for _ in range(4):
        clock.sleep(sweep.cfg.defer_delay + 1)
        sweep.schedule(clock.now())
    assert sum(c[:3] == ("profile", "load", "--yes") for _, c in fleet.calls) == before
    entry = sweep.state.entry("vonk-forge/a")
    assert entry.get("attempts", 0) == 0
    assert (
        entry["wait_deferral"]["wait"]["controller"]["reason"]
        == record["controller_document"]["wait"]["reason"]
    )
    hold = ControllerWaitHold.model_validate(entry["controller_wait"])
    assert hold.next_check > clock.now()
    write_status(sweep)
    assert (
        record["controller_document"]["wait"]["reason"]
        in (tmp_path / "status.md").read_text()
    )
    assert build_status(sweep)["controller_waits"]
    # Fault recovery changes observed behaviour, then permits canonical re-planning.
    monkeypatch.setattr(fleet, "_app_doc", original)
    clock.sleep(sweep.cfg.defer_delay + 1)
    sweep._refresh_controller_waits(clock.now())
    assert not sweep.controller_wait_holds()


def test_shared_wait_is_a_platform_issue_by_content_not_a_retry_count(
    tmp_path, monkeypatch
) -> None:
    sweep, fleet, clock, original = prepare(tmp_path, monkeypatch, second=True)
    # Expire both lanes against the same typed platform dependency.
    for slot in sweep.state.slots.values():
        sweep._observe_load(slot, clock.now())
    clock.t = max(s["watch"]["wait_deadline"] for s in sweep.state.slots.values()) + 1
    for key, slot in list(sweep.state.slots.items()):
        sweep._advance_loading(key, slot, clock.now())
    sweep.enqueue_finished_cleanup()
    assert sweep.controller_wait_issue() is not None
    assert "recipe.run-switch.v2" in sweep.controller_wait_issue()
    assert not sweep.ready_candidates()
    assert all(e.get("attempts", 0) == 0 for e in sweep.state.recipes.values())
    # Repeated observations of one content identity alone are not shared evidence.
    holds = sweep.controller_wait_holds()
    same = holds["vonk-forge/b"].model_copy(
        update={"identity": holds["vonk-forge/a"].identity}
    )
    sweep.state.entry("vonk-forge/b")["controller_wait"] = same.model_dump(mode="json")
    assert sweep.controller_wait_issue() is None
    sweep.state.entry("vonk-forge/b")["controller_wait"] = holds[
        "vonk-forge/b"
    ].model_dump(mode="json")
    # Supersession cannot clear a proven shared incident.
    monkeypatch.setattr(
        fleet,
        "_app_doc",
        lambda app: {"id": app["id"], "state": LifecycleState.SUPERSEDED},
    )
    clock.sleep(sweep.cfg.defer_delay + 1)
    sweep._refresh_controller_waits(clock.now())
    assert sweep.controller_wait_issue() is not None
    # One original receipt genuinely resumes: both observations converge away.
    monkeypatch.setattr(fleet, "_app_doc", original)
    clock.sleep(sweep.cfg.defer_delay + 1)
    sweep._refresh_controller_waits(clock.now())
    assert not sweep.controller_wait_holds()


def test_hold_follows_a_successor_and_replans_once_nothing_waits(
    tmp_path, monkeypatch
) -> None:
    sweep, fleet, clock, _ = prepare(tmp_path, monkeypatch)
    expire(sweep, clock)
    held = fleet._app_doc
    (first,) = fleet.apps
    successor = "8cb1d3a7-0000-4000-8000-0000000000b1"
    fleet.apps[successor] = dict(fleet.apps[first], id=successor)
    ended: dict[str, tuple[LifecycleState, str | None]] = {
        first: (LifecycleState.SUPERSEDED, successor)
    }

    def document(app):
        if app["id"] not in ended:
            return held(app)
        state, by = ended[app["id"]]
        return {"id": app["id"], "state": state, "superseded_by": by}

    monkeypatch.setattr(fleet, "_app_doc", document)
    clock.sleep(sweep.cfg.defer_delay + 1)
    sweep._refresh_controller_waits(clock.now())
    # The Controller continued the receipt: the successor carries the same wait.
    assert sweep.controller_wait_holds()["vonk-forge/a"].application_id == successor
    clock.sleep(sweep.cfg.defer_delay + 1)
    sweep._refresh_controller_waits(clock.now())
    assert sweep.controller_wait_holds()["vonk-forge/a"].application_id == successor
    assert not sweep.ready_candidates()
    # Ended without a successor (a newer save, a cancel): nothing waits any more.
    ended[successor] = (LifecycleState.CANCELLED, None)
    clock.sleep(sweep.cfg.defer_delay + 1)
    sweep._refresh_controller_waits(clock.now())
    assert not sweep.controller_wait_holds()
    assert [r.key for r in sweep.ready_candidates()] == ["vonk-forge/a"]
    assert sweep.state.entry("vonk-forge/a").get("attempts", 0) == 0


def test_latest_content_gets_a_new_review(tmp_path, monkeypatch) -> None:
    sweep, _, clock, _ = prepare(tmp_path, monkeypatch)
    expire(sweep, clock)
    recipe = sweep.recipes["vonk-forge/a"]
    sweep.recipes[recipe.key] = replace(recipe, content_sha256="d" * 64)
    sweep._refresh_controller_waits(clock.now())
    assert not sweep.controller_wait_holds()
    assert content_identity(sweep.recipes[recipe.key]) != content_identity(recipe)


def test_same_reason_with_different_operation_kind_is_not_shared(
    tmp_path, monkeypatch
) -> None:
    sweep, _, clock, _ = prepare(tmp_path, monkeypatch, second=True)
    for slot in sweep.state.slots.values():
        sweep._observe_load(slot, clock.now())
    clock.t = max(s["watch"]["wait_deadline"] for s in sweep.state.slots.values()) + 1
    for key, slot in list(sweep.state.slots.items()):
        sweep._advance_loading(key, slot, clock.now())
    hold = sweep.controller_wait_holds()["vonk-forge/b"]
    assert hold.wait.controller is not None
    operation = hold.wait.controller.blockers[0].operation
    assert operation is not None
    operation.kind = "another-operation"
    sweep.state.entry("vonk-forge/b")["controller_wait"] = hold.model_dump(mode="json")
    assert sweep.controller_wait_issue() is None


def test_unavailable_receipt_keeps_a_visible_next_observation(
    tmp_path, monkeypatch
) -> None:
    from spark_sweep.vonkctl import Reply

    sweep, _, clock, _ = prepare(tmp_path, monkeypatch)
    expire(sweep, clock)
    monkeypatch.setattr(sweep.vk, "run", lambda *a, **kw: Reply((), 1, None, "offline"))
    clock.sleep(sweep.cfg.defer_delay + 1)
    sweep._refresh_controller_waits(clock.now())
    holds = sweep.controller_wait_holds()
    assert holds["vonk-forge/a"].next_check > clock.now()
    assert not sweep.ready_candidates()


def test_observed_platform_content_change_replans_without_counting_attempts(
    tmp_path, monkeypatch
) -> None:
    from spark_sweep.recovery import Fingerprint

    sweep, _, clock, _ = prepare(tmp_path, monkeypatch)
    before = Fingerprint("worker", "a" * 40, "c" * 64)
    after = Fingerprint("worker", "b" * 40, "c" * 64)
    monkeypatch.setattr(sweep, "_component_observations", lambda: {"worker": before})
    for load in sweep.state.data["loads"].values():
        load.pop("wait_platform", None)
    sweep.poll_application()
    # The original observation ages out while the dependency is still waiting.
    monkeypatch.setattr(sweep, "_component_observations", lambda: {"worker": None})
    expire(sweep, clock)
    assert sweep.controller_wait_holds()["vonk-forge/a"].platform["worker"] == before
    sweep._refresh_controller_waits(clock.now())
    assert (
        sweep.controller_wait_holds()
    )  # unavailable provenance is not changed content
    monkeypatch.setattr(sweep, "_component_observations", lambda: {"worker": after})
    sweep._refresh_controller_waits(clock.now())
    assert not sweep.controller_wait_holds()
    assert sweep.state.entry("vonk-forge/a").get("attempts", 0) == 0


def test_stalled_alarm_cannot_replace_a_declared_dependency(
    tmp_path, monkeypatch
) -> None:
    sweep, _, clock, _ = prepare(tmp_path, monkeypatch)
    slot = sweep.state.slots["vonk-forge/a"]
    load = sweep.state.data["loads"][slot["request_key"]]
    load["stalled"] = verdict.STALLED_CODE
    sweep._advance_loading("vonk-forge/a", slot, clock.now())
    assert slot["watch"]["wait"]["controller"]["reason"] == "observation-unavailable"
    clock.t = slot["watch"]["wait_deadline"] + 1
    sweep._advance_loading("vonk-forge/a", slot, clock.now())
    assert sweep.controller_wait_holds()
    assert sweep.state.entry("vonk-forge/a").get("attempts", 0) == 0


def test_unscoped_wait_does_not_hold_alternative_placement(
    tmp_path, monkeypatch
) -> None:
    record = deepcopy(COPY)
    del record["controller_document"]["wait"]["blockers"][0]["operation"]
    sweep, _, clock, _ = prepare(tmp_path, monkeypatch, record=record)
    expire(sweep, clock)
    assert not sweep.controller_wait_holds()
    assert sweep.state.entry("vonk-forge/a").get("attempts", 0) == 0


def test_operation_identity_does_not_depend_on_blocker_code_spelling() -> None:
    document = deepcopy(COPY["controller_document"])
    document["wait"]["reason"] = "dependency-pending"
    signal = verdict.application_signal(document)
    assert signal.wait is not None and signal.wait.controller is not None
    assert signal.wait.controller.identity == (
        "dependency-pending",
        "recipe.run-switch.v2",
    )


def test_publication_alone_does_not_release_unknown_image_wait(
    tmp_path, monkeypatch
) -> None:
    sweep, _, clock, _ = prepare(tmp_path, monkeypatch)
    recipe = sweep.recipes["vonk-forge/a"]
    unknown = replace(recipe, runtime_image=None, library_commit="publication-a")
    sweep.recipes[recipe.key] = unknown
    slot = sweep.state.slots[recipe.key]
    accepted = AcceptedRequest.read(slot)
    assert accepted is not None
    slot["accepted_content"] = accepted.model_copy(
        update={"recipe": unknown}
    ).model_dump(mode="json")
    expire(sweep, clock)
    sweep.recipes[recipe.key] = replace(unknown, library_commit="publication-b")
    sweep._refresh_controller_waits(clock.now())
    assert sweep.controller_wait_holds()
    assert not sweep.ready_candidates()
