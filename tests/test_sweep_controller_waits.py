"""Replay the 2026-10-23 waits at the Controller document/scheduler boundary."""

from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from uuid import NAMESPACE_URL, uuid5

import pytest
from sweep_fakes import FakeModel, FakeRecipe, make_sweep
from vonk_forge_contracts.sweep import ControllerWaitOperation, SweepWaitReason

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


def controller_document(record, application_id):
    """A recorded wait as the Controller sends it for one application.

    The recordings keep only the blocker operation's kind; the Controller's
    ``BlockingOperation`` always names its id too (its own child operation), so
    each application waits behind its own operation unless the record names one.
    """
    document = deepcopy(record["controller_document"])
    document["id"] = application_id
    for blocker in document["wait"]["blockers"]:
        if "operation" in blocker:
            blocker["operation"].setdefault(
                "id", str(uuid5(NAMESPACE_URL, f"{application_id}/child"))
            )
    return document


#: One Controller operation that two applications both wait behind.
SHARED_OPERATION = "5d7f0b8e-4c1a-4e7b-9a52-0d6c3b1e2f40"
SHARED = deepcopy(COPY)
SHARED["controller_document"]["wait"]["blockers"][0]["operation"]["id"] = (
    SHARED_OPERATION
)


@pytest.mark.parametrize("row", RECORDS, ids=lambda row: row["recorded_at"])
def test_exact_records_preserve_the_declared_wait_and_deadline(row) -> None:
    document = controller_document(row, row["controller_document"]["id"])
    signal = verdict.application_signal(document)
    assert signal.wait is not None
    assert signal.wait.reason is SweepWaitReason.CONTROLLER_WAIT
    assert signal.wait.controller is not None
    assert signal.wait.controller.reason == document["wait"]["reason"]
    operation = document["wait"]["blockers"][0]["operation"]
    assert signal.wait.controller.identity == (
        document["wait"]["reason"],
        frozenset({(operation["kind"], operation["id"])}),
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
    recipes = [
        FakeRecipe(
            "a", content="a" * 64, digests=("1" * 64,), local="cached", blocked=True
        )
    ]
    if second:
        recipes.append(
            FakeRecipe(
                "b", content="b" * 64, digests=("1" * 64,), local="cached", blocked=True
            )
        )
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
        document.update(controller_document(record, app["id"]))
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
    for app in fleet.apps.values():
        app["blocked"] = False
    clock.sleep(sweep.cfg.defer_delay + 1)
    sweep._refresh_controller_waits(clock.now())
    assert not sweep.controller_wait_holds()


def test_shared_wait_is_a_platform_issue_by_content_not_a_retry_count(
    tmp_path, monkeypatch
) -> None:
    sweep, fleet, clock, original = prepare(
        tmp_path, monkeypatch, second=True, record=SHARED
    )
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
    for app in fleet.apps.values():
        app["blocked"] = False
    clock.sleep(sweep.cfg.defer_delay + 1)
    sweep._refresh_controller_waits(clock.now())
    assert not sweep.controller_wait_holds()


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
    document = controller_document(SHARED, "app")
    document["wait"]["reason"] = "dependency-pending"
    signal = verdict.application_signal(document)
    assert signal.wait is not None and signal.wait.controller is not None
    assert signal.wait.controller.identity == (
        "dependency-pending",
        frozenset({("recipe.run-switch.v2", SHARED_OPERATION)}),
    )


def test_every_operation_field_separates_wait_identity() -> None:
    # Guard: a correlating identity that drops an operation field (as it once
    # dropped the id) makes unrelated waits look like one shared dependency.
    signal = verdict.application_signal(controller_document(SHARED, "app"))
    assert signal.wait is not None and signal.wait.controller is not None
    wait = signal.wait.controller
    for name in ControllerWaitOperation.model_fields:
        other = wait.model_copy(deep=True)
        operation = other.blockers[0].operation
        assert operation is not None
        setattr(operation, name, "another")
        assert other.identity != wait.identity, name


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


def waits_behind_applications(tmp_path, monkeypatch, record):
    """Through tick(): a and b expire behind their applications' waits; c is free."""
    slow = {"digests": ("1" * 64,), "local": "cached", "load_seconds": 10**6}
    recipes = [FakeRecipe(slug, content=slug * 64, **slow) for slug in "ab"]
    recipes.append(
        FakeRecipe("c", content="c" * 64, digests=("1" * 64,), local="cached")
    )
    sweep, fleet, clock = make_sweep(
        tmp_path, recipes, [FakeModel("1" * 64, local="cached")]
    )
    clock.t = datetime.fromisoformat(record["recorded_at"]).timestamp()
    original = fleet._app_doc
    resumed: set[str] = set()

    def observed(app):
        document = original(app)
        held = {row["alias"] for row in app.get("assign", [])} & {"a", "b"}
        if held and app["id"] not in resumed:
            document.update(controller_document(record, app["id"]))
        return document

    monkeypatch.setattr(fleet, "_app_doc", observed)
    for _ in range(1000):
        sweep.tick()
        if set(sweep.controller_wait_holds()) == {"vonk-forge/a", "vonk-forge/b"}:
            return sweep, clock, resumed
        clock.sleep(sweep.cfg.poll_seconds)
    raise AssertionError("the waits never expired into Controller wait holds")


def places(sweep, clock, key) -> bool:
    for _ in range(100):
        sweep.tick()
        if key in sweep.state.slots:
            return True
        clock.sleep(sweep.cfg.poll_seconds)
    return False


def test_unrelated_waits_with_one_reason_do_not_pause_placement(
    tmp_path, monkeypatch
) -> None:
    # 2026-10-24: two unrelated loads, each behind its own Run/Switch child.
    sweep, clock, _ = waits_behind_applications(tmp_path, monkeypatch, RECEIPT)
    holds = sweep.controller_wait_holds()
    assert holds["vonk-forge/a"].wait.description == (
        holds["vonk-forge/b"].wait.description
    )
    assert sweep.controller_wait_issue() is None
    assert places(sweep, clock, "vonk-forge/c")


def test_one_shared_dependency_is_reported_and_holds_only_its_recipes(
    tmp_path, monkeypatch
) -> None:
    shared = deepcopy(RECEIPT)
    shared["controller_document"]["wait"]["blockers"][0]["operation"]["id"] = (
        SHARED_OPERATION
    )
    sweep, clock, _ = waits_behind_applications(tmp_path, monkeypatch, shared)
    assert sweep.controller_wait_issue() is not None
    assert places(sweep, clock, "vonk-forge/c")
    assert {"vonk-forge/a", "vonk-forge/b"} <= set(sweep.controller_wait_holds())
    issue = sweep.controller_wait_issue()
    assert issue is not None and SHARED_OPERATION in issue
    write_status(sweep)
    assert SHARED_OPERATION in (tmp_path / "status.md").read_text()


def test_one_receipt_resuming_releases_only_its_own_dependency(
    tmp_path, monkeypatch
) -> None:
    sweep, clock, resumed = waits_behind_applications(tmp_path, monkeypatch, RECEIPT)
    holds = sweep.controller_wait_holds()
    resumed.add(holds["vonk-forge/a"].application_id)
    clock.sleep(sweep.cfg.defer_delay + 1)
    sweep.tick()
    holds = sweep.controller_wait_holds()
    assert "vonk-forge/a" not in holds
    assert "vonk-forge/b" in holds


def test_hold_saved_without_operation_ids_heals_on_its_next_observation(
    tmp_path, monkeypatch
) -> None:
    # State written before the sweep read operation ids survives the upgrade.
    sweep, clock, _ = waits_behind_applications(tmp_path, monkeypatch, RECEIPT)
    for entry in sweep.state.recipes.values():
        hold = entry.get("controller_wait")
        for blocker in hold["wait"]["controller"]["blockers"] if hold else []:
            blocker["operation"].pop("id")
    sweep.state.save()
    sweep.state = State.load(tmp_path / "state.json", clock.now)
    assert sweep.controller_wait_issue() is None
    clock.sleep(sweep.cfg.defer_delay + 1)
    sweep.tick()
    assert not sweep.controller_wait_holds()
    assert all(e.get("attempts", 0) == 0 for e in sweep.state.recipes.values())
