"""Replay preparation refusals that caused identical loads on 2026-10-23.

The fixture projects the exact record timestamps, recipe content and Controller
state/blockers; infrastructure identifiers and unrelated progress are omitted.
"""

from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import replace
from datetime import datetime
from pathlib import Path

import pytest
from sweep_fakes import FakeModel, FakeRecipe, make_sweep
from vonk_forge_contracts import QualificationCauseCode
from vonk_forge_contracts.sweep import (
    SweepPlacementCode,
    SweepSlotPhase,
    SweepWaitReason,
)

from spark_sweep import policy, verdict
from spark_sweep.catalog import Presence
from spark_sweep.lifecycle import LifecycleState, RecipeStatus
from spark_sweep.report import build_status, write_status
from spark_sweep.run import REFUSAL_BACKOFF_SECONDS
from spark_sweep.state import State

REFUSAL = json.loads(
    (
        Path(__file__).parent
        / "fixtures/sweep-verdicts/preparation-refusal-2026-10-23.json"
    ).read_text()
)
SAVED_RECORDS = [{**REFUSAL, "recorded_at": at} for at in REFUSAL["recorded_at"]]
# The saved receipt is superseded. A deterministic refusal requires FAILED,
# or a bounded wait expiring while that refusal remains on an active receipt.
RECORDS = [
    {
        **record,
        "controller_document": {
            **record["controller_document"],
            "state": LifecycleState.FAILED,
        },
    }
    for record in SAVED_RECORDS
]


def prepare(tmp_path, monkeypatch, record, *, active=None, second=False, batched=False):
    recipe = FakeRecipe(
        record["recipe"].split("/", 1)[1],
        content=record["content_sha256"],
        digests=("1" * 64,),
        local="cached",
        blocked=True,
    )
    recipes = [recipe]
    if second:
        recipes.append(
            FakeRecipe(
                "second", ("1" * 64,), content="b" * 64, local="cached", blocked=True
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
    sweep.observe_platform()
    sweep.schedule(clock.now())
    if batched:
        # Fault injection: reload an accepted multi-slot receipt from a saved sweep.
        slots = list(sweep.state.slots.values())
        request = slots[0]["request_key"]
        for slot in slots[1:]:
            sweep.state.data["loads"].pop(slot["request_key"])
            slot["request_key"] = request
            slot["accepted_content"]["request_key"] = request
        sweep.state.save()
        sweep.state = State.load(tmp_path / "state.json", clock.now)
    original = fleet._app_doc
    document = deepcopy(record["controller_document"])
    if active:
        document["state"] = active

    for app in fleet.apps.values():
        app["state"] = document["state"]

    def refused(app):
        return {**original(app), **deepcopy(document)}

    monkeypatch.setattr(fleet, "_app_doc", refused)
    sweep._tick()
    key = recipe.key
    return sweep, fleet, clock, key, document


def loads(fleet):
    return sum(c[:3] == ("profile", "load", "--yes") for _, c in fleet.calls)


@pytest.mark.parametrize("record", RECORDS, ids=lambda r: r["recorded_at"])
def test_exact_refusal_backs_off_until_deadline_or_controller_change(
    tmp_path, monkeypatch, record
):
    sweep, fleet, clock, key, _ = prepare(tmp_path, monkeypatch, record)
    sweep.enqueue_finished_cleanup()
    before = loads(fleet)
    sweep.state.save()
    sweep.state = State.load(tmp_path / "state.json", clock.now)
    for _ in range(4):
        clock.sleep(sweep.cfg.defer_delay + 1)
        sweep.observe_platform()  # same software, fresh observations do not mean recovery
        sweep.schedule(clock.now())
    assert loads(fleet) == before
    entry = sweep.state.entry(key)
    assert entry["status"] == RecipeStatus.PENDING
    assert entry.get("attempts", 0) == 0
    hold = sweep.controller_wait_holds()[key]
    assert hold.blockers == verdict.refusal_identity(record["controller_document"])
    assert hold.next_check > clock.now()
    assert build_status(sweep)["controller_waits"][key]["blockers"]
    write_status(sweep)
    text = (tmp_path / "status.md").read_text()
    assert "platform refusal" in text and hold.blockers[0][0] in text
    assert "next observation" in text and "next allowed retry at" in text
    # A software fix changes evidence, without operator retry flags or a count cap.
    fleet.api_source_sha = "b" * 40
    sweep.observe_platform()
    sweep.schedule(clock.now())
    assert loads(fleet) == before + 1
    assert entry.get("attempts", 0) == 0


@pytest.mark.parametrize("change", ["catalog", "detail", "code", "resumed"])
def test_changed_evidence_replans(tmp_path, monkeypatch, change):
    sweep, _, clock, key, document = prepare(tmp_path, monkeypatch, RECORDS[0])
    sweep.enqueue_finished_cleanup()
    if change == "catalog":
        sweep.recipes[key] = replace(sweep.recipes[key], content_sha256="e" * 64)
    elif change == "detail":
        document["progress"]["blockers"][0]["detail"] += " changed"
    elif change == "code":
        document["progress"]["blockers"][0]["code"] = "external.changed_blocker"
    else:
        document["state"] = LifecycleState.SUCCEEDED
        document["progress"]["blockers"] = []
    clock.sleep(sweep.cfg.defer_delay + 1)
    sweep.schedule(clock.now())
    assert not sweep.controller_wait_holds()
    assert sweep.state.entry(key).get("attempts", 0) == 0


@pytest.mark.parametrize("severity", ["warning", "info"])
def test_preparation_warning_never_holds_placement(tmp_path, monkeypatch, severity):
    record = deepcopy(RECORDS[0])
    record["controller_document"]["progress"]["blockers"][0]["severity"] = severity
    sweep, _, _, _, _ = prepare(tmp_path, monkeypatch, record)
    assert not sweep.controller_wait_holds()


def test_supersession_and_unavailable_observations_are_not_recovery(
    tmp_path, monkeypatch
):
    from spark_sweep.vonkctl import Reply

    sweep, _, clock, key, document = prepare(tmp_path, monkeypatch, RECORDS[0])
    sweep.enqueue_finished_cleanup()
    # A receipt losing its blockers on supersession is not a fixed dependency.
    document["progress"]["blockers"] = []
    clock.sleep(sweep.cfg.defer_delay + 1)
    sweep._refresh_controller_waits(clock.now())
    assert key in sweep.controller_wait_holds()
    monkeypatch.setattr(sweep, "_component_observations", dict)
    monkeypatch.setattr(sweep.vk, "run", lambda *a, **kw: Reply((), 1, None, "offline"))
    clock.sleep(sweep.cfg.defer_delay + 1)
    sweep._refresh_controller_waits(clock.now())
    assert key in sweep.controller_wait_holds()
    assert sweep.controller_wait_holds()[key].next_check > clock.now()


def test_changing_another_blocker_cannot_hide_the_same_refusal(tmp_path, monkeypatch):
    sweep, _, clock, key, document = prepare(tmp_path, monkeypatch, RECORDS[0])
    sweep.enqueue_finished_cleanup()
    document["progress"]["blockers"].append(
        {"code": "external.other", "detail": "changed", "severity": "error"}
    )
    clock.sleep(sweep.cfg.defer_delay + 1)
    sweep._refresh_controller_waits(clock.now())
    assert key in sweep.controller_wait_holds()


def test_refusal_identity_uses_full_details_and_ignores_informational_churn():
    document = deepcopy(RECORDS[0]["controller_document"])
    before = verdict.refusal_identity(document)
    assert len(before) == 1
    document["progress"]["blockers"][0]["detail"] += (
        " changed beyond the 300-byte excerpt"
    )
    assert verdict.refusal_identity(document) != before
    document["progress"]["blockers"][0]["severity"] = "info"
    assert not verdict.refusal_identity(document)


def test_unrelated_recipe_is_placed_during_refusal_holds(tmp_path, monkeypatch):
    sweep, fleet, clock, key, _ = prepare(
        tmp_path, monkeypatch, RECORDS[0], second=True
    )
    holds = sweep.controller_wait_holds()
    assert len(holds) == 2
    assert len({h.application_id for h in holds.values()}) == 2
    assert sweep.controller_wait_issue() is None
    before = loads(fleet)
    fresh = FakeRecipe("fresh", ("1" * 64,), content="e" * 64, local="cached")
    fleet.recipes[fresh.key] = fresh
    sweep.refresh_catalog()
    sweep.queue = list(sweep.recipes)
    sweep.schedule(clock.now())
    assert loads(fleet) == before + 1
    assert fresh.key in sweep.state.slots
    assert key not in sweep.state.slots
    assert all(e.get("attempts", 0) == 0 for e in sweep.state.recipes.values())


def test_another_name_cannot_bypass_a_content_bound_refusal(tmp_path, monkeypatch):
    sweep, _, _, key, _ = prepare(tmp_path, monkeypatch, RECORDS[0])
    sweep.enqueue_finished_cleanup()
    renamed = replace(sweep.recipes.pop(key), key="vonk-forge/renamed")
    sweep.recipes[renamed.key] = renamed
    sweep.queue = [renamed.key]
    assert not sweep.ready_candidates()


@pytest.mark.parametrize("settled", [LifecycleState.FAILED])
def test_static_refusal_allows_one_fresh_placement_at_persisted_deadline(
    tmp_path, monkeypatch, settled
):
    record = deepcopy(RECORDS[0])
    record["controller_document"]["state"] = settled
    sweep, fleet, clock, key, _ = prepare(tmp_path, monkeypatch, record)
    hold = sweep.controller_wait_holds()[key]
    assert hold.retry_at == clock.now() + REFUSAL_BACKOFF_SECONDS
    before = loads(fleet)
    sweep.state.save()
    sweep.state = State.load(tmp_path / "state.json", clock.now)
    for elapsed in (60, 600, REFUSAL_BACKOFF_SECONDS - 1):
        clock.t = hold.retry_at - REFUSAL_BACKOFF_SECONDS + elapsed
        sweep.observe_platform()
        sweep.schedule(clock.now())
        assert loads(fleet) == before
        assert sweep.controller_wait_holds()[key].retry_at == hold.retry_at
    clock.t = hold.retry_at
    sweep.observe_platform()
    sweep.schedule(clock.now())
    sweep.schedule(clock.now())
    assert loads(fleet) == before + 1
    assert sweep.state.entry(key).get("attempts", 0) == 0


def test_preparation_can_heal_without_redeploy(tmp_path, monkeypatch):
    sweep, fleet, clock, key, document = prepare(tmp_path, monkeypatch, RECORDS[0])
    before = loads(fleet)
    hold = sweep.controller_wait_holds()[key]
    # The old settled receipt keeps its refusal; only a fresh load sees the healed builder.
    original = fleet._app_doc
    application_id = hold.application_id

    def healed(app):
        if app["id"] == application_id:
            return original(app)
        return {**original(app), "state": LifecycleState.SUCCEEDED, "progress": {}}

    monkeypatch.setattr(fleet, "_app_doc", healed)
    clock.t = hold.retry_at
    sweep.observe_platform()
    sweep.schedule(clock.now())
    assert loads(fleet) == before + 1
    assert document["progress"]["blockers"]  # no false recovery from the old receipt
    sweep._tick()
    assert key not in sweep.controller_wait_holds()
    assert sweep.state.entry(key).get("attempts", 0) == 0


def test_unknown_platform_capture_does_not_release_refusal(tmp_path, monkeypatch):
    from spark_sweep.run import Sweep

    observations = Sweep._component_observations
    monkeypatch.setattr(Sweep, "_component_observations", lambda self: {})
    sweep, fleet, clock, key, _ = prepare(tmp_path, monkeypatch, RECORDS[0])
    hold = sweep.controller_wait_holds()[key]
    assert hold.platform == {}
    before = loads(fleet)
    monkeypatch.setattr(Sweep, "_component_observations", observations)
    sweep.observe_platform()
    sweep.schedule(clock.now())
    assert loads(fleet) == before
    assert sweep.controller_wait_holds()[key].retry_at == hold.retry_at


def test_even_own_placements_on_held_node_do_not_release_refusal(tmp_path, monkeypatch):
    sweep, fleet, clock, key, _ = prepare(tmp_path, monkeypatch, RECORDS[0])
    hold = sweep.controller_wait_holds()[key]
    before = loads(fleet)
    node = next(iter(sweep.sparks())).id
    for elapsed in (60, 780, 1560, 2340, REFUSAL_BACKOFF_SECONDS - 1):
        clock.t = hold.retry_at - REFUSAL_BACKOFF_SECONDS + elapsed
        # The sweep's own new receipt/run on precisely the refused node.
        sweep.fleet = replace(
            sweep.fleet,
            presences=(
                Presence(
                    node,
                    sweep._alias(sweep.recipes[key]),
                    f"sweep-run-{elapsed}",
                    True,
                    "ready",
                    "running",
                ),
            ),
        )
        sweep.observe_platform()
        sweep.schedule(clock.now())
        assert loads(fleet) == before
        assert sweep.controller_wait_holds()[key].retry_at == hold.retry_at
    clock.t = hold.retry_at
    sweep.schedule(clock.now())
    assert loads(fleet) == before + 1
    sweep._tick()  # the new receipt refuses identical content again
    renewed = sweep.controller_wait_holds()[key]
    assert renewed.retry_at == hold.retry_at + REFUSAL_BACKOFF_SECONDS
    clock.sleep(780)
    sweep.schedule(clock.now())
    assert loads(fleet) == before + 1


def test_static_refusal_blocks_replay_until_one_hour_then_retries(
    tmp_path, monkeypatch
):
    sweep, fleet, clock, _, _ = prepare(tmp_path, monkeypatch, RECORDS[0])
    assert loads(fleet) == 1
    clock.sleep(REFUSAL_BACKOFF_SECONDS - 1)
    sweep.observe_platform()
    sweep.schedule(clock.now())
    assert loads(fleet) == 1
    clock.sleep(1)
    sweep.schedule(clock.now())
    sweep.schedule(clock.now())
    assert loads(fleet) == 2


@pytest.mark.parametrize(
    "settled", [LifecycleState.SUPERSEDED, LifecycleState.CANCELLED]
)
def test_superseded_saved_receipt_and_cancelled_receipt_requeue_without_hold(
    tmp_path, monkeypatch, settled
):
    record = deepcopy(SAVED_RECORDS[0])
    record["controller_document"]["state"] = settled
    sweep, fleet, clock, key, _ = prepare(tmp_path, monkeypatch, record)
    assert not sweep.controller_wait_holds()
    assert sweep.state.entry(key).get("attempts", 0) == 0
    sweep.enqueue_finished_cleanup()
    clock.sleep(sweep.cfg.retry_delay + 1)
    sweep.schedule(clock.now())
    assert loads(fleet) == 2  # one original receipt and its fresh requeue
    assert key in sweep.state.slots
    assert sweep.state.slots[key]["deadline"] > clock.now()


def test_named_refusal_on_shared_receipt_replaces_only_collateral(
    tmp_path, monkeypatch
):
    record = deepcopy(RECORDS[0])
    record["controller_document"]["progress"]["blockers"][0]["recipe"] = record[
        "recipe"
    ]
    sweep, fleet, _, key, _ = prepare(
        tmp_path, monkeypatch, record, second=True, batched=True
    )
    assert set(sweep.controller_wait_holds()) == {key}
    other = "vonk-forge/second"
    assert any(
        row.get("recipe") == other and "collateral" in row.get("reason", "")
        for row in sweep.log.entries()
    )
    # _tick already re-planned B after the shared refusal, without waiting for A.
    assert loads(fleet) == 3  # two initial lane loads, one collateral replacement
    assert other in sweep.state.slots
    assert key not in sweep.state.slots


@pytest.mark.parametrize("kind", ["placement", "takeover", "replay"])
def test_submit_guard_catches_held_content_without_ready_candidates(
    tmp_path, monkeypatch, kind
):
    sweep, fleet, _, key, _ = prepare(tmp_path, monkeypatch, RECORDS[0])
    before = loads(fleet)
    # Reintroduce the refused assignment after normal cleanup. Bypass the
    # scheduler entirely, as same-key replay and cleanup/takeover re-entry do.
    sweep.vk.call(
        "profile",
        "add",
        key,
        "--spark",
        sweep.sparks()[0].name,
        "--as",
        sweep._alias(sweep.recipes[key]),
        "--state",
        LifecycleState.RUNNING,
        "--yes",
        profile=sweep.cfg.sweep_profile,
    )
    if kind == "replay":
        assert sweep._poll_application({"request_key": "guard-replay", "app_id": None})
    else:
        assert sweep._submit_load(sweep.cfg.sweep_profile, "guard-replay", kind) is None
    assert loads(fleet) == before
    row = sweep.log.entries()[-1]
    assert row["code"] == SweepPlacementCode.REFUSAL_HELD
    assert row["recipe"] == key


def test_apply_guard_rejects_renamed_held_content_without_scheduler(
    tmp_path, monkeypatch
):
    sweep, fleet, clock, key, _ = prepare(tmp_path, monkeypatch, RECORDS[0])
    sweep.enqueue_finished_cleanup()
    renamed = replace(sweep.recipes[key], key="vonk-forge/renamed")
    sweep.recipes[renamed.key] = renamed
    fleet.recipes[renamed.key] = replace(fleet.recipes[key], slug="renamed")
    before = loads(fleet)
    assert (
        sweep._apply([policy.Placement(renamed, (sweep.sparks()[0].id,))], clock.now())
        is None
    )
    assert loads(fleet) == before
    assert sweep.log.entries()[-1]["code"] == SweepPlacementCode.REFUSAL_HELD


@pytest.mark.parametrize(
    "active", [LifecycleState.RUNNING, LifecycleState.NEEDS_OPERATOR]
)
def test_active_refusal_hold_requires_bounded_wait_expiry(
    tmp_path, monkeypatch, active
):
    sweep, _, clock, key, _ = prepare(tmp_path, monkeypatch, RECORDS[0], active=active)
    assert not sweep.controller_wait_holds()
    slot = sweep.state.slots[key]
    clock.t = slot["watch"]["wait_deadline"] + 1
    sweep._tick()
    assert key in sweep.controller_wait_holds()


def test_reobserving_same_receipt_cannot_renew_retry_after_restart(
    tmp_path, monkeypatch
):
    sweep, _, clock, key, document = prepare(
        tmp_path, monkeypatch, RECORDS[0], active=LifecycleState.RUNNING
    )
    load = next(iter(sweep.state.data["loads"].values()))
    document["state"] = LifecycleState.FAILED
    sweep._hold_controller_refusal(load, document)
    deadline = sweep.controller_wait_holds()[key].retry_at
    # Interrupt after persisting the hold, before the loading lane was requeued.
    sweep.state.save()
    sweep.state = State.load(tmp_path / "state.json", clock.now)
    for elapsed in (60, REFUSAL_BACKOFF_SECONDS - 1, REFUSAL_BACKOFF_SECONDS + 1):
        clock.t = deadline - REFUSAL_BACKOFF_SECONDS + elapsed
        sweep._hold_controller_refusal(load, document)
        assert sweep.controller_wait_holds()[key].retry_at == deadline


def test_collateral_wait_expires_first_and_replans_without_held_assignment(
    tmp_path, monkeypatch
):
    record = deepcopy(RECORDS[0])
    record["controller_document"]["progress"]["blockers"][0]["recipe"] = record[
        "recipe"
    ]
    sweep, fleet, clock, key, _ = prepare(
        tmp_path,
        monkeypatch,
        record,
        active=LifecycleState.RUNNING,
        second=True,
        batched=True,
    )
    other = "vonk-forge/second"
    culprit = sweep.state.slots[key]
    collateral = sweep.state.slots[other]
    # Preserve a later budget for A; only B's shared-receipt wait expires now.
    culprit["watch"]["wait_deadline"] = clock.now() + 1000
    collateral["watch"]["wait_deadline"] = clock.now() - 1
    before = loads(fleet)
    for _ in range(5):
        start = len(fleet.calls)
        sweep._tick()
        calls = [args for _, args in fleet.calls[start:]]
        assert sum(args[:3] == ("profile", "load", "--review") for args in calls) <= 1
        assert set(sweep.controller_wait_holds()) == {key}
        assert key not in sweep.state.slots
        assert sweep.state.slots[other]["phase"] == SweepSlotPhase.LOADING
        assert loads(fleet) == before + 1
        assignments = fleet.profiles[sweep.cfg.sweep_profile]["assignments"]
        assert {item["recipe_selector"] for item in assignments} == {other}
        clock.sleep(sweep.cfg.poll_seconds)
    assert all(entry.get("attempts", 0) == 0 for entry in sweep.state.recipes.values())
    row = next(
        row
        for row in sweep.log.entries()
        if row.get("recipe") == key and row.get("evidence", {}).get("retry_at")
    )
    assert row["evidence"]["wait"]["reason"] == SweepWaitReason.BLOCKER


@pytest.mark.parametrize("state", [LifecycleState.FAILED, LifecycleState.RUNNING])
def test_unattributed_shared_refusal_uses_normal_bounded_requeue(
    tmp_path, monkeypatch, state
):
    sweep, _, clock, _, _ = prepare(
        tmp_path, monkeypatch, RECORDS[0], active=state, second=True, batched=True
    )
    if state == LifecycleState.RUNNING:
        slots = list(sweep.state.slots.values())
        slots[0]["watch"]["wait_deadline"] = clock.now() + 1000
        slots[1]["watch"]["wait_deadline"] = clock.now() - 1
        sweep._tick()
    assert not sweep.controller_wait_holds()
    # Preserve main's observational retry for terminal unattributed failures;
    # active wait expiry spends no recipe attempt.
    attempts = 1 if state == LifecycleState.FAILED else 0
    assert all(
        entry.get("attempts", 0) == attempts for entry in sweep.state.recipes.values()
    )
    assert all(
        entry["status"] == RecipeStatus.PENDING
        for entry in sweep.state.recipes.values()
    )
    assert all(
        entry["not_before"] > clock.now() for entry in sweep.state.recipes.values()
    )


@pytest.mark.parametrize("attribution", ["document", "assignment"])
@pytest.mark.parametrize("interrupted", [False, True])
def test_recipe_attributed_failure_precedes_refusal_hold(
    tmp_path, monkeypatch, attribution, interrupted
):
    record = deepcopy(RECORDS[0])
    doc = record["controller_document"]
    doc["progress"]["blockers"][0]["recipe"] = record["recipe"]
    if attribution == "document":
        doc["failure"] = {"code": QualificationCauseCode.RUNTIME_EXIT}
    else:
        doc["progress"]["switch_adapter"] = {
            "assignment_failures": [
                {
                    "recipe": record["recipe"],
                    "code": QualificationCauseCode.RUNTIME_EXIT,
                    "reason": "injected runtime failure",
                }
            ]
        }
    if interrupted:
        sweep, _, _, key, observed = prepare(
            tmp_path, monkeypatch, RECORDS[0], active=LifecycleState.RUNNING
        )
        load = next(iter(sweep.state.data["loads"].values()))
        observed["state"] = LifecycleState.FAILED
        sweep._hold_controller_refusal(load, observed)
        sweep.state.slots[key]["phase"] = SweepSlotPhase.LOADING
        observed.update(doc)
        sweep._tick()
    else:
        sweep, _, _, key, _ = prepare(tmp_path, monkeypatch, record)
        assert not sweep.controller_wait_holds()
    entry = sweep.state.entry(key)
    assert entry["status"] == RecipeStatus.FAILED
    assert entry["code"] == QualificationCauseCode.RUNTIME_EXIT
    assert entry["attempts"] == 1


def test_submit_guard_ignores_adopted_live_assignment(tmp_path, monkeypatch):
    sweep, fleet, _clock, key, doc = prepare(
        tmp_path, monkeypatch, RECORDS[0], active=LifecycleState.RUNNING
    )
    load = next(iter(sweep.state.data["loads"].values()))
    doc["state"] = LifecycleState.FAILED
    sweep._hold_controller_refusal(load, doc)
    assert key in sweep.controller_wait_holds()
    # Inject the interruption window: the hold persisted before its slot ended.
    sweep.state.slots[key]["phase"] = SweepSlotPhase.LOADING
    before = loads(fleet)
    assert (
        sweep._submit_load(sweep.cfg.sweep_profile, "independent", "placement")
        is not None
    )
    assert loads(fleet) == before + 1


@pytest.mark.parametrize("collateral", [False, True])
def test_submission_refusal_requeues_with_description_and_retry_deadline(
    tmp_path, monkeypatch, collateral
):
    sweep, fleet, clock, held_key, doc = prepare(
        tmp_path, monkeypatch, RECORDS[0], active=LifecycleState.RUNNING
    )
    sweep.requeue(held_key, "injected re-plan")
    sweep.enqueue_finished_cleanup()
    key = held_key
    if collateral:
        fresh = FakeRecipe("second", ("1" * 64,), content="b" * 64, local="cached")
        fleet.recipes[fresh.key] = fresh
        sweep.refresh_catalog()
        key = fresh.key
    original = sweep._submit_load

    def hold_before_submission(profile, request, kind, **kwargs):
        doc["state"] = LifecycleState.FAILED
        if collateral:
            # A late stale assignment must defer B with A's actual hold evidence.
            sweep._open_slot(
                policy.Placement(sweep.recipes[held_key], (sweep.sparks()[1].id,)),
                "late-refusal",
                clock.now(),
            )
            sweep._hold_controller_refusal(
                {"request_key": "late-refusal", "app_id": "late-refusal"}, doc
            )
            sweep.vk.call(
                "profile",
                "add",
                held_key,
                "--spark",
                sweep.sparks()[1].name,
                "--as",
                sweep._alias(sweep.recipes[held_key]),
                "--state",
                LifecycleState.RUNNING,
                "--yes",
                profile=profile,
            )
        else:
            sweep._hold_controller_refusal(
                {**sweep.state.data["loads"][request], "app_id": "late-refusal"}, doc
            )
        return original(profile, request, kind, **kwargs)

    monkeypatch.setattr(sweep, "_submit_load", hold_before_submission)
    placement = policy.Placement(sweep.recipes[key], (sweep.sparks()[0].id,))
    # Inject at the submission boundary, after admission review.
    assert sweep._apply([placement], clock.now()) is False
    hold = sweep.controller_wait_holds()[held_key]
    row = next(
        row
        for row in reversed(sweep.log.entries())
        if row.get("step") == "requeue" and row.get("recipe") == key
    )
    assert row["reason"] == hold.wait.description
    assert row["evidence"]["retry_at"] == hold.retry_at
    assert sweep.state.entry(key)["not_before"] == hold.retry_at
