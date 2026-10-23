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

from spark_sweep import verdict
from spark_sweep.catalog import Presence
from spark_sweep.lifecycle import LifecycleState, RecipeStatus
from spark_sweep.report import build_status, write_status
from spark_sweep.state import State

REFUSAL = json.loads(
    (
        Path(__file__).parent
        / "fixtures/sweep-verdicts/preparation-refusal-2026-10-23.json"
    ).read_text()
)
RECORDS = [{**REFUSAL, "recorded_at": at} for at in REFUSAL["recorded_at"]]


def prepare(
    tmp_path, monkeypatch, record, *, active=False, second=False, batched=False
):
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
        document["state"] = LifecycleState.RUNNING

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


@pytest.mark.parametrize("change", ["fleet", "catalog", "detail", "code", "resumed"])
def test_changed_evidence_replans(tmp_path, monkeypatch, change):
    sweep, _, clock, key, document = prepare(tmp_path, monkeypatch, RECORDS[0])
    sweep.enqueue_finished_cleanup()
    if change == "fleet":
        sweep.fleet = replace(
            sweep.fleet,
            presences=(
                Presence(
                    sweep.controller_wait_holds()[key].node_ids[0],
                    "other",
                    "run-new",
                    True,
                    "ready",
                    "running",
                ),
            ),
        )
    elif change == "catalog":
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


@pytest.mark.parametrize("batched", [False, True])
def test_unrelated_recipe_is_placed_during_refusal_holds(
    tmp_path, monkeypatch, batched
):
    sweep, fleet, clock, key, _ = prepare(
        tmp_path, monkeypatch, RECORDS[0], second=True, batched=batched
    )
    holds = sweep.controller_wait_holds()
    assert len(holds) == 2
    assert len({h.application_id for h in holds.values()}) == (1 if batched else 2)
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


@pytest.mark.parametrize("settled", [LifecycleState.SUPERSEDED, LifecycleState.FAILED])
def test_static_refusal_allows_one_fresh_placement_at_persisted_deadline(
    tmp_path, monkeypatch, settled
):
    record = deepcopy(RECORDS[0])
    record["controller_document"]["state"] = settled
    sweep, fleet, clock, key, _ = prepare(tmp_path, monkeypatch, record)
    hold = sweep.controller_wait_holds()[key]
    assert hold.retry_at == clock.now() + 3600
    before = loads(fleet)
    sweep.state.save()
    sweep.state = State.load(tmp_path / "state.json", clock.now)
    for elapsed in (60, 600, 3600 - 1):
        clock.t = hold.retry_at - 3600 + elapsed
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


@pytest.mark.parametrize("known", [(), ("api",), ("worker",)])
def test_unknown_platform_capture_releases_when_deployment_is_observed(
    tmp_path, monkeypatch, known
):
    from spark_sweep.recovery import Fingerprint

    capture = {owner: Fingerprint(owner, "a" * 40, "c" * 64) for owner in known}
    # Hide provenance throughout capture, including the _tick refresh.
    from spark_sweep.run import Sweep

    monkeypatch.setattr(Sweep, "_component_observations", lambda self: capture)
    sweep, fleet, clock, key, _ = prepare(tmp_path, monkeypatch, RECORDS[0])
    assert sweep.controller_wait_holds()[key].platform == capture
    before = loads(fleet)
    capture = {
        **capture,
        "api": Fingerprint("api", "a" * 40, "c" * 64),
        "worker": Fingerprint("worker", "a" * 40, "c" * 64),
    }
    sweep.schedule(clock.now())
    assert loads(fleet) == before + 1
    assert not sweep.controller_wait_holds()


def test_unrelated_fleet_churn_does_not_release_refusal(tmp_path, monkeypatch):
    sweep, fleet, clock, key, _ = prepare(tmp_path, monkeypatch, RECORDS[0])
    hold = sweep.controller_wait_holds()[key]
    before = loads(fleet)
    assert hold.node_ids
    sweep.fleet = replace(
        sweep.fleet,
        presences=(
            *sweep.fleet.presences,
            Presence("unrelated-node", "other", "run-new", True, "ready", "running"),
        ),
    )
    clock.sleep(sweep.cfg.defer_delay + 1)
    sweep.observe_platform()
    sweep.schedule(clock.now())
    assert loads(fleet) == before
    sweep.fleet = replace(
        sweep.fleet,
        presences=(
            Presence(hold.node_ids[0], "other", "run-new", True, "ready", "running"),
        ),
    )
    sweep.schedule(clock.now())
    assert key not in sweep.controller_wait_holds()


def test_static_refusal_blocks_replay_until_one_hour_then_retries(
    tmp_path, monkeypatch
):
    sweep, fleet, clock, _, _ = prepare(tmp_path, monkeypatch, RECORDS[0])
    assert loads(fleet) == 1
    clock.sleep(3599)
    sweep.observe_platform()
    sweep.schedule(clock.now())
    assert loads(fleet) == 1
    clock.sleep(1)
    sweep.schedule(clock.now())
    sweep.schedule(clock.now())
    assert loads(fleet) == 2
