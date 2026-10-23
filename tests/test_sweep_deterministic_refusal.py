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


def prepare(tmp_path, monkeypatch, record, *, active=False):
    recipe = FakeRecipe(
        record["recipe"].split("/", 1)[1],
        content=record["content_sha256"],
        digests=("1" * 64,),
        local="cached",
        blocked=True,
    )
    sweep, fleet, clock = make_sweep(
        tmp_path, [recipe], [FakeModel("1" * 64, local="cached")]
    )
    clock.t = datetime.fromisoformat(record["recorded_at"]).timestamp()
    sweep.preflight()
    sweep.refresh_catalog()
    sweep.queue = list(sweep.recipes)
    sweep.cached_models = sweep._cached_models()
    sweep.observe_platform()
    sweep.schedule(clock.now())
    original = fleet._app_doc
    document = deepcopy(record["controller_document"])
    if active:
        document["state"] = LifecycleState.RUNNING

    for app in fleet.apps.values():
        app["state"] = document["state"]

    def refused(app):
        return {**original(app), **deepcopy(document)}

    monkeypatch.setattr(fleet, "_app_doc", refused)
    sweep.poll_application()
    key = recipe.key
    sweep._advance_loading(key, sweep.state.slots[key], clock.now())
    return sweep, fleet, clock, key, document


def loads(fleet):
    return sum(c[:3] == ("profile", "load", "--yes") for _, c in fleet.calls)


@pytest.mark.parametrize("record", RECORDS, ids=lambda r: r["recorded_at"])
def test_exact_refusal_never_resubmits_until_controller_content_changes(
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
    assert "next observation" in text
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
                Presence("node-a", "other", "run-new", True, "ready", "running"),
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
    sweep._refresh_controller_waits(clock.now())
    assert not sweep.controller_wait_holds()
    assert sweep.state.entry(key).get("attempts", 0) == 0


def test_preparation_warning_expires_without_blame_and_keeps_other_work_eligible(
    tmp_path, monkeypatch
):
    record = deepcopy(RECORDS[0])
    record["controller_document"]["progress"]["blockers"][0]["severity"] = "warning"
    sweep, fleet, clock, key, _ = prepare(tmp_path, monkeypatch, record, active=True)
    slot = sweep.state.slots[key]
    assert "wait_deadline" in slot["watch"]
    clock.t = slot["watch"]["wait_deadline"] + 1
    sweep._advance_loading(key, slot, clock.now())
    sweep.enqueue_finished_cleanup()
    assert key in sweep.controller_wait_holds()
    assert sweep.state.entry(key).get("attempts", 0) == 0
    other = FakeRecipe("other", ("1" * 64,), local="cached")
    fleet.recipes[other.key] = other
    sweep.refresh_catalog()
    sweep.queue = list(sweep.recipes)
    assert [r.key for r in sweep.ready_candidates()] == [other.key]
    sweep.schedule(clock.now())
    assert other.key in sweep.state.slots
    assert key not in sweep.state.slots


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


def test_shared_refusal_uses_the_existing_platform_pause(tmp_path, monkeypatch):
    sweep, fleet, clock, _, _ = prepare(tmp_path, monkeypatch, RECORDS[0])
    sweep.enqueue_finished_cleanup()
    other = FakeRecipe("other", ("1" * 64,), content="e" * 64, local="cached")
    fleet.recipes[other.key] = other
    sweep.refresh_catalog()
    sweep.queue = list(sweep.recipes)
    sweep.schedule(clock.now())
    sweep.poll_application()
    slot = sweep.state.slots[other.key]
    sweep._advance_loading(other.key, slot, clock.now())
    sweep.enqueue_finished_cleanup()
    assert len(sweep.controller_wait_holds()) == 2
    assert "shared Controller refusal" in sweep.controller_wait_issue()
    assert not sweep.ready_candidates()
    assert all(e.get("attempts", 0) == 0 for e in sweep.state.recipes.values())
