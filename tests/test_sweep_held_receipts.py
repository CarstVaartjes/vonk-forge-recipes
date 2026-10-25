"""A bound receipt the Controller still works on keeps its lane (2026-10-25).

Production 2026-10-24T20:13Z on: Controller waits carry no forward marks, so a
lane's local wait window expired while its bound application still waited
(``receipt-missing``, ``run-switch.phase-retry``). The lane was requeued, pruned
from the sweep profile, and the next placement's whole-profile load superseded the
receipt the Controller was still working on (30 of 35 applies superseded, 27 of 33
run-switches cancelled by the sweep's own loads). Only the Controller ends such a
wait: the receipt's own end, its stall alarm, or newer content.
"""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime

from sweep_fakes import FakeModel, FakeRecipe, make_sweep
from test_sweep_controller_waits import RECEIPT, controller_document

from spark_sweep import verdict
from spark_sweep.lifecycle import LifecycleState
from spark_sweep.report import write_status

DIGEST = "1" * 64


def held_lane(tmp_path, monkeypatch):
    """Lane A is bound and its Run/Switch child waits on the Controller; B is not queued."""
    a = FakeRecipe("a", content="a" * 64, digests=(DIGEST,), local="cached")
    a.load_seconds = 10**6
    b = FakeRecipe("b", content="b" * 64, digests=(DIGEST,), local="cached")
    sweep, fleet, clock = make_sweep(
        tmp_path, [a, b], [FakeModel(DIGEST, local="cached")]
    )
    clock.t = datetime.fromisoformat(RECEIPT["recorded_at"]).timestamp()
    sweep.preflight()
    sweep.refresh_catalog()
    sweep.cached_models = sweep._cached_models()
    sweep.queue = [a.key]
    sweep.schedule(clock.now())
    (application,) = fleet.apps
    original = fleet._app_doc
    waiting = {application}

    def observed(app):
        document = original(app)
        if app["id"] in waiting and app["state"] == LifecycleState.RUNNING:
            document.update(controller_document(RECEIPT, app))
        return document

    monkeypatch.setattr(fleet, "_app_doc", observed)
    sweep.advance_slots(clock.now())
    slot = sweep.state.slots[a.key]
    assert slot["watch"]["wait"]["controller"]["reason"] == "receipt-missing"
    assert sweep.state.data["loads"][slot["request_key"]]["admission_unbound"] is False
    return sweep, fleet, clock, a, b, application, waiting


def expire_window_and_place_b(sweep, clock, a, b):
    slot = sweep.state.slots[a.key]
    clock.t = slot["watch"]["wait_deadline"] + 1
    sweep.advance_slots(clock.now())
    sweep.enqueue_finished_cleanup()
    sweep.queue = [a.key, b.key]
    sweep.schedule(clock.now())


def test_window_expiry_never_supersedes_the_sweeps_own_held_receipt(
    tmp_path, monkeypatch
) -> None:
    sweep, fleet, clock, a, b, application, _ = held_lane(tmp_path, monkeypatch)
    expire_window_and_place_b(sweep, clock, a, b)
    # A keeps its lane, its Spark and its assignment; B takes the free Spark.
    assert sweep.state.slots[a.key]["phase"] == "loading"
    assert sweep.state.slots[b.key]["node_ids"] == ["spk_b"]
    exported = {x["assignment_name"] for x in fleet.profiles[10]["assignments"]}
    assert {"a", "b"} <= exported
    assert not [c for c in fleet.writes(10) if c[:3] == ("profile", "remove", "a")]
    # B's whole-profile load adopted A's unchanged child: the receipt continues.
    assert fleet.apps[application]["state"] == LifecycleState.RUNNING
    assert any(run["alias"] == "a" for run in fleet.runs)
    assert sweep.state.entry(a.key).get("attempts", 0) == 0
    assert "controller_wait" not in sweep.state.entry(a.key)
    # Never silent: the status names the held receipt, not a due re-plan.
    write_status(sweep)
    status = (tmp_path / "status.md").read_text()
    assert f"held on Controller receipt {application}" in status
    assert "re-plan due" not in status


def test_a_held_receipt_ends_on_its_own_end(tmp_path, monkeypatch) -> None:
    sweep, fleet, clock, a, b, application, waiting = held_lane(tmp_path, monkeypatch)
    expire_window_and_place_b(sweep, clock, a, b)
    # The Controller's child receipt arrives and the run comes up: A goes on.
    waiting.clear()
    for item in fleet.apps[application]["assign"]:
        item["at"] = clock.now()
    clock.sleep(sweep.cfg.poll_seconds)
    sweep.advance_slots(clock.now())
    assert sweep.state.slots[a.key]["phase"] == "smoking"
    assert sweep.state.entry(a.key).get("attempts", 0) == 0


def test_a_held_receipt_ends_on_the_controllers_stall_alarm(
    tmp_path, monkeypatch
) -> None:
    sweep, _, clock, a, b, _, _ = held_lane(tmp_path, monkeypatch)
    expire_window_and_place_b(sweep, clock, a, b)
    slot = sweep.state.slots[a.key]
    load = sweep.state.data["loads"][slot["request_key"]]
    load["stalled"] = verdict.STALLED_CODE
    sweep._advance_loading(a.key, slot, clock.now())
    assert slot["phase"] == "finished"
    assert sweep.controller_wait_holds()[a.key].live
    assert sweep.state.entry(a.key).get("attempts", 0) == 0


def test_a_blind_pass_does_not_release_a_held_receipt(tmp_path, monkeypatch) -> None:
    sweep, _, clock, a, _, _, _ = held_lane(tmp_path, monkeypatch)
    slot = sweep.state.slots[a.key]
    clock.t = slot["watch"]["wait_deadline"] + 1
    sweep.blind = True
    sweep._advance_loading(a.key, slot, clock.now())
    assert slot["phase"] == "loading"


def test_newer_content_steers_past_a_held_receipt(tmp_path, monkeypatch) -> None:
    sweep, _, clock, a, b, _, _ = held_lane(tmp_path, monkeypatch)
    expire_window_and_place_b(sweep, clock, a, b)
    recipe = sweep.recipes[a.key]
    sweep.recipes[a.key] = replace(recipe, content_sha256="d" * 64)
    clock.sleep(sweep.cfg.poll_seconds)
    sweep.advance_slots(clock.now())
    assert sweep.state.slots[a.key]["phase"] == "finished"
    sweep.enqueue_finished_cleanup()
    sweep._refresh_controller_waits(clock.now())
    clock.sleep(sweep.cfg.defer_delay + 1)
    assert a.key in {r.key for r in sweep.ready_candidates()}
    assert sweep.state.entry(a.key).get("attempts", 0) == 0
