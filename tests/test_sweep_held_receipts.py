"""A bound receipt the Controller still works on keeps its lane (2026-10-25).

Production 2026-10-24T20:13Z on: Controller waits carry no forward marks, so a
lane's local wait window expired while its bound application still waited
(``receipt-missing``, ``run-switch.phase-retry``). The lane was requeued, pruned
from the sweep profile, and the next placement's whole-profile load superseded the
receipt the Controller was still working on (30 of 35 applies superseded, 27 of 33
run-switches cancelled by the sweep's own loads). Only the Controller ends such a
wait: the receipt's own end, its stall alarm, or newer content.

Every scenario runs against the Controller's stall alarm as vonk-forge #1899 raises
it (``sweep_fakes.StallAlarm``): one ``PROGRESS_STALL_WINDOW`` after a wait last moved
forward, also for wait-only applications such as ``receipt-missing``. Today's
Controller raises it only from ``progress.blockers``, so the stall-alarm endings depend
on #1899 being deployed. A release today's Controller already produces: the child or
application fails (reconcile.py:440-450).
"""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime

from sweep_fakes import (
    PROGRESS_STALL_WINDOW,
    STALLED,
    FakeModel,
    FakeRecipe,
    StallAlarm,
    make_sweep,
)
from test_sweep_controller_waits import RECEIPT, controller_document

from spark_sweep.lifecycle import LifecycleState
from spark_sweep.policy import TimeoutPolicy
from spark_sweep.report import write_status

DIGEST = "1" * 64
#: Each lane's local window: shorter than the Controller's stall window, so a lane's
#: window runs out while the Controller still works its receipt without alarm.
WINDOW = 600.0


def held_lane(tmp_path, monkeypatch, node_count=1):
    """Lane A is bound and its Run/Switch child waits on the Controller; B is not queued."""
    a = FakeRecipe(
        "a",
        content="a" * 64,
        digests=(DIGEST,),
        local="cached",
        node_count=node_count,
        load_seconds=10**6,
    )
    b = FakeRecipe("b", content="b" * 64, digests=(DIGEST,), local="cached")
    sweep, fleet, clock = make_sweep(
        tmp_path,
        [a, b],
        [FakeModel(DIGEST, local="cached")],
        timeouts=TimeoutPolicy(
            first_start_seconds=WINDOW, floor_seconds=WINDOW, cap_seconds=WINDOW
        ),
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
    alarm = StallAlarm(fleet).install(monkeypatch)
    sweep.advance_slots(clock.now())
    slot = sweep.state.slots[a.key]
    assert slot["watch"]["wait"]["controller"]["reason"] == "receipt-missing"
    assert sweep.state.data["loads"][slot["request_key"]]["admission_unbound"] is False
    assert slot["deadline"] - slot["started_at"] == WINDOW < PROGRESS_STALL_WINDOW
    return sweep, fleet, clock, a, b, application, waiting, alarm


def until_alarm(sweep, clock, alarm, slot):
    """Observe each poll until the lane ends; it is held until the Controller alarms."""
    for _ in range(1000):
        clock.sleep(sweep.cfg.poll_seconds)
        sweep.advance_slots(clock.now())
        if slot["phase"] != "loading":
            assert alarm.alarmed, "released before the Controller's stall alarm"
            return
    raise AssertionError("the hold never ended")


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
    sweep, fleet, clock, a, b, application, _, _ = held_lane(tmp_path, monkeypatch)
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
    sweep, fleet, clock, a, b, application, waiting, _ = held_lane(
        tmp_path, monkeypatch
    )
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
    """Depends on vonk-forge #1899: the alarm on a receipt-missing application."""
    sweep, _, clock, a, b, application, _, alarm = held_lane(tmp_path, monkeypatch)
    expire_window_and_place_b(sweep, clock, a, b)
    slot = sweep.state.slots[a.key]
    until_alarm(sweep, clock, alarm, slot)
    assert slot["phase"] == "finished"
    # The alarm comes one stall window after the wait last moved, never earlier.
    (since,) = {at for at, app in alarm.alarmed if app == application}
    assert since - slot["started_at"] > PROGRESS_STALL_WINDOW
    assert sweep.controller_wait_holds()[a.key].live
    assert sweep.state.entry(a.key).get("attempts", 0) == 0


def test_a_blind_pass_does_not_release_a_held_receipt(tmp_path, monkeypatch) -> None:
    sweep, _, clock, a, _, _, _, _ = held_lane(tmp_path, monkeypatch)
    slot = sweep.state.slots[a.key]
    clock.t = slot["watch"]["wait_deadline"] + 1
    sweep.blind = True
    sweep._advance_loading(a.key, slot, clock.now())
    assert slot["phase"] == "loading"


def test_newer_content_steers_past_a_held_receipt(tmp_path, monkeypatch) -> None:
    sweep, _, clock, a, b, _, _, _ = held_lane(tmp_path, monkeypatch)
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


def test_a_hold_that_clears_without_movement_is_no_recipe_silence(
    tmp_path, monkeypatch
) -> None:
    sweep, fleet, clock, a, _, _, _, _ = held_lane(tmp_path, monkeypatch)
    shown = fleet._app_doc
    declared = {"wait": False}

    def document(app):
        # The recorded child snapshot throughout; only the Controller's wait comes and goes.
        observed = shown(app)
        if not declared["wait"]:
            observed.pop("wait", None)
            observed.pop("next_attempt_at", None)
        return observed

    monkeypatch.setattr(fleet, "_app_doc", document)
    slot = sweep.state.slots[a.key]
    sweep.advance_slots(clock.now())  # own work seen moving
    assert "progressed_at" in slot["watch"]
    declared["wait"] = True
    for _ in range(2):  # past the lane window, inside the Controller's
        clock.sleep(WINDOW)
        sweep.advance_slots(clock.now())
        assert slot["phase"] == "loading"
    # The wait clears and the child shows exactly what it showed before the hold.
    declared["wait"] = False
    clock.sleep(sweep.cfg.poll_seconds)
    sweep.advance_slots(clock.now())
    assert slot["phase"] == "loading"
    assert sweep.state.entry(a.key).get("attempts", 0) == 0


def test_a_held_receipt_ends_when_the_controller_fails_its_child(
    tmp_path, monkeypatch
) -> None:
    # Today's Controller: a load follows its child, and a child that ended failed is
    # a definite end of the application (reconcile.py:440-450).
    sweep, fleet, clock, a, b, application, _, _ = held_lane(tmp_path, monkeypatch)
    expire_window_and_place_b(sweep, clock, a, b)
    fleet.apps[application].update(
        state=LifecycleState.FAILED, reason="Profile step 1 ended in failed"
    )
    clock.sleep(sweep.cfg.poll_seconds)
    sweep.advance_slots(clock.now())
    assert sweep.state.slots[a.key]["phase"] == "finished"
    assert "spk_a" not in {
        node
        for slot in sweep.state.slots.values()
        if slot["phase"] == "loading"
        for node in slot["node_ids"]
    }
    (row,) = [row for row in sweep.log.entries() if row.get("recipe") == a.key]
    assert row["evidence"]["application_state"] == LifecycleState.FAILED


def test_an_offline_lane_spark_keeps_a_held_dual_until_the_stall_alarm(
    tmp_path, monkeypatch
) -> None:
    """Depends on vonk-forge #1899: the alarm on a receipt-missing application."""
    sweep, fleet, clock, a, _, application, _, alarm = held_lane(
        tmp_path, monkeypatch, node_count=2
    )
    slot = sweep.state.slots[a.key]
    assert sorted(slot["node_ids"]) == ["spk_a", "spk_b"]
    fleet.offline_nodes.add("spk_b")
    # The receipt cannot move; the Controller alarms on it and the hold ends, bounded.
    until_alarm(sweep, clock, alarm, slot)
    assert slot["phase"] == "finished"
    assert fleet.apps[application]["state"] == LifecycleState.RUNNING
    assert sweep.controller_wait_holds()[a.key].live
    assert sweep.state.entry(a.key).get("attempts", 0) == 0


def test_the_fake_alarm_has_the_controllers_shape(tmp_path, monkeypatch) -> None:
    """vonk-forge #1899: only the stall is stored; the alarm is read-time, last."""
    _, fleet, clock, _, _, application, _, alarm = held_lane(tmp_path, monkeypatch)
    app = fleet.apps[application]
    fresh = fleet._app_doc(app)
    assert all(b["code"] != STALLED for b in fresh["wait"]["blockers"])
    clock.sleep(PROGRESS_STALL_WINDOW + 1)
    view = fleet._app_doc(app)
    for blockers in (
        view["blockers"],
        view["progress"]["blockers"],
        view["wait"]["blockers"],
    ):
        assert [b["code"] for b in blockers] == ["receipt-missing", STALLED]
        assert blockers[-1]["severity"] == "error"
        assert blockers[-1]["detail"].startswith("No progress since ")
        assert blockers[-1]["detail"].endswith(
            ": receipt-missing: Waiting for the exact Run/Switch child receipt"
        )
    assert view["state"] == LifecycleState.RUNNING
    assert app["state"] == LifecycleState.RUNNING and "stalled" not in app
    assert alarm.alarmed
