"""Work under a Controller wait on a bound receipt ends only with the Controller.

Replays the 2026-10-24 incident: a lane held by the Controller's info ``receipt-missing``
wait (its run-switch child pending) while the distribution moved, froze, retried or
flapped. The Controller still works that receipt, so neither the lane's local window nor
a dependency deadline the Controller does not act on releases it: releasing would prune
its assignment and the sweep's next load would supersede the receipt. The receipt's own
end or the Controller's stall alarm does: one ``PROGRESS_STALL_WINDOW`` after the last
new high or new child phase (``StallAlarm``, the alarm of vonk-forge #1899, which this
behaviour depends on for wait-only receipts). Forward-only marks still renew other
waits (``test_only_a_rising_mark_renews_a_wait``).
"""

from __future__ import annotations

import json
from copy import deepcopy
from datetime import datetime
from itertools import pairwise
from pathlib import Path

import pytest
from sweep_fakes import (
    GB,
    PROGRESS_STALL_WINDOW,
    STALLED,
    FakeModel,
    FakeRecipe,
    Gateway,
    StallAlarm,
    make_sweep,
)
from vonk_forge_contracts.sweep import SweepWait, SweepWaitCode, SweepWaitReason

from spark_sweep import verdict
from spark_sweep.lifecycle import LifecycleState
from spark_sweep.policy import TimeoutPolicy

pytest_plugins = ["sweep_bounds"]
pytestmark = pytest.mark.usefixtures("bounded_clock")

WINDOW = 600.0
MODEL = "1" * 64
RECORDS = json.loads(
    (
        Path(__file__).parent
        / "fixtures/sweep-verdicts/controller-waits-2026-10-23.json"
    ).read_text()
)
# The recorded info receipt-missing wait without a Controller dependency deadline.
RECEIPT_WAIT = next(
    row["controller_document"]["wait"]
    for row in RECORDS
    if row["controller_document"]["wait"]["reason"] == "receipt-missing"
    and "dependency_deadline" not in row["controller_document"]["wait"]
)
# The same wait as recorded with the Controller's own dependency deadline.
DEADLINE_RECORD = next(
    row
    for row in RECORDS
    if row["controller_document"]["wait"]["reason"] == "receipt-missing"
    and "dependency_deadline" in row["controller_document"]["wait"]
)


@pytest.fixture
def gateway():
    gateway = Gateway()
    yield gateway
    gateway.stop()


def _timeouts() -> TimeoutPolicy:
    return TimeoutPolicy(
        first_start_seconds=WINDOW, floor_seconds=WINDOW, cap_seconds=WINDOW
    )


def _held_by_receipt_wait(fleet, monkeypatch, wait=RECEIPT_WAIT) -> list[dict]:
    """The Controller declares its wait on every running application, as recorded.

    Returns the child progress the sweep was shown, in order.
    """
    original = fleet._app_doc
    shown: list[dict] = []

    def held(app):
        document = original(app)
        if document["state"] == LifecycleState.RUNNING:
            document["wait"] = deepcopy(wait)
            shown.append(deepcopy(document["progress"]["child_progress"]))
        return document

    monkeypatch.setattr(fleet, "_app_doc", held)
    return shown


def _until(sweep, clock, done) -> None:
    for _ in range(1000):
        sweep.tick()
        if done():
            return
        clock.sleep(sweep.cfg.poll_seconds)
    raise AssertionError("the sweep never got there")


def _expired(tmp_path: Path, key: str) -> list[dict]:
    if not (tmp_path / "results.jsonl").exists():
        return []
    return [
        row["evidence"]
        for row in _rows(tmp_path)
        if row.get("recipe") == key
        and row.get("code") == SweepWaitCode.BUDGET_EXHAUSTED
    ]


def _rows(tmp_path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in (tmp_path / "results.jsonl").read_text().splitlines()
    ]


def _loads(fleet) -> list[float]:
    return [
        t
        for t, (profile, call) in zip(fleet.call_times, fleet.calls, strict=True)
        if profile == 10 and call[:3] == ("profile", "load", "--yes")
    ]


def test_bytes_moving_under_a_controller_wait_outlast_the_lane_budget(
    tmp_path: Path, gateway: Gateway, monkeypatch
) -> None:
    big = FakeRecipe(
        "big",
        content="a" * 64,
        digests=(MODEL,),
        copy_seconds=4 * WINDOW,
        load_seconds=60,
    )
    sweep, fleet, clock = make_sweep(
        tmp_path, [big], [FakeModel(MODEL)], gateway=gateway, timeouts=_timeouts()
    )
    _held_by_receipt_wait(fleet, monkeypatch)
    sweep.preflight()
    for _ in range(600):
        sweep.tick()
        if sweep.state.status(big.key) == "passed":
            break
        clock.sleep(sweep.cfg.poll_seconds)
    else:
        raise AssertionError("the moving copy never completed under the wait")
    assert not [
        row
        for row in _rows(tmp_path)
        if row.get("code") == SweepWaitCode.BUDGET_EXHAUSTED
    ]
    assert not any(c[:2] == ("profile", "cancel") for _, c in fleet.calls)


def _alarm_since(evidence: dict) -> float:
    """When the Controller's alarm says the wait last moved (``No progress since``)."""
    (alarm,) = [
        b for b in evidence["wait"]["controller"]["blockers"] if b["code"] == STALLED
    ]
    since = alarm["detail"].removeprefix("No progress since ").split(": ", 1)[0]
    return datetime.fromisoformat(since).timestamp()


def test_a_frozen_copy_keeps_its_held_lane_until_the_controller_alarms(
    tmp_path: Path, gateway: Gateway, monkeypatch
) -> None:
    # The copy moves for 2 windows (20 % of 10), then freezes under the same wait.
    stuck = FakeRecipe(
        "a-stuck",
        content="a" * 64,
        digests=(MODEL,),
        copy_seconds=10 * WINDOW,
        copy_stalls=True,
        memory=90 * GB,
    )
    good = FakeRecipe(
        "b-good", content="b" * 64, digests=(MODEL,), load_seconds=60, memory=90 * GB
    )
    sweep, fleet, clock = make_sweep(
        tmp_path,
        [stuck, good],
        [FakeModel(MODEL)],
        gateway=gateway,
        timeouts=_timeouts(),
    )
    fleet.sparks = fleet.sparks[:1]  # one lane: the next recipe needs the held Spark
    _held_by_receipt_wait(fleet, monkeypatch)
    alarm = StallAlarm(fleet).install(monkeypatch)
    sweep.preflight()
    _until(sweep, clock, lambda: stuck.key in sweep.state.slots)
    first_load = _loads(fleet)[0]
    frozen = first_load + 2 * WINDOW
    # Past the lane window, before the Controller's: still held, nothing else placed.
    _until(sweep, clock, lambda: clock.now() > frozen + WINDOW)
    assert not alarm.alarmed
    assert sweep.state.slots[stuck.key]["phase"] == "loading"
    assert not _expired(tmp_path, stuck.key)
    assert _loads(fleet) == [first_load]
    # The Controller alarms one stall window after the last new high; the freed
    # lane goes to the next recipe.
    _until(sweep, clock, lambda: sweep.state.status(good.key) == "passed")
    (evidence,) = _expired(tmp_path, stuck.key)
    assert evidence["wait"]["reason"] == SweepWaitReason.CONTROLLER_WAIT
    assert evidence["wait"]["controller"]["reason"] == RECEIPT_WAIT["reason"]
    since = _alarm_since(evidence)
    poll = sweep.cfg.poll_seconds
    assert frozen - 2 * poll <= since <= frozen + poll
    assert evidence["observed_at"] - since > PROGRESS_STALL_WINDOW
    assert sweep.state.entry(stuck.key).get("attempts", 0) == 0
    assert _loads(fleet)[-1] >= evidence["observed_at"]


@pytest.mark.parametrize("case", ["retried-copy", "flapping-phase"])
def test_a_retried_or_flapping_child_reaches_the_controllers_alarm(
    tmp_path: Path, gateway: Gateway, monkeypatch, case: str
) -> None:
    # Every copy attempt fails at 20 % and restarts from 0, rising again below its
    # old mark; or the child's operation flaps prepare <-> runtime-preflight with 0
    # bytes and a fresh progress time each flip. Neither is a new high or a new
    # child phase, so the Controller's window runs from the last real movement.
    recipe = FakeRecipe(
        case,
        content="a" * 64,
        digests=(MODEL,),
        copy_seconds=0 if case == "flapping-phase" else 10 * WINDOW,
        copy_retries=case == "retried-copy",
        load_seconds=10 * WINDOW,
        phase_flap=case == "flapping-phase",
    )
    sweep, fleet, clock = make_sweep(
        tmp_path, [recipe], [FakeModel(MODEL)], gateway=gateway, timeouts=_timeouts()
    )
    shown = _held_by_receipt_wait(fleet, monkeypatch)
    alarm = StallAlarm(fleet).install(monkeypatch)
    sweep.preflight()
    _until(sweep, clock, lambda: _expired(tmp_path, recipe.key))
    first_load = _loads(fleet)[0]
    poll = sweep.cfg.poll_seconds
    if case == "retried-copy":
        attempts = [child["operation"]["completed_bytes"] for child in shown]
        assert any(b < a for a, b in pairwise(attempts))
        last_high = first_load + 2 * WINDOW  # the first attempt's 20 %
    else:
        assert len({child["operation"]["phase"] for child in shown}) == 2
        assert len({child["operation"]["last_progress_at"] for child in shown}) > 2
        # The second operation phase is the last one not seen before.
        last_high = first_load + 30
    (evidence,) = _expired(tmp_path, recipe.key)
    since = _alarm_since(evidence)
    assert last_high - 2 * poll <= since <= last_high + 2 * poll
    assert alarm.alarmed[0][0] - since > PROGRESS_STALL_WINDOW
    # Held until the Controller's alarm: no earlier release, no second load.
    assert evidence["observed_at"] >= alarm.alarmed[0][0]
    assert _loads(fleet) == [first_load]
    assert sweep.state.entry(recipe.key).get("attempts", 0) == 0


#: The 2026-10-24 incident's rotating lead wait: each stays this long.
ROTATION = 900.0
ROTATING = [
    {
        "reason": "receipt-missing",
        "blockers": [
            {
                "code": "receipt-missing",
                "detail": "Waiting for the exact Run/Switch child receipt",
                "operation": {"kind": "run-switch"},
                "severity": "info",
            }
        ],
    },
    {
        "reason": "run-switch.phase-retry",
        "blockers": [
            {
                "code": "run-switch.phase-retry",
                "detail": "admission.capacity_busy",
                "node_ids": ["spk_a"],
                "severity": "warning",
            }
        ],
    },
    {
        "reason": "runtime_preflight.stale",
        "blockers": [
            {
                "code": "runtime_preflight.stale",
                "detail": "runtime_preflight.stale",
                "operation": {"kind": "fleet-profile"},
                "severity": "warning",
            }
        ],
    },
]


def test_a_rotating_wait_on_a_retried_copy_alarms_one_window_after_its_cycle(
    tmp_path: Path, gateway: Gateway, monkeypatch
) -> None:
    # receipt-missing -> run-switch.phase-retry -> runtime_preflight.stale ->
    # receipt-missing ..., each copy attempt failing at 20 % and retried from 0.
    # A wait recurring is not movement: once the cycle showed every wait, the
    # Controller alarms one stall window after the last of them first showed.
    recipe = FakeRecipe(
        "rotating",
        content="a" * 64,
        digests=(MODEL,),
        copy_seconds=10 * WINDOW,
        copy_retries=True,
        load_seconds=10 * WINDOW,
    )
    sweep, fleet, clock = make_sweep(
        tmp_path, [recipe], [FakeModel(MODEL)], gateway=gateway, timeouts=_timeouts()
    )
    original = fleet._app_doc

    def rotating(app):
        document = original(app)
        if document["state"] == LifecycleState.RUNNING:
            turn = int((clock.now() - app["created"]) // ROTATION) % len(ROTATING)
            document["wait"] = deepcopy(ROTATING[turn])
        return document

    monkeypatch.setattr(fleet, "_app_doc", rotating)
    alarm = StallAlarm(fleet).install(monkeypatch)
    sweep.preflight()
    _until(sweep, clock, lambda: _expired(tmp_path, recipe.key))
    first_load = _loads(fleet)[0]
    cycle = first_load + 2 * ROTATION  # the third wait first shows
    assert cycle > first_load + 2 * WINDOW  # after the copy's last new high
    poll = sweep.cfg.poll_seconds
    (evidence,) = _expired(tmp_path, recipe.key)
    since = _alarm_since(evidence)
    assert cycle - poll <= since <= cycle + 2 * poll
    assert alarm.alarmed[0][0] - since > PROGRESS_STALL_WINDOW
    # Held until then; the alarm follows the wait the load shows at that moment.
    assert _loads(fleet) == [first_load]
    turn = int((alarm.alarmed[0][0] - first_load) // ROTATION) % len(ROTATING)
    blockers = evidence["wait"]["controller"]["blockers"]
    assert [b["code"] for b in blockers][-2:] == [
        ROTATING[turn]["blockers"][0]["code"],
        STALLED,
    ]
    assert sweep.state.entry(recipe.key).get("attempts", 0) == 0


def test_an_agent_re_execution_mid_copy_stays_held_without_an_alarm(
    tmp_path: Path, gateway: Gateway, monkeypatch
) -> None:
    # Half-way the agent restarts the copy: a new checkpoint key under the same
    # attempt, its bytes from 0 while the child's max-merged bytes stand still for
    # longer than the Controller's stall window. The re-execution's counters start
    # over, so each new byte is movement.
    recipe = FakeRecipe(
        "re-executed",
        content="a" * 64,
        digests=(MODEL,),
        copy_seconds=8 * WINDOW,
        copy_reexecutes=True,
    )
    sweep, fleet, clock = make_sweep(
        tmp_path, [recipe], [FakeModel(MODEL)], gateway=gateway, timeouts=_timeouts()
    )
    shown = _held_by_receipt_wait(fleet, monkeypatch)
    alarm = StallAlarm(fleet).install(monkeypatch)
    sweep.preflight()
    _until(sweep, clock, lambda: recipe.key in sweep.state.slots)
    first_load = _loads(fleet)[0]
    restart = first_load + 4 * WINDOW
    assert 4 * WINDOW > PROGRESS_STALL_WINDOW  # the child bytes stand still that long
    _until(sweep, clock, lambda: clock.now() > restart + 4 * WINDOW - 3 * 10)
    keys = [child["operation"].get("checkpoint", {}).get("key") for child in shown]
    assert {"copy-1", "copy-2"} <= set(keys)
    attempts = [child["operation"]["completed_bytes"] for child in shown]
    assert any(b < a for a, b in pairwise(attempts))
    assert len({child["retry_attempt"] for child in shown}) == 1
    assert not alarm.alarmed
    assert sweep.state.slots[recipe.key]["phase"] == "loading"
    assert not _expired(tmp_path, recipe.key)
    assert _loads(fleet) == [first_load]


@pytest.mark.parametrize("case", ["rising-copy", "deadline"])
def test_a_copy_reaching_new_highs_stays_held_without_an_alarm(
    tmp_path: Path, gateway: Gateway, monkeypatch, case: str
) -> None:
    # The copy keeps reaching new highs past the lane window and past the
    # Controller's stall window; in "deadline" the Controller also declares a
    # dependency deadline it does not act on (no fleet-profile code reads it).
    recipe = FakeRecipe(
        case, content="a" * 64, digests=(MODEL,), copy_seconds=20 * WINDOW
    )
    sweep, fleet, clock = make_sweep(
        tmp_path, [recipe], [FakeModel(MODEL)], gateway=gateway, timeouts=_timeouts()
    )
    wait = RECEIPT_WAIT
    if case == "deadline":
        clock.t = datetime.fromisoformat(DEADLINE_RECORD["recorded_at"]).timestamp()
        wait = DEADLINE_RECORD["controller_document"]["wait"]
    _held_by_receipt_wait(fleet, monkeypatch, wait)
    alarm = StallAlarm(fleet).install(monkeypatch)
    sweep.preflight()
    _until(sweep, clock, lambda: recipe.key in sweep.state.slots)
    first_load = _loads(fleet)[0]
    _until(sweep, clock, lambda: clock.now() > first_load + 6 * WINDOW)
    assert 6 * WINDOW > PROGRESS_STALL_WINDOW
    if case == "deadline":
        assert (
            clock.now()
            > datetime.fromisoformat(wait["dependency_deadline"]).timestamp()
        )
    assert not alarm.alarmed
    assert sweep.state.slots[recipe.key]["phase"] == "loading"
    assert not _expired(tmp_path, recipe.key)
    assert _loads(fleet) == [first_load]
    assert not any(
        c[:2] in (("profile", "cancel"), ("profile", "remove")) for _, c in fleet.calls
    )
    assert sweep.state.entry(recipe.key).get("attempts", 0) == 0


@pytest.mark.parametrize("marks", [True, False], ids=["with-marks", "older-watch"])
def test_a_restart_adopts_a_moving_wait(
    tmp_path: Path, gateway: Gateway, monkeypatch, marks: bool
) -> None:
    big = FakeRecipe(
        "big",
        content="a" * 64,
        digests=(MODEL,),
        copy_seconds=3 * WINDOW,
        load_seconds=60,
    )
    sweep, fleet, clock = make_sweep(
        tmp_path, [big], [FakeModel(MODEL)], gateway=gateway, timeouts=_timeouts()
    )
    _held_by_receipt_wait(fleet, monkeypatch)
    sweep.preflight()
    start = clock.now()
    _until(sweep, clock, lambda: clock.now() > start + 1.5 * WINDOW)
    slot = sweep.state.slots[big.key]
    assert slot["watch"]["wait_marks"]
    assert slot["watch"]["wait_started_at"] > start + WINDOW
    if not marks:  # as a sweep before this rule left it
        slot["watch"].pop("wait_marks")
        sweep.state.data["loads"][slot["request_key"]]["signal"].pop("marks")
    sweep.state.save()
    again, _, _ = make_sweep(
        tmp_path, [big], [FakeModel(MODEL)], fleet=fleet, timeouts=_timeouts()
    )
    again.preflight()
    _until(again, clock, lambda: again.state.status(big.key) == "passed")
    assert not _expired(tmp_path, big.key)


def test_only_a_rising_mark_renews_a_wait() -> None:
    receipt = SweepWait(reason=SweepWaitReason.CONTROLLER_RETRY)
    watch: dict = {}

    def observe(marks, now, wait=receipt, signature=None):
        signal = verdict.Signal((now,) if signature is None else signature, wait, marks)
        return verdict.progress_verdict(signal, watch, window=10, now=now)

    assert observe((1, None), 0) is verdict.Progress.WAITING
    # A changed reason or a changed signature with no rising mark is no progress.
    assert observe((1, None), 5, SweepWait(reason=SweepWaitReason.OBSERVATION))
    assert observe((1, None), 11) is verdict.Progress.WAIT_EXPIRED
    watch.clear()
    assert observe((1, None), 0) is verdict.Progress.WAITING
    for now in (8, 16, 24):  # the bytes keep rising past the window
        assert observe((now, None), now) is verdict.Progress.WAITING
    assert watch["wait_started_at"] == 24
    # Bytes reset by a retry, and back to the old mark: neither renews.
    assert observe((0, None), 28) is verdict.Progress.WAITING
    assert observe((24, None), 30) is verdict.Progress.WAITING
    # A counter that stops being reported keeps its mark and renews nothing.
    assert observe((None, None), 32) is verdict.Progress.WAITING
    assert watch["wait_marks"] == [24, None]
    assert observe((24, None), 35) is verdict.Progress.WAIT_EXPIRED
    # A counter first reported, an own-work pass in between, an unreadable pass.
    watch.clear()
    assert observe((1, None), 0) is verdict.Progress.WAITING
    assert observe((1, 3), 4) is verdict.Progress.WAITING
    verdict.progress_verdict(
        verdict.Signal((9,), None, (1, 3)), watch, window=10, now=6
    )
    verdict.progress_verdict(None, watch, window=10, now=8)
    assert observe((1, 3), 11) is verdict.Progress.WAIT_EXPIRED
    # The route publication wait has no counter: a new run or route state never renews.
    watch.clear()
    publication = SweepWait(reason=SweepWaitReason.ROUTE_PUBLICATION)
    for now, run in ((0, "r1"), (6, "r2"), (11, "r3")):
        progress = observe((), now, publication, (("n1", run, "running", True),))
    assert progress is verdict.Progress.WAIT_EXPIRED
    # An older watch (no marks, or another shape) is adopted: it records, renews nothing.
    for older in ({}, {"wait_marks": [5]}, {"wait_signature": [1]}):
        watch.clear()
        watch.update(older, wait_started_at=0)
        assert observe((7, 7), 5) is verdict.Progress.WAITING
        assert watch["wait_marks"] == [7, 7]
        assert observe((7, 7), 11) is verdict.Progress.WAIT_EXPIRED
