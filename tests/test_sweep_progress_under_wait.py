"""Work that moves forward under a Controller wait is progress; anything else still ends.

Replays the 2026-10-24 incident: a lane held by the Controller's info ``receipt-missing``
wait (its run-switch child pending) while the distribution kept moving bytes. The lane's
window is measured from the last time a forward-only counter rose (the Controller's
max-merged child bytes, completed steps, items), never as a fixed budget from slot open,
and never renewed by a flapping phase, a retry that re-copies the same bytes or a restart.
"""

from __future__ import annotations

import json
from copy import deepcopy
from datetime import datetime
from itertools import pairwise
from pathlib import Path

import pytest
from sweep_fakes import GB, FakeModel, FakeRecipe, Gateway, make_sweep
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


def test_a_wait_without_movement_expires_one_window_after_the_last_bytes(
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
    sweep.preflight()
    for _ in range(600):
        sweep.tick()
        if sweep.state.status(good.key) == "passed":
            break
        clock.sleep(sweep.cfg.poll_seconds)
    else:
        raise AssertionError("the next recipe was never placed after the wait expired")
    first_load = _loads(fleet)[0]
    (expired,) = [
        row
        for row in _rows(tmp_path)
        if row.get("recipe") == stuck.key
        and row.get("code") == SweepWaitCode.BUDGET_EXHAUSTED
    ]
    evidence = expired["evidence"]
    assert evidence["wait"]["reason"] == SweepWaitReason.CONTROLLER_WAIT
    assert evidence["wait"]["controller"]["reason"] == RECEIPT_WAIT["reason"]
    # Measured from the last movement (the copy froze 2 windows in), not from slot open.
    assert evidence["started_at"] >= first_load + 2 * WINDOW - sweep.cfg.poll_seconds
    assert evidence["observed_at"] - evidence["started_at"] > WINDOW
    assert sweep.state.entry(stuck.key).get("attempts", 0) == 0
    # The freed lane goes to the next recipe in the same pass.
    assert _loads(fleet)[-1] >= evidence["observed_at"]


def test_a_retried_copy_expires_one_window_after_its_bytes_last_rose(
    tmp_path: Path, gateway: Gateway, monkeypatch
) -> None:
    # Every attempt fails at 20 % (2 windows in) and the retry restarts from 0: the
    # operation snapshot resets and moves again, the max-merged bytes stay at 20 %.
    retried = FakeRecipe(
        "retried",
        content="a" * 64,
        digests=(MODEL,),
        copy_seconds=10 * WINDOW,
        copy_retries=True,
    )
    sweep, fleet, clock = make_sweep(
        tmp_path, [retried], [FakeModel(MODEL)], gateway=gateway, timeouts=_timeouts()
    )
    shown = _held_by_receipt_wait(fleet, monkeypatch)
    sweep.preflight()
    _until(sweep, clock, lambda: _expired(tmp_path, retried.key))
    attempts = [child["operation"]["completed_bytes"] for child in shown]
    assert any(b < a for a, b in pairwise(attempts))
    assert [child["bytes"] for child in shown] == sorted(c["bytes"] for c in shown)
    (evidence,) = _expired(tmp_path, retried.key)
    first_load = _loads(fleet)[0]
    poll = sweep.cfg.poll_seconds
    # The window runs from the last rise of the bytes, not from the resets after it.
    assert first_load + 2 * WINDOW - 2 * poll <= evidence["started_at"]
    assert evidence["started_at"] <= first_load + 2 * WINDOW + poll
    assert WINDOW < evidence["observed_at"] - evidence["started_at"] <= WINDOW + poll
    assert sweep.state.entry(retried.key).get("attempts", 0) == 0


def test_a_flapping_phase_without_bytes_expires_one_window_after_the_wait_began(
    tmp_path: Path, gateway: Gateway, monkeypatch
) -> None:
    # prepare <-> runtime-preflight with 0 bytes and a fresh progress time each flip.
    flapping = FakeRecipe(
        "flapping",
        content="a" * 64,
        digests=(MODEL,),
        load_seconds=10 * WINDOW,
        phase_flap=True,
    )
    sweep, fleet, clock = make_sweep(
        tmp_path, [flapping], [FakeModel(MODEL)], gateway=gateway, timeouts=_timeouts()
    )
    shown = _held_by_receipt_wait(fleet, monkeypatch)
    sweep.preflight()
    _until(sweep, clock, lambda: _expired(tmp_path, flapping.key))
    assert {child["operation"]["phase"] for child in shown} == {
        "prepare",
        "runtime-preflight",
    }
    assert len({child["operation"]["last_progress_at"] for child in shown}) > 2
    (evidence,) = _expired(tmp_path, flapping.key)
    first_load = _loads(fleet)[0]
    assert evidence["started_at"] <= first_load
    assert evidence["observed_at"] - first_load <= WINDOW + 2 * sweep.cfg.poll_seconds
    assert sweep.state.entry(flapping.key).get("attempts", 0) == 0


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


def test_the_controller_deadline_caps_a_renewing_wait(
    tmp_path: Path, gateway: Gateway, monkeypatch
) -> None:
    big = FakeRecipe(
        "big", content="a" * 64, digests=(MODEL,), copy_seconds=20 * WINDOW
    )
    sweep, fleet, clock = make_sweep(
        tmp_path, [big], [FakeModel(MODEL)], gateway=gateway, timeouts=_timeouts()
    )
    clock.t = datetime.fromisoformat(DEADLINE_RECORD["recorded_at"]).timestamp()
    wait = DEADLINE_RECORD["controller_document"]["wait"]
    deadline = datetime.fromisoformat(wait["dependency_deadline"]).timestamp()
    assert deadline - clock.now() > 2 * WINDOW  # the copy renews before the cap
    _held_by_receipt_wait(fleet, monkeypatch, wait)
    sweep.preflight()
    _until(sweep, clock, lambda: _expired(tmp_path, big.key))
    (evidence,) = _expired(tmp_path, big.key)
    assert evidence["started_at"] > _loads(fleet)[0] + WINDOW  # it did renew
    assert evidence["deadline"] == deadline
    assert deadline < evidence["observed_at"] <= deadline + sweep.cfg.poll_seconds


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
