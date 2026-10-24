"""Work that moves under a Controller wait is progress; a wait without movement still ends.

Replays the 2026-10-24 incident: a lane held by the Controller's info ``receipt-missing``
wait (its run-switch child pending) while the distribution kept moving bytes. The lane's
window is measured from the last movement, never as a fixed budget from slot open.
"""

from __future__ import annotations

import json
from copy import deepcopy
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


@pytest.fixture
def gateway():
    gateway = Gateway()
    yield gateway
    gateway.stop()


def _timeouts() -> TimeoutPolicy:
    return TimeoutPolicy(
        first_start_seconds=WINDOW, floor_seconds=WINDOW, cap_seconds=WINDOW
    )


def _held_by_receipt_wait(fleet, monkeypatch) -> None:
    """The Controller declares its wait on every running application, as recorded."""
    original = fleet._app_doc

    def held(app):
        document = original(app)
        if document["state"] == LifecycleState.RUNNING:
            document["wait"] = deepcopy(RECEIPT_WAIT)
        return document

    monkeypatch.setattr(fleet, "_app_doc", held)


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


def test_advancing_signature_renews_a_wait_but_reasons_and_flaps_do_not() -> None:
    receipt = SweepWait(reason=SweepWaitReason.CONTROLLER_RETRY)
    watch: dict = {}

    def observe(signature, now, wait=receipt):
        return verdict.progress_verdict(
            verdict.Signal(signature, wait), watch, window=10, now=now
        )

    assert observe((1,), 0) is verdict.Progress.WAITING
    # A changed reason with the same signature is no progress: the budget holds.
    assert (
        observe((1,), 11, SweepWait(reason=SweepWaitReason.OBSERVATION))
        is verdict.Progress.WAIT_EXPIRED
    )
    watch.clear()
    assert observe((1,), 0) is verdict.Progress.WAITING
    for now in (8, 16, 24):  # bytes keep moving past the window
        assert observe((now,), now) is verdict.Progress.WAITING
    assert watch["wait_started_at"] == 24
    assert observe((24,), 34) is verdict.Progress.WAITING
    assert observe((24,), 35) is verdict.Progress.WAIT_EXPIRED
    # A signature that returns between waits without moving is no progress: an
    # own-work observation in between does not renew the wait either.
    watch.clear()
    assert observe((1,), 0) is verdict.Progress.WAITING
    verdict.progress_verdict(verdict.Signal((2,)), watch, window=10, now=4)
    assert observe((1,), 8) is verdict.Progress.WAITING
    assert observe((1,), 11) is verdict.Progress.WAIT_EXPIRED
    # Unreadable observations neither renew nor forget the last seen movement.
    watch.clear()
    assert observe((1,), 0) is verdict.Progress.WAITING
    verdict.progress_verdict(None, watch, window=10, now=5)
    assert observe((1,), 11) is verdict.Progress.WAIT_EXPIRED
