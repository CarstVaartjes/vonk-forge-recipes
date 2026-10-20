"""The sweep follows the Controller's verdicts: waits never expire, only a stall ends work.

Unit cases drive ``progress_verdict`` and ``disposition`` on vonkctl JSON in the published
shapes (``tests/fixtures/sweep-verdicts``: ``FleetProfileApplicationView`` and the CLI's
error document); behaviour cases run the whole sweep on a fake clock.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from sweep_fakes import FakeModel, FakeRecipe, Gateway, make_sweep

from spark_sweep import policy, verdict
from spark_sweep.policy import TimeoutPolicy
from spark_sweep.verdict import Disposition, Progress
from spark_sweep.vonkctl import Reply

FIXTURES = Path(__file__).parent / "fixtures" / "sweep-verdicts"
HOUR = 3600.0


def _document(name: str) -> dict[str, Any]:
    return json.loads((FIXTURES / f"{name}.json").read_text())


@pytest.fixture
def gateway():
    gateway = Gateway()
    yield gateway
    gateway.stop()


def _timeouts(seconds: float = 600.0) -> TimeoutPolicy:
    return TimeoutPolicy(
        first_start_seconds=seconds, floor_seconds=seconds, cap_seconds=seconds
    )


def _drive(sweep, clock, *, until, limit: int = 2000) -> None:
    """Tick on the fake clock until ``until()`` holds (a bounded number of steps)."""
    for _ in range(limit):
        sweep.tick()
        if until():
            return
        clock.sleep(sweep.cfg.poll_seconds)
    raise AssertionError("the scenario did not settle")


# -- progress_verdict on Controller documents --------------------------------------------


@pytest.mark.parametrize(
    "name", ["queued-retry", "backoff", "admission-pending", "needs-operator"]
)
def test_a_controller_owned_wait_retains_age_and_reconciliation(name: str) -> None:
    watch: dict[str, Any] = {}
    signal = verdict.application_signal(_document(name))
    assert signal.waiting_on
    for hour in range(48):  # two days of the same document: past every old cap
        progress = verdict.progress_verdict(signal, watch, window=600, now=hour * HOUR)
        assert progress is Progress.WAITING
    assert watch["waiting_on"] == signal.waiting_on
    assert watch["progressed_at"] == watch["wait"]["first_seen_at"] == 0
    assert watch["wait"]["next_check_at"] > 47 * HOUR
    assert watch["wait"]["next_check"] == "reconcile-request"


@pytest.mark.parametrize("name", ["running-silent", "phase-retry"])
def test_the_works_own_silence_for_a_window_is_stalled(name: str) -> None:
    watch: dict[str, Any] = {}
    signal = verdict.application_signal(_document(name))
    assert signal.waiting_on is None  # a retried phase is the load's own work
    assert verdict.progress_verdict(signal, watch, window=600, now=0) is (
        Progress.PROGRESSING
    )
    assert verdict.progress_verdict(signal, watch, window=600, now=600) is (
        Progress.PROGRESSING
    )
    assert verdict.progress_verdict(signal, watch, window=600, now=601) is (
        Progress.STALLED
    )


def test_a_warning_blocker_is_a_brief_hold_not_a_wait() -> None:
    document = _document("running-silent")
    document["progress"]["blockers"] = [
        {"code": "run-switch.something", "detail": "brief", "severity": "warning"}
    ]
    assert verdict.application_signal(document).waiting_on is None


def test_movement_restarts_the_window_and_a_wait_never_ages_it() -> None:
    watch: dict[str, Any] = {}
    silent = verdict.application_signal(_document("running-silent"))
    moved = verdict.application_signal(_document("running-moved"))
    queued = verdict.application_signal(_document("queued-retry"))
    verdict.progress_verdict(silent, watch, window=600, now=0)
    assert verdict.progress_verdict(moved, watch, window=600, now=500) is (
        Progress.PROGRESSING
    )
    # Ten hours queued, then the same silent work again: the window starts afresh.
    assert verdict.progress_verdict(queued, watch, window=600, now=10 * HOUR) is (
        Progress.WAITING
    )
    assert verdict.progress_verdict(moved, watch, window=600, now=10 * HOUR + 599) is (
        Progress.PROGRESSING
    )


def test_an_unreadable_observation_is_unknown_and_waits() -> None:
    watch: dict[str, Any] = {"signature": [1], "progressed_at": 0.0}
    assert verdict.progress_verdict(None, watch, window=600, now=5000) is (
        Progress.WAITING
    )
    assert watch["progressed_at"] == 0
    assert watch["wait"]["last_progress_at"] == 0
    assert watch["wait"]["next_check_at"] > 5000


def test_the_controllers_typed_end_stops_following_a_failed_application() -> None:
    assert verdict.retry_ended(_document("failed-repeated"))
    assert not verdict.retry_ended({"state": "failed", "progress": {}})


# -- disposition ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("cli-transport-timeout", Disposition.WAIT),
        ("cli-busy-429", Disposition.WAIT),
        ("cli-own-field-refusal", Disposition.WAIT),  # the sweep's own field
        ("cli-recipe-refusal", Disposition.FAIL),  # the recipe's own data
    ],
)
def test_cli_errors_are_decided_by_their_published_fields(
    name: str, expected: Disposition
) -> None:
    reply = Reply(("vonkctl",), 2, _document(name), "")
    assert verdict.reply_disposition(reply) is expected


@pytest.mark.parametrize(
    ("failure", "expected"),
    [
        (policy.classify("smoke", "case.M0", "HTTP 503", verdict.SMOKE_BUSY), "wait"),
        (policy.classify("smoke", "case.M0", "401", policy.CLIENT_CREDENTIAL), "wait"),
        (policy.classify("download", "storage.insufficient_capacity", "full"), "wait"),
        (
            policy.classify("smoke", "case.M0", "socket timed out", "smoke-timeout"),
            "observe",
        ),
        (policy.classify("install", "copy.stalled", "x", "copy-stalled"), "observe"),
        (
            policy.classify("review", "run-switch.resource.insufficient_capacity", "x"),
            "observe",
        ),
        (policy.classify("start", "recipe.runtime_oom", "x"), "fail"),
        (
            policy.classify("start", "recipe.runtime_oom", "x", verdict.UNATTRIBUTED),
            "observe",
        ),
        (policy.classify("smoke", "case.M0", "wrong", "smoke-assertion"), "fail"),
        (policy.classify("timeout", "load.timeout", "silent"), "fail"),
    ],
)
def test_outcomes_spend_attempts_only_when_uncertain(
    failure: policy.Failure, expected: str
) -> None:
    assert verdict.disposition(failure).value == expected


# -- behaviour on a fake clock -------------------------------------------------------------


def test_a_lane_held_by_the_controller_past_every_old_cap_waits_then_passes(
    tmp_path: Path, gateway: Gateway
) -> None:
    # 30 hours of an admission hold: past the 900 s blocked cap, the 1800 s platform
    # wait, the 7200 s longest load window and the 24 h load wall.
    held = FakeRecipe("big", blocked=True, blocked_for=30 * HOUR)
    sweep, fleet, clock = make_sweep(
        tmp_path,
        [held],
        [FakeModel("m1")],
        gateway=gateway,
        timeouts=_timeouts(600),
        poll_seconds=300.0,
    )
    start = clock.now()
    seen_waiting = []

    def observe(_now: float) -> None:
        slot = sweep.state.slots.get(held.key)
        if slot and slot.get("watch", {}).get("waiting_on"):
            seen_waiting.append(clock.now() - start)

    clock.hooks.append(observe)
    _drive(sweep, clock, until=lambda: sweep.state.status(held.key) == "passed")
    assert max(seen_waiting) > 29 * HOUR
    entry = sweep.state.entry(held.key)
    assert entry["attempts"] == 1 and not entry.get("failure_class")
    assert not any(c[:2] == ("profile", "cancel") for _, c in fleet.calls)


def test_a_lane_whose_own_progress_is_silent_ends_stalled_with_evidence(
    tmp_path: Path, gateway: Gateway
) -> None:
    silent = FakeRecipe("silent", load_seconds=10**7)
    sweep, fleet, clock = make_sweep(
        tmp_path,
        [silent],
        [FakeModel("m1")],
        gateway=gateway,
        timeouts=_timeouts(600),
    )
    _drive(sweep, clock, until=lambda: sweep.state.status(silent.key) == "deferred")
    entry = sweep.state.entry(silent.key)
    assert entry["failure_class"] == "timeout"
    assert "no progress for" in entry["error"]
    row = next(
        r
        for r in sweep.log.entries()
        if r["recipe"] == silent.key and r["status"] == "deferred"
    )
    assert row["controller_document"]["state"] == "running"  # the document it ended on
    assert any(c[:2] == ("profile", "cancel") for _, c in fleet.calls)


def test_a_busy_smoke_spends_no_attempt(tmp_path: Path, gateway: Gateway) -> None:
    gateway.busy = 3  # 503 three times, then the engine answers
    sweep, _, clock = make_sweep(
        tmp_path, [FakeRecipe("a")], [FakeModel("m1")], gateway=gateway
    )
    _drive(sweep, clock, until=lambda: sweep.state.status("vonk-forge/a") == "passed")
    entry = sweep.state.entry("vonk-forge/a")
    assert entry["attempts"] == 1  # only the pass
    waits = [
        r
        for r in sweep.log.entries()
        if r["step"] == "smoke" and r["status"] == "waiting"
    ]
    assert waits and all(r["failure_class"] == verdict.SMOKE_BUSY for r in waits)
    assert waits[0]["evidence"]["http_status"] == 503


def test_a_refused_load_is_recorded_never_resubmitted_identically(
    tmp_path: Path, gateway: Gateway
) -> None:
    sweep, fleet, clock = make_sweep(
        tmp_path, [FakeRecipe("a")], [FakeModel("m1")], gateway=gateway
    )
    real = fleet.runner
    refusal = _document("cli-recipe-refusal")

    def runner(argv, timeout):
        args = list(argv)
        placement = (
            "load" in args
            and "--request-key" in args
            and "--review" not in args
            and args[args.index("--profile") + 1] == str(sweep.cfg.sweep_profile)
        )
        if placement:
            fleet.calls.append((sweep.cfg.sweep_profile, ("profile", "load")))
            return 2, json.dumps(refusal), ""
        return real(argv, timeout)

    sweep.vk.runner = runner
    _drive(sweep, clock, until=lambda: sweep.state.status("vonk-forge/a") == "failed")
    for _ in range(30):  # the sweep keeps running: nothing places the same plan again
        sweep.tick()
        clock.sleep(sweep.cfg.poll_seconds)
    submits = [c for p, c in fleet.calls if c == ("profile", "load")]
    assert len(submits) == 1
    entry = sweep.state.entry("vonk-forge/a")
    assert entry["code"] == "recipe.invalid_option"
    assert "option_choices" in entry["evidence"]["reply"]


def test_every_requeue_leaves_a_results_row(tmp_path: Path, gateway: Gateway) -> None:
    recipes = [FakeRecipe("a", load_seconds=600), FakeRecipe("b", load_seconds=600)]
    sweep, fleet, clock = make_sweep(
        tmp_path, recipes, [FakeModel("m1")], gateway=gateway
    )

    def owner_takes_the_fleet(_now: float) -> None:
        if sweep.state.slots and not fleet.profiles.get(2, {}).get("latest"):
            fleet.owner_load(2, duration=300)

    clock.hooks.append(owner_takes_the_fleet)
    _drive(
        sweep,
        clock,
        until=lambda: any(r["step"] == "requeue" for r in sweep.log.entries()),
    )
    requeues = [r for r in sweep.log.entries() if r["step"] == "requeue"]
    assert requeues
    for row in requeues:
        assert row["reason"] and row["status"] == "requeued"
        assert row["controller_document"]["id"]  # the application it was last seen in
