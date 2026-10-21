"""Owner observation outages expire locally; pauses never retain finished lanes."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import pytest
from sweep_fakes import FakeModel, FakeRecipe, make_sweep
from vonk_forge_contracts.sweep import (
    InfrastructureIncident,
    SweepWaitCode,
    SweepWaitExpiry,
    SweepWaitReason,
)

from spark_sweep.lifecycle import LifecycleState
from spark_sweep.owner import OwnerGuard
from spark_sweep.state import State
from spark_sweep.vonkctl import VonkctlTimeout


def unreadable_owner(sweep, monkeypatch, *, timeout: bool = False) -> list[int]:
    runner = sweep.vk.runner
    reads: list[int] = []

    def run(argv: Sequence[str], limit: float) -> tuple[int, str, str]:
        if "--profile" in argv and tuple(argv[-2:]) == ("profile", "progress"):
            profile = int(argv[argv.index("--profile") + 1])
            if profile in (2, 3):
                reads.append(profile)
                if timeout:
                    raise VonkctlTimeout(argv, None, "injected Controller restart")
                return 2, "", "HTTP 502: injected Controller restart"
        return runner(argv, limit)

    monkeypatch.setattr(sweep.vk, "runner", run)
    return reads


@pytest.mark.parametrize("timeout", [False, True])
def test_two_hour_owner_outage_does_not_stop_scheduling(
    tmp_path: Path, monkeypatch, timeout: bool
) -> None:
    recipe = FakeRecipe("a", local="cached", blocked=True)
    sweep, fleet, clock = make_sweep(
        tmp_path, [recipe], [FakeModel("m1", local="cached")]
    )
    reads = unreadable_owner(sweep, monkeypatch, timeout=timeout)
    sweep.preflight()
    sweep.refresh_catalog()
    start = clock.now()
    sweep._tick()
    assert sweep.paused and not sweep.state.slots
    initial_reason = sweep.pause_reason
    # Watch history survives a restart; transport failures cannot renew it.
    sweep.state.save()
    sweep.state = State.load(sweep.state.path, clock.now)
    sweep.guard = OwnerGuard(sweep.vk, sweep.state, sweep.cfg.owner_hold_seconds)
    for _ in range(120):
        clock.sleep(60)
        sweep._tick()
        if clock.now() - start >= 180:
            assert not sweep.paused
    assert "until" in initial_reason
    assert clock.now() - start == 7200
    assert reads.count(2) > 120 and reads.count(3) > 120
    submissions = [
        at
        for (_, args), at in zip(fleet.calls, fleet.call_times)
        if args[:2] == ("profile", "load") and "--yes" in args
    ]
    assert submissions and min(submissions) <= start + 180
    assert max(submissions) > start + 3600
    for number in (2, 3):
        watch = sweep.guard.state.data["owner"]["unreadable"][str(number)]
        expiry = SweepWaitExpiry.model_validate(watch["expiry"])
        assert expiry.code is SweepWaitCode.BUDGET_EXHAUSTED
        assert expiry.wait.reason is SweepWaitReason.OBSERVATION
        assert expiry.started_at == start
        assert expiry.deadline < start + 180
        incident = InfrastructureIncident.model_validate(
            sweep.guard.state.data["infra"][f"owner-profile-{number}"]
        )
        assert SweepWaitCode.BUDGET_EXHAUSTED in incident.message


def test_over_budget_lane_released_and_download_observed_during_pause(
    tmp_path: Path, monkeypatch
) -> None:
    held = FakeRecipe("held", local="cached", blocked=True)
    downloading = FakeRecipe("downloading", ("m2",))
    sweep, fleet, clock = make_sweep(
        tmp_path,
        [held, downloading],
        [FakeModel("m1", local="cached"), FakeModel("m2")],
    )
    sweep.preflight()
    sweep.refresh_catalog()
    sweep._tick()
    slot = sweep.state.slots[held.key]
    unreadable_owner(sweep, monkeypatch)
    clock.sleep(slot["deadline"] - slot["started_at"] + 1)
    sweep._tick()
    assert sweep.paused
    assert held.key not in sweep.state.slots
    row = next(
        r
        for r in sweep.log.entries()
        if r.get("code") == SweepWaitCode.BUDGET_EXHAUSTED
    )
    assert row["evidence"]["observed_at"] > row["evidence"]["deadline"]
    assert sweep.state.entry(held.key)["not_before"] > clock.now()
    assert downloading.key in sweep.state.downloads
    assert fleet.models["m2"].local == "cached"


def test_readable_active_owner_still_pauses_after_observation_budget(
    tmp_path: Path,
) -> None:
    sweep, fleet, clock = make_sweep(
        tmp_path,
        [FakeRecipe("pending", local="cached")],
        [FakeModel("m1", local="cached")],
    )
    fleet.owner_load(2, 3 * 3600)
    sweep.preflight()
    for _ in range(120):
        clock.sleep(60)
        sweep._tick()
        assert sweep.paused
        assert not sweep.state.slots
    assert sweep.state.data["owner"]["baseline"]["2"]["state"] == LifecycleState.RUNNING


def test_profile_budgets_are_independent_and_recovery_clears_the_incident(
    tmp_path: Path, monkeypatch
) -> None:
    sweep, _, clock = make_sweep(tmp_path, [], [])
    start = clock.now()
    guard = OwnerGuard(sweep.vk, sweep.state, 1800, profiles=[2])
    with monkeypatch.context() as faults:
        unreadable_owner(sweep, faults)
        assert guard.check(start).paused
        assert not guard.check(start + 121).paused
        guard.profiles.append(3)
        assert guard.check(start + 121).paused
        assert not guard.check(start + 242).paused
        first = sweep.state.data["owner"]["unreadable"]["2"]["expiry"]
        second = sweep.state.data["owner"]["unreadable"]["3"]["expiry"]
        assert first["started_at"] == start
        assert second["started_at"] == start + 121
    # The real fake CLI returns declared 404s for these idle drafts.
    assert not guard.check(start + 243).paused
    assert not sweep.state.data["owner"]["unreadable"]
    assert not sweep.state.data["infra"]
    with monkeypatch.context() as faults:
        unreadable_owner(sweep, faults, timeout=True)
        assert guard.check(start + 244).paused
        assert (
            sweep.state.data["owner"]["unreadable"]["2"]["wait_started_at"]
            == start + 244
        )
