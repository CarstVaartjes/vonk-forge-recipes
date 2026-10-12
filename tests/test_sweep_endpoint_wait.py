"""A route the platform is still publishing is waited for, not failed."""

from __future__ import annotations

from pathlib import Path

from sweep_fakes import FakeModel, FakeRecipe, make_sweep

from spark_sweep.run import SMOKE_WALL_SECONDS


def _smoking(tmp_path: Path):
    sweep, fleet, clock = make_sweep(
        tmp_path,
        [FakeRecipe("a", local="cached", copy_seconds=9000)],
        [FakeModel("m1", local="cached")],
    )
    sweep.preflight()
    sweep.refresh_catalog()
    sweep.queue = list(sweep.recipes)
    sweep.cached_models = sweep._cached_models()
    sweep.schedule(clock.now())
    key = next(iter(sweep.state.slots))
    slot = sweep.state.slots[key]
    slot.update(phase="smoking", run_id="run-1", ready_at=clock.now())
    return sweep, fleet, clock, key, slot


def _unpublished(slot) -> dict:
    return {"assignments": [{"alias": slot["alias"], "state": "unavailable"}]}


def test_endpoint_published_later_proceeds_to_smoke(tmp_path: Path) -> None:
    sweep, fleet, clock, key, slot = _smoking(tmp_path)
    fleet.observations.append((("profile", "endpoint"), 0, _unpublished(slot)))
    sweep._submit_smoke(key, slot)
    assert key not in sweep.futures
    assert slot["phase"] == "smoking"
    assert slot["smoke_not_before"] > clock.now()
    sweep._advance_smoking(key, slot)  # too early: not asked again
    assert key not in sweep.futures
    clock.t = slot["smoke_not_before"]
    sweep._advance_smoking(key, slot)
    assert key in sweep.futures
    assert slot["phase"] == "smoking"


def test_endpoint_never_published_fails_with_state_after_budget(
    tmp_path: Path,
) -> None:
    sweep, fleet, clock, key, slot = _smoking(tmp_path)
    clock.t += SMOKE_WALL_SECONDS
    fleet.observations.append((("profile", "endpoint"), 0, _unpublished(slot)))
    sweep._submit_smoke(key, slot)
    assert key not in sweep.futures
    assert slot["phase"] == "finished"
    assert sweep.state.entry(key)["attempts"] == 1
    assert "endpoint.unpublished" in str(sweep.state.entry(key))
    assert "unavailable" in str(sweep.state.entry(key))
