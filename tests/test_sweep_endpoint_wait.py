"""A route the platform is still publishing is waited for, not failed."""

from __future__ import annotations

from pathlib import Path

from sweep_fakes import FakeModel, FakeRecipe, make_sweep


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


def test_an_endpoint_the_controller_has_not_published_is_waited_for_not_failed(
    tmp_path: Path,
) -> None:
    sweep, fleet, clock, key, slot = _smoking(tmp_path)
    clock.t += 24 * 3600.0  # past the old 1800 s publication cap
    fleet.observations.append((("profile", "endpoint"), 0, _unpublished(slot)))
    sweep._submit_smoke(key, slot)
    assert key not in sweep.futures
    assert slot["phase"] == "finished"
    assert not sweep.state.entry(key).get("attempts")
    assert sweep.state.entry(key)["not_before"] > clock.now()
    assert "unavailable" in sweep.state.entry(key)["wait_deferral"]["wait"]["detail"]
