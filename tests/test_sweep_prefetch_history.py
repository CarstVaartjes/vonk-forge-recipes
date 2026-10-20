"""Old download receipts are evidence, not reservations for absent models."""

from pathlib import Path

import pytest
from sweep_fakes import GB, FakeModel, FakeRecipe, make_sweep

from spark_sweep.lifecycle import LifecycleState
from spark_sweep.prefetch import PrefetchConfig


@pytest.mark.parametrize(
    "state",
    [LifecycleState.SUCCEEDED, LifecycleState.FAILED, LifecycleState.CANCELLED],
)
def test_evicted_download_history_cannot_starve_new_models(
    tmp_path: Path, state: LifecycleState
) -> None:
    old = FakeRecipe("evicted", ("old",))
    new = FakeRecipe("new", ("new",))
    sweep, fleet, clock = make_sweep(
        tmp_path,
        [old, new],
        [FakeModel("old"), FakeModel("new")],
        prefetch=PrefetchConfig(nas_budget_bytes=30 * GB),
    )
    sweep.refresh_catalog()
    old_recipe = sweep.recipes[old.key]
    sweep.state.downloads[old.key] = {
        "state": state,
        "done_at": clock.now(),
        "next_check": clock.now() + 1000,
    }
    # History stays durable, but a model no longer present or arriving consumes
    # neither the byte budget nor an eviction pin.
    sweep.prefetcher.tick(
        sweep.recipes,
        sweep.models,
        [old_recipe, sweep.recipes[new.key]],
        sweep.sizes,
        set(),
        sweep.boost,
    )
    calls = [c for _, c in fleet.calls if c[:2] == ("recipe", "download")]
    assert any(c[2] == new.key for c in calls)
    assert sweep.state.downloads[old.key]["state"] == state
    plans = sweep.prefetcher.last_plans
    pins = sweep.prefetcher.desired_pins(
        plans, sweep.recipes, sweep.models, sweep.sizes, set()
    )
    assert old.key not in pins
    assert new.key in pins


def test_inflight_models_still_consume_the_budget(tmp_path: Path) -> None:
    old = FakeRecipe("arriving", ("old",))
    new = FakeRecipe("new", ("new",))
    sweep, fleet, _clock = make_sweep(
        tmp_path,
        [old, new],
        [FakeModel("old"), FakeModel("new")],
        prefetch=PrefetchConfig(nas_budget_bytes=30 * GB),
    )
    sweep.refresh_catalog()
    sweep.prefetcher._request(sweep.recipes[old.key], "model")
    count = len([c for _, c in fleet.calls if c[:2] == ("recipe", "download")])
    sweep.prefetcher.tick(
        sweep.recipes,
        sweep.models,
        list(sweep.recipes.values()),
        sweep.sizes,
        set(),
        sweep.boost,
    )
    assert len([c for _, c in fleet.calls if c[:2] == ("recipe", "download")]) == count
