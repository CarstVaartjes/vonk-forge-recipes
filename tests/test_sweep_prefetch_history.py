"""Old download receipts are evidence, not reservations for absent models."""

from pathlib import Path

import pytest
from sweep_fakes import FakeModel, FakeRecipe, make_sweep

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
        prefetch=PrefetchConfig(),
    )
    sweep.refresh_catalog()
    old_recipe = sweep.recipes[old.key]
    sweep.state.downloads[old.key] = {
        "state": state,
        "done_at": clock.now(),
        "next_check": clock.now() + 1000,
    }
    # History stays durable, but a model no longer present or arriving holds
    # no eviction pin.
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
    pins = sweep.prefetcher.desired_pins(plans, sweep.recipes, sweep.models, set())
    assert old.key not in pins
    assert new.key in pins


@pytest.mark.parametrize("restart", [False, True])
def test_new_cache_observation_overrides_completed_download_hint(
    tmp_path: Path, restart: bool
) -> None:
    old = FakeRecipe("evicted", ("old",))
    new = FakeRecipe("new", ("new",))
    sweep, fleet, clock = make_sweep(
        tmp_path,
        [old, new],
        [FakeModel("old"), FakeModel("new")],
        prefetch=PrefetchConfig(),
    )
    sweep.refresh_catalog()
    sweep.prefetcher._request(sweep.recipes[old.key], "model")
    clock.sleep(fleet.download_seconds + 1)
    sweep.prefetcher._poll()
    # A completion newer than the cached listing remains useful immediately.
    assert "old" in sweep._cached_models()
    fleet.models["old"].local = "not_cached"
    fleet.recipes[old.key].local = "not_cached"
    clock.sleep(1)
    sweep.state.save()
    if restart:
        sweep, _, _ = make_sweep(
            tmp_path, [old, new], [], fleet=fleet, prefetch=sweep.prefetcher.config
        )
    sweep.refresh_catalog()
    cached = sweep._cached_models()
    assert "old" not in cached
    sweep.cached_models = cached
    assert not sweep.ready(sweep.recipes[old.key])
    sweep.prefetcher.tick(
        sweep.recipes,
        sweep.models,
        list(sweep.recipes.values()),
        sweep.sizes,
        cached,
        sweep.boost,
        cached,
    )
    calls = [c for _, c in fleet.calls if c[:2] == ("recipe", "download")]
    assert any(c[2] == new.key for c in calls)
