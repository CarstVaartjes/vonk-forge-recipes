"""Queue priority and independent Spark lanes, using only the fake clock."""

from pathlib import Path

from sweep_fakes import GB, FakeModel, FakeRecipe, make_sweep

from spark_sweep.prefetch import PrefetchConfig


def test_model_cached_image_missing_groups_do_not_starve_model_downloads(
    tmp_path: Path,
) -> None:
    window = 16
    warm = [FakeRecipe(f"warm-{i}", (f"m{i}",)) for i in range(window + 1)]
    cold = FakeRecipe("cold", ("missing",))
    sweep, fleet, clock = make_sweep(
        tmp_path,
        warm + [cold],
        [FakeModel(f"m{i}", local="cached") for i in range(window + 1)]
        + [FakeModel("missing")],
        prefetch=PrefetchConfig(group_window=window, pin_profile=None),
    )
    sweep.preflight()
    sweep.start_takeover()
    sweep.refresh_catalog()
    # Models are cached; recipes are not cached because their images are missing.
    assert all(sweep.recipes[r.key].local == "not_cached" for r in warm)
    sweep.tick()
    plans = sweep.prefetcher.last_plans
    assert all(p.new_bytes == 0 for p in plans[: window + 1])
    assert cold.key in [c[2] for _, c in fleet.calls if c[:2] == ("recipe", "download")]
    assert sweep.state.downloads[cold.key]["kind"] == "model"
    assert not sweep.state.slots
    assert not any(e.get("status") == "passed" for e in sweep.state.recipes.values())
    assert clock.sleeps == 0


def test_cached_singles_run_together_before_cached_duals_and_downloads(
    tmp_path: Path,
) -> None:
    recipes = [
        FakeRecipe("large", ("large",), local="cached", memory=100 * GB),
        FakeRecipe("small", ("small",), local="cached", memory=30 * GB),
        FakeRecipe("dual", ("dual",), local="cached", node_count=2),
        FakeRecipe("cold", ("cold",)),
        FakeRecipe("giant", ("giant",)),
    ]
    sweep, fleet, clock = make_sweep(
        tmp_path,
        recipes,
        [
            FakeModel("large", bytes=100 * GB, local="cached"),
            FakeModel("small", bytes=GB, local="cached"),
            FakeModel("dual", bytes=GB, local="cached"),
            FakeModel("cold", bytes=GB),
            FakeModel("giant", bytes=1000 * GB),
        ],
    )
    sweep.preflight()
    sweep.start_takeover()
    sweep.refresh_catalog()
    sweep.tick()
    assert sweep.queue == [
        "vonk-forge/small",
        "vonk-forge/large",
        "vonk-forge/dual",
        "vonk-forge/cold",
        "vonk-forge/giant",
    ]
    slots = sweep.state.slots
    assert set(slots) == {"vonk-forge/small", "vonk-forge/large"}
    assert {tuple(slot["node_ids"]) for slot in slots.values()} == {
        ("spk_a",),
        ("spk_b",),
    }
    assert {slot["started_at"] for slot in slots.values()} == {clock.now()}
    assert {run["recipe"] for run in fleet.runs} == set(slots)
    assert clock.sleeps == 0


def test_dual_waits_for_both_lanes_without_refilling_a_free_lane(
    tmp_path: Path,
) -> None:
    recipes = [
        FakeRecipe("a", ("a",), local="cached"),
        FakeRecipe("b", ("b",), local="cached"),
        FakeRecipe("dual", ("dual",), local="cached", node_count=2),
        FakeRecipe("later", ("later",)),
    ]
    sweep, fleet, clock = make_sweep(
        tmp_path,
        recipes,
        [FakeModel(r.digests[0], local="cached") for r in recipes],
    )
    sweep.preflight()
    sweep.start_takeover()
    sweep.refresh_catalog()
    sweep.tick()
    assert set(sweep.state.slots) == {recipes[0].key, recipes[1].key}
    # A later single becomes ready while the two initial lanes drain.
    fleet.recipes[recipes[3].key].local = "cached"
    sweep.refresh_catalog()
    sweep.queue = [recipes[2].key, recipes[3].key]
    sweep.state.slots[recipes[0].key]["phase"] = "finished"
    sweep.enqueue_finished_cleanup()
    sweep.cleanup_finished()
    sweep.observe_cleanup()
    sweep.schedule(clock.now())
    assert recipes[2].key not in sweep.state.slots
    assert recipes[3].key not in sweep.state.slots
    assert sweep.state.data["mode"] == "single"
    sweep.state.slots[recipes[1].key]["phase"] = "finished"
    sweep.enqueue_finished_cleanup()
    sweep.cleanup_finished()
    sweep.observe_cleanup()
    sweep.schedule(clock.now())
    assert set(sweep.state.slots) == {recipes[2].key}
    assert set(sweep.state.slots[recipes[2].key]["node_ids"]) == {"spk_a", "spk_b"}
    assert sweep.state.data["mode"] == "dual"
    assert clock.sleeps == 0
