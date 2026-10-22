"""Prefetch: bounded concurrency, in-flight downloads, eviction pins, measured throughput."""

from __future__ import annotations

from pathlib import Path

import pytest
from sweep_fakes import GB, FakeModel, FakeRecipe, Gateway, make_sweep

from spark_sweep.prefetch import PrefetchConfig


@pytest.fixture
def gateway():
    gateway = Gateway()
    yield gateway
    gateway.stop()


def _active_ops(fleet) -> int:
    return sum(1 for op in fleet.ops.values() if op["state"] == "running")


def _many(n: int) -> tuple[list[FakeRecipe], list[FakeModel]]:
    return [FakeRecipe(f"r{i}", (f"m{i}",)) for i in range(n)], [
        FakeModel(f"m{i}") for i in range(n)
    ]


def test_model_downloads_never_exceed_the_slot_budget(
    tmp_path: Path, gateway: Gateway
) -> None:
    recipes, models = _many(6)
    sweep, fleet, clock = make_sweep(
        tmp_path,
        recipes,
        models,
        gateway=gateway,
        prefetch=PrefetchConfig(max_model_downloads=2, pin_profile=13),
    )
    peak = []
    clock.hooks.append(lambda _t: peak.append(_active_ops(fleet)))
    sweep.run()
    assert max(peak) == 2  # saturated, never above the budget
    assert {e["status"] for e in sweep.state.recipes.values()} == {"passed"}


def test_preparing_siblings_share_one_model_download_slot(tmp_path: Path) -> None:
    """One active model with many preparing recipe views cannot exhaust the pool."""
    recipes = [FakeRecipe(f"shared-{i}", ("shared",)) for i in range(8)]
    recipes += [FakeRecipe(f"cold-{i}", (f"cold-{i}",)) for i in range(3)]
    models = [FakeModel("shared"), *(FakeModel(f"cold-{i}") for i in range(3))]
    sweep, fleet, _ = make_sweep(
        tmp_path,
        recipes,
        models,
        prefetch=PrefetchConfig(max_model_downloads=3, pin_profile=13),
    )
    sweep.preflight()
    sweep.refresh_catalog()
    sweep.prefetcher._request(sweep.recipes[recipes[0].key], "model")
    # All recipes sharing that model report preparing; only one parent exists.
    for recipe in recipes[:8]:
        recipe.local = "preparing"
    sweep.refresh_catalog()
    result = sweep.prefetcher.tick(
        sweep.recipes,
        sweep.models,
        list(sweep.recipes.values()),
        sweep.sizes,
        sweep._present(),
        sweep.boost,
    )
    assert result["model_ops"] == 3
    assert _active_ops(fleet) == 3
    assert sum(args[:2] == ("recipe", "download") for _, args in fleet.calls) == 3


def test_the_byte_budget_bounds_what_is_downloaded_but_not_yet_tested(
    tmp_path: Path, gateway: Gateway
) -> None:
    recipes, models = _many(4)  # 20 GB each
    sweep, fleet, clock = make_sweep(
        tmp_path,
        recipes,
        models,
        gateway=gateway,
        prefetch=PrefetchConfig(
            max_model_downloads=4, nas_budget_bytes=30 * GB, pin_profile=13
        ),
    )
    peak = []
    clock.hooks.append(lambda _t: peak.append(_active_ops(fleet)))
    sweep.run()
    assert max(peak) == 1
    assert {e["status"] for e in sweep.state.recipes.values()} == {"passed"}


def test_a_download_someone_else_started_is_not_requested_again(
    tmp_path: Path, gateway: Gateway
) -> None:
    recipes, models = _many(2)
    sweep, fleet, clock = make_sweep(tmp_path, recipes, models, gateway=gateway)
    # The operator already requested r0 (in flight); the library reports it as preparing.
    fleet.ops["op-ext"] = {
        "id": "op-ext",
        "recipe": "vonk-forge/r0",
        "state": "running",
        "started": clock.now(),
        "duration": 100.0,
        "request": "ext",
        "bytes": 20 * GB,
    }
    sweep.run()
    requested = [c[2] for _, c in fleet.calls if c[:2] == ("recipe", "download")]
    assert "vonk-forge/r0" not in requested and "vonk-forge/r1" in requested
    assert (
        sweep.state.recipes["vonk-forge/r0"]["status"] == "passed"
    )  # tested once it landed


def test_cached_recipes_are_tested_before_ones_that_need_a_download(
    tmp_path: Path, gateway: Gateway
) -> None:
    cold, warm = (
        FakeRecipe("cold", ("m1",)),
        FakeRecipe("warm", ("m2",), local="cached"),
    )
    sweep, fleet, _ = make_sweep(
        tmp_path,
        [cold, warm],
        [FakeModel("m1"), FakeModel("m2", local="cached")],
        gateway=gateway,
        sparks=("spark-a",),
    )
    sweep.run()
    adds = [c[2] for p, c in fleet.calls if p == 10 and c[:2] == ("profile", "add")]
    assert adds == ["vonk-forge/warm", "vonk-forge/cold"]


def test_seed_list_recipes_go_first(tmp_path: Path, gateway: Gateway) -> None:
    recipes, models = _many(4)
    sweep, fleet, _ = make_sweep(
        tmp_path,
        recipes,
        models,
        gateway=gateway,
        seed=frozenset({"r3"}),
        prefetch=PrefetchConfig(max_model_downloads=1, pin_profile=None),
    )
    sweep.run()
    downloads = [c[2] for _, c in fleet.calls if c[:2] == ("recipe", "download")]
    assert downloads[0] == "vonk-forge/r3"


def test_downloaded_recipes_are_pinned_until_tested_and_the_pin_profile_is_never_loaded(
    tmp_path: Path, gateway: Gateway
) -> None:
    recipes, models = _many(3)
    sweep, fleet, clock = make_sweep(tmp_path, recipes, models, gateway=gateway)
    pinned_during = []
    clock.hooks.append(
        lambda _t: pinned_during.append(
            list(fleet.profiles.get(13, {}).get("assignments", []))
        )
    )
    sweep.run()
    assert max(len(p) for p in pinned_during) >= 2
    assert {a["desired_state"] for p in pinned_during for a in p} == {
        "installed"
    }  # named, never started
    assert (
        fleet.profiles[13]["assignments"] == []
    )  # released when everything was tested
    assert not [c for p, c in fleet.calls if p == 13 and c[:2] == ("profile", "load")]
    assert sweep.state.data["pins"] == []


def test_pin_profile_trouble_does_not_stop_the_sweep(
    tmp_path: Path, gateway: Gateway
) -> None:
    recipes, models = _many(2)
    sweep, fleet, _ = make_sweep(tmp_path, recipes, models, gateway=gateway)
    fleet.reject_pins = True
    sweep.run()
    assert {e["status"] for e in sweep.state.recipes.values()} == {"passed"}
    assert "assignment limit" in (sweep.prefetcher.pin_error or "")
    attempts = [c for p, c in fleet.calls if p == 13 and c[:2] == ("profile", "add")]
    assert 1 <= len(attempts) <= 6  # backs off, never hammers, never gives up for good


def test_nas_full_pauses_new_model_downloads_and_is_a_wait(
    tmp_path: Path, gateway: Gateway
) -> None:
    full = FakeRecipe(
        "full",
        ("m1",),
        fail_download="insufficient free space on the NAS",
        fail_download_code="storage.insufficient_capacity",
    )
    sweep, _, clock = make_sweep(
        tmp_path,
        [full, FakeRecipe("next", ("m2",))],
        [FakeModel("m1"), FakeModel("m2")],
        gateway=gateway,
        prefetch=PrefetchConfig(max_model_downloads=1, pin_profile=None),
    )
    start = clock.now()

    def collector_frees_space(now: float) -> None:
        if now - start > 3600:
            full.fail_download = None

    clock.hooks.append(collector_frees_space)
    sweep.run()
    assert any("NAS pressure" in e["message"] for e in sweep.state.data["events"])
    # A full NAS is a wait, never a verdict: once there is room the recipe is tested.
    assert sweep.state.recipes["vonk-forge/full"]["status"] == "passed"
    assert not sweep.state.recipes["vonk-forge/full"].get("failure_class")
    assert sweep.state.recipes["vonk-forge/next"]["status"] == "passed"


def test_throughput_is_measured_from_the_operations(
    tmp_path: Path, gateway: Gateway
) -> None:
    recipes, models = _many(2)
    sweep, _, _ = make_sweep(tmp_path, recipes, models, gateway=gateway)
    sweep.run()
    assert sweep.rate.rate == pytest.approx(
        2 * 20 * GB / 60, rel=0.01
    )  # two operations moving at once
    assert sweep.state.data["rate"]["samples"] > 0


def test_one_pin_protects_a_model_shared_by_several_recipes(
    tmp_path: Path, gateway: Gateway
) -> None:
    recipes = [
        FakeRecipe("a1", ("m1",)),
        FakeRecipe("a2", ("m1",), engine="sglang"),
        FakeRecipe("a3", ("m1",), engine="trtllm"),
        FakeRecipe("b", ("m2",)),
    ]
    sweep, fleet, clock = make_sweep(
        tmp_path, recipes, [FakeModel("m1"), FakeModel("m2")], gateway=gateway
    )
    peak: list[int] = []
    clock.hooks.append(
        lambda _t: peak.append(len(fleet.profiles.get(13, {}).get("assignments", [])))
    )
    sweep.run()
    assert max(peak) == 2  # one per model set, not one per recipe
