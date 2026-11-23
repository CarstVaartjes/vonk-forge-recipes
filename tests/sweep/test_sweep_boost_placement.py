"""The owner's explicit boost reaches placement through the sweep's own tick."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from sweep_fakes import GB, FakeModel, FakeRecipe, make_sweep

from spark_sweep.report import build_status

MODELS = [  # the favourite's model is the biggest, so size order alone would put it last
    FakeModel("m1", GB, local="cached"),
    FakeModel("m2", 80 * GB, local="cached"),
    FakeModel("m3", local="cached"),
]


def _sweep(tmp_path: Path, recipes: list[FakeRecipe], **overrides):
    sweep, _, clock = make_sweep(tmp_path, recipes, MODELS, **overrides)
    sweep.preflight()
    sweep.refresh_catalog()
    return sweep, clock


def _lane_taken_then_all(sweep, clock, ticks: int = 3) -> None:
    sweep._tick()  # the "busy" recipe takes one lane
    sweep.cfg = replace(sweep.cfg, only=())
    sweep.refresh_catalog()
    for _ in range(ticks):
        clock.sleep(10)
        sweep._tick()


def test_boosted_favourite_is_placed_before_a_smaller_ready_single(
    tmp_path: Path,
) -> None:
    busy = FakeRecipe("busy", ("m3",), local="cached", load_seconds=5000)
    small = FakeRecipe("small", ("m1",), local="cached", load_seconds=5000)
    favourite = FakeRecipe("favourite", ("m2",), local="cached", load_seconds=5000)
    sweep, clock = _sweep(
        tmp_path, [busy, small, favourite], only=("busy",), boost=("favourite",)
    )
    _lane_taken_then_all(sweep, clock, 1)
    assert favourite.key in sweep.state.slots
    assert small.key not in sweep.state.slots


def test_without_a_boost_the_smaller_single_is_placed_first(tmp_path: Path) -> None:
    busy = FakeRecipe("busy", ("m3",), local="cached", load_seconds=5000)
    small = FakeRecipe("small", ("m1",), local="cached", load_seconds=5000)
    favourite = FakeRecipe("favourite", ("m2",), local="cached", load_seconds=5000)
    sweep, clock = _sweep(tmp_path, [busy, small, favourite], only=("busy",))
    _lane_taken_then_all(sweep, clock, 1)
    assert sweep.queue.index(small.key) < sweep.queue.index(favourite.key)
    assert favourite.key not in sweep.state.slots


def test_boosted_dual_holds_the_free_lane_for_itself(tmp_path: Path) -> None:
    """An explicit owner boost may reserve the free lane for a dual; nothing else may."""
    busy = FakeRecipe("busy", ("m1",), local="cached", load_seconds=5000)
    dual = FakeRecipe("favourite", ("m2",), local="cached", node_count=2)
    single = FakeRecipe("later", ("m3",), local="cached", load_seconds=5000)
    sweep, clock = _sweep(
        tmp_path, [busy, dual, single], only=("busy",), boost=("favourite",)
    )
    _lane_taken_then_all(sweep, clock)
    assert dual.key not in sweep.state.slots
    assert single.key not in sweep.state.slots
    assert dual.key in build_status(sweep)["dual_reservation"]


def test_recent_update_alone_leaves_a_free_lane_to_the_single(tmp_path: Path) -> None:
    busy = FakeRecipe("busy", ("m1",), local="cached", load_seconds=5000)
    dual = FakeRecipe(
        "fresh",
        ("m2",),
        local="cached",
        node_count=2,
        updated="1970-01-12T00:00:00Z",  # inside the recent window of the fake clock
    )
    single = FakeRecipe("later", ("m3",), local="cached", load_seconds=5000)
    sweep, clock = _sweep(tmp_path, [busy, dual, single], only=("busy",))
    assert sweep.boost(sweep.recipes[dual.key]) > 1.0  # it does download early
    _lane_taken_then_all(sweep, clock)
    assert single.key in sweep.state.slots
    assert not sweep.state.data.get("dual_reservation")
