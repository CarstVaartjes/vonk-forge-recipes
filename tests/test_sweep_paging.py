"""Library paging: a slow page or an invalidated cursor must not stall the sweep."""

from __future__ import annotations

from pathlib import Path

import pytest
from sweep_fakes import FakeModel, FakeRecipe, Gateway, make_sweep

from spark_sweep.catalog import fetch_models, fetch_recipes, recipe_listing
from spark_sweep.vonkctl import Vonkctl


@pytest.fixture
def gateway():
    gateway = Gateway()
    yield gateway
    gateway.stop()


def _fleet_with(n: int):
    from sweep_fakes import FakeClock, FakeFleet

    recipes = [FakeRecipe(f"r{i}", (f"m{i}",)) for i in range(n)]
    return FakeFleet(FakeClock(), recipes, [FakeModel(f"m{i}") for i in range(n)])


def test_an_invalid_cursor_restarts_the_pass_from_the_first_page() -> None:
    fleet = _fleet_with(6)  # three pages of two
    fleet.library_faults = [None, "cursor"]  # page 2 finds its cursor invalidated
    recipes, commit = fetch_recipes(Vonkctl("vonkctl", runner=fleet.runner))
    assert (
        sorted(r.key for r in recipes) == sorted(f"vonk-forge/r{i}" for i in range(6))
        and commit == "lib-commit-1"
    )
    library_calls = [c for _, c in fleet.calls if c[:2] == ("recipe", "library")]
    assert "--cursor" not in library_calls[2]  # the restart began at the first page


def test_a_timed_out_page_is_retried_not_fatal() -> None:
    fleet = _fleet_with(4)
    fleet.library_faults = ["timeout", "timeout", None]
    models = fetch_models(Vonkctl("vonkctl", runner=fleet.runner))
    assert len(models) == 4


def test_rows_already_read_survive_a_failed_page_and_a_complete_pass_prunes_the_gone() -> (
    None
):
    fleet = _fleet_with(4)
    listing = recipe_listing(Vonkctl("vonkctl", runner=fleet.runner))
    listing.begin(0)
    assert not listing.step()  # page 1: two rows
    fleet.library_faults = ["timeout"]
    assert not listing.step() and "transport_timeout" in listing.last_error
    assert len(listing.rows) == 2  # still usable
    assert listing.step(1.0) and len(listing.rows) == 4 and listing.passes == 1
    del fleet.recipes["vonk-forge/r3"]
    listing.begin(2)
    while not listing.step(3.0):
        pass
    assert "vonk-forge/r3" not in listing.rows  # dropped only after a complete pass


def test_scheduling_continues_on_the_cached_listing_while_pages_fail(
    tmp_path: Path, gateway: Gateway
) -> None:
    recipes = [FakeRecipe(f"r{i}", (f"m{i}",)) for i in range(4)]
    sweep, fleet, _ = make_sweep(
        tmp_path,
        recipes,
        [FakeModel(f"m{i}") for i in range(4)],
        gateway=gateway,
        catalog_seconds=30,
    )
    # After the first pass, every refresh page fails for a long time; a restart is also forced.
    fleet.library_faults = [None] * 8 + ["timeout", "cursor", "timeout", "timeout"] * 6
    assert sweep.run() == 0
    assert {e["status"] for e in sweep.state.recipes.values()} == {"passed"}
    assert any(
        "library page failed" in e["message"] for e in sweep.state.data["events"]
    )
