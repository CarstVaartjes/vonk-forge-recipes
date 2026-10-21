"""Library paging: a slow page or an invalidated cursor must not stall the sweep."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import pytest
from pydantic import ValidationError
from sweep_fakes import FakeModel, FakeRecipe, Gateway, make_sweep

from spark_sweep.catalog import (
    fetch_models,
    fetch_recipes,
    parse_recipe,
    recipe_listing,
)
from spark_sweep.catalog_contract import RecipeRow
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


def test_shared_platform_list_vector_keeps_pins_ports_and_serving_checks():
    """Consume the same projection-only list vector as vonkctl's renderer."""
    import json

    from spark_sweep.catalog import parse_recipe

    page = json.loads(
        (Path(__file__).parent / "fixtures/library-list.json").read_text()
    )
    row = page["recipes"][0]
    recipe = parse_recipe(row, page["library"]["commit"])
    assert recipe is not None
    assert recipe.key == "vonk-forge/synthetic-tiny-build"
    assert recipe.library_commit == "a" * 40
    assert recipe.model_digests == tuple(
        sorted(selection["content_sha256"] for selection in row["projected"]["models"])
    )
    assert recipe.adapter == row["projected"]["interfaces"][0]["adapter"]
    assert recipe.ports == (row["projected"]["interfaces"][0]["port"],)
    assert recipe.base_image == row["projected"]["base_image_digest"]
    assert recipe.checks == tuple(row["projected"]["validation"]["serving"]["checks"])


@pytest.mark.parametrize("image_pin", [False, True])
@pytest.mark.parametrize("summary_fields", [False, True])
def test_legacy_document_shape_stays_supported_until_coordinator_contracts(
    image_pin, summary_fields
):
    """Keep both vectors until the coordinator ends old Controller support.

    The legacy vector captures LibraryRecipeProjection.document: RecipeDefinition
    from production contract tree 120f0e7cb7de236f7ec27e98d075652ed7671c7f,
    using the same canonical synthetic recipe as the platform's projected vector.
    No production read or sweep state is needed to replay that wire contract.
    """
    recipes = []
    projections = []
    for filename in ["library-list.json", "library-list-legacy.json"]:
        page = json.loads((Path(__file__).parent / "fixtures" / filename).read_text())
        row = page["recipes"][0]
        if not summary_fields:
            row.pop("engine")
            row.pop("node_count")
        if image_pin:
            row["prebuilt_image"] = {
                "build_key": "b" * 64,
                "reference": "ghcr.io/test/runtime@sha256:" + "c" * 64,
            }
        recipe = parse_recipe(row, page["library"]["commit"])
        assert recipe is not None
        recipes.append(asdict(recipe))  # Recipe equality deliberately excludes checks.
        projections.append(RecipeRow.model_validate(row).projected)
        assert recipe.model_digests == (
            "05a481f76890725bfb498309a58b7464846ac4853b8e4a5db6495d40ce631ac4",
        )
        assert recipe.aliases == ("synthetic-tiny",)
        assert recipe.ports == (8000,)
        assert recipe.artifact_bytes == 1024
        assert recipe.node_count == 1 and recipe.engine == "vllm"
        assert recipe.base_image == "f" * 64
        assert (
            recipe.checks
            and recipe.checks[0]["request"]["path"] == "/v1/chat/completions"
        )
        assert (recipe.runtime_image is not None) == image_pin
        if recipe.runtime_image is not None:
            assert recipe.runtime_image.build_key == "b" * 64
            assert recipe.runtime_image.image_digest == "sha256:" + "c" * 64
    assert recipes[0] == recipes[1]
    assert projections[0] == projections[1]


def test_projected_shape_wins_even_when_legacy_document_is_invalid():
    row = {"selector": "vonk-forge/current", "projected": {}, "document": None}
    recipe = parse_recipe(row)
    assert recipe is not None and recipe.base_image == ""


@pytest.mark.parametrize("projected", [None, {"runtime_engine": 123}])
def test_invalid_present_projection_does_not_fall_back_to_legacy(projected):
    with pytest.raises(ValidationError):
        parse_recipe(
            {"selector": "vonk-forge/current", "projected": projected, "document": {}}
        )
