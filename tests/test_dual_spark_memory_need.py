"""Nonfitting dual recipes must expose their demand instead of waiving it."""

from pathlib import Path

import pytest
from catalog_documents import catalog_models
from vonk_forge_contracts import RecipeDefinition
from vonk_forge_contracts.recipe import RecipeRuntimeEnvironment

from qualification.memory_need import (
    LIMIT_BYTES,
    OS_FLOOR_BYTES,
    OVERHEAD_BYTES,
    calculate_need,
    declaration_problems,
    load_inputs,
    memory_variants,
    per_rank_need,
    require_memory_need,
)

ROOT = Path(__file__).resolve().parents[1]
RECIPES = ("glm-5-3-flash-exl3-dflash2-reederey87-vllm-dual",)


def recipe_for(slug: str) -> RecipeDefinition:
    return RecipeDefinition.model_validate_json(
        (ROOT / "recipes" / f"{slug}.json").read_text()
    )


@pytest.mark.parametrize("slug", RECIPES)
def test_capacity_profiles_declare_the_loading_floor(slug: str) -> None:
    recipe = recipe_for(slug)
    variants = [recipe]
    variants.extend(
        recipe.with_option_choices({option.name: choice.value})
        for option in recipe.options
        for choice in option.choices
    )
    for variant in variants:
        need = per_rank_need(variant, catalog_models(), load_inputs())
        assert not declaration_problems(variant, need)
        assert (
            require_memory_need(
                variant, catalog_models(), load_inputs(), document_digest=""
            )
            is None
        )


@pytest.mark.parametrize("slug", RECIPES)
@pytest.mark.parametrize("rank", [0, 1])
def test_guard_rejects_hidden_demand_and_headroom(slug: str, rank: int) -> None:
    recipe = recipe_for(slug)
    need = per_rank_need(recipe, catalog_models(), load_inputs())
    memory = recipe.topology.roles[rank].resources.memory
    memory.peak_bytes = need.peak_bytes - 1
    assert any("peak understates" in p for p in declaration_problems(recipe, need))
    with pytest.raises(ValueError, match="peak understates"):
        require_memory_need(recipe, catalog_models(), load_inputs(), document_digest="")
    memory.peak_bytes = need.peak_bytes
    memory.reserve_bytes = OS_FLOOR_BYTES - 1
    assert any("reserve understates" in p for p in declaration_problems(recipe, need))
    with pytest.raises(ValueError, match="reserve understates"):
        require_memory_need(recipe, catalog_models(), load_inputs(), document_digest="")


def test_loading_floor_does_not_double_count_the_fraction_draft_term() -> None:
    recipe = recipe_for(RECIPES[0])
    need = per_rank_need(recipe, catalog_models(), load_inputs())
    # The old 17 GB measurement was outside a utilization budget, not another
    # model beside the explicitly selected target/draft weight bytes.
    assert need.peak_bytes > 100_000_000_000
    assert need.peak_bytes < 125_650_657_280


@pytest.mark.parametrize("slug", RECIPES)
def test_lower_fraction_cannot_hide_selected_weight_or_absolute_kv_floor(
    slug: str,
) -> None:
    recipe = recipe_for(slug)
    fraction = next(
        a for a in recipe.runtime.arguments if a.name == "gpu-memory-utilization"
    )
    fraction.value = "0.10"
    need = per_rank_need(recipe, catalog_models(), load_inputs())
    assert need.peak_bytes > LIMIT_BYTES


@pytest.mark.parametrize("slug", RECIPES)
def test_four_rank_memory_projection_with_retuned_budget(slug: str) -> None:
    # Memory projection only. A distinct four-Spark recipe must also port the
    # GLM TP2-only launcher and validate the actual startup/context fit.
    recipe = recipe_for(slug)
    recipe.topology.node_count = 4
    recipe.topology.parallelism.tensor = 4
    recipe.topology.roles[1].count = 3
    next(
        a for a in recipe.runtime.arguments if a.name == "tensor-parallel-size"
    ).value = 4
    next(
        a for a in recipe.runtime.arguments if a.name == "gpu-memory-utilization"
    ).value = "0.60"
    recipe = RecipeDefinition.model_validate(recipe.model_dump(mode="json"))
    need = per_rank_need(recipe, catalog_models(), load_inputs())
    assert need.fits
    assert need.verdict == "fits 4 Sparks"


ENGRAM = "deepseek-v4-1-flash-exl3-mia-dual"


def test_every_selected_engram_file_is_excluded_from_per_rank_need() -> None:
    # Owner rule (2026-10-26): Engram tables are never memory usage. Any recipe
    # selecting a known Engram file is covered without registration.
    models, inputs = catalog_models(), load_inputs()
    inflated = {
        digest: model.model_copy(
            update={
                "files": [
                    item.model_copy(update={"size_bytes": item.size_bytes * 10})
                    if item.sha256 in inputs.engram_files
                    else item
                    for item in model.files
                ]
            }
        )
        for digest, model in models.items()
    }
    checked = 0
    for path in sorted((ROOT / "recipes").glob("*.json")):
        recipe = RecipeDefinition.model_validate_json(path.read_text())
        if recipe.runtime.engine not in {"vllm", "sglang"} or not any(
            item.sha256 in inputs.engram_files
            for selection in recipe.models
            for item in models[selection.model.content_sha256].files
            if item.id in {f.file_id for f in selection.files}
        ):
            continue
        for variant in memory_variants(recipe):
            if variant.topology.node_count == 1:
                assert calculate_need(variant, models, inputs) == calculate_need(
                    variant, inflated, inputs
                ), path.stem
            else:
                assert (
                    per_rank_need(variant, models, inputs).peak_bytes
                    == per_rank_need(variant, inflated, inputs).peak_bytes
                ), path.stem
        checked += 1
    assert checked > 0


def test_single_spark_calculator_excludes_engram_files() -> None:
    # No single-Spark vLLM/SGLang recipe selects an Engram file today; mark
    # the Mosaic recipe's largest weight file as one to exercise the path.
    models, inputs = catalog_models(), load_inputs()
    recipe = recipe_for("glm-5-3-flash-exl3-mosaic-12l-0xsero-sglang-single")
    need = calculate_need(recipe, models, inputs)
    ids = {f.file_id for f in recipe.models[0].files}
    largest = max(
        (
            f
            for f in models[recipe.models[0].model.content_sha256].files
            if f.id in ids and "weights" in f.roles
        ),
        key=lambda f: f.size_bytes,
    )
    marked = inputs.model_copy(
        update={"engram_files": {**inputs.engram_files, largest.sha256: "test"}}
    )
    assert (
        need.weights_bytes - calculate_need(recipe, models, marked).weights_bytes
        == largest.size_bytes
    )


def test_proven_mia_engram_dual_fits_without_its_engram_shards() -> None:
    # Hardware-passed (aa385fa2) with a 120,370,379,081-byte peak.
    recipe = recipe_for(ENGRAM)
    models, inputs = catalog_models(), load_inputs()
    need = per_rank_need(recipe, models, inputs)
    target = models[recipe.models[0].model.content_sha256]
    ids = {f.file_id for f in recipe.models[0].files}
    loaded = sum(
        f.size_bytes for f in target.files if f.id in ids and "weights" in f.roles
    )
    assert need.fits
    assert need.peak_bytes == (loaded + 1) // 2 + 2_684_354_560 + OVERHEAD_BYTES
    assert not declaration_problems(recipe, need)
    # Fault injection: counted as weights, the Engram shards cannot fit.
    assert not per_rank_need(
        recipe, models, inputs.model_copy(update={"engram_files": {}})
    ).fits


def test_explicit_row_cache_is_charged_per_rank() -> None:
    models, inputs = catalog_models(), load_inputs()
    four = recipe_for("deepseek-v4-1-flash-1017-mia-sglang-four")
    floor = per_rank_need(four, models, inputs).peak_bytes
    cache = next(e for e in four.runtime.environment if e.name == "DSV41_CACHE_GIB")
    assert cache.value == "4"
    cache.value = "0"
    # Four Sparks, one rank each: 4 GiB of row cache per rank.
    assert floor - per_rank_need(four, models, inputs).peak_bytes == 4 * 2**30
    # Unset or zero counts nothing; an explicit size is a memory setting.
    recipe = recipe_for(ENGRAM)
    floor = per_rank_need(recipe, models, inputs).peak_bytes
    recipe.runtime.environment = [
        e for e in recipe.runtime.environment if e.name != "DSV41_CACHE_GIB"
    ]
    assert per_rank_need(recipe, models, inputs).peak_bytes == floor
    recipe.runtime.environment.append(
        RecipeRuntimeEnvironment(name="DSV41_CACHE_GIB", value="8")
    )
    assert per_rank_need(recipe, models, inputs).peak_bytes - floor == 8 * 2**30
