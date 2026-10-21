"""Nonfitting dual recipes must expose their demand instead of waiving it."""

from pathlib import Path

import pytest
from catalog_documents import catalog_models
from vonk_forge_contracts import RecipeDefinition
from vonk_forge_contracts.recipe import RecipeRuntimeArgument

from qualification.memory_need import (
    GUARD_BYTES,
    LIMIT_BYTES,
    calculate_need,
    load_inputs,
    per_rank_need,
    require_memory_need,
)

ROOT = Path(__file__).resolve().parents[1]
RECIPES = (
    "llama-3-1-405b-instruct-awq-int4-nvidia-vllm-dual",
    "glm-5-3-flash-exl3-dflash2-reederey87-vllm-dual",
)


def recipe_for(slug: str) -> RecipeDefinition:
    return RecipeDefinition.model_validate_json(
        (ROOT / "recipes" / f"{slug}.json").read_text()
    )


@pytest.mark.parametrize("slug", RECIPES)
def test_capacity_profiles_declare_need_and_report_nonfit(slug: str) -> None:
    recipe = recipe_for(slug)
    variants = [recipe]
    variants.extend(
        recipe.with_option_choices({option.name: choice.value})
        for option in recipe.options
        for choice in option.choices
    )
    for variant in variants:
        need = per_rank_need(variant, catalog_models())
        assert need.verdict == "does not fit 2 Sparks"
        assert need.peak_bytes > LIMIT_BYTES
        assert (
            require_memory_need(
                variant, catalog_models(), load_inputs(), document_digest=""
            )
            is not None
        )


@pytest.mark.parametrize("slug", RECIPES)
@pytest.mark.parametrize("rank", [0, 1])
def test_guard_rejects_hidden_demand_and_headroom(slug: str, rank: int) -> None:
    recipe = recipe_for(slug)
    need = per_rank_need(recipe, catalog_models())
    memory = recipe.topology.roles[rank].resources.memory
    memory.peak_bytes = need.peak_bytes - 1
    with pytest.raises(ValueError, match="peak understates"):
        require_memory_need(recipe, catalog_models(), load_inputs(), document_digest="")
    memory.peak_bytes = need.peak_bytes
    memory.reserve_bytes = GUARD_BYTES - 1
    with pytest.raises(ValueError, match="reserve understates"):
        require_memory_need(recipe, catalog_models(), load_inputs(), document_digest="")


def test_draft_budget_is_not_waived() -> None:
    recipe = recipe_for(RECIPES[1])
    need = per_rank_need(recipe, catalog_models())
    absolute = calculate_need(recipe, catalog_models(), load_inputs().geometries)
    assert absolute.overhead_bytes == 25_000_000_000
    assert need.peak_bytes == absolute.total_bytes


@pytest.mark.parametrize("slug", RECIPES)
def test_lower_fraction_cannot_hide_selected_weight_or_absolute_kv_floor(
    slug: str,
) -> None:
    recipe = recipe_for(slug)
    recipe.runtime.arguments.append(
        RecipeRuntimeArgument(name="gpu-memory-utilization", value="0.10")
    )
    need = per_rank_need(recipe, catalog_models())
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
    recipe.runtime.arguments.append(
        RecipeRuntimeArgument(name="gpu-memory-utilization", value="0.60")
    )
    for arg in recipe.runtime.arguments:
        if arg.name == "kv-cache-memory-bytes":
            arg.value = (
                calculate_need(
                    recipe, catalog_models(), load_inputs().geometries
                ).kv_bytes
                + calculate_need(
                    recipe, catalog_models(), load_inputs().geometries
                ).state_bytes
            )
    recipe = RecipeDefinition.model_validate(recipe.model_dump(mode="json"))
    need = per_rank_need(recipe, catalog_models())
    assert need.fits
    assert need.verdict == "fits 4 Sparks"
