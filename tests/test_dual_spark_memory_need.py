"""Nonfitting dual recipes must expose their demand instead of waiving it."""

from pathlib import Path

import pytest
from catalog_documents import catalog_models
from vonk_forge_contracts import RecipeDefinition

from qualification.memory_need import (
    LIMIT_BYTES,
    OS_FLOOR_BYTES,
    OVERHEAD_BYTES,
    declaration_problems,
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
    recipe = recipe_for(RECIPES[1])
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
# Every vLLM/SGLang recipe that selects Engram shards a pinned reader keeps on disk.
FILE_BACKED = (
    ENGRAM,
    "deepseek-v4-1-flash-exl3-sfxnz-vllm-dual",
    "deepseek-v4-1-flash-tr3-hybrid-drowzeys-vllm-four",
    "deepseek-v4-1-flash-exl3-0xsero-vllm-dual",
    "deepseek-v4-1-flash-r0b0tlab-sglang-four",
    "deepseek-v4-1-flash-1017-mia-sglang-triple",
    "deepseek-v4-1-flash-1017-mia-sglang-four",
)


def selected_weights(recipe: RecipeDefinition) -> dict[str, int]:
    models = catalog_models()
    weights = {}
    for selection in recipe.models:
        ids = {f.file_id for f in selection.files}
        for item in models[selection.model.content_sha256].files:
            if item.id in ids and "weights" in item.roles:
                weights[item.sha256] = item.size_bytes
    return weights


def with_facts(inputs, **update):
    return inputs.model_copy(
        update={
            "file_backed_weights": [
                fact.model_copy(update=update) for fact in inputs.file_backed_weights
            ]
        }
    )


@pytest.mark.parametrize("slug", FILE_BACKED)
def test_file_backed_engram_files_are_not_resident_weights(slug: str) -> None:
    recipe = recipe_for(slug)
    models, inputs = catalog_models(), load_inputs()
    weights = selected_weights(recipe)
    engram = {
        digest
        for fact in inputs.file_backed_weights
        if fact.runtime_image_digest == recipe.execution.build.base_image.digest
        for digest in fact.file_sha256
    }
    assert engram <= weights.keys()
    shards = recipe.topology.parallelism.tensor * recipe.topology.parallelism.pipeline
    loaded = sum(size for digest, size in weights.items() if digest not in engram)
    floor = per_rank_need(recipe, models, inputs).peak_bytes
    assert floor >= (loaded + shards - 1) // shards + OVERHEAD_BYTES
    # Fault injection: without the facts the shards count as loaded again.
    bare = inputs.model_copy(update={"file_backed_weights": []})
    assert per_rank_need(recipe, models, bare).peak_bytes > floor


def test_proven_mia_engram_dual_fits_without_its_engram_shards() -> None:
    # Hardware-passed (aa385fa2) with a 120,370,379,081-byte peak.
    recipe = recipe_for(ENGRAM)
    models, inputs = catalog_models(), load_inputs()
    need = per_rank_need(recipe, models, inputs)
    target = selected_weights(recipe.model_copy(update={"models": recipe.models[:1]}))
    assert need.fits
    assert (
        need.peak_bytes
        == (sum(target.values()) + 1) // 2 + 2_684_354_560 + OVERHEAD_BYTES
    )
    assert not declaration_problems(recipe, need)
    assert not per_rank_need(
        recipe, models, inputs.model_copy(update={"file_backed_weights": []})
    ).fits


def test_stale_or_unpinned_file_backed_fact_fails_fast() -> None:
    recipe = recipe_for(ENGRAM)
    models, inputs = catalog_models(), load_inputs()
    with pytest.raises(ValueError, match="registered for image"):
        per_rank_need(recipe, models, with_facts(inputs, runtime_image_digest="0" * 64))
    environment = next(
        e for e in recipe.runtime.environment if e.name == "DSV41_RESIDENT_SCALES"
    )
    environment.value = "1"
    with pytest.raises(ValueError, match="require DSV41_RESIDENT_SCALES=0"):
        per_rank_need(recipe, models, inputs)
    environment.value = "0"
    cache = next(e for e in recipe.runtime.environment if e.name == "DSV41_CACHE_GIB")
    recipe.runtime.environment.remove(cache)
    # Unset, the overlay keeps an 8 GiB row cache: the recipe must pin it.
    with pytest.raises(ValueError, match="pin DSV41_CACHE_GIB"):
        per_rank_need(recipe, models, inputs)


def test_resident_row_cache_is_charged_per_rank() -> None:
    models, inputs = catalog_models(), load_inputs()
    four = recipe_for("deepseek-v4-1-flash-1017-mia-sglang-four")
    floor = per_rank_need(four, models, inputs).peak_bytes
    cache = next(e for e in four.runtime.environment if e.name == "DSV41_CACHE_GIB")
    assert cache.value == "4"
    cache.value = "0"
    # Four Sparks, one rank each: 4 GiB of row cache per rank.
    assert floor - per_rank_need(four, models, inputs).peak_bytes == 4 * 2**30
    recipe = recipe_for(ENGRAM)
    floor = per_rank_need(recipe, models, inputs).peak_bytes
    next(
        e for e in recipe.runtime.environment if e.name == "DSV41_CACHE_GIB"
    ).value = "8"
    assert per_rank_need(recipe, models, inputs).peak_bytes - floor == 8 * 2**30
