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
        need = per_rank_need(variant, catalog_models())
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
    need = per_rank_need(recipe, catalog_models())
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
    need = per_rank_need(recipe, catalog_models())
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
    next(
        a for a in recipe.runtime.arguments if a.name == "gpu-memory-utilization"
    ).value = "0.60"
    recipe = RecipeDefinition.model_validate(recipe.model_dump(mode="json"))
    need = per_rank_need(recipe, catalog_models())
    assert need.fits
    assert need.verdict == "fits 4 Sparks"


ENGRAM = "deepseek-v4-1-flash-exl3-mia-dual"


def test_file_backed_engram_tables_are_not_resident_weights(monkeypatch) -> None:
    # Hardware-passed (aa385fa2) with a 120,370,379,081-byte peak: only the
    # EXL3 target is loaded; the official shards 47/48 are read row by row.
    import qualification.memory_need as memory

    recipe = recipe_for(ENGRAM)
    models = catalog_models()
    target = models[recipe.models[0].model.content_sha256]
    selected = {f.file_id for f in recipe.models[0].files}
    target_bytes = sum(
        f.size_bytes for f in target.files if f.id in selected and "weights" in f.roles
    )
    need = per_rank_need(recipe, models)
    assert need.fits
    assert need.peak_bytes == (target_bytes + 1) // 2 + 2_684_354_560 + OVERHEAD_BYTES
    assert not declaration_problems(recipe, need)

    # Fault injection: without the pinned fact, or under another image, the
    # same selection counts as loaded weights again and cannot fit.
    inputs = load_inputs()
    for facts in (
        {},
        {
            key: fact.model_copy(update={"runtime_image_digest": "0" * 64})
            for key, fact in inputs.file_backed_weights.items()
        },
    ):
        changed = inputs.model_copy(update={"file_backed_weights": facts})
        monkeypatch.setattr(
            memory, "load_inputs", lambda root, changed=changed: changed
        )
        assert not per_rank_need(recipe, models).fits
        assert any(
            "peak understates" in p
            for p in declaration_problems(recipe, per_rank_need(recipe, models))
        )
