"""Reject fixed cache pools that cannot admit the advertised context."""

import json
import runpy
import subprocess
from pathlib import Path

import pytest
from catalog_documents import catalog_models
from vonk_forge_contracts import RecipeDefinition
from vonk_forge_contracts.recipe import RecipeRuntimeArgument

from qualification.memory_need import (
    LIMIT_BYTES,
    absolute_kv_need,
    argument_value,
    cache_bytes,
    load_inputs,
    memory_variants,
    per_rank_need,
    require_absolute_kv_capacity,
    require_memory_need,
)

ROOT = Path(__file__).resolve().parents[1]
VISION = "deepseek-v4-flash-vision-exp-mia-dual"
LAGUNA = "laguna-m-1-nvfp4-r0b0tlab-vllm-dual"


def recipe_for(slug: str = VISION) -> RecipeDefinition:
    return RecipeDefinition.model_validate_json(
        (ROOT / "recipes" / f"{slug}.json").read_text()
    )


@pytest.mark.parametrize(
    "slug",
    [VISION, "deepseek-v4-flash-vision-exp-dspark-tonyd2wild-vllm-dual"],
)
def test_reported_startup_fault_is_rejected_before_catalog_publication(
    slug: str,
) -> None:
    recipe = recipe_for(slug)
    assert recipe.settings.kind == "generation"
    recipe.settings.context_tokens.value = 1_048_576
    next(
        arg for arg in recipe.runtime.arguments if arg.name == "kv-cache-memory-bytes"
    ).value = 495_976_448
    with pytest.raises(ValueError, match="cannot hold one full-context request"):
        require_memory_need(recipe, catalog_models(), load_inputs(), document_digest="")


@pytest.mark.parametrize("context", [None, "auto", -1, "-1", 1_048_576])
def test_explicit_and_model_default_context_have_the_same_need(context) -> None:
    recipe = recipe_for()
    recipe.runtime.arguments = [
        arg for arg in recipe.runtime.arguments if arg.name != "max-model-len"
    ]
    if context is not None:
        recipe.runtime.arguments.append(
            RecipeRuntimeArgument(name="max-model-len", value=context)
        )
    need = absolute_kv_need(recipe, catalog_models(), load_inputs())
    assert need == 1_048_576 * 8192 + 64 * 1024**2
    # MLA is replicated on the two tensor ranks, never divided by TP=2.
    assert need > 7.54 * 1024**3


def test_pool_boundary_and_wrapper_alias_are_checked() -> None:
    recipe = recipe_for()
    need = absolute_kv_need(recipe, catalog_models(), load_inputs())
    pool = next(
        arg for arg in recipe.runtime.arguments if arg.name == "kv-cache-memory-bytes"
    )
    pool.name = "kv-cache-memory"
    pool.value = need
    require_absolute_kv_capacity(recipe, catalog_models(), load_inputs())
    pool.value = need - 1
    with pytest.raises(ValueError, match="cannot hold one full-context request"):
        require_absolute_kv_capacity(recipe, catalog_models(), load_inputs())


def test_option_dtype_can_invalidate_a_base_pool() -> None:
    recipe = recipe_for("hy3-295b-nvfp4-vllm-mia-dual")
    need = absolute_kv_need(recipe, catalog_models(), load_inputs())
    next(
        arg for arg in recipe.runtime.arguments if arg.name == "kv-cache-memory-bytes"
    ).value = need
    with pytest.raises(ValueError, match="cannot hold one full-context request"):
        require_absolute_kv_capacity(recipe, catalog_models(), load_inputs())


def test_gqa_sharding_and_nvfp4_block_scales() -> None:
    recipe = recipe_for("hy3-295b-nvfp4-a4q-drowzeys-vllm-dual")
    assert recipe.settings.kind == "generation"
    recipe.settings.context_tokens.value = 16
    assert absolute_kv_need(recipe, catalog_models(), load_inputs()) == (
        16 * 80 * 4 * (128 // 2 + 128 // 16) * 2
    )
    recipe.topology.parallelism.tensor = 1
    assert absolute_kv_need(recipe, catalog_models(), load_inputs()) == (
        16 * 80 * 8 * (128 // 2 + 128 // 16) * 2
    )


def test_pipeline_layers_shard_while_mla_stays_tensor_replicated() -> None:
    recipe = recipe_for()
    assert recipe.settings.kind == "generation"
    recipe.settings.context_tokens.value = 256
    recipe.topology.parallelism.pipeline = 2
    # The largest stage has 22 of the 43 layers; round its per-token bound up.
    assert absolute_kv_need(recipe, catalog_models(), load_inputs()) == (
        256 * 4192 + 64 * 1024**2
    )


def test_geometry_must_match_selected_configuration() -> None:
    inputs = load_inputs()
    recipe = recipe_for()
    inputs.geometries[recipe.models[0].model.slug].configuration_sha256 = "0" * 64
    with pytest.raises(ValueError, match="configuration drift"):
        absolute_kv_need(recipe, catalog_models(), inputs)


def test_sglang_token_pool_must_hold_one_context() -> None:
    recipe = recipe_for("deepseek-v4-1-flash-1017-mia-sglang-triple")
    context = argument_value(recipe, "context-length")
    assert isinstance(context, int)
    pool = next(
        arg for arg in recipe.runtime.arguments if arg.name == "max-total-tokens"
    )
    pool.value = context
    require_absolute_kv_capacity(recipe, catalog_models(), load_inputs())
    pool.value = context - 1
    with pytest.raises(ValueError, match="cannot hold one full-context request"):
        require_absolute_kv_capacity(recipe, catalog_models(), load_inputs())


def test_catalog_absolute_pools_hold_one_full_context_for_every_option() -> None:
    for path in sorted((ROOT / "recipes").glob("*.json")):
        recipe = RecipeDefinition.model_validate_json(path.read_text())
        require_absolute_kv_capacity(recipe, catalog_models(), load_inputs())


def test_corrected_pools_cover_all_declared_concurrency() -> None:
    # The source records the operator-visible correction; no pass/failure ledger.
    for path in sorted((ROOT / "recipes").glob("*.json")):
        document = json.loads(path.read_text())
        recipe = RecipeDefinition.model_validate(document)
        if "Absolute KV correction:" not in recipe.metadata.description:
            continue
        for variant in memory_variants(recipe):
            concurrency = argument_value(variant, "max-num-seqs")
            assert isinstance(concurrency, int)
            pool = argument_value(variant, "kv-cache-memory-bytes")
            assert pool is not None
            assert cache_bytes(pool) >= absolute_kv_need(
                variant, catalog_models(), load_inputs(), concurrency=concurrency
            )


def test_touched_multi_spark_vllm_pool_holds_the_declared_workload() -> None:
    recipe = recipe_for(LAGUNA)
    models, inputs = catalog_models(), load_inputs()
    require_memory_need(recipe, models, inputs, document_digest="", touched=True)
    pool = next(
        arg for arg in recipe.runtime.arguments if arg.name == "kv-cache-memory-bytes"
    )
    pool.value = cache_bytes(pool.value) - 1
    with pytest.raises(ValueError, match="set absolute kv-cache-memory-bytes"):
        require_memory_need(recipe, models, inputs, document_digest="", touched=True)
    recipe.runtime.arguments.remove(pool)
    with pytest.raises(ValueError, match="set absolute kv-cache-memory-bytes"):
        require_memory_need(recipe, models, inputs, document_digest="", touched=True)
    # Files the change does not touch keep their utilization-sized pools.
    require_memory_need(recipe, models, inputs, document_digest="")


# Hardware-passed in the sweep on their current content (2026-10-24..26),
# with sharded weights + absolute pool + 8 GB above LIMIT_BYTES.
PROVEN_ABOVE_PLANNING_LIMIT = [
    "deepseek-v4-flash-vision-exp-eugr-vllm-dual",
    "hy3-295b-nvfp4-mtp-tonyd2wild-vllm-dual",
    "hy3-295b-nvfp4-vllm-mia-dual",
]


@pytest.mark.parametrize("slug", PROVEN_ABOVE_PLANNING_LIMIT)
def test_touching_a_proven_multi_spark_vllm_document_keeps_it_valid(
    slug: str,
) -> None:
    recipe = recipe_for(slug)
    models, inputs = catalog_models(), load_inputs()
    assert per_rank_need(recipe, models).peak_bytes > LIMIT_BYTES
    require_memory_need(recipe, models, inputs, document_digest="", touched=True)


def test_only_recipe_files_changed_since_the_merge_base_are_touched(
    tmp_path: Path,
) -> None:
    def git(*args: str) -> None:
        subprocess.run(
            ["git", "-C", str(tmp_path), "-c", "user.name=t", "-c", "user.email=t@t"]
            + list(args),
            check=True,
            capture_output=True,
        )

    (tmp_path / "recipes").mkdir()
    for name in ("kept", "edited"):
        (tmp_path / f"recipes/{name}.json").write_text("{}")
    git("init", "-q", "-b", "main")
    git("add", "-A")
    git("commit", "-q", "-m", "base")
    (tmp_path / "recipes/edited.json").write_text('{"x": 1}')
    (tmp_path / "recipes/new.json").write_text("{}")
    touched = runpy.run_path(str(ROOT / "tools/check-recipe-memory-need"))[
        "touched_recipes"
    ]
    assert touched(tmp_path, "HEAD") == [
        tmp_path / "recipes/edited.json",
        tmp_path / "recipes/new.json",
    ]
