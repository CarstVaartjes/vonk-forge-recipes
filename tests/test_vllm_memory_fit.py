"""vLLM's GPU budget excludes unified-memory overhead on each GB10 node."""

import json
import math
from pathlib import Path

import pytest
from catalog_documents import catalog_models
from vonk_forge_contracts import RecipeDefinition, document_sha256

from qualification.memory_need import (
    GUARD_BYTES,
    OVERHEAD_BYTES,
    SPARK_3542_IDLE_AVAILABLE_BYTES,
    SPARK_MEMORY_BYTES,
    absolute_kv_need,
    argument_value,
    cache_bytes,
    calculate_need,
    load_inputs,
    require_memory_need,
)
from qualification.vllm_memory_fit import (
    PENDING_CREATOR,
    assert_memory_fit,
    loads_draft_model,
    non_utilization_bytes,
)

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    "path", sorted((ROOT / "recipes").glob("*.json")), ids=lambda path: path.stem
)
def test_vllm_recipe_fits_each_target_node(path: Path) -> None:
    document = json.loads(path.read_text())
    recipe = RecipeDefinition.model_validate(document)
    if recipe.runtime.engine != "vllm":
        return
    require_memory_need(
        recipe,
        catalog_models(),
        load_inputs(),
        document_digest=document_sha256(document),
    )


def test_incident_recipe_keeps_one_million_context_with_headroom() -> None:
    recipe = RecipeDefinition.model_validate_json(
        (
            ROOT / "recipes/qwen3-8-27b-aday777-abliterated-drowzeys-vllm-single.json"
        ).read_text()
    )
    assert recipe.settings.kind == "generation"
    assert recipe.settings.context_tokens.value == 1_048_576
    require_memory_need(recipe, catalog_models(), load_inputs(), document_digest="")
    need = calculate_need(recipe, catalog_models(), load_inputs())
    assert need.kv_bytes == 16 * 4 * (256 + 256) * 1_048_576
    # vLLM's one-request check (_check_enough_kv_cache_memory): each of the 48
    # recurrent layers' FP32 SSM + BF16 conv page pads to the attention page,
    # prefix caching (mamba_cache_mode align) holds two pages per layer, and
    # full-attention layers round to that block. Operation c2940bb5 refused
    # the old 34,527,510,528-byte pool with this exact need on 2026-10-27.
    token_bytes = 4 * (256 + 256)
    recurrent_page = 48 * 128 * 128 * 4 + 3 * (2 * 16 * 128 + 48 * 128) * 2
    block = 16 * math.ceil(math.ceil(recurrent_page / token_bytes) / 16)
    page = block * token_bytes
    vllm_need = 16 * math.ceil(1_048_576 / block) * page + 48 * 2 * page
    assert vllm_need == 34_681_651_200
    pool = cache_bytes(argument_value(recipe, "kv-cache-memory-bytes"))
    assert absolute_kv_need(recipe, catalog_models(), load_inputs()) >= vllm_need
    # Passing that check is not enough to serve the request: the image's vLLM
    # shares each tensor across one layer per group (16 attention + 48 GDN ->
    # group size 16, one attention and three recurrent groups), sizes blocks as
    # pool // page // 16 (kv_cache_utils.py:1062) and takes one block as the
    # null block (block_pool.py:190). A max_model_len request must still get
    # every attention block plus two blocks per recurrent group.
    usable_blocks = pool // page // 16 - 1
    assert usable_blocks >= math.ceil(1_048_576 / block) + 3 * 2
    memory = recipe.topology.roles[0].resources.memory
    assert memory.reserve_bytes == GUARD_BYTES
    assert memory.peak_bytes >= need.total_bytes + 3_000_000_000
    assert_memory_fit(recipe)
    assert memory.peak_bytes + memory.reserve_bytes <= SPARK_3542_IDLE_AVAILABLE_BYTES


@pytest.mark.parametrize("defect", ["torch-only", "os-floor", "worker", "implicit"])
def test_guard_rejects_unsafe_envelopes(defect: str) -> None:
    recipe = RecipeDefinition.model_validate_json(
        (ROOT / "recipes/deepseek-v4-flash-vision-exp-eugr-vllm-dual.json").read_text()
    )
    # Exercise this guard's utilization-sized allocation, rather than the
    # recipe's now-absolute KV pool and independent startup-only budget.
    recipe.runtime.arguments = [
        item
        for item in recipe.runtime.arguments
        if item.name != "kv-cache-memory-bytes"
    ]
    for item in recipe.runtime.arguments:
        if item.name == "gpu-memory-utilization":
            item.value = "0.85"
    for role in recipe.topology.roles:
        role.resources.memory.peak_bytes = (
            SPARK_MEMORY_BYTES * 85 + 99
        ) // 100 + OVERHEAD_BYTES
    assert_memory_fit(recipe)
    if defect == "torch-only":
        recipe.topology.roles[0].resources.memory.peak_bytes = 118_000_000_000
    elif defect == "os-floor":
        recipe.topology.roles[0].resources.memory.peak_bytes = 121_000_000_000
    elif defect == "worker":
        recipe.topology.roles[1].resources.memory.peak_bytes = 118_000_000_000
    else:
        recipe.runtime.arguments = [
            item
            for item in recipe.runtime.arguments
            if item.name != "gpu-memory-utilization"
        ]
    with pytest.raises(AssertionError):
        assert_memory_fit(recipe)


@pytest.mark.parametrize("slug", sorted(PENDING_CREATOR))
def test_pending_creator_recipes_still_need_their_creator(slug: str) -> None:
    path = ROOT / "recipes" / f"{slug}.json"
    assert path.exists(), f"{slug} is gone; remove it from PENDING_CREATOR"
    recipe = RecipeDefinition.model_validate_json(path.read_text())
    assert loads_draft_model(recipe)
    assert_memory_fit(recipe, count_draft_model=False)
    with pytest.raises(AssertionError):
        assert_memory_fit(recipe)


def test_pending_creator_still_meets_the_os_floor() -> None:
    """Waiving the draft term never waives the OS floor."""
    slug = min(PENDING_CREATOR)
    recipe = RecipeDefinition.model_validate_json(
        (ROOT / "recipes" / f"{slug}.json").read_text()
    )
    recipe.topology.roles[0].resources.memory.peak_bytes = 125_000_000_000
    with pytest.raises(AssertionError):
        assert_memory_fit(recipe, count_draft_model=False)


def _incident_recipe() -> RecipeDefinition:
    return RecipeDefinition.model_validate_json(
        (ROOT / "recipes/qwen3-8-27b-nvfp4-dflash2-eugr-vllm-single.json").read_text()
    )


def _with_utilization(recipe: RecipeDefinition, value: str) -> None:
    for item in recipe.runtime.arguments:
        if item.name == "gpu-memory-utilization":
            item.value = value


def test_draft_model_overhead_rejects_the_frozen_envelope() -> None:
    """0.7 with peak 0.7 x memory + 8e9 froze a Spark; the guard must refuse it."""
    recipe = _incident_recipe()
    assert loads_draft_model(recipe)
    _with_utilization(recipe, "0.7")
    recipe.topology.roles[0].resources.memory.peak_bytes = 99_464_262_042
    with pytest.raises(AssertionError):
        assert_memory_fit(recipe)


def test_draft_model_overhead_caps_utilization() -> None:
    """With the draft term no honest peak fits a Spark at the old 0.85."""
    recipe = _incident_recipe()
    _with_utilization(recipe, "0.85")
    recipe.topology.roles[0].resources.memory.peak_bytes = 120_600_000_000
    with pytest.raises(AssertionError):
        assert_memory_fit(recipe)


def test_target_only_speculation_keeps_the_plain_overhead() -> None:
    """Native MTP (no separate model) and no speculation use the plain 8e9 term."""
    recipe = _incident_recipe()
    for item in recipe.runtime.arguments:
        if item.name == "speculative-config":
            item.value = '{"method":"mtp","num_speculative_tokens":2}'
    assert not loads_draft_model(recipe)
    assert non_utilization_bytes(recipe) == OVERHEAD_BYTES
    recipe.runtime.arguments = [
        item for item in recipe.runtime.arguments if item.name != "speculative-config"
    ]
    assert non_utilization_bytes(recipe) == OVERHEAD_BYTES


def test_self_referencing_draft_is_not_a_second_model() -> None:
    recipe = _incident_recipe()
    for item in recipe.runtime.arguments:
        if item.name == "speculative-config":
            item.value = '{"method":"deepseek_mtp","model":"/models/target"}'
    assert not loads_draft_model(recipe)
