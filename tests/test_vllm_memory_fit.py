"""vLLM's GPU budget excludes unified-memory overhead on each GB10 node."""

import math
from pathlib import Path

import pytest
from catalog_documents import catalog_models
from vonk_forge_contracts import RecipeDefinition
from warning_asserts import assert_not_warns, declare_peak

from qualification.memory_need import (
    ADMISSION_IDLE_USE_BYTES,
    OS_FLOOR_BYTES,
    SPARK_MEMORY_BYTES,
    absolute_kv_need,
    argument_value,
    budget_margin_bytes,
    cache_bytes,
    calculate_need,
    load_inputs,
    memory_estimate_warnings,
)
from qualification.vllm_memory_fit import assert_memory_fit, loads_draft_model

ROOT = Path(__file__).resolve().parents[1]


def test_incident_recipe_keeps_one_million_context_with_headroom() -> None:
    recipe = RecipeDefinition.model_validate_json(
        (
            ROOT / "recipes/qwen3-8-27b-aday777-abliterated-drowzeys-vllm-single.json"
        ).read_text()
    )
    declare_peak(recipe)
    assert recipe.settings.kind == "generation"
    assert recipe.settings.context_tokens.value == 1_048_576
    assert_not_warns(
        memory_estimate_warnings(
            recipe, catalog_models(), load_inputs(), document_digest=""
        ),
        "peak understates|not admissible|full-context request|cannot hold the selected weights|set absolute|mem-fraction-static|state slots",
    )
    need = calculate_need(recipe, catalog_models(), load_inputs())
    assert need.kv_bytes > 16 * 4 * (256 + 256) * 1_048_576
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
    # One sequence: the workload is vLLM's hybrid allocation for one request.
    assert need.kv_bytes + need.state_bytes == absolute_kv_need(
        recipe, catalog_models(), load_inputs()
    )
    memory = recipe.topology.roles[0].resources.memory
    assert memory.peak_bytes >= need.total_bytes + 3_000_000_000
    assert_memory_fit(recipe)


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
        ) // 100 + budget_margin_bytes(recipe)
    assert_memory_fit(recipe)
    if defect == "torch-only":
        recipe.topology.roles[0].resources.memory.peak_bytes = 118_000_000_000
    elif defect == "os-floor":
        recipe.topology.roles[0].resources.memory.peak_bytes = (
            SPARK_MEMORY_BYTES - ADMISSION_IDLE_USE_BYTES - OS_FLOOR_BYTES + 1
        )
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


def _incident_recipe() -> RecipeDefinition:
    return RecipeDefinition.model_validate_json(
        (ROOT / "recipes/qwen3-8-27b-nvfp4-dflash2-eugr-vllm-single.json").read_text()
    )


def test_target_only_speculation_is_not_a_second_model() -> None:
    recipe = _incident_recipe()
    for item in recipe.runtime.arguments:
        if item.name == "speculative-config":
            item.value = '{"method":"mtp","num_speculative_tokens":2}'
    assert not loads_draft_model(recipe)


def test_self_referencing_draft_is_not_a_second_model() -> None:
    recipe = _incident_recipe()
    for item in recipe.runtime.arguments:
        if item.name == "speculative-config":
            item.value = '{"method":"deepseek_mtp","model":"/models/target"}'
    assert not loads_draft_model(recipe)
