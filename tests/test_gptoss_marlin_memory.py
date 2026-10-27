"""Bound the gpt-oss-120b vLLM need: weights, absolute KV and Marlin expansion."""

from __future__ import annotations

from pathlib import Path

import pytest
from catalog_documents import catalog_models
from vonk_forge_contracts import RecipeDefinition
from vonk_forge_contracts.memory_admission import evaluate_memory_admission
from vonk_forge_contracts.recipe import RecipeRuntimeArgument

from qualification.memory_need import (
    ADMISSION_IDLE_USE_BYTES,
    TOTAL_BYTES,
    argument_value,
    calculate_need,
    load_inputs,
    require_memory_need,
    vllm_utilization_envelope,
)

ROOT = Path(__file__).resolve().parents[1]


def test_full_context_need_and_absolute_pool_replace_the_failed_fraction():
    recipe = RecipeDefinition.model_validate_json(
        (ROOT / "recipes/gpt-oss-120b-nvidia-vllm-single.json").read_text()
    )
    inputs = load_inputs()
    need = calculate_need(recipe, catalog_models(), inputs)
    assert need.weights_bytes == 65_248_893_184
    assert need.kv_bytes == 9_663_676_416
    assert need.resident_expansion_bytes == 10_000_000_000
    assert need.total_bytes == 92_912_569_600 < need.limit_bytes
    assert argument_value(recipe, "max-model-len") == 131_072
    assert argument_value(recipe, "max-num-seqs") == 1
    assert argument_value(recipe, "enforce-eager") is True
    assert argument_value(recipe, "max-num-batched-tokens") == 2048
    assert argument_value(recipe, "gpu-memory-utilization") == "0.649858"
    assert argument_value(recipe, "kv-cache-memory-bytes") == need.kv_bytes
    assert recipe.topology.roles[0].resources.memory.peak_bytes == need.total_bytes
    require_memory_need(recipe, catalog_models(), inputs, document_digest="")
    # The actual shared admission contract charges host occupancy exactly once.
    admission = evaluate_memory_admission(
        pool_bytes=TOTAL_BYTES,
        occupied_bytes=ADMISSION_IDLE_USE_BYTES,
        demand_bytes=need.total_bytes,
    )
    assert admission.admissible
    recipe.runtime.arguments = [
        item
        for item in recipe.runtime.arguments
        if item.name != "kv-cache-memory-bytes"
    ] + [RecipeRuntimeArgument(name="gpu-memory-utilization", value="0.85")]
    # Content 4036b68d's launch exhausted host memory on 2026-10-21: its peak
    # covered 0.85 plus overhead, not also the 10 GB Marlin repack expansion.
    envelope = vllm_utilization_envelope(recipe)
    assert envelope is not None
    recipe.topology.roles[0].resources.memory.peak_bytes = envelope
    require_memory_need(recipe, catalog_models(), inputs, document_digest="")
    with pytest.raises(ValueError, match="absolute kv-cache-memory-bytes"):
        require_memory_need(
            recipe, catalog_models(), inputs, document_digest="", touched=True
        )


def test_resident_expansion_rejects_graph_and_image_drift():
    recipe = RecipeDefinition.model_validate_json(
        (ROOT / "recipes/gpt-oss-120b-nvidia-vllm-single.json").read_text()
    )
    inputs = load_inputs()
    eager = next(
        item for item in recipe.runtime.arguments if item.name == "enforce-eager"
    )
    eager.value = False
    with pytest.raises(ValueError, match="requires disabled graphs"):
        calculate_need(recipe, catalog_models(), inputs)
    eager.value = True
    recipe.execution.build.base_image.digest = "0" * 64
    with pytest.raises(ValueError, match="does not match the pinned image"):
        calculate_need(recipe, catalog_models(), inputs)
