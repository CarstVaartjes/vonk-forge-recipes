"""vLLM's GPU budget excludes unified-memory overhead on each GB10 node."""

import json
from pathlib import Path

import pytest
from catalog_documents import catalog_models
from vonk_forge_contracts import RecipeDefinition, document_sha256

from qualification.memory_need import GUARD_BYTES, load_inputs, require_memory_need
from qualification.vllm_memory_fit import (
    OVERHEAD_BYTES,
    PENDING_CREATOR,
    SPARK_3542_IDLE_AVAILABLE_BYTES,
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
    key = f"{recipe.identity.publisher}/{recipe.identity.slug}"
    inputs = load_inputs()
    if key not in inputs.hardware_passes or recipe.topology.node_count != 1:
        require_memory_need(
            recipe,
            catalog_models(),
            load_inputs(),
            document_digest=document_sha256(document),
        )
        return
    count = path.stem not in PENDING_CREATOR
    assert_memory_fit(recipe, count)
    # Options cannot share an argument name, so each utilization override is
    # independent. Resolve each choice through the actual contract merger.
    for option in recipe.options:
        for choice in option.choices:
            assert_memory_fit(
                recipe.with_option_choices({option.name: choice.value}), count
            )


def test_unproven_incident_recipe_bounds_context_with_headroom() -> None:
    recipe = RecipeDefinition.model_validate_json(
        (
            ROOT / "recipes/qwen3-8-27b-aday777-abliterated-drowzeys-vllm-single.json"
        ).read_text()
    )
    assert recipe.settings.kind == "generation"
    assert recipe.settings.context_tokens.value == 8192
    assert any(
        item.name == "kv-cache-memory-bytes" for item in recipe.runtime.arguments
    )
    memory = recipe.topology.roles[0].resources.memory
    assert memory.reserve_bytes == GUARD_BYTES
    assert_memory_fit(recipe)
    assert memory.peak_bytes <= SPARK_3542_IDLE_AVAILABLE_BYTES - memory.reserve_bytes


@pytest.mark.parametrize("defect", ["torch-only", "os-floor", "worker", "implicit"])
def test_guard_rejects_unsafe_envelopes(defect: str) -> None:
    recipe = RecipeDefinition.model_validate_json(
        (ROOT / "recipes/deepseek-v4-flash-vision-exp-eugr-vllm-dual.json").read_text()
    )
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
