"""vLLM's GPU budget excludes unified-memory overhead on each GB10 node."""

from decimal import Decimal
from pathlib import Path

import pytest
from vonk_forge_contracts import RecipeDefinition

ROOT = Path(__file__).resolve().parents[1]
# Physical GB10 unified memory, not aggregate memory across the topology.
SPARK_MEMORY_BYTES = 130_663_231_488
# Lowest idle MemAvailable observed on a Spark; admission keeps the floor above it.
SPARK_IDLE_AVAILABLE_BYTES = 126_600_000_000
OVERHEAD_BYTES = 8_000_000_000
OS_FLOOR_BYTES = 6_000_000_000
VLLM_DEFAULT_UTILIZATION = Decimal("0.90")


def assert_memory_fit(recipe: RecipeDefinition) -> None:
    arguments = [
        argument
        for argument in recipe.runtime.arguments
        if argument.name == "gpu-memory-utilization"
    ]
    utilization = VLLM_DEFAULT_UTILIZATION
    if arguments:
        argument = arguments[-1]
        assert argument.setting is None, "Memory utilization must be a fixed budget"
        utilization = Decimal(str(argument.value))
    assert 0 < utilization <= 1
    required_peak = utilization * SPARK_MEMORY_BYTES + OVERHEAD_BYTES
    for role in recipe.topology.roles:
        memory = role.resources.memory
        label = f"{recipe.identity.slug}/{role.name} at utilization {utilization}"
        assert memory.peak_bytes >= required_peak, (
            f"{label}: peak {memory.peak_bytes} understates GPU budget plus overhead "
            f"{required_peak}"
        )
        assert memory.peak_bytes + OS_FLOOR_BYTES <= SPARK_IDLE_AVAILABLE_BYTES, (
            f"{label}: peak plus OS floor exceeds an idle Spark's available memory"
        )


@pytest.mark.parametrize(
    "path", sorted((ROOT / "recipes").glob("*.json")), ids=lambda path: path.stem
)
def test_vllm_recipe_fits_each_target_node(path: Path) -> None:
    recipe = RecipeDefinition.model_validate_json(path.read_text())
    if recipe.runtime.engine != "vllm":
        return
    assert_memory_fit(recipe)
    # Options cannot share an argument name, so each utilization override is
    # independent. Resolve each choice through the actual contract merger.
    for option in recipe.options:
        for choice in option.choices:
            assert_memory_fit(recipe.with_option_choices({option.name: choice.value}))


def test_incident_recipe_keeps_one_million_context_with_headroom() -> None:
    recipe = RecipeDefinition.model_validate_json(
        (
            ROOT / "recipes/qwen3-8-27b-aday777-abliterated-drowzeys-vllm-single.json"
        ).read_text()
    )
    assert recipe.settings.kind == "generation"
    assert recipe.settings.context_tokens.value == 1_048_576
    argument = next(
        item
        for item in recipe.runtime.arguments
        if item.name == "gpu-memory-utilization"
    )
    assert Decimal(str(argument.value)) == Decimal("0.85")
    assert recipe.topology.roles[0].resources.memory.reserve_bytes == OVERHEAD_BYTES
    assert_memory_fit(recipe)


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
