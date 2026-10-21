"""vLLM's GPU budget excludes unified-memory overhead on each GB10 node."""

import json
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
# A separately loaded speculative draft model (DFlash, DSpark, EAGLE, an external
# MTP checkpoint) holds memory outside `gpu-memory-utilization`. Calibrated from
# qwen3-8-27b-nvfp4-dflash2-eugr-vllm-single on a GB10 (2026-10-13): at 0.7 it
# drove ~116.4e9 bytes attributable to the run (121e9 used less 4.6e9 idle) within
# twelve minutes and froze the Spark, 17e9 above 0.7 x memory + 8e9. That sample
# was still climbing, so this is a floor; the sweep's memory-envelope note refines it.
DRAFT_MODEL_OVERHEAD_BYTES = 17_000_000_000
# Draft-model recipes whose weights leave no KV room once the draft overhead is
# counted on a fitting envelope: each needs its creator (more Sparks, a smaller
# context or drafter). Until then only the draft term is waived for them: the
# plain GPU budget and the OS floor still apply, and the entry goes stale, and
# fails, as soon as the recipe fits with the term counted.
PENDING_CREATOR = frozenset(
    {
        "glm-5-3-flash-exl3-dflash2-reederey87-vllm-dual",
        "glm-5-3-flash-exl3-dflash2-vllm-dual",
        "glm-5-3-flash-nvidia-nvfp4-dflash2-vllm-dual",
        "minimax-m3-w4a16-gptq-eagle3-tonyd2wild-vllm-dual",
    }
)


def loads_draft_model(recipe: RecipeDefinition) -> bool:
    """Whether the speculative config names a model other than the served target."""
    # SparkInfer builds and selects its DSpark draft inside the recipe wrapper,
    # rather than an authored speculative-config argument. Budget it too.
    wrapper = ROOT / recipe.execution.build.context.path / "vllm-wrapper.sh"
    if wrapper.is_file() and "export SPEC_MODEL_PATH=" in wrapper.read_text():
        return True
    arguments = recipe.runtime.arguments
    served = set(recipe.runtime.entrypoint) | {
        str(item.value) for item in arguments if item.name == "model"
    }
    for argument in arguments:
        if argument.name != "speculative-config":
            continue
        config = argument.value
        if isinstance(config, str):
            try:
                config = json.loads(config)
            except json.JSONDecodeError:
                continue  # "None" switches speculation off
        draft = config.get("model") if isinstance(config, dict) else None
        if draft is not None and str(draft) not in served:
            return True
    return False


def non_utilization_bytes(
    recipe: RecipeDefinition, count_draft_model: bool = True
) -> int:
    """Memory a vLLM node holds beside its `gpu-memory-utilization` budget."""
    if count_draft_model and loads_draft_model(recipe):
        return OVERHEAD_BYTES + DRAFT_MODEL_OVERHEAD_BYTES
    return OVERHEAD_BYTES


def assert_memory_fit(recipe: RecipeDefinition, count_draft_model: bool = True) -> None:
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
    required_peak = utilization * SPARK_MEMORY_BYTES + non_utilization_bytes(
        recipe, count_draft_model
    )
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
    count = path.stem not in PENDING_CREATOR
    assert_memory_fit(recipe, count)
    # Options cannot share an argument name, so each utilization override is
    # independent. Resolve each choice through the actual contract merger.
    for option in recipe.options:
        for choice in option.choices:
            assert_memory_fit(
                recipe.with_option_choices({option.name: choice.value}), count
            )


# The incident recipe at 0.85 measured 122_005_303_296 bytes on spark-3542
# (2026-10-15/16): its overhead beside the utilization share, and the idle
# available memory of that Spark, which the peak and the OS floor must fit.
INCIDENT_MEASURED_OVERHEAD_BYTES = 122_005_303_296 - int(
    Decimal("0.85") * SPARK_MEMORY_BYTES
)
SPARK_3542_IDLE_AVAILABLE_BYTES = 125_650_657_280


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
    utilization = Decimal(str(argument.value))
    memory = recipe.topology.roles[0].resources.memory
    assert memory.reserve_bytes == OVERHEAD_BYTES
    assert_memory_fit(recipe)
    # Declares what it really uses, and that fits the Spark it froze.
    assert memory.peak_bytes >= (
        utilization * SPARK_MEMORY_BYTES + INCIDENT_MEASURED_OVERHEAD_BYTES
    )
    assert memory.peak_bytes + OS_FLOOR_BYTES <= SPARK_3542_IDLE_AVAILABLE_BYTES


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
