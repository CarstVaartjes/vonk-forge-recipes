"""GB10 engine reservations must leave the host pressure guard room to recover."""

from decimal import Decimal
from pathlib import Path

import pytest
from catalog_documents import catalog_models
from test_vllm_memory_fit import (
    OVERHEAD_BYTES,
    ROOT,
    SPARK_3542_IDLE_AVAILABLE_BYTES,
    SPARK_MEMORY_BYTES,
    non_utilization_bytes,
)
from vonk_forge_contracts import RecipeDefinition

# vonk-forge 040c51b8531523bebe542b5e1f7b9403e0767f72:
# rust/crates/vonk-agent-protocol/src/host_memory_guard_policy.rs, re-exported
# by rust/crates/vonk-agent-helper/src/host_memory_guard.rs. Full PSI >30% for
# 3 seconds below 16 GiB triggers the guard (also with an NVIDIA allocation
# failure). The old 6 GB OS floor in test_vllm_memory_fit is insufficient.
STALL_AVAILABLE_BYTES = 16 * 1024**3
MAX_WORKLOAD_BYTES = SPARK_3542_IDLE_AVAILABLE_BYTES - STALL_AVAILABLE_BYTES


def cache_bytes(value: object) -> int:
    """vLLM human-readable integers: G is binary, lowercase g is decimal."""
    text = str(value)
    multipliers = {
        "k": 1000,
        "m": 1000**2,
        "g": 1000**3,
        "K": 1024,
        "M": 1024**2,
        "G": 1024**3,
    }
    if text[-1:] in multipliers:
        return int(Decimal(text[:-1]) * multipliers[text[-1]])
    return int(text)


def selected_weight_bytes(recipe: RecipeDefinition) -> int:
    models = catalog_models()
    weights: dict[str, int] = {}
    for selection in recipe.models:
        selected = {item.file_id for item in selection.files}
        for item in models[selection.model.content_sha256].files:
            if item.id in selected and "weights" in item.roles:
                weights[item.sha256] = item.size_bytes
    return sum(weights.values())


def assert_host_headroom(recipe: RecipeDefinition) -> None:
    if recipe.topology.node_count != 1 or recipe.runtime.engine not in {
        "vllm",
        "sglang",
    }:
        return
    name = (
        "gpu-memory-utilization"
        if recipe.runtime.engine == "vllm"
        else "mem-fraction-static"
    )
    arguments = [item for item in recipe.runtime.arguments if item.name == name]
    assert arguments, (
        f"{recipe.identity.slug}: set {name} explicitly; engine defaults change"
    )
    argument = arguments[-1]
    assert argument.setting is None, "Memory reservation must be a fixed budget"
    fraction = Decimal(str(argument.value))
    assert fraction.is_finite() and 0 < fraction <= 1
    overhead = (
        non_utilization_bytes(recipe)
        if recipe.runtime.engine == "vllm"
        else OVERHEAD_BYTES
    )
    reserved = fraction * SPARK_MEMORY_BYTES + overhead
    assert reserved <= MAX_WORKLOAD_BYTES, (
        f"{recipe.identity.slug}: {name}={fraction} plus overhead reserves {reserved}; "
        f"maximum {MAX_WORKLOAD_BYTES} leaves the host guard's 16 GiB floor"
    )
    # Explicit vLLM KV bytes bypass gpu-memory-utilization. Count the selected
    # weights (including any drafter) instead of accepting a cosmetic fraction.
    for item in recipe.runtime.arguments:
        if item.name == "kv-cache-memory-bytes":
            assert item.setting is None
            reserved = (
                selected_weight_bytes(recipe) + cache_bytes(item.value) + OVERHEAD_BYTES
            )
            assert reserved <= MAX_WORKLOAD_BYTES, (
                f"{recipe.identity.slug}: explicit KV cache plus weights/overhead "
                f"reserves {reserved}, above {MAX_WORKLOAD_BYTES}"
            )


@pytest.mark.parametrize(
    "path", sorted((ROOT / "recipes").glob("*.json")), ids=lambda path: path.stem
)
def test_single_spark_engine_reservation_leaves_host_headroom(path: Path) -> None:
    recipe = RecipeDefinition.model_validate_json(path.read_text())
    assert_host_headroom(recipe)
    for option in recipe.options:
        for choice in option.choices:
            assert_host_headroom(
                recipe.with_option_choices({option.name: choice.value})
            )


@pytest.mark.parametrize(
    "engine,name",
    [("vllm", "gpu-memory-utilization"), ("sglang", "mem-fraction-static")],
)
@pytest.mark.parametrize("value", ["0.80", "0.85", "0.90", "0.95", None])
def test_guard_rejects_unsafe_and_implicit_reservations(
    engine: str, name: str, value: str | None
) -> None:
    recipe = RecipeDefinition.model_validate_json(
        (
            ROOT / f"recipes/llama-3-1-8b-instruct-nvfp4-nvidia-{engine}-single.json"
        ).read_text()
    )
    if value is None:
        recipe.runtime.arguments = [
            item for item in recipe.runtime.arguments if item.name != name
        ]
    else:
        next(
            item for item in recipe.runtime.arguments if item.name == name
        ).value = value
    with pytest.raises(AssertionError):
        assert_host_headroom(recipe)


def test_guard_rejects_absolute_kv_cache_bypassing_fraction() -> None:
    recipe = RecipeDefinition.model_validate_json(
        (
            ROOT / "recipes/nemotron-3-5-lightning-dspark-lowmem-canary-single.json"
        ).read_text()
    )
    next(
        item
        for item in recipe.runtime.arguments
        if item.name == "kv-cache-memory-bytes"
    ).value = "120G"
    with pytest.raises(AssertionError, match="explicit KV cache"):
        assert_host_headroom(recipe)


def test_bound_counts_guard_floor_os_and_runtime_overhead() -> None:
    assert Decimal("0.75") * SPARK_MEMORY_BYTES + OVERHEAD_BYTES <= MAX_WORKLOAD_BYTES
    assert Decimal("0.80") * SPARK_MEMORY_BYTES + OVERHEAD_BYTES > MAX_WORKLOAD_BYTES
