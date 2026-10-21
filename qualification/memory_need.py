"""Shared GB10 vLLM demand estimates; a capacity warning is not an admission ban."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from decimal import ROUND_CEILING, Decimal
from pathlib import Path

from vonk_forge_contracts import ModelDefinition, RecipeDefinition

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


# Lower observed idle MemAvailable and the platform's sustained-pressure floor.
SPARK_3542_IDLE_AVAILABLE_BYTES = 125_650_657_280
HOST_HEADROOM_BYTES = 16 * 1024**3
SAFE_WORKLOAD_BYTES = SPARK_3542_IDLE_AVAILABLE_BYTES - HOST_HEADROOM_BYTES


def selected_weight_bytes(
    recipe: RecipeDefinition, models: Mapping[str, ModelDefinition]
) -> int:
    weights: dict[str, int] = {}
    for selection in recipe.models:
        selected = {item.file_id for item in selection.files}
        for item in models[selection.model.content_sha256].files:
            if item.id in selected and "weights" in item.roles:
                weights[item.sha256] = item.size_bytes
    return sum(weights.values())


def cache_bytes(value: object) -> int:
    """vLLM integers: uppercase suffixes are binary, lowercase are decimal."""
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


@dataclass(frozen=True)
class MemoryNeed:
    peak_bytes: int
    safe_limit_bytes: int
    node_count: int

    @property
    def fits(self) -> bool:
        return self.peak_bytes <= self.safe_limit_bytes

    @property
    def verdict(self) -> str:
        return f"{'fits' if self.fits else 'does not fit'} {self.node_count} Sparks"


def memory_need(
    recipe: RecipeDefinition, models: Mapping[str, ModelDefinition]
) -> MemoryNeed:
    """Lower bound, not a measurement: engine budget and selected weight/KV floor.

    Dividing all weights evenly is deliberately optimistic: replicated tensors,
    quantization expansion and startup peaks can only increase the loading floor.
    A separate vLLM drafter adds the calibrated 17 GB outside the GPU budget.
    """
    if recipe.runtime.engine != "vllm":
        raise ValueError("the vLLM need calculator requires a vLLM recipe")
    utilization = VLLM_DEFAULT_UTILIZATION
    kv_bytes = 0
    for argument in recipe.runtime.arguments:
        if argument.name == "gpu-memory-utilization":
            if argument.setting is not None:
                raise ValueError("memory utilization must be a fixed budget")
            utilization = Decimal(str(argument.value))
        elif argument.name == "kv-cache-memory-bytes":
            if argument.setting is not None:
                raise ValueError("KV bytes must be a fixed budget")
            kv_bytes = cache_bytes(argument.value)
    if not utilization.is_finite() or not 0 < utilization <= 1:
        raise ValueError("memory utilization must be in (0, 1]")
    parallel = recipe.topology.parallelism
    shards = parallel.tensor * parallel.pipeline
    loading_floor = (
        (selected_weight_bytes(recipe, models) + shards - 1) // shards
        + kv_bytes
        + OVERHEAD_BYTES
    )
    budget = utilization * SPARK_MEMORY_BYTES + non_utilization_bytes(recipe)
    peak = max(int(budget.to_integral_value(rounding=ROUND_CEILING)), loading_floor)
    return MemoryNeed(peak, SAFE_WORKLOAD_BYTES, recipe.topology.node_count)


def declaration_problems(recipe: RecipeDefinition, need: MemoryNeed) -> list[str]:
    """An oversized estimate is valid; hiding demand or headroom is not."""
    problems: list[str] = []
    for role in recipe.topology.roles:
        memory = role.resources.memory
        if memory.peak_bytes < need.peak_bytes:
            problems.append(f"{role.name}: peak understates need {need.peak_bytes}")
        if memory.reserve_bytes < HOST_HEADROOM_BYTES:
            problems.append(
                f"{role.name}: reserve understates host headroom {HOST_HEADROOM_BYTES}"
            )
    return problems
