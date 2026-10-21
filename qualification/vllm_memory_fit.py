"""Hardware-calibrated per-node vLLM envelopes, shared by producer guards."""

import json
from decimal import Decimal
from pathlib import Path

from vonk_forge_contracts import ModelDefinition, RecipeDefinition, document_sha256

from qualification.memory_need import (
    DRAFT_MODEL_OVERHEAD_BYTES,
    OS_FLOOR_BYTES,
    OVERHEAD_BYTES,
    SPARK_IDLE_AVAILABLE_BYTES,
    SPARK_MEMORY_BYTES,
)

VLLM_DEFAULT_UTILIZATION = Decimal("0.90")
# A separately loaded speculative draft model (DFlash, DSpark, EAGLE, an external
# MTP checkpoint) holds memory outside `gpu-memory-utilization`. Calibrated from
# qwen3-8-27b-nvfp4-dflash2-eugr-vllm-single on a GB10 (2026-10-13): at 0.7 it
# drove ~116.4e9 bytes attributable to the run (121e9 used less 4.6e9 idle) within
# twelve minutes and froze the Spark, 17e9 above 0.7 x memory + 8e9. That sample
# was still climbing, so this is a floor; the sweep's memory-envelope note refines it.
# Draft-model recipes whose weights leave no KV room once the draft overhead is
# counted on a fitting envelope: each needs its creator (more Sparks, a smaller
# context or drafter). Until then only the draft term is waived for them: the
# plain GPU budget and the OS floor still apply, and the entry goes stale, and
# fails, as soon as the recipe fits with the term counted.
PENDING_CREATOR = frozenset(
    {
        "glm-5-3-flash-exl3-dflash2-vllm-dual",
        "glm-5-3-flash-exl3-k2-0xsero-vllm-single",
        "glm-5-3-flash-nvidia-nvfp4-dflash2-vllm-dual",
        "laguna-s-2-1-nvfp4-dflash-vllm-mia-single",
        "minimax-m3-w4a16-gptq-eagle3-tonyd2wild-vllm-dual",
    }
)


def loads_draft_model(recipe: RecipeDefinition) -> bool:
    """Whether the speculative config names a model other than the served target."""
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
    if not arguments and any(
        item.name == "kv-cache-memory-bytes" for item in recipe.runtime.arguments
    ):
        # Need-based documents omit utilization: do not invent an implicit 0.9
        # reservation. Keep explicitly authored historical budgets calibrated.
        from qualification.memory_need import load_inputs, require_memory_need

        assert recipe.topology.node_count == 1, (
            "Absolute multi-Spark pools need a per-rank need model"
        )
        root = Path(__file__).resolve().parents[1]
        models: dict[str, ModelDefinition] = {}
        for selection in recipe.models:
            document = json.loads(
                (root / "models" / f"{selection.model.slug}.json").read_text()
            )
            models[document_sha256(document)] = ModelDefinition.model_validate(document)
        need = require_memory_need(recipe, models, load_inputs(), document_digest="")
        assert need is not None
        required_peak = Decimal(need.total_bytes)
    else:
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


# The incident recipe at 0.85 measured 122_005_303_296 bytes on spark-3542
# (2026-10-15/16): its overhead beside the utilization share, and the idle
# available memory of that Spark, which the peak and the OS floor must fit.
INCIDENT_MEASURED_OVERHEAD_BYTES = 122_005_303_296 - int(
    Decimal("0.85") * SPARK_MEMORY_BYTES
)
