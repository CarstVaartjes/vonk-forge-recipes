"""Hardware-calibrated per-node vLLM envelopes, shared by producer guards."""

import json
from decimal import Decimal
from pathlib import Path

from vonk_forge_contracts import ModelDefinition, RecipeDefinition, document_sha256
from vonk_forge_contracts.memory_admission import evaluate_memory_admission

from qualification.memory_need import (
    ADMISSION_IDLE_USE_BYTES,
    OS_FLOOR_BYTES,
    OVERHEAD_BYTES,
    SPARK_MEMORY_BYTES,
    VLLM_DEFAULT_UTILIZATION,
    speculative_config,
)

# A speculative drafter holds nothing beside `gpu-memory-utilization`, whatever
# mechanism loads it (vLLM v0.30.0 lines; the same accounting at af5b4857e,
# the DFlash2 image's vLLM): the worker snapshots free memory (gpu_worker.py:451, psutil
# available on unified memory, mem_utils.py:143-150) before load_model loads the
# target and then the drafter (gpu_model_runner.py:5347-5350, every
# speculative-config method incl. custom proposers, :629-682); profile_run runs
# drafter.dummy_run (:6225) and the CUDA graph estimate captures it (default
# on, envs.py:2109); the KV budget is the requested share less everything
# consumed since the snapshot (mem_utils.py:312-321, gpu_worker.py:621-625), and
# the drafter's attention layers draw on that pool. Its weights therefore count
# inside the budget (memory_need.require_bounded_kv_pool). On hardware the
# pool-less single-Spark runs with a separate drafter held at most 9.9 GB beside
# utilization x memory with the OS included, inside the 8 GB overhead plus the
# 6 GB OS floor; the 2026-10-13 DFlash2 freeze at 0.7 (image 155ce16b, blamed
# on the platform) did not recur: that recipe passed at 0.6 on 431593a0 with
# 3.7 GB beside its budget.


def loads_draft_model(recipe: RecipeDefinition) -> bool:
    """Whether the speculative config names a model other than the served target."""
    served = set(recipe.runtime.entrypoint) | {
        str(item.value) for item in recipe.runtime.arguments if item.name == "model"
    }
    config = speculative_config(recipe)
    draft = config.get("model") if config is not None else None
    return draft is not None and str(draft) not in served


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
    if not arguments and any(
        item.name == "kv-cache-memory-bytes" for item in recipe.runtime.arguments
    ):
        # Need-based documents omit utilization: do not invent an implicit 0.9
        # reservation. Keep explicitly authored historical budgets calibrated.
        from qualification.memory_need import calculate_need, load_inputs

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
        need = calculate_need(recipe, models, load_inputs())
        required_peak = Decimal(need.total_bytes)
    else:
        required_peak = utilization * SPARK_MEMORY_BYTES + OVERHEAD_BYTES
    for role in recipe.topology.roles:
        memory = role.resources.memory
        label = f"{recipe.identity.slug}/{role.name} at utilization {utilization}"
        assert memory.peak_bytes >= required_peak, (
            f"{label}: peak {memory.peak_bytes} understates GPU budget plus overhead "
            f"{required_peak}"
        )
        assert evaluate_memory_admission(
            memory.peak_bytes,
            SPARK_MEMORY_BYTES,
            occupied_bytes=ADMISSION_IDLE_USE_BYTES,
            memory_floor_bytes=OS_FLOOR_BYTES,
        ).admissible, (
            f"{label}: peak plus OS floor exceeds an idle Spark's available memory"
        )


# The incident recipe at 0.85 measured 122_005_303_296 bytes on spark-3542
# (2026-10-15/16): its overhead beside the utilization share, and the idle
# available memory of that Spark, which the peak and the OS floor must fit.
INCIDENT_MEASURED_OVERHEAD_BYTES = 122_005_303_296 - int(
    Decimal("0.85") * SPARK_MEMORY_BYTES
)
