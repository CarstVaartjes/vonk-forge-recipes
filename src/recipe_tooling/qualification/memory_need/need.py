"""Offline memory demand, separate from historical engine reservation fractions.

Unchanged catalog documents retain their behavior. A new document's fixed pool
must hold one full-context request by source-backed cache geometry, and its
declaration must cover the loading floor and be admissible.
The committed content snapshot is not a qualification status list.
"""

from __future__ import annotations

from collections.abc import Mapping
from decimal import Decimal

from recipe_tooling.qualification.memory_need import args as mn_args
from recipe_tooling.qualification.memory_need import sglang as mn_sglang
from recipe_tooling.qualification.memory_need import vllm as mn_vllm
from recipe_tooling.qualification.memory_need.constants import (
    IDLE_AVAILABLE_BYTES,
    OVERHEAD_BYTES,
    TOTAL_BYTES,
)
from recipe_tooling.qualification.memory_need.geometry import (
    MemoryInputs,
    resident_weight_bytes,
)
from recipe_tooling.qualification.memory_need.records import (
    MemoryNeed,
    PerRankMemoryNeed,
)
from vonk_forge_contracts import (
    ModelDefinition,
    RecipeDefinition,
)
from vonk_forge_contracts.api_checks import CONTEXT_ARGUMENTS


def calculate_need(
    recipe: RecipeDefinition,
    models: Mapping[str, ModelDefinition],
    inputs: MemoryInputs,
) -> MemoryNeed:
    """Informative full workload: declared context times engine concurrency,
    including selected drafts. Warnings only.

    Currently supports single-node ordinary attention and supplied compressed
    geometry. A new topology/automatic scheduler needs explicit inputs rather
    than borrowing a benchmark concurrency or a fraction of physical memory.
    """
    label = recipe.identity.slug
    if recipe.topology.node_count != 1:
        raise ValueError(
            f"{label}: supply a per-rank need model for this multi-Spark recipe id"
        )
    settings = recipe.settings
    if settings.kind != "generation" or settings.concurrency is None:
        raise ValueError(
            f"{label}: declare engine context and concurrency for memory sizing"
        )
    if mn_args.loads_draft_model(recipe) and len(recipe.models) == 1:
        raise ValueError(
            f"{label}: include the wrapper's draft weights in the need model"
        )
    context_flag = CONTEXT_ARGUMENTS[recipe.runtime.engine][0]
    concurrency_flag = (
        "max-num-seqs" if recipe.runtime.engine == "vllm" else "max-running-requests"
    )
    context = mn_args.positive_argument(recipe, context_flag)
    concurrency = mn_args.positive_argument(recipe, concurrency_flag)
    # Each choice can override a setting-bound argument through the contract's
    # option merger; calculate the actual variant, not just the base settings.
    block_flag = "block-size" if recipe.runtime.engine == "vllm" else "page-size"
    block = (
        mn_args.positive_argument(recipe, block_flag)
        if mn_args.argument_value(recipe, block_flag) is not None
        else (16 if recipe.runtime.engine == "vllm" else 1)
    )
    context = ((context + block - 1) // block) * block
    weights: list[int] = []
    kv = state = expansion = 0
    for selection in recipe.models:
        model = models[selection.model.content_sha256]
        selected = {item.file_id for item in selection.files}
        weights.append(
            sum(
                resident_weight_bytes(item.size_bytes, item.sha256, recipe, inputs)
                for item in model.files
                if item.id in selected and "weights" in item.roles
            )
        )
        geometry = mn_args.pinned_geometry(
            recipe, selection.model.slug, model, inputs.geometries
        )
        if geometry.sliding_layers > geometry.layers or (
            geometry.sliding_layers and not geometry.sliding_window
        ):
            raise ValueError(f"{label}: invalid sliding-window cache geometry")
        graph_flag = (
            "enforce-eager" if recipe.runtime.engine == "vllm" else "disable-cuda-graph"
        )
        if (
            geometry.resident_expansion_bytes
            and mn_args.argument_value(recipe, graph_flag) is not True
        ):
            raise ValueError(
                f"{label}: calibrated resident bound requires disabled graphs"
            )
        expansion += geometry.resident_expansion_bytes
        recurrent = mn_vllm.vllm_recurrent(recipe, selection.model.slug, geometry)
        if recurrent is not None:
            kv += mn_vllm.vllm_hybrid_need(
                recipe, geometry, recurrent, context, concurrency
            )
            continue
        kv += mn_vllm.cache_need_for_context(recipe, geometry, context) * concurrency
        slots = concurrency
        if geometry.state_padding:
            slots = mn_sglang.sglang_mamba_slots(recipe, label)
        state += geometry.state_bytes_per_request * (slots + geometry.state_padding)
    # A draft's weights are counted above, like the target's; vLLM holds
    # nothing else for it beside its budget (vllm_memory_fit.py).
    return MemoryNeed(
        weights[0], sum(weights[1:]), kv, state, resident_expansion_bytes=expansion
    )


def per_rank_need(
    recipe: RecipeDefinition,
    models: Mapping[str, ModelDefinition],
    inputs: MemoryInputs,
) -> PerRankMemoryNeed:
    """Optimistic loading floor, never a proof of full-context or runtime fit.

    Selected weights and explicit KV pools are counted once. Adaptive pool
    ceilings and the host pressure planning margin are warnings, not proof that
    a workload cannot run. See the calibration sources in recipe-authoring.md.
    """
    if recipe.runtime.engine not in {"vllm", "sglang"}:
        raise ValueError("memory need requires a vLLM/SGLang recipe")
    kv_bytes = 0
    pool = mn_args.argument_value(recipe, "kv-cache-memory-bytes")
    if pool is None:
        pool = mn_args.argument_value(recipe, "kv-cache-memory")
    if pool is not None:
        kv_bytes = mn_args.cache_bytes(pool)
    loading_floor = (
        mn_args.rank_weight_bytes(recipe, models, inputs) + kv_bytes + OVERHEAD_BYTES
    )
    # Utilization sizes an adaptive pool; it is not irreducible workload demand.
    # Physical fit uses observed idle availability; a changed document's peak
    # must cover this floor and be admissible on an idle Spark. For multi-Spark
    # recipes the 16 GiB planning guard (LIMIT_BYTES) is only a warning: the
    # hardware-proven set runs above it.
    return PerRankMemoryNeed(
        loading_floor, IDLE_AVAILABLE_BYTES, recipe.topology.node_count
    )


def declaration_problems(
    recipe: RecipeDefinition, need: PerRankMemoryNeed
) -> list[str]:
    """An oversized estimate is valid; hiding demand is not.

    ``reserve_bytes`` is informational and never charged (admission holds the
    platform floor itself), so it is not judged here.
    """
    problems: list[str] = []
    fraction = mn_args.argument_value(recipe, "mem-fraction-static")
    static_peak = (
        Decimal(str(fraction)) * TOTAL_BYTES + OVERHEAD_BYTES
        if recipe.runtime.engine == "sglang" and fraction is not None
        else None
    )
    for role in recipe.topology.roles:
        memory = role.resources.memory
        if memory.peak_bytes < need.peak_bytes:
            problems.append(f"{role.name}: peak understates need {need.peak_bytes}")
        if static_peak is not None and (
            not static_peak.is_finite() or static_peak > memory.peak_bytes
        ):
            problems.append(
                f"{role.name}: peak understates SGLang static allocation plus overhead"
            )
    return problems
