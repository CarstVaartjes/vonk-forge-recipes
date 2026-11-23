"""Memory estimate warnings and findings."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from decimal import Decimal
from functools import cache

from recipe_tooling.qualification.memory_need import args as mn_args
from recipe_tooling.qualification.memory_need import constants as mn_constants
from recipe_tooling.qualification.memory_need import evidence as mn_evidence
from recipe_tooling.qualification.memory_need import need as mn_need
from recipe_tooling.qualification.memory_need import sglang as mn_sglang
from recipe_tooling.qualification.memory_need import swap as mn_swap
from recipe_tooling.qualification.memory_need import vllm as mn_vllm
from recipe_tooling.qualification.memory_need.constants import (
    ADMISSION_IDLE_USE_BYTES,
    ADMISSION_MAXIMUM_PEAK_BYTES,
    LIMIT_BYTES,
    OVERHEAD_BYTES,
    SPARK_MEMORY_BYTES,
    TOTAL_BYTES,
    VLLM_DEFAULT_UTILIZATION,
    budget_margin_bytes,
)
from recipe_tooling.qualification.memory_need.geometry import MemoryInputs
from recipe_tooling.qualification.memory_need.records import (
    EnvelopeException,
    EstimateFinding,
    EstimateFindingKind,
    MissingEstimateInput,
)
from vonk_forge_contracts import (
    ModelDefinition,
    RecipeDefinition,
)
from vonk_forge_contracts.memory_admission import (
    evaluate_memory_admission,
)


@cache
def envelope_exceptions() -> dict[str, EnvelopeException]:
    document = json.loads(
        (mn_constants.ROOT / "policy/memory-envelope-exceptions.json").read_text()
    )
    return {
        slug: EnvelopeException.model_validate(entry)
        for slug, entry in document.items()
    }


def envelope_exception(slug: str, document_digest: str) -> bool:
    """Whether this exact recipe content has a recorded owner-approved exception.

    Bound to the content digest: any edit to the recipe expires the entry.
    """
    entry = envelope_exceptions().get(slug)
    return (
        entry is not None
        and bool(document_digest)
        and (entry.content_sha256 == document_digest)
    )


def estimate_problems(
    label: str, rule: Callable[[], list[str]]
) -> list[EstimateFinding]:
    """One estimate rule's findings; an estimate that cannot be computed is one too."""
    try:
        return [
            EstimateFinding(EstimateFindingKind.DISAGREEMENT, text) for text in rule()
        ]
    except (ValueError, TypeError, KeyError) as error:
        kind = (
            EstimateFindingKind.UNAVAILABLE
            if isinstance(error, MissingEstimateInput)
            else EstimateFindingKind.DISAGREEMENT
        )
        return [EstimateFinding(kind, f"{label}: estimate unavailable: {error}")]


def memory_estimate_warnings(
    recipe: RecipeDefinition,
    models: Mapping[str, ModelDefinition],
    inputs: MemoryInputs,
    *,
    document_digest: str,
    touched: bool = False,
) -> list[str]:
    """The text of ``memory_estimate_findings``."""
    return [
        finding.text
        for finding in memory_estimate_findings(
            recipe, models, inputs, document_digest=document_digest, touched=touched
        )
    ]


def memory_estimate_findings(
    recipe: RecipeDefinition,
    models: Mapping[str, ModelDefinition],
    inputs: MemoryInputs,
    *,
    document_digest: str,
    touched: bool = False,
) -> list[EstimateFinding]:
    """Where the declared memory disagrees with the estimate; none of it blocks.

    The recipes are estimates until the hardware try measures them, so every
    finding here is a warning (the contract shape stays validated by
    RecipeMemoryResources). Calibrated on the proven set: a fixed pool holds one
    full-context request, not context x concurrency, and the declared peak
    covers the loading floor (and a vLLM utilization budget, on any Spark count,
    with the margin vLLM holds beside it) and is admissible. ``touched`` marks a
    recipe file the change under review edits; only those are asked to bound a
    utilization-sized KV pool. Recorded content is judged by hardware, so only
    its unproven multi-Spark SGLang startup room is reported.
    """
    key = f"{recipe.identity.publisher}/{recipe.identity.slug}"
    warnings = estimate_problems(
        key, lambda: mn_vllm.absolute_kv_warnings(recipe, models, inputs)
    )
    multi_sglang = [
        variant
        for variant in mn_args.memory_variants(recipe)
        if variant.runtime.engine == "sglang" and variant.topology.node_count > 1
    ]
    if mn_evidence.unchanged_recipe(recipe, document_digest, inputs):
        if document_digest and not mn_evidence.hardware_proven(recipe, document_digest):
            for variant in multi_sglang:
                warnings += estimate_problems(
                    key,
                    lambda variant=variant: mn_sglang.sglang_rank_budget_warnings(
                        variant, models, inputs, key
                    ),
                )
        return warnings
    warnings += [
        EstimateFinding(EstimateFindingKind.DISAGREEMENT, text)
        for text in mn_swap.admissible_declaration_warnings(recipe)
    ]
    if recipe.runtime.engine not in {"vllm", "sglang"}:
        return warnings
    single = recipe.topology.node_count == 1
    measured = mn_evidence.measured_budget(recipe)
    for variant in mn_args.memory_variants(recipe):
        warnings += estimate_problems(
            key,
            lambda variant=variant: variant_estimate_warnings(
                variant,
                recipe,
                models,
                inputs,
                key,
                document_digest,
                touched,
                single,
                mn_evidence.keeps_measured_peaks(variant, measured),
            ),
        )
    return warnings


def variant_estimate_warnings(
    variant: RecipeDefinition,
    recipe: RecipeDefinition,
    models: Mapping[str, ModelDefinition],
    inputs: MemoryInputs,
    key: str,
    document_digest: str,
    touched: bool,
    single: bool,
    keep: bool = False,
) -> list[str]:
    need = mn_need.per_rank_need(variant, models, inputs)
    # Proven single-Spark documents run below the multi-Spark SGLang
    # static-peak declaration; the loading floor, vLLM's budget and
    # SGLang's weights, expansion and capped pools hold there, because the
    # Controller reserves the declared peak.
    vllm = variant.runtime.engine == "vllm"
    envelope = (
        None
        if not single
        or envelope_exception(recipe.identity.slug, document_digest)
        or (keep and vllm)
        else mn_evidence.declarable_vllm_envelope(variant)
        if vllm
        else min(
            mn_sglang.sglang_static_need(variant, models, inputs, key, pinned=False)
            + OVERHEAD_BYTES,
            ADMISSION_MAXIMUM_PEAK_BYTES,
        )
    )
    engine = "vLLM budget" if vllm else "SGLang static need"
    problems = (
        [
            f"{role.name}: peak understates need {need.peak_bytes}"
            for role in variant.topology.roles
            if role.resources.memory.peak_bytes < need.peak_bytes
        ]
        + [
            f"{role.name}: peak understates {engine} plus overhead {envelope}"
            for role in variant.topology.roles
            if envelope is not None and role.resources.memory.peak_bytes < envelope
        ]
        if single
        else mn_need.declaration_problems(variant, need)
        + [
            f"{role.name}: peak understates vLLM budget plus overhead {multi}"
            for role in variant.topology.roles
            if not keep
            and (multi := mn_evidence.declarable_vllm_envelope(variant)) is not None
            and role.resources.memory.peak_bytes < multi
        ]
    )
    warnings = [f"{key}: " + "; ".join(problems)] if problems else []
    if not single and variant.runtime.engine == "sglang":
        warnings += mn_sglang.sglang_rank_budget_warnings(variant, models, inputs, key)
    elif touched and not envelope_exception(recipe.identity.slug, document_digest):
        warnings += bounded_kv_warnings(variant, models, inputs, key)
    if not single:
        warnings += startup_budget_warnings(variant)
    return warnings


def bounded_kv_warnings(
    recipe: RecipeDefinition,
    models: Mapping[str, ModelDefinition],
    inputs: MemoryInputs,
    key: str,
) -> list[str]:
    """A touched recipe should bound KV; an estimate, calibrated on the proven set.

    A touched pool-less vLLM budget, on one Spark or several, must be
    admissible on an idle Spark. The 2026-10-26 Laguna M.1 stop (TP=2, 0.82,
    475,392 pool tokens, 16 GiB MemAvailable) was the host memory guard's
    per-sample 16 GiB bar, which the swap-aware guard (vonk-forge #1936)
    removed; GLM 5.3 Flash NVFP4 DFlash2 dual passed pool-less at the
    creator's 0.86 on hardware (7cb340f9a04b, 2026-10-08).
    Proven single-Spark documents run up to 0.86 (SGLang 0.85), whose share
    plus overhead is the largest admitted peak, 120,370,379,081 bytes. The
    budget, plus the source-backed resident expansion of geometry pinned to
    the launch, is what admission checks: gpt-oss-120b 4036b68d at
    0.85 beside its 10 GB Marlin repack exhausted host memory on 2026-10-21.
    A declared SGLang single fraction, capped or not, must be admissible and
    hold its weights, expansion and capped pools: GLM a0 fa86311a at 0.75 was
    OOM-killed on 2026-10-21.

    A pool-less vLLM budget must also hold the selected weights per rank plus
    the runtime overhead, or vLLM finds no memory for KV cache blocks: Laguna
    S 2.1 r0b0tlab 7b2c74ef at 0.6 left 4.25 GB beside 74.15 GB of target and
    draft weights (2026-10-19, 2026-10-27). The proven set leaves at least
    17.89 GB (d3-lite at 0.15). Where the launch pins cache geometry, the
    budget also holds one full-context request, which vLLM requires before it
    starts: Laguna S 2.1 Mia b72a8ee1 at 0.85 had 1.12 GB beside 101.94 GB of
    weights for a 13.0 GB BF16 262,144-token request (vLLM reported -23.95 GiB
    of KV cache memory on 2026-10-27).

    An absolute pool must hold one full-context request
    (absolute_kv_warnings), not context x max-num-seqs: vLLM schedules
    sequences within the pool. Proven duals hold fewer: DeepSeek V4 Flash 0731
    r0b0tlab (16.0 GB for 16 x 327,680 tokens), GLM 5.3 Flash NVFP4 Spark eugr
    (14.69 GB for 4 x 500,000) and the EXL3 Mia dual (2.5 GiB, 774,400 tokens
    measured upstream, for 2 x 600,000). Sizing the DSpark vLLM 0.25 dual pool
    for 16 x 200,000 tokens (27.31 GB) exhausted host memory on both Sparks.
    SGLang derives an omitted mem-fraction-static at launch; the proven
    Nemotron 3 Nano NVFP4 NVIDIA SGLang single runs on it, so none is required.
    """
    warnings: list[str] = []
    vllm = recipe.runtime.engine == "vllm"
    name = "gpu-memory-utilization" if vllm else "mem-fraction-static"
    fraction = mn_args.argument_value(recipe, name)
    if not vllm and recipe.topology.node_count == 1:
        if fraction is None:
            return warnings
        # SGLang profiles after loading: the static budget must hold the
        # weights, their expansion and any capped pools (GLM a0 fa86311a at
        # 0.75 was OOM-killed), and a cap bounds the pools, not the budget the
        # fraction reserves.
        share = Decimal(str(fraction))
        resident = mn_sglang.sglang_static_need(
            recipe, models, inputs, key, pinned=False
        )
        base = mn_sglang.sglang_preload_free_bytes(recipe, models, inputs)
        if share * base < resident:
            warnings.append(
                f"{key}: mem-fraction-static {fraction} x SGLang pre-load free "
                f"{base} cannot hold the selected weights, resident "
                f"expansion and capped pools ({resident} bytes)"
            )
        capped = mn_sglang.sglang_capped_static_need(recipe, models, inputs, key)
        demand = mn_sglang.sglang_demand(recipe, capped)
        if (
            demand is None
            or not evaluate_memory_admission(
                int(demand),
                SPARK_MEMORY_BYTES,
                occupied_bytes=ADMISSION_IDLE_USE_BYTES,
            ).admissible
        ):
            warnings.append(
                f"{key}: mem-fraction-static {fraction} share of {TOTAL_BYTES} bytes "
                f"plus {OVERHEAD_BYTES if capped is not None else budget_margin_bytes(recipe)} "
                "overhead is not admissible on an idle Spark"
            )
        return warnings
    pools = (
        ("kv-cache-memory-bytes", "kv-cache-memory") if vllm else ("max-total-tokens",)
    )
    if any(mn_args.argument_value(recipe, name) is not None for name in pools):
        return warnings
    if fraction is None and not vllm:
        return warnings  # Multi-Spark SGLang uses sglang_rank_budget_warnings.
    share = Decimal(str(fraction)) if fraction is not None else VLLM_DEFAULT_UTILIZATION
    weights = mn_args.rank_weight_bytes(recipe, models, inputs) if vllm else 0
    primary = recipe.models[0]
    pinned = mn_args.applicable_geometry(
        recipe,
        primary.model.slug,
        models[primary.model.content_sha256],
        inputs,
    )
    context = mn_vllm.absolute_kv_need(recipe, models, inputs) if vllm and pinned else 0
    if vllm and share * TOTAL_BYTES < weights + OVERHEAD_BYTES + context:
        request = (
            f" and one full-context request ({context} bytes/rank)" if context else ""
        )
        warnings.append(
            f"{key}: {name} {share} share of {TOTAL_BYTES} bytes cannot hold the "
            f"selected weights ({weights} bytes/rank) plus {OVERHEAD_BYTES} "
            f"overhead{request}, leaving vLLM no memory for KV cache blocks; raise "
            f"it within admission, shorten the context or set absolute {pools[0]}"
        )
    budget = (mn_evidence.vllm_utilization_envelope(recipe) or 0) + sum(
        geometry.resident_expansion_bytes
        for selection in recipe.models
        if (
            geometry := mn_args.applicable_geometry(
                recipe,
                selection.model.slug,
                models[selection.model.content_sha256],
                inputs,
            )
        )
        is not None
    )
    bounded = evaluate_memory_admission(
        budget, SPARK_MEMORY_BYTES, occupied_bytes=ADMISSION_IDLE_USE_BYTES
    ).admissible
    if not bounded:
        warnings.append(
            f"{key}: set absolute {pools[0]}, or a {name} whose share of "
            f"{TOTAL_BYTES} bytes plus {budget_margin_bytes(recipe)} overhead is admissible on "
            "an idle Spark"
        )
    return warnings


def startup_budget_warnings(recipe: RecipeDefinition) -> list[str]:
    """Absolute KV overrides cache sizing, not vLLM's earlier startup check.

    Beside a fixed pool the fraction sizes nothing; vLLM only requires free
    memory >= utilization x total before loading weights (v0.30.0 gpu_worker.py:451
    free-memory snapshot, mem_utils.py:143-150; request_memory in worker/utils.py
    at dda4668b). The Controller admits a role only when its declared peak fits
    observed free memory, so utilization x total <= peak guarantees that check
    on an admitted Spark; no overhead term belongs here. Creator evidence: Mia
    (DeepSeek-v4.1-Flash-EXL3-2x-DGX-Sparks .env.example#L143 at 6f7d1590, "only
    vLLM's startup sanity check, not a budget"), MiniMax M3 (DEFAULT-CONFIG.md
    #L137-L142 at a8101fa4: 0.93 failed by 0.02 GiB), Step 3.7
    (evidence/runtime/vllm-stepfun-tp2-failure.md#L19 at e79e192d).
    """
    if recipe.runtime.engine != "vllm" or (
        mn_args.argument_value(recipe, "kv-cache-memory-bytes") is None
        and mn_args.argument_value(recipe, "kv-cache-memory") is None
    ):
        return []
    fraction = mn_args.argument_value(recipe, "gpu-memory-utilization")
    if fraction is None:
        return [
            (
                f"{recipe.identity.slug}: absolute KV needs an explicit startup "
                "gpu-memory-utilization budget"
            )
        ]
    utilization = Decimal(str(fraction))
    if not utilization.is_finite() or not 0 < utilization <= 1:
        return [f"{recipe.identity.slug}: invalid vLLM startup budget"]
    return [
        f"{recipe.identity.slug}/{role.name}: vLLM startup free-memory "
        "check (utilization x total) exceeds declared peak"
        for role in recipe.topology.roles
        if utilization * TOTAL_BYTES > role.resources.memory.peak_bytes
    ]


def memory_warnings(
    recipe: RecipeDefinition,
    models: Mapping[str, ModelDefinition],
    inputs: MemoryInputs,
    document_digest: str = "",
    *,
    touched: bool = False,
) -> list[str]:
    """Catalog estimates and where the declaration disagrees, then the notes.

    Neither changes hardware qualification: the try decides, and the measured
    use replaces the estimate.
    """
    return [
        f"WARNING {item}"
        for item in memory_estimate_warnings(
            recipe, models, inputs, document_digest=document_digest, touched=touched
        )
    ] + memory_notes(recipe, models, inputs)


def memory_notes(
    recipe: RecipeDefinition,
    models: Mapping[str, ModelDefinition],
    inputs: MemoryInputs,
) -> list[str]:
    """The informational estimate lines: swap, full workload and loading floor."""
    if recipe.runtime.engine not in {"vllm", "sglang"}:
        return mn_swap.swap_warnings(recipe)
    missing = {
        selection.model.content_sha256 for selection in recipe.models
    } - models.keys()
    if missing:
        return [
            (
                f"WARNING {recipe.identity.slug}: memory estimate unavailable; "
                f"missing Model inputs: {', '.join(sorted(missing))}"
            ),
            *mn_swap.swap_warnings(recipe),
        ]
    warnings = mn_swap.swap_warnings(recipe)
    for index, variant in enumerate(mn_args.memory_variants(recipe)):
        prefix = f"WARNING {recipe.identity.slug} [{index}]"
        try:
            need = mn_need.calculate_need(variant, models, inputs)
        except (ValueError, TypeError) as error:
            warnings.append(f"{prefix}: full-workload estimate unavailable: {error}")
        else:
            warnings.append(
                f"{prefix}: estimated workload {need.total_bytes} bytes/rank; "
                f"planning limit {LIMIT_BYTES}; hardware test required"
            )
        try:
            rank = mn_need.per_rank_need(variant, models, inputs)
        except (ValueError, TypeError) as error:
            warnings.append(f"{prefix}: loading floor unavailable: {error}")
        else:
            warnings.append(
                f"{prefix}: loading floor {rank.peak_bytes} bytes/rank; "
                f"observed idle capacity {rank.safe_limit_bytes}; no hardware verdict"
            )
    return warnings
