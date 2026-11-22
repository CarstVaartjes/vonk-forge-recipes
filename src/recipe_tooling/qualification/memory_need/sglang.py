"""SGLang static memory need."""

from __future__ import annotations

from collections.abc import Mapping
from decimal import ROUND_CEILING, Decimal

from vonk_forge_contracts import (
    ModelDefinition,
    RecipeDefinition,
)

from recipe_tooling.qualification.memory_need import args as mn_args
from recipe_tooling.qualification.memory_need import vllm as mn_vllm
from recipe_tooling.qualification.memory_need.constants import (
    OVERHEAD_BYTES,
    SGLANG_MAMBA_FULL_MEMORY_RATIO,
    SGLANG_MAMBA_SLOTS_PER_REQUEST,
    SGLANG_PRELOAD_FREE_BYTES,
    TOTAL_BYTES,
    budget_margin_bytes,
)
from recipe_tooling.qualification.memory_need.geometry import (
    HYBRID_ARCHITECTURES,
    CacheGeometry,
    MemoryInputs,
)


def sglang_token_pools(
    recipe: RecipeDefinition, geometry: CacheGeometry, tokens: int, key: str
) -> int:
    """Per-rank bytes of the full-attention pool and the pooled sliding layers."""
    need = mn_vllm.cache_need_for_context(recipe, geometry, tokens)
    if geometry.sglang_swa_layers:
        ratio = mn_args.argument_value(recipe, "swa-full-tokens-ratio")
        if ratio is None:
            raise ValueError(f"{key}: declare swa-full-tokens-ratio")
        swa = geometry.model_copy(
            update={
                "layers": geometry.sglang_swa_layers,
                "kv_heads": geometry.sglang_swa_kv_heads or geometry.kv_heads,
                "sliding_layers": 0,
                "token_bytes": None,
                "token_bytes_by_dtype": {},
                "replicated_token_bytes": 0,
            }
        )
        swa_tokens = int(
            (Decimal(str(ratio)) * tokens).to_integral_value(ROUND_CEILING)
        )
        need += mn_vllm.cache_need_for_context(recipe, swa, swa_tokens)
    return need


def sglang_mamba_slots(recipe: RecipeDefinition, label: str) -> int:
    """max-mamba-cache-size, which must hold five state slots per running request.

    kv_cache_configurator.resolve_max_num_reqs sets max_num_reqs to
    max-mamba-cache-size // 5 and raises at 0, so fewer slots either fail the
    start (operation 31e3de45, 2 slots) or silently lower the concurrency.
    """
    slots = mn_args.positive_argument(recipe, "max-mamba-cache-size")
    running = mn_args.positive_argument(recipe, "max-running-requests")
    if slots < SGLANG_MAMBA_SLOTS_PER_REQUEST * running:
        raise ValueError(
            f"{label}: max-mamba-cache-size {slots} cannot hold the "
            f"{SGLANG_MAMBA_SLOTS_PER_REQUEST} state slots of each of "
            f"{running} running requests"
        )
    return slots


def sglang_static_need(
    recipe: RecipeDefinition,
    models: Mapping[str, ModelDefinition],
    inputs: MemoryInputs,
    key: str,
    *,
    pinned: bool,
) -> int:
    """Per-rank bytes SGLang holds inside mem-fraction-static once profiled.

    Weights, the resident expansion of the selected geometry and, when
    max-total-tokens caps the pools, the capped token and state pools.
    ``pinned`` requires geometry for every Model (multi-Spark); otherwise only
    geometry pinned to this launch is charged and an uncapped pool, sized
    from what remains, adds nothing.
    """
    capped = mn_args.argument_value(recipe, "max-total-tokens") is not None
    tokens = mn_args.positive_argument(recipe, "max-total-tokens") if capped else 0
    need = mn_args.rank_weight_bytes(recipe, models, inputs)
    for selection in recipe.models:
        model = models[selection.model.content_sha256]
        geometry = (
            mn_args.pinned_geometry(
                recipe, selection.model.slug, model, inputs.geometries
            )
            if pinned
            else mn_args.applicable_geometry(
                recipe, selection.model.slug, model, inputs
            )
        )
        if geometry is None:
            continue
        need += geometry.resident_expansion_bytes
        if not capped:
            continue
        need += sglang_token_pools(recipe, geometry, tokens, key)
        if geometry.sglang_state_bytes_per_slot:
            if mn_args.argument_value(recipe, "max-mamba-cache-size") is None:
                raise ValueError(f"{key}: set absolute max-mamba-cache-size")
            slots = sglang_mamba_slots(recipe, key)
            need += geometry.sglang_state_bytes_per_slot * (slots + 1)
            if mn_args.argument_value(recipe, "speculative-algorithm") is not None:
                # _handle_max_mamba_cache: per-step intermediate state for the
                # running requests the slots admit, plus the padding slot.
                running = min(
                    mn_args.positive_argument(recipe, "max-running-requests"),
                    slots // SGLANG_MAMBA_SLOTS_PER_REQUEST,
                )
                need += (
                    geometry.sglang_state_bytes_per_slot
                    * (running + 1)
                    * mn_args.positive_argument(recipe, "speculative-num-draft-tokens")
                )
        if geometry.state_padding:
            # The pinned image's allocator charges max-mamba-cache-size plus
            # its padding slots (glm-a0-cache-sizing.py.txt).
            if mn_args.argument_value(recipe, "max-mamba-cache-size") is None:
                raise ValueError(f"{key}: set absolute max-mamba-cache-size")
            need += geometry.state_bytes_per_request * (
                sglang_mamba_slots(recipe, key) + geometry.state_padding
            )
    return need


def sglang_capped_static_need(
    recipe: RecipeDefinition,
    models: Mapping[str, ModelDefinition],
    inputs: MemoryInputs,
    key: str,
) -> int | None:
    """Per-rank static need when the recipe caps every SGLang pool, else None.

    Capped means max-total-tokens is set, the geometry of every Model is known
    (an unknown one could hide an uncapped pool) and a hybrid Model's
    max-mamba-cache-size is set: without it SGLang sizes the Mamba pool by
    mamba_full_memory_ratio from the fraction, whichever engine the geometry
    was pinned for (sglang_static_need only knows SGLang state fields).
    """
    if mn_args.argument_value(recipe, "max-total-tokens") is None:
        return None
    mamba_capped = mn_args.argument_value(recipe, "max-mamba-cache-size") is not None
    for selection in recipe.models:
        model = models[selection.model.content_sha256]
        geometry = mn_args.applicable_geometry(
            recipe, selection.model.slug, model, inputs
        )
        if geometry is None or (
            geometry.architecture in HYBRID_ARCHITECTURES and not mamba_capped
        ):
            return None
    return sglang_static_need(recipe, models, inputs, key, pinned=False)


def sglang_demand(recipe: RecipeDefinition, capped_need: int | None) -> Decimal | None:
    """Per-rank bytes an SGLang launch takes from a Spark, None without a fraction.

    The base on GB10 is host MemAvailable after CUDA init (sglang
    utils/common.py is_integrated path), so the admission model charges the
    Spark's total memory. With every pool capped (``capped_need`` from
    sglang_capped_static_need; kv_cache_configurator._apply_token_constraints
    takes min(capacity, max-total-tokens), _handle_max_mamba_cache the
    max-mamba-cache-size slots) SGLang allocates only that need and the
    fraction merely gates the remainder > 0: demand is the need plus the
    runtime overhead. This rule is derived from the SGLang source and is not
    calibrated on a fully capped proven run; capped recipes with CUDA graphs or
    speculation rest on OVERHEAD_BYTES. An uncapped pool grows into what the fraction
    leaves: mem-fraction-static x total plus overhead, calibrated on the proven
    0.75 recipes (105.6e9 to 109.4e9 measured against 106.0e9).
    """
    fraction = mn_args.argument_value(recipe, "mem-fraction-static")
    if fraction is None:
        return None
    if capped_need is not None:
        return Decimal(capped_need + OVERHEAD_BYTES)
    return Decimal(str(fraction)) * TOTAL_BYTES + budget_margin_bytes(recipe)


def sglang_rank_budget_warnings(
    recipe: RecipeDefinition,
    models: Mapping[str, ModelDefinition],
    inputs: MemoryInputs,
    key: str,
) -> list[str]:
    """Multi-Spark SGLang profiles its pools after the weights are resident.

    sglang 94602c9c kv_cache_configurator._profile_available_bytes charges
    everything resident against mem-fraction-static times pre-load free memory,
    so that budget on the lowest idle Spark must hold the sharded weights, their
    resident expansion and the capped token and state pools.

    The token cap must hold one full-context request
    (absolute_kv_warnings), not context x max-running-requests: SGLang
    schedules requests within it. The proven Qwen3.8 Flash Next W4A16 r0b0tlab
    dual caps one 262,144-token context at concurrency 4. The state cap must
    hold five slots per running request: at startup kv_cache_configurator
    .resolve_max_num_reqs lowers max-running-requests to max-mamba-cache-size
    // 5 and fails only below one, so 5 slots serve concurrency 1. With
    speculation, _handle_max_mamba_cache adds intermediate state for those
    running requests plus the padding slot, per draft token.
    """
    fraction = mn_args.argument_value(recipe, "mem-fraction-static")
    if fraction is None:
        return [f"{key}: declare mem-fraction-static for the static budget"]
    uncapped = mn_args.argument_value(recipe, "max-total-tokens") is None
    need = sglang_static_need(recipe, models, inputs, key, pinned=not uncapped)
    base = sglang_preload_free_bytes(recipe, models, inputs)
    static_bytes = Decimal(str(fraction)) * base
    if uncapped:
        # Uncapped, SGLang sizes the pools from what the fraction leaves after
        # the resident weights: the declared peak (fraction x total + overhead)
        # bounds them and the Controller's admission of that peak is the memory
        # authority. The check guards only whether SGLang can start.
        minimum = sglang_startup_rest(recipe, models, inputs, key)
        rest = static_bytes - need
        if not rest.is_finite() or rest < minimum:
            return [
                (
                    f"{key}: mem-fraction-static {fraction} x SGLang pre-load free "
                    f"{base} leaves {rest:.0f} bytes/rank after "
                    f"sharded weights and resident expansion ({need} bytes), below the "
                    f"{minimum} bytes SGLang needs to start (one request's state slots "
                    "and one KV token); increase mem-fraction-static"
                )
            ]
        return []
    if not static_bytes.is_finite() or static_bytes < need:
        return [
            (
                f"{key}: static budget {static_bytes:.0f} bytes (mem-fraction-static x "
                f"SGLang pre-load free {base}) cannot hold sharded weights, "
                f"resident expansion and capped pools ({need} bytes/rank); increase "
                "mem-fraction-static while retaining the absolute caps"
            )
        ]
    return []


def sglang_preload_free_bytes(
    recipe: RecipeDefinition,
    models: Mapping[str, ModelDefinition],
    inputs: MemoryInputs,
) -> int:
    """SGLang's pre-load free memory: the primary Model's image measurement, else the default."""
    selection = recipe.models[0]
    geometry = mn_args.applicable_geometry(
        recipe, selection.model.slug, models[selection.model.content_sha256], inputs
    )
    if geometry is None or geometry.sglang_preload_free_bytes is None:
        return SGLANG_PRELOAD_FREE_BYTES
    return geometry.sglang_preload_free_bytes


def sglang_startup_rest(
    recipe: RecipeDefinition,
    models: Mapping[str, ModelDefinition],
    inputs: MemoryInputs,
    key: str,
) -> int:
    """Least per-rank rest memory (fraction x pre-load free minus resident) SGLang starts on.

    kv_cache_configurator._handle_max_mamba_cache gives the state pool
    mamba-full-memory-ratio / (1 + ratio) of the rest and solves, per rank,
    (K + 1) x state + (K / 5 + 1) x D x state <= that budget for the slots K
    (D draft tokens, 0 without speculation); resolve_max_num_reqs then needs
    K // 5 >= 1 (SGLANG_MAMBA_SLOTS_PER_REQUEST). The pool also needs room
    for one KV token beside the state. Only the primary Model's geometry is
    required; a draft Model is resident weights (rank_weight_bytes) and
    expansion without pools of its own.
    """
    ratio = Decimal(
        str(
            mn_args.argument_value(recipe, "mamba-full-memory-ratio")
            or SGLANG_MAMBA_FULL_MEMORY_RATIO
        )
    )
    draft = 0
    if mn_args.argument_value(recipe, "speculative-algorithm") is not None:
        # DSPARK's verify window is the block size plus one (server_args).
        draft = (
            mn_args.positive_argument(recipe, "speculative-num-draft-tokens")
            if mn_args.argument_value(recipe, "speculative-num-draft-tokens")
            is not None
            else mn_args.positive_argument(recipe, "speculative-dspark-block-size") + 1
            if mn_args.argument_value(recipe, "speculative-dspark-block-size")
            is not None
            else 1
        )
    minimum = Decimal(0)
    for index, selection in enumerate(recipe.models):
        model = models[selection.model.content_sha256]
        geometry = (
            mn_args.pinned_geometry(
                recipe, selection.model.slug, model, inputs.geometries
            )
            if index == 0
            else mn_args.applicable_geometry(
                recipe, selection.model.slug, model, inputs
            )
        )
        if geometry is None:
            continue
        kv = sglang_token_pools(recipe, geometry, 1, key)
        slot = geometry.sglang_state_bytes_per_slot
        if slot:
            slots = SGLANG_MAMBA_SLOTS_PER_REQUEST
            budget = slot * (1 + draft) + slots * slot * (
                1 + Decimal(draft) / SGLANG_MAMBA_SLOTS_PER_REQUEST
            )
            held = slot * (slots + 1) + slot * (1 + 1) * draft
            minimum += max(budget * (1 + ratio) / ratio, held + kv)
        else:
            minimum += kv
        if geometry.state_padding:
            minimum += geometry.state_bytes_per_request * (1 + geometry.state_padding)
    return int(minimum.to_integral_value(ROUND_CEILING))
