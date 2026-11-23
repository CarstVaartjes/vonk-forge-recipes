"""vLLM hybrid and absolute KV need."""

from __future__ import annotations

from collections.abc import Mapping
from math import lcm

from recipe_tooling.qualification.memory_need import args as mn_args
from recipe_tooling.qualification.memory_need.geometry import (
    HYBRID_ARCHITECTURES,
    MAMBA_DTYPE_BYTES,
    VLLM_UNMODELED_HYBRID,
    CacheGeometry,
    MambaCacheMode,
    MemoryInputs,
    RecurrentCache,
)
from recipe_tooling.qualification.memory_need.records import MissingEstimateInput
from vonk_forge_contracts import (
    ModelDefinition,
    RecipeDefinition,
)
from vonk_forge_contracts.api_checks import CONTEXT_ARGUMENTS


def cache_need_for_context(
    recipe: RecipeDefinition, geometry: CacheGeometry, context: int
) -> int:
    """One cache-layout calculation for single-node and per-rank workload sizing."""
    dtype = str(mn_args.argument_value(recipe, "kv-cache-dtype") or "auto")
    parallel = recipe.topology.parallelism
    layers, token_bytes = attention_token_bytes(recipe, geometry)
    needed = context * token_bytes
    if not geometry.token_bytes_by_dtype.get(dtype) and geometry.token_bytes is None:
        sliding_layers = (
            geometry.sliding_layers + parallel.pipeline - 1
        ) // parallel.pipeline
        needed -= (
            (context - min(context, geometry.sliding_window))
            * sliding_layers
            * (token_bytes // layers)
        )
    replicated_bytes = (
        geometry.replicated_token_bytes * layers + geometry.layers - 1
    ) // geometry.layers
    return needed + context * replicated_bytes


def attention_token_bytes(
    recipe: RecipeDefinition, geometry: CacheGeometry
) -> tuple[int, int]:
    """Attention layers on the largest pipeline stage and their bytes per token."""
    dtype = str(mn_args.argument_value(recipe, "kv-cache-dtype") or "auto")
    parallel = recipe.topology.parallelism
    layers = (geometry.layers + parallel.pipeline - 1) // parallel.pipeline
    heads = (
        geometry.kv_heads
        if geometry.tensor_replicated
        else (geometry.kv_heads + parallel.tensor - 1) // parallel.tensor
    )
    if dtype not in geometry.token_bytes_by_dtype and dtype not in {
        "auto",
        "bfloat16",
        "float16",
        "fp8",
        "fp8_e4m3",
        "fp8_e5m2",
        "nvfp4",
        "fp4_mx_block16",
    }:
        raise MissingEstimateInput(
            f"{recipe.identity.slug}: supply cache layout calibration for dtype {dtype}"
        )
    if dtype in geometry.token_bytes_by_dtype:
        token_bytes = (
            geometry.token_bytes_by_dtype[dtype] * layers + geometry.layers - 1
        ) // geometry.layers
    elif geometry.token_bytes is not None:
        token_bytes = (
            geometry.token_bytes * layers + geometry.layers - 1
        ) // geometry.layers
    elif dtype in ("auto", "bfloat16", "float16"):
        token_bytes = layers * heads * (geometry.key_dim + geometry.value_dim) * 2
    elif dtype in ("fp8", "fp8_e4m3", "fp8_e5m2"):
        token_bytes = layers * heads * (geometry.key_dim + geometry.value_dim)
    elif dtype in ("nvfp4", "fp4_mx_block16"):
        # Packed data plus one scale byte per 16 elements, separately for K/V.
        # fp4_mx_block16 is the triton lane's layout: packed e2m1 plus a uint8
        # biased-exponent scale per 16 elements, 0.5625 bytes/element
        # (adapters/inkling/drowzeys-kvquant-dual/scripts/nvfp4-kv-boot.sh:11-12).
        token_bytes = (
            layers
            * heads
            * sum(
                (dim + 1) // 2 + (dim + 15) // 16
                for dim in (geometry.key_dim, geometry.value_dim)
            )
        )
    return layers, token_bytes


def mamba_dtype_bytes(recipe: RecipeDefinition, flag: str, default: int) -> int:
    value = mn_args.argument_value(recipe, flag)
    if value is None or value == "auto":
        return default
    if str(value) not in MAMBA_DTYPE_BYTES:
        raise ValueError(
            f"{recipe.identity.slug}: supply state size for {flag} {value}"
        )
    return MAMBA_DTYPE_BYTES[str(value)]


def recurrent_layer_bytes(recipe: RecipeDefinition, recurrent: RecurrentCache) -> int:
    """One request's unpadded state in one recurrent layer on one tensor rank."""
    tensor = recipe.topology.parallelism.tensor
    conv = -(-recurrent.conv_channels // tensor) * (
        recurrent.conv_kernel - 1 + mn_args.speculative_tokens(recipe)
    )
    return conv * mamba_dtype_bytes(recipe, "mamba-cache-dtype", 2) + -(
        -recurrent.state_elements // tensor
    ) * mamba_dtype_bytes(recipe, "mamba-ssm-cache-dtype", recurrent.state_dtype_bytes)


def mamba_cache_mode(
    recipe: RecipeDefinition, recurrent: RecurrentCache
) -> MambaCacheMode:
    """vLLM MambaModelConfig: prefix caching selects "all" where the loaded
    class supports it, else "align"; off holds one page per recurrent group.

    An omitted flag counts as on: v0.27 keeps hybrid prefix caching opt-in
    (arg_utils.py default_prefix_caching), but nightly 3f1bcfd8 defaults it
    to is_prefix_caching_supported, and "align" never needs fewer blocks.
    """
    if mn_args.argument_value(recipe, "enable-prefix-caching") is False:
        return MambaCacheMode.NONE
    mode = MambaCacheMode(
        str(mn_args.argument_value(recipe, "mamba-cache-mode") or MambaCacheMode.NONE)
    )
    if mode is MambaCacheMode.ALIGN or not recurrent.mamba_prefix_caching:
        return MambaCacheMode.ALIGN
    return MambaCacheMode.ALL


def vllm_recurrent(
    recipe: RecipeDefinition, slug: str, geometry: CacheGeometry
) -> RecurrentCache | None:
    """Typed recurrent state a vLLM launch is sized by; a hybrid architecture
    without it is rejected unless VLLM_UNMODELED_HYBRID names the geometry."""
    if recipe.runtime.engine != "vllm":
        return None
    if (
        geometry.recurrent is None
        and geometry.architecture in HYBRID_ARCHITECTURES
        and slug not in VLLM_UNMODELED_HYBRID
    ):
        raise MissingEstimateInput(
            f"{recipe.identity.slug}: {slug} is a hybrid {geometry.architecture}; "
            "add its typed recurrent geometry"
        )
    return geometry.recurrent


def vllm_hybrid_need(
    recipe: RecipeDefinition,
    geometry: CacheGeometry,
    recurrent: RecurrentCache,
    context: int,
    concurrency: int,
) -> int:
    """Per-rank pool vLLM v0.27's hybrid allocator needs for the requests.

    Platform._align_hybrid_block_size raises the block to the smallest
    kernel-aligned size (mode "all": aligned to the Mamba chunk) whose
    attention page holds one recurrent state, and pads that state to the page.
    _get_kv_cache_groups_uniform_page_size groups layers by the smallest
    bucket (the largest when within 1.5x), padding each type's last group;
    get_num_blocks sizes pool // page // group blocks and BlockPool takes one
    as the null block. Per request an attention group holds cdiv(context,
    block) blocks and a recurrent group (MambaSpec.max_memory_usage_bytes)
    1 (mode none), 2 (align) or cdiv(context, block) (all), plus one per
    speculative token. The startup check (_max_memory_usage_bytes_from_groups)
    counts the same padded groups without the null block, so this covers it.
    """
    stage_layers, token_bytes = attention_token_bytes(recipe, geometry)
    per_token = token_bytes // stage_layers
    state = recurrent_layer_bytes(recipe, recurrent)
    align = recurrent.kernel_block_tokens
    if mn_args.argument_value(recipe, "block-size") is not None:
        align = max(align, mn_args.positive_argument(recipe, "block-size"))
    mode = mamba_cache_mode(recipe, recurrent)
    if mode is MambaCacheMode.ALL:
        assert recurrent.chunk_tokens is not None
        chunk = lcm(recurrent.chunk_tokens, align)
        tokens = -(-state // per_token)
        block = chunk * -(-tokens // chunk)
    else:
        block = align * -(-state // (align * per_token))
    stages = recipe.topology.parallelism.pipeline
    attention_layers = -(-recurrent.attention_layers // stages)
    recurrent_layers = -(-recurrent.layers // stages)
    smallest, largest = sorted((attention_layers, recurrent_layers))
    group = largest if 2 * largest < 3 * smallest else smallest
    attention_blocks = -(-context // block)
    recurrent_blocks = (
        mn_args.speculative_tokens(recipe)
        + {
            MambaCacheMode.NONE: 1,
            MambaCacheMode.ALIGN: 2,
            MambaCacheMode.ALL: attention_blocks,
        }[mode]
    )
    blocks = (
        concurrency
        * (
            -(-attention_layers // group) * attention_blocks
            + -(-recurrent_layers // group) * recurrent_blocks
        )
        + 1
    )
    return blocks * group * block * per_token


def absolute_kv_need(
    recipe: RecipeDefinition,
    models: Mapping[str, ModelDefinition],
    inputs: MemoryInputs,
    *,
    concurrency: int = 1,
) -> int:
    """Per-rank pool for full context, including block rounding and packed scales.

    MLA latents replicate over tensor ranks; ordinary GQA heads shard, with at
    least one head per rank. Geometry is bound to the selected configuration.
    Padded/hybrid layouts use a source-backed conservative allocation bound.
    This is an allocation estimate, not a hardware qualification verdict.
    """
    selection = recipe.models[0]
    model = models[selection.model.content_sha256]
    geometry = mn_args.pinned_geometry(
        recipe, selection.model.slug, model, inputs.geometries
    )
    context_flag = CONTEXT_ARGUMENTS[recipe.runtime.engine][0]
    context_value = mn_args.argument_value(recipe, context_flag)
    context = (
        geometry.max_context_tokens
        if context_value in (None, "auto", -1, "-1")
        else mn_args.positive_argument(recipe, context_flag)
    )
    if context is None:
        raise MissingEstimateInput(
            f"{recipe.identity.slug}: declare max_context_tokens for the geometry in "
            "memory-inputs.json"
        )
    block_flag = "block-size" if recipe.runtime.engine == "vllm" else "page-size"
    block = (
        mn_args.positive_argument(recipe, block_flag)
        if mn_args.argument_value(recipe, block_flag) is not None
        else 16
    )
    context = ((context + block - 1) // block) * block
    recurrent = vllm_recurrent(recipe, selection.model.slug, geometry)
    if recurrent is not None:
        return vllm_hybrid_need(recipe, geometry, recurrent, context, concurrency)
    return (
        cache_need_for_context(recipe, geometry, context)
        + geometry.state_bytes_per_request
    ) * concurrency


def absolute_kv_warnings(
    recipe: RecipeDefinition,
    models: Mapping[str, ModelDefinition],
    inputs: MemoryInputs,
) -> list[str]:
    """A fixed pool should hold at least one full-context request, even on TP/PP."""
    warnings: list[str] = []
    if recipe.runtime.engine not in {"vllm", "sglang"}:
        return warnings
    for variant in mn_args.memory_variants(recipe):
        if variant.runtime.engine == "sglang":
            tokens = mn_args.argument_value(variant, "max-total-tokens")
            if tokens is not None and mn_args.positive_argument(
                variant, "max-total-tokens"
            ) < mn_args.positive_argument(variant, CONTEXT_ARGUMENTS["sglang"][0]):
                warnings.append(
                    f"{recipe.identity.slug}: absolute KV token pool cannot hold one full-context request"
                )
        elif variant.runtime.engine == "vllm":
            pool = mn_args.argument_value(variant, "kv-cache-memory-bytes")
            if pool is None:
                pool = mn_args.argument_value(variant, "kv-cache-memory")
            if pool is not None:
                need = absolute_kv_need(variant, models, inputs)
                if mn_args.cache_bytes(pool) < need:
                    warnings.append(
                        f"{recipe.identity.slug}: absolute KV pool {mn_args.cache_bytes(pool)} bytes cannot hold one full-context request ({need} bytes/rank required)"
                    )
    return warnings
