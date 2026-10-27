"""vLLM's hybrid allocator sizes Mamba/GDN state; a hand state bound does not."""

import json
from pathlib import Path

import pytest
from catalog_documents import catalog_models
from pydantic import ValidationError
from vonk_forge_contracts import RecipeDefinition, document_sha256
from vonk_forge_contracts.recipe import RecipeRuntimeArgument

from qualification.memory_need import (
    HYBRID_ARCHITECTURES,
    VLLM_UNMODELED_HYBRID,
    CacheGeometry,
    absolute_kv_need,
    argument_value,
    cache_bytes,
    cache_need_for_context,
    hardware_proven,
    load_inputs,
    memory_variants,
    positive_argument,
    require_memory_need,
)

ROOT = Path(__file__).resolve().parents[1]
INPUTS = load_inputs()
DTYPE_BYTES = {"float32": 4, "float16": 2, "bfloat16": 2}


def recipes() -> list[tuple[RecipeDefinition, str]]:
    found = []
    for path in sorted((ROOT / "recipes").glob("*.json")):
        document = json.loads(path.read_text())
        found.append(
            (RecipeDefinition.model_validate(document), document_sha256(document))
        )
    return found


def recurrent(slug: str) -> bool:
    geometry = INPUTS.geometries.get(slug)
    return geometry is not None and geometry.recurrent is not None


def hybrid_vllm_recipes() -> list[RecipeDefinition]:
    return [
        recipe
        for recipe, _ in recipes()
        if recipe.runtime.engine == "vllm" and recurrent(recipe.models[0].model.slug)
    ]


def allocator_walk(
    recipe: RecipeDefinition, geometry: CacheGeometry, pool: int
) -> bool:
    """Whether vLLM v0.27 serves one max_model_len request from ``pool`` bytes.

    Walks the source step by step from the config dimensions and launch
    arguments: MambaStateShapeCalculator and MambaStateDtypeCalculator,
    MambaModelConfig's cache mode, Platform._align_hybrid_block_size,
    _get_kv_cache_groups_uniform_page_size, the startup check
    _max_memory_usage_bytes_from_groups, get_num_blocks and BlockPool's null
    block. Only the attention bytes per token come from the shared helper.
    """
    state = geometry.recurrent
    assert state is not None
    tp = recipe.topology.parallelism.tensor
    speculative = argument_value(recipe, "speculative-config")
    if isinstance(speculative, str):
        speculative = json.loads(speculative)
    spec = (
        int(speculative.get("num_speculative_tokens", 0))
        if isinstance(speculative, dict)
        else 0
    )
    conv_dtype = DTYPE_BYTES.get(str(argument_value(recipe, "mamba-cache-dtype")), 2)
    ssm_dtype = DTYPE_BYTES.get(
        str(argument_value(recipe, "mamba-ssm-cache-dtype")), state.state_dtype_bytes
    )
    mamba_page = (state.conv_channels // tp) * (
        state.conv_kernel - 1 + spec
    ) * conv_dtype + (state.state_elements // tp) * ssm_dtype
    if argument_value(recipe, "enable-prefix-caching") is not True:
        mode = "none"
    elif argument_value(recipe, "mamba-cache-mode") == "align":
        mode = "align"
    else:
        mode = "all" if state.mamba_prefix_caching else "align"
    layers = {
        "attention": [f"attention.{i}" for i in range(state.attention_layers)],
        "mamba": [f"mamba.{i}" for i in range(state.layers)],
    }
    attention_1_token = cache_need_for_context(recipe, geometry, 1) // geometry.layers
    kernel_alignment = state.kernel_block_tokens
    if argument_value(recipe, "block-size") is not None:
        kernel_alignment = max(
            kernel_alignment, positive_argument(recipe, "block-size")
        )
    if mode == "all":
        assert state.chunk_tokens is not None
        chunk = kernel_alignment
        while chunk % state.chunk_tokens:
            chunk += kernel_alignment
        tokens_per_state = -(-mamba_page // attention_1_token)
        block = chunk * -(-tokens_per_state // chunk)
    else:
        block = kernel_alignment * -(
            -mamba_page // (kernel_alignment * attention_1_token)
        )
    page = block * attention_1_token
    assert page >= mamba_page
    group_size = min(len(names) for names in layers.values())
    if max(len(names) for names in layers.values()) < group_size * 1.5:
        group_size = max(len(names) for names in layers.values())
    groups = []
    for kind, names in layers.items():
        count = -(-len(names) // group_size)
        groups += [(kind, names[i::count]) for i in range(count)]
    context = int(
        str(argument_value(recipe, "max-model-len") or geometry.max_context_tokens)
    )
    per_mamba_group = {
        "none": 1 + spec,
        "align": 2 + spec,
        "all": -(-context // block) + spec,
    }[mode]
    needed = sum(
        -(-context // block) if kind == "attention" else per_mamba_group
        for kind, _ in groups
    )
    startup = group_size * page * needed
    usable = pool // page // group_size - 1  # one block is the null block
    return usable >= needed and startup <= pool


def other_modes(recipe: RecipeDefinition) -> list[RecipeDefinition]:
    """The launch under each prefix-caching mode."""
    found = []
    for caching, mode in ((False, None), (True, "align"), (True, "all")):
        copy = recipe.model_copy(deep=True)
        copy.runtime.arguments = [
            item
            for item in copy.runtime.arguments
            if item.name not in {"enable-prefix-caching", "mamba-cache-mode"}
        ]
        copy.runtime.arguments.append(
            RecipeRuntimeArgument(name="enable-prefix-caching", value=caching)
        )
        if mode is not None:
            copy.runtime.arguments.append(
                RecipeRuntimeArgument(name="mamba-cache-mode", value=mode)
            )
        found.append(copy)
    return found


@pytest.mark.parametrize(
    "recipe", hybrid_vllm_recipes(), ids=lambda recipe: recipe.identity.slug
)
def test_hybrid_bound_is_the_smallest_pool_vllm_serves(
    recipe: RecipeDefinition,
) -> None:
    models = catalog_models()
    geometry = INPUTS.geometries[recipe.models[0].model.slug]
    authored = memory_variants(recipe)
    for variant in [*authored, *other_modes(recipe)]:
        need = absolute_kv_need(variant, models, INPUTS)
        assert allocator_walk(variant, geometry, need)
        assert not allocator_walk(variant, geometry, need - 1)
    for variant in authored:
        pool = argument_value(variant, "kv-cache-memory-bytes")
        if pool is not None:
            assert allocator_walk(variant, geometry, cache_bytes(pool))


def test_proven_hybrid_documents_pass_unchanged() -> None:
    # Calibration: hardware-proven hybrid pools hold vLLM's allocation as authored.
    models, proven = catalog_models(), []
    for recipe, digest in recipes():
        if not any(recurrent(item.model.slug) for item in recipe.models):
            continue
        if not hardware_proven(recipe, digest):
            continue
        proven.append(recipe.identity.slug)
        require_memory_need(recipe, models, INPUTS, document_digest=digest)
        require_memory_need(recipe, models, INPUTS, document_digest="", touched=True)
    assert "nemotron-3-5-super-vl-lilicorr-sfxnz-vllm-single" in proven
    assert "nemotron-3-nano-omni-30b-a3b-bf16-nvidia-vllm-single" in proven


def test_vllm_hybrids_are_typed_or_named_unmodeled() -> None:
    pinned = {
        item.model.slug
        for recipe, _ in recipes()
        if recipe.runtime.engine == "vllm"
        for item in recipe.models
    }
    untyped = {
        key
        for key, geometry in INPUTS.geometries.items()
        if key in pinned
        and geometry.architecture in HYBRID_ARCHITECTURES
        and geometry.recurrent is None
    }
    assert untyped == set(VLLM_UNMODELED_HYBRID)
    # Fault injection: an untyped hybrid geometry is rejected on vLLM.
    recipe = next(iter(hybrid_vllm_recipes()))
    slug = recipe.models[0].model.slug
    inputs = INPUTS.model_copy(deep=True)
    inputs.geometries[slug].recurrent = None
    with pytest.raises(ValueError, match="add its typed recurrent geometry"):
        absolute_kv_need(recipe, catalog_models(), inputs)


def test_other_engines_keep_the_hand_state_bound() -> None:
    recipe = next(
        recipe
        for recipe, _ in recipes()
        if recipe.runtime.engine == "sglang" and recurrent(recipe.models[0].model.slug)
    )
    geometry = INPUTS.geometries[recipe.models[0].model.slug]
    context = positive_argument(recipe, "context-length")
    assert absolute_kv_need(recipe, catalog_models(), INPUTS) == (
        cache_need_for_context(recipe, geometry, -(-context // 16) * 16)
        + geometry.state_bytes_per_request
    )


ADAY777 = "qwen3-8-27b-ara-abliterated-nvfp4-mtp-df03f976"


@pytest.mark.parametrize(
    "update",
    [
        {"architecture": "UnclassifiedForCausalLM"},
        {"architecture": "LlamaForCausalLM"},
        {"token_bytes": 4096},
        {"sliding_layers": 4, "sliding_window": 128},
        {"replicated_token_bytes": 768},
        {"state_padding": 1},
        {"recurrent": {"mamba_prefix_caching": True, "chunk_tokens": 128}},
        {"recurrent": {"chunk_tokens": 128}},
        {"recurrent": {"attention_layers": 17}},
    ],
)
def test_recurrent_geometry_fields_are_exclusive(update: dict[str, object]) -> None:
    document = INPUTS.geometries[ADAY777].model_dump()
    for key, value in update.items():
        if isinstance(value, dict):
            document[key] |= value
        else:
            document[key] = value
    with pytest.raises(ValidationError):
        CacheGeometry.model_validate(document)
