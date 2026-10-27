"""vLLM's hybrid allocator sizes Mamba/GDN state; a hand state bound does not."""

import json
import re
from pathlib import Path

import pytest
from catalog_documents import catalog_models
from vonk_forge_contracts import RecipeDefinition, document_sha256
from vonk_forge_contracts.recipe import RecipeRuntimeArgument

from qualification.memory_need import (
    CacheGeometry,
    MambaCacheMode,
    absolute_kv_need,
    argument_value,
    cache_bytes,
    cache_need_for_context,
    hardware_proven,
    load_inputs,
    mamba_cache_mode,
    memory_variants,
    positive_argument,
    recurrent_layer_bytes,
    require_memory_need,
    speculative_tokens,
)

ROOT = Path(__file__).resolve().parents[1]
INPUTS = load_inputs()
# Hand state bounds pinned by vLLM recipes although the model has recurrent
# layers, each because its vLLM layout holds cache types vllm_hybrid_need
# cannot express. A new entry needs the same evidence.
EXPLICIT_HAND_BOUNDS = {
    # Its only vLLM recipe (sfxnz dual, proven on 9f77f966) runs vLLM v0.30.0:
    # Qwen4Exp adds a second tp_replicated PLE MambaSpec, a QSA
    # CircularBufferSpec ring, an MLA compressed-key cache and an MTP QSA layer.
    "qwen3-8-flash-next-nvfp4-fab0aecb",
    # Sliding-window plus short-convolution layers on creator images whose
    # vLLM Inkling layout is not pinned; the state term is SGLang-measured.
    "inkling-small-nvfp4",
}
RECURRENT_WORDS = re.compile(
    r"recurrent|mamba|linear.attention|gated.delta|\bGDN\b|\bSSM\b|sconv", re.IGNORECASE
)


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

    Walks the source step by step: Platform._align_hybrid_block_size,
    _get_kv_cache_groups_uniform_page_size, get_num_blocks, BlockPool's null
    block and MambaSpec/FullAttentionSpec block demand.
    """
    recurrent = geometry.recurrent
    assert recurrent is not None
    layers = {
        "attention": [f"attention.{i}" for i in range(geometry.layers)],
        "mamba": [f"mamba.{i}" for i in range(recurrent.layers)],
    }
    attention_1_token = cache_need_for_context(recipe, geometry, 1) // geometry.layers
    mamba_page = recurrent_layer_bytes(recipe, recurrent)
    kernel_alignment = recurrent.kernel_block_tokens
    if argument_value(recipe, "block-size") is not None:
        kernel_alignment = max(
            kernel_alignment, positive_argument(recipe, "block-size")
        )
    mode = mamba_cache_mode(recipe, recurrent)
    if mode is MambaCacheMode.ALL:
        assert recurrent.all_mode_chunk_tokens is not None
        chunk = kernel_alignment
        while chunk % recurrent.all_mode_chunk_tokens:
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
    usable = pool // page // group_size - 1  # one block is the null block
    context = int(
        str(argument_value(recipe, "max-model-len") or geometry.max_context_tokens)
    )
    spec = speculative_tokens(recipe)
    per_mamba_group = {
        MambaCacheMode.NONE: 1 + spec,
        MambaCacheMode.ALIGN: 2 + spec,
        MambaCacheMode.ALL: -(-context // block) + spec,
    }[mode]
    needed = sum(
        -(-context // block) if kind == "attention" else per_mamba_group
        for kind, _ in groups
    )
    # _check_enough_kv_cache_memory sums layers without padding or null block.
    startup = (
        geometry.layers * -(-context // block) + recurrent.layers * per_mamba_group
    ) * page
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


def test_every_recurrent_geometry_is_exercised_by_a_vllm_recipe() -> None:
    covered = {recipe.models[0].model.slug for recipe in hybrid_vllm_recipes()}
    typed = {key for key, item in INPUTS.geometries.items() if item.recurrent}
    assert typed and typed <= covered


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


def test_vllm_recurrent_state_is_never_a_hand_bound() -> None:
    pinned = {
        item.model.slug
        for recipe, _ in recipes()
        if recipe.runtime.engine == "vllm"
        for item in recipe.models
    }
    hand = {
        key
        for key, geometry in INPUTS.geometries.items()
        if key in pinned
        and geometry.state_bytes_per_request
        and RECURRENT_WORDS.search(geometry.source)
    }
    assert hand == EXPLICIT_HAND_BOUNDS
