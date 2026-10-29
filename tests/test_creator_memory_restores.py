"""Recipes follow the creator's memory settings (owner decision 2026-10-29).

One row per restored recipe. Our pre-emptive corrections are reverted; the
sweep re-proves each recipe, and a restored recipe that then fails on hardware
goes back to a lowered value with that failure cited.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import pytest
from catalog_documents import catalog_models
from documents import recipe as load_recipe
from documents import recipe_display
from vonk_forge_contracts import document_sha256

from qualification.memory_need import (
    OS_FLOOR_BYTES,
    argument_value,
    load_inputs,
    memory_variants,
    per_rank_need,
    require_memory_need,
    vllm_utilization_envelope,
)

ROOT = Path(__file__).resolve().parents[1]
MIA = "https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks/blob/94ae731f4619f769be1f4dfe6f6607ce0a3e869d"
FORBIDDEN = (
    "GB10 memory correction",
    "GB10 fit:",
    "Draft-model correction",
    "draft-model allowance",
    "Absolute KV correction",
)


@dataclass(frozen=True)
class Creator:
    source: str
    arguments: dict[str, str]
    absent: tuple[str, ...] = ()
    context_tokens: int | None = None
    options: dict[str, dict[str, str]] = field(default_factory=dict)
    vendored: tuple[str, str] | None = None
    peak: int | None = None  # None: the pool-less vLLM envelope
    os_floor_reserve: bool = False


CREATOR = {
    "glm-5-3-flash-exl3-dflash2-vllm-mia-four": Creator(
        source=f"{MIA}/.env.tp4.example#L115",
        arguments={"gpu-memory-utilization": "0.75"},
        absent=("kv-cache-memory-bytes",),
        os_floor_reserve=True,
    ),
    "glm-5-3-flash-exl3-dflash2-vllm-mia-triple": Creator(
        source=f"{MIA}/.env.tp3.example#L172",
        arguments={"gpu-memory-utilization": "0.80"},
        absent=("kv-cache-memory-bytes",),
        os_floor_reserve=True,
    ),
    "glm-5-3-flash-nvfp4-vllm-four": Creator(
        source="https://github.com/tonyd2wild/GLM-5.3-Flash-NVFP4-1M-KV-4x-DGX-Spark/blob/2ac4e8dd661b12e6a1d7e69df711707abd6b9cd4/launch-glm53-tp4-24g.sh#L86",
        arguments={
            "gpu-memory-utilization": "0.85",
            "kv-cache-memory": "25769803776",
        },
        # Multi-Spark startup budget: 0.85 x total + 8 GB, beside the pinned pool.
        peak=119063746765,
    ),
    "laguna-s-2-1-nvfp4-r0b0tlab-vllm-single": Creator(
        source="https://github.com/r0b0tlab/laguna-s-2.1-nvfp4-sm121-vllm/blob/4cf44cf36b88b564031ddb19c41d7cb9cfe0f52a/sparkrun/recipes/laguna-s-2.1-nvfp4-vllm-r0b0tlab.yaml#L27",
        arguments={"gpu-memory-utilization": "0.85"},
        absent=("kv-cache-memory-bytes",),
    ),
}


@pytest.mark.parametrize("slug", sorted(CREATOR))
def test_recipe_follows_the_creators_memory_settings(slug: str) -> None:
    row, recipe = CREATOR[slug], load_recipe(slug)
    for name, expected in row.arguments.items():
        if row.vendored is not None:
            path, pattern = row.vendored
            match = re.search(pattern, (ROOT / path).read_text(encoding="utf-8"))
            assert match, (path, pattern)
            expected = match.group(1)
        assert str(argument_value(recipe, name)) == expected, (name, row.source)
    for name in row.absent:
        assert argument_value(recipe, name) is None, name
    if row.context_tokens is not None:
        assert argument_value(recipe, "max-model-len") == row.context_tokens
    for choice, values in row.options.items():
        option, value = choice.split("/")
        chosen = recipe.with_option_choices({option: value})
        for name, expected in values.items():
            assert str(argument_value(chosen, name)) == expected, (choice, name)
    if row.peak is None:
        expected_peak = max(
            envelope
            for variant in memory_variants(recipe)
            if (envelope := vllm_utilization_envelope(variant)) is not None
        )
    else:
        expected_peak = row.peak
        need = per_rank_need(recipe, catalog_models(), load_inputs())
        assert expected_peak >= need.peak_bytes
    for role in recipe.topology.roles:
        memory = role.resources.memory
        assert memory.peak_bytes == expected_peak, role.name
        if row.os_floor_reserve:
            assert memory.reserve_bytes >= OS_FLOOR_BYTES, role.name
    description = recipe_display(slug).description
    assert not [text for text in FORBIDDEN if text in description]
    assert "Memory follows the creator's settings" in description


def test_lowering_a_restored_peak_by_one_byte_is_refused() -> None:
    slug = "laguna-s-2-1-nvfp4-r0b0tlab-vllm-single"
    recipe = load_recipe(slug)
    models, inputs = catalog_models(), load_inputs()
    digest = document_sha256(recipe.model_dump(mode="json", by_alias=True))
    require_memory_need(recipe, models, inputs, document_digest=digest, touched=True)
    for role in recipe.topology.roles:
        role.resources.memory.peak_bytes -= 1
    with pytest.raises(ValueError, match="peak understates"):
        require_memory_need(
            recipe, models, inputs, document_digest=digest, touched=True
        )
