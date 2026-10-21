"""Admission declarations, engine budgets and frozen hardware calibration bytes."""

import hashlib
import json
from decimal import Decimal
from pathlib import Path

import pytest
from catalog_documents import catalog_models
from vonk_forge_contracts import RecipeDefinition, document_sha256

from qualification.memory_need import (
    IDLE_AVAILABLE_BYTES,
    IDLE_USE_BYTES,
    OS_FLOOR_BYTES,
    OVERHEAD_BYTES,
    TOTAL_BYTES,
    argument_value,
    declaration_problems,
    load_inputs,
    memory_variants,
    per_rank_need,
    require_admissible_declaration,
    require_memory_need,
)

ROOT = Path(__file__).resolve().parents[1]
PROVEN = json.loads((ROOT / "tests/fixtures/proven-recipe-bytes.json").read_text())
REPAIRED = (
    "deepseek-v4-flash-0731-dspark-tonyd2wild-vllm-dual",
    "deepseek-v4-flash-0731-eugr-vllm-dual",
    "deepseek-v4-flash-0731-r0b0tlab-vllm-dual",
    "deepseek-v4-flash-dspark-nvfp4-kv-1-5m-drowzeys-vllm-dual",
    "deepseek-v4-flash-dspark-r0b0tlab-vllm-025-dual",
    "deepseek-v4-flash-fp8-r0b0tlab-vllm-dual",
    "deepseek-v4-flash-mtp-tonyd2wild-vllm-dual",
    "deepseek-v4-flash-vision-exp-dspark-tonyd2wild-vllm-dual",
    "deepseek-v4-flash-vision-exp-eugr-vllm-dual",
    "deepseek-v4-flash-vision-exp-mia-dual",
    "deepseek-v4-flash-vision-exp-r0b0tlab-vllm-dual",
    "deepseek-v4-flash-vision-exp-sfxnz-vllm-dual",
    "deepseek-v4-flash-vision-exp-sglang-tonyd2wild-dual",
    "deepseek-v4-flash-vllm-mia-dual",
    "glm-5-2-aqlm-vllm-triple",
    "glm-5-2-quanttrio-vllm-four",
    "glm-5-3-flash-nvfp4-spark-eugr-vllm-dual",
    "hy3-295b-nvfp4-a4q-drowzeys-vllm-dual",
    "hy3-295b-nvfp4-mtp-tonyd2wild-vllm-dual",
    "hy3-295b-nvfp4-vllm-mia-dual",
    "inkling-975b-a41b-nvfp4-sglang-eight",
    "inkling-small-nvfp4-dspark-tonyd2wild-vllm-dual",
    "inkling-small-nvfp4-kv-1m-drowzeys-sglang-dual",
    "inkling-small-nvfp4-sglang-dual",
    "inkling-small-nvfp4-sglang-mia-dual",
    "leanstral-1-5-119b-a6b-vllm-mia-dual",
    "mimo-v2-5-omni-nvfp4-tonyd2wild-vllm-dual",
    "mimo-v2-5-omni-nvfp4-tonyd2wild-vllm-triple",
    "mimo-v2-5-omni-nvfp4-vllm-mia-dual",
    "mimo-v2-6-flash-rl-nvfp4-r0b0tlab-sglang-dual",
    "mimo-v2-6-flash-rl-vllm-dual",
    "mimo-v2-6-pro-mopd-arvq-drowzeys-vllm-four",
    "minimax-m2-7-nvfp4-r0b0tlab-vllm-dual",
    "qwen3-8-flash-next-nvfp4-sfxnz-vllm-dual",
    "qwen3-8-flash-next-w4a16-r0b0tlab-sglang-dual",
    "step-3-7-flash-nvfp4-r0b0tlab-vllm-dual",
    "step-5-preview-exl3-vllm-four",
)


def recipe_for(slug: str) -> RecipeDefinition:
    return RecipeDefinition.model_validate_json(
        (ROOT / "recipes" / f"{slug}.json").read_text()
    )


def test_proven_recipe_documents_remain_byte_identical() -> None:
    assert len(PROVEN) == 180
    for key, digest in PROVEN.items():
        path = ROOT / "recipes" / f"{key.split('/')[1]}.json"
        assert hashlib.sha256(path.read_bytes()).hexdigest() == digest, key


def test_proven_set_calibrates_need_without_false_rejections() -> None:
    checked = 0
    for key in PROVEN:
        recipe = recipe_for(key.split("/")[1])
        if recipe.runtime.engine not in {"vllm", "sglang"}:
            continue
        for variant in memory_variants(recipe):
            need = per_rank_need(variant, catalog_models())
            assert all(
                need.peak_bytes + role.resources.memory.reserve_bytes
                <= IDLE_AVAILABLE_BYTES
                for role in variant.topology.roles
            ), key
        checked += 1
    assert checked == 115


@pytest.mark.parametrize("slug", REPAIRED)
def test_repaired_declarations_and_engine_allocations_fit_every_variant(
    slug: str,
) -> None:
    recipe = recipe_for(slug)
    assert f"{recipe.identity.publisher}/{slug}" not in PROVEN
    for variant in memory_variants(recipe):
        require_admissible_declaration(variant)
        need = per_rank_need(variant, catalog_models())
        assert not declaration_problems(variant, need)
        if variant.runtime.engine == "vllm":
            assert argument_value(variant, "gpu-memory-utilization") is None
            assert all(
                role.resources.memory.peak_bytes == need.peak_bytes
                for role in variant.topology.roles
            )
        else:
            fraction = argument_value(variant, "mem-fraction-static")
            assert fraction is not None
            allocated = Decimal(str(fraction)) * TOTAL_BYTES + OVERHEAD_BYTES
            assert all(
                allocated <= role.resources.memory.peak_bytes
                for role in variant.topology.roles
            )
        require_memory_need(
            variant, catalog_models(), load_inputs(), document_digest=""
        )


def test_catalog_checks_each_supported_document() -> None:
    for path in sorted((ROOT / "recipes").glob("*.json")):
        document = json.loads(path.read_text())
        recipe = RecipeDefinition.model_validate(document)
        require_memory_need(
            recipe,
            catalog_models(),
            load_inputs(),
            document_digest=document_sha256(document),
        )


@pytest.mark.parametrize("rank", [0, 1])
def test_original_deferral_fails_before_any_controller_operation(rank: int) -> None:
    recipe = recipe_for("deepseek-v4-flash-vision-exp-sglang-tonyd2wild-dual")
    recipe.topology.roles[rank].resources.memory.peak_bytes = 124_000_000_000
    with pytest.raises(ValueError, match="declared peak plus reserve"):
        require_memory_need(recipe, catalog_models(), load_inputs(), document_digest="")


def test_static_budget_drift_cannot_hide_behind_a_smaller_declaration() -> None:
    recipe = recipe_for("deepseek-v4-flash-vision-exp-sglang-tonyd2wild-dual")
    next(
        a for a in recipe.runtime.arguments if a.name == "mem-fraction-static"
    ).value = "0.80"
    with pytest.raises(ValueError, match="static allocation"):
        require_memory_need(recipe, catalog_models(), load_inputs(), document_digest="")


def test_admission_uses_the_owned_host_baseline_once_and_checks_reserve() -> None:
    recipe = recipe_for("deepseek-v4-flash-vision-exp-sglang-tonyd2wild-dual")
    # The deferral's host use is lower than the owner's historical baseline.
    assert 4_515_061_760 < IDLE_USE_BYTES
    assert IDLE_AVAILABLE_BYTES == TOTAL_BYTES - IDLE_USE_BYTES
    for role in recipe.topology.roles:
        role.resources.memory.reserve_bytes = OS_FLOOR_BYTES
        role.resources.memory.peak_bytes = IDLE_AVAILABLE_BYTES - OS_FLOOR_BYTES
    require_admissible_declaration(recipe)
    recipe.topology.roles[1].resources.memory.reserve_bytes += 1
    with pytest.raises(ValueError, match="worker.*declared peak plus reserve"):
        require_admissible_declaration(recipe)


def test_wrapper_cache_alias_counts_toward_actual_rank_demand() -> None:
    recipe = recipe_for("deepseek-v4-flash-vision-exp-sfxnz-vllm-dual")
    initial = per_rank_need(recipe, catalog_models()).peak_bytes
    pin = next(a for a in recipe.runtime.arguments if a.name == "kv-cache-memory")
    assert isinstance(pin.value, int)
    assert pin.value == 12 * 1024**3
    pin.value += 1
    assert per_rank_need(recipe, catalog_models()).peak_bytes == initial + 1
    assert declaration_problems(recipe, per_rank_need(recipe, catalog_models()))
