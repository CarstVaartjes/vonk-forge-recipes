"""Admission declarations, engine budgets and frozen hardware calibration bytes."""

import hashlib
import json
import re
from decimal import Decimal
from pathlib import Path

import pytest
from catalog_documents import catalog_models
from vonk_forge_contracts import RecipeDefinition, document_sha256
from vonk_forge_contracts.memory_admission import evaluate_memory_admission
from vonk_forge_contracts.recipe import RecipeRuntimeArgument

from qualification.memory_need import (
    ADMISSION_IDLE_USE_BYTES,
    IDLE_AVAILABLE_BYTES,
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

# The multi-Spark SGLang rank budget rejects these repairs as new content; their
# unchanged bytes stay baselined and each is a hardware-led follow-up (#1761).
RANK_BUDGET_FOLLOW_UPS = {
    "deepseek-v4-flash-vision-exp-sglang-tonyd2wild-dual",
    "inkling-975b-a41b-nvfp4-sglang-eight",
    "inkling-small-nvfp4-sglang-mia-dual",
    "mimo-v2-6-flash-rl-nvfp4-r0b0tlab-sglang-dual",
}


def recipe_for(slug: str) -> RecipeDefinition:
    return RecipeDefinition.model_validate_json(
        (ROOT / "recipes" / f"{slug}.json").read_text()
    )


def test_proven_recipe_documents_remain_byte_identical() -> None:
    """Proven bytes change only with their entry here; an upstream refresh writes both.

    Owner (2026-10-26): proven recipes move to the newest upstream version and the sweep
    re-checks them (``tools/refresh-upstream`` ``follow_proven_bytes``). Any other edit
    of a proven document must update this fixture deliberately.
    """
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
            need = per_rank_need(variant, catalog_models(), load_inputs())
            assert all(
                need.peak_bytes + role.resources.memory.reserve_bytes
                <= IDLE_AVAILABLE_BYTES
                for role in variant.topology.roles
            ), key
        checked += 1
    assert checked == 115


def require_repaired_declaration(slug: str, variant: RecipeDefinition) -> None:
    """A repaired role declares its honest startup peak: at least the loading floor.

    Owner (2026-10-27): the peak may exceed the floor where hardware measured
    more, per role; it must still be admissible, never shaved to fit.
    """
    require_admissible_declaration(variant)
    need = per_rank_need(variant, catalog_models(), load_inputs())
    assert not declaration_problems(variant, need)
    if variant.runtime.engine == "vllm":
        fraction = argument_value(variant, "gpu-memory-utilization")
        assert fraction is not None
        budget = Decimal(str(fraction)) * TOTAL_BYTES + OVERHEAD_BYTES
        assert all(
            budget <= role.resources.memory.peak_bytes
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
    if slug in RANK_BUDGET_FOLLOW_UPS:
        with pytest.raises((TypeError, ValueError)):
            require_memory_need(
                variant, catalog_models(), load_inputs(), document_digest=""
            )
    else:
        require_memory_need(
            variant, catalog_models(), load_inputs(), document_digest=""
        )


@pytest.mark.parametrize("slug", REPAIRED)
def test_repaired_declarations_and_engine_allocations_fit_every_variant(
    slug: str,
) -> None:
    recipe = recipe_for(slug)
    assert f"{recipe.identity.publisher}/{slug}" not in PROVEN
    for variant in memory_variants(recipe):
        require_repaired_declaration(slug, variant)


def test_repaired_role_declares_its_measured_peak_above_the_floor() -> None:
    slug = "mimo-v2-5-omni-nvfp4-tonyd2wild-vllm-dual"
    recipe = recipe_for(slug)
    floor = per_rank_need(recipe, catalog_models(), load_inputs()).peak_bytes
    entrypoint, worker = recipe.topology.roles
    # Hardware showed more than the floor, and the roles may differ.
    entrypoint.resources.memory.peak_bytes = floor + 9_000_000_000
    worker.resources.memory.peak_bytes = floor + 13_000_000_000
    require_repaired_declaration(slug, recipe)
    # 2026-10-27: ~120.9 GB at the end of target load, before its 7.46 GB KV
    # pool. An honest peak that is not admissible is reported, never shaved.
    worker.resources.memory.peak_bytes = 128_400_000_000
    with pytest.raises(ValueError, match="worker.*not admissible"):
        require_repaired_declaration(slug, recipe)
    # Below the loading floor still hides demand.
    worker.resources.memory.peak_bytes = floor - 1
    with pytest.raises(AssertionError):
        require_repaired_declaration(slug, recipe)


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
    with pytest.raises(ValueError, match="not admissible"):
        require_memory_need(recipe, catalog_models(), load_inputs(), document_digest="")


def test_static_budget_drift_cannot_hide_behind_a_smaller_declaration() -> None:
    recipe = recipe_for("deepseek-v4-flash-vision-exp-sglang-tonyd2wild-dual")
    next(
        a for a in recipe.runtime.arguments if a.name == "mem-fraction-static"
    ).value = "0.80"
    with pytest.raises(ValueError, match="static allocation"):
        require_memory_need(recipe, catalog_models(), load_inputs(), document_digest="")


def test_repaired_deferral_uses_shared_admission_with_owned_idle_host() -> None:
    recipe = recipe_for("deepseek-v4-flash-vision-exp-sglang-tonyd2wild-dual")
    # Proven 120,370,379,081-byte peaks were admitted live; the offline host
    # baseline must admit them too (it is not the estimator's 3542 incident).
    assert ADMISSION_IDLE_USE_BYTES <= TOTAL_BYTES - OS_FLOOR_BYTES - 120_370_379_081
    for role in recipe.topology.roles:
        assert role.resources.memory.peak_bytes == 92_405_678_632
        assert evaluate_memory_admission(
            role.resources.memory.peak_bytes,
            TOTAL_BYTES,
            occupied_bytes=ADMISSION_IDLE_USE_BYTES,
        ).admissible
        # Informational reserve is not another claim in the shared predicate.
        role.resources.memory.reserve_bytes = TOTAL_BYTES
    require_admissible_declaration(recipe)
    recipe.topology.roles[1].resources.memory.peak_bytes = (
        TOTAL_BYTES - ADMISSION_IDLE_USE_BYTES - OS_FLOOR_BYTES + 1
    )
    with pytest.raises(ValueError, match="worker.*not admissible"):
        require_admissible_declaration(recipe)


def test_tests_judge_admission_only_through_its_owner() -> None:
    # One owner: require_admissible_declaration / evaluate_memory_admission. A
    # hand-written peak + reserve bound of 128 GB was a second, stricter copy:
    # it refused the proven 120,370,379,081-byte peak beside an 8 GB reserve.
    # Pinning a declared peak to the loading floor was another: it forbade the
    # honest measured peak (owner 2026-10-27; declaration_problems owns >= for
    # multi-Spark roles).
    recipe = recipe_for("deepseek-v4-flash-vision-exp-sglang-tonyd2wild-dual")
    for role in recipe.topology.roles:
        role.resources.memory.peak_bytes = 120_370_379_081
        role.resources.memory.reserve_bytes = 8_000_000_000
    require_admissible_declaration(recipe)
    # The 128-GB literal, and a peak + reserve sum bounded by hand.
    restated = re.compile(
        r"\b128(?:_000_000_000|000000000|e9)\b"
        r"|assertLess(?:Equal)?\(\s*(?:envelope|admission)\b"
        r"|reserve_bytes\W*\s*<=?\s*\d"
        r"|peak_bytes\s*==\s*need\.peak_bytes"
    )
    copies = [
        f"{path.name}:{number}"
        for path in sorted((ROOT / "tests").glob("*.py"))
        for number, line in enumerate(path.read_text().splitlines(), 1)
        if restated.search(line)
    ]
    assert not copies, f"restated admission limit: {copies}"


def test_recorded_content_precedes_admission(monkeypatch) -> None:
    from qualification import memory_need

    recipe = recipe_for("deepseek-v4-flash-vision-exp-sglang-tonyd2wild-dual")
    document = json.loads(
        (ROOT / "recipes" / f"{recipe.identity.slug}.json").read_text()
    )
    digest = document_sha256(document)
    inputs = load_inputs()
    key = f"{recipe.identity.publisher}/{recipe.identity.slug}"
    inputs.historical_documents[key] = [digest]
    calls = []

    def reject(*args, **kwargs):
        calls.append((args, kwargs))
        return evaluate_memory_admission(1, 0)

    monkeypatch.setattr(memory_need, "evaluate_memory_admission", reject)
    assert require_memory_need(recipe, {}, inputs, document_digest=digest) is None
    assert not calls
    with pytest.raises(ValueError, match="not admissible"):
        require_memory_need(recipe, {}, inputs, document_digest="")
    assert calls


def test_wrapper_cache_alias_counts_toward_actual_rank_demand() -> None:
    recipe = recipe_for("deepseek-v4-flash-vision-exp-sfxnz-vllm-dual")
    initial = per_rank_need(recipe, catalog_models(), load_inputs()).peak_bytes
    pin = next(a for a in recipe.runtime.arguments if a.name == "kv-cache-memory")
    assert isinstance(pin.value, int)
    assert pin.value == 12 * 1024**3
    pin.value += 1
    assert (
        per_rank_need(recipe, catalog_models(), load_inputs()).peak_bytes == initial + 1
    )
    assert declaration_problems(
        recipe, per_rank_need(recipe, catalog_models(), load_inputs())
    )


@pytest.mark.parametrize("rank", [0, 1])
def test_absolute_kv_launch_survives_observed_startup_memory(rank: int) -> None:
    import math

    recipe = recipe_for("deepseek-v4-flash-fp8-r0b0tlab-vllm-dual")

    # vLLM dda4668b worker/utils.py::request_memory checks this fraction even
    # with absolute KV bytes, before the checkpoint loader runs. Both ranks
    # receive the same arguments. Reproduce the 2026-10-22 startup refusal.
    def startup(fraction: Decimal) -> None:
        if math.ceil(TOTAL_BYTES * fraction) > 111.79 * 1024**3:
            raise ValueError("Free memory on device cuda:0")

    with pytest.raises(ValueError, match="Free memory"):
        startup(Decimal("0.92"))
    fraction = argument_value(recipe, "gpu-memory-utilization")
    assert fraction is not None
    startup(Decimal(str(fraction)))
    assert (
        Decimal(str(fraction)) * TOTAL_BYTES + OVERHEAD_BYTES
        <= recipe.topology.roles[rank].resources.memory.peak_bytes
    )
    assert argument_value(recipe, "kv-cache-memory-bytes") == 3_414_163_456


@pytest.mark.parametrize("defect", ["omitted", "oversized"])
@pytest.mark.parametrize(
    "slug",
    [
        "deepseek-v4-flash-fp8-r0b0tlab-vllm-dual",
        "deepseek-v4-flash-vision-exp-sfxnz-vllm-dual",
    ],
)
def test_absolute_kv_cannot_hide_an_implicit_startup_reservation(
    defect: str, slug: str
) -> None:
    recipe = recipe_for(slug)
    recipe.runtime.arguments = [
        a for a in recipe.runtime.arguments if a.name != "gpu-memory-utilization"
    ]
    if defect == "oversized":
        recipe.runtime.arguments.append(
            RecipeRuntimeArgument(name="gpu-memory-utilization", value="0.92")
        )
    with pytest.raises(ValueError, match="startup"):
        require_memory_need(recipe, catalog_models(), load_inputs(), document_digest="")


def test_unproven_absolute_kv_recipes_bound_startup_for_every_option() -> None:
    from qualification.memory_need import require_vllm_startup_budget

    for path in sorted((ROOT / "recipes").glob("*.json")):
        recipe = RecipeDefinition.model_validate_json(path.read_text())
        if f"{recipe.identity.publisher}/{recipe.identity.slug}" in PROVEN:
            continue
        for variant in memory_variants(recipe):
            require_vllm_startup_budget(variant)
