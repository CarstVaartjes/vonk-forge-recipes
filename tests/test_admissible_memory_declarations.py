"""Admission declarations, engine budgets and frozen hardware calibration bytes."""

import hashlib
import json
import re
from decimal import Decimal
from pathlib import Path

import documents
import pytest
from catalog_documents import catalog_models
from vonk_forge_contracts import (
    RecipeDefinition,
)
from vonk_forge_contracts.memory_admission import evaluate_memory_admission
from vonk_forge_contracts.recipe import RecipeRuntimeArgument
from warning_asserts import assert_not_warns, assert_warns, declare_peak

from qualification.memory_need import (
    ADMISSION_IDLE_USE_BYTES,
    IDLE_AVAILABLE_BYTES,
    OS_FLOOR_BYTES,
    TOTAL_BYTES,
    admissible_declaration_warnings,
    argument_value,
    declarable_vllm_envelope,
    declaration_problems,
    load_inputs,
    memory_estimate_warnings,
    memory_variants,
    per_rank_need,
    sglang_demand,
    sglang_rank_budget_warnings,
)

ROOT = Path(__file__).resolve().parents[1]
PROVEN = json.loads((ROOT / "tests/fixtures/proven-recipe-bytes.json").read_text())


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
    assert PROVEN
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
            assert need.peak_bytes <= IDLE_AVAILABLE_BYTES, key
        checked += 1
    assert checked == 114


def require_repaired_declaration(slug: str, variant: RecipeDefinition) -> None:
    """A repaired role declares its honest startup peak: at least the loading floor.

    Owner (2026-10-27): the peak may exceed the floor where hardware measured
    more, per role; it must still be admissible, never shaved to fit.
    """
    need = per_rank_need(variant, catalog_models(), load_inputs())
    assert not declaration_problems(variant, need)
    if variant.runtime.engine == "vllm":
        fraction = argument_value(variant, "gpu-memory-utilization")
        assert fraction is not None
        # Pool-less: the envelope; absolute pool: the startup request (utilization x total).
        budget = declarable_vllm_envelope(variant) or (
            Decimal(str(fraction)) * TOTAL_BYTES
        )
        assert all(
            budget <= role.resources.memory.peak_bytes
            for role in variant.topology.roles
        )
    else:
        fraction = argument_value(variant, "mem-fraction-static")
        assert fraction is not None
        allocated = sglang_demand(variant, None)
        assert allocated is not None and all(
            allocated <= role.resources.memory.peak_bytes
            for role in variant.topology.roles
        )


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
    assert_warns(admissible_declaration_warnings(recipe), "worker.*not admissible")
    # Below the loading floor still hides demand.
    worker.resources.memory.peak_bytes = floor - 1
    with pytest.raises(AssertionError):
        require_repaired_declaration(slug, recipe)


@pytest.mark.parametrize(
    ("slug", "shaved"),
    [
        ("deepseek-v4-flash-vision-exp-sglang-tonyd2wild-dual", "0.645978808810"),
        ("mimo-v2-6-flash-rl-nvfp4-r0b0tlab-sglang-dual", "0.718702620795"),
    ],
)
def test_sglang_rank_budget_refuses_the_shaved_fraction(slug: str, shaved: str) -> None:
    """#1647 cut the fraction below the per-rank weights plus one request's pools."""
    recipe = recipe_for(slug)
    declare_peak(recipe)
    key = f"vonk-forge/{slug}"
    assert_not_warns(
        sglang_rank_budget_warnings(recipe, catalog_models(), load_inputs(), key),
        "static budget",
    )
    next(
        a for a in recipe.runtime.arguments if a.name == "mem-fraction-static"
    ).value = shaved
    assert_warns(
        sglang_rank_budget_warnings(recipe, catalog_models(), load_inputs(), key),
        "static budget",
    )


@pytest.mark.parametrize("rank", [0, 1])
def test_original_deferral_fails_before_any_controller_operation(rank: int) -> None:
    recipe = recipe_for("deepseek-v4-flash-vision-exp-sglang-tonyd2wild-dual")
    recipe.topology.roles[rank].resources.memory.peak_bytes = (
        TOTAL_BYTES - ADMISSION_IDLE_USE_BYTES - OS_FLOOR_BYTES + 1
    )
    assert_warns(
        memory_estimate_warnings(
            recipe, catalog_models(), load_inputs(), document_digest=""
        ),
        "not admissible",
    )


def test_static_budget_drift_cannot_hide_behind_a_smaller_declaration() -> None:
    recipe = recipe_for("deepseek-v4-flash-vision-exp-sglang-tonyd2wild-dual")
    declare_peak(recipe, 112_530_585_191)
    next(
        a for a in recipe.runtime.arguments if a.name == "mem-fraction-static"
    ).value = "0.85"
    assert_warns(
        memory_estimate_warnings(
            recipe, catalog_models(), load_inputs(), document_digest=""
        ),
        "static allocation",
    )


def test_repaired_deferral_uses_shared_admission_with_owned_idle_host() -> None:
    recipe = recipe_for("deepseek-v4-flash-vision-exp-sglang-tonyd2wild-dual")
    # Proven 120,370,379,081-byte peaks were admitted live; the offline host
    # baseline must admit them too (it is not the estimator's 3542 incident).
    assert ADMISSION_IDLE_USE_BYTES <= TOTAL_BYTES - OS_FLOOR_BYTES - 120_370_379_081
    declare_peak(recipe, 112_530_585_191)
    for role in recipe.topology.roles:
        assert role.resources.memory.peak_bytes == 112_530_585_191
        assert evaluate_memory_admission(
            role.resources.memory.peak_bytes,
            TOTAL_BYTES,
            occupied_bytes=ADMISSION_IDLE_USE_BYTES,
        ).admissible
        # Informational reserve is not another claim in the shared predicate.
        role.resources.memory.reserve_bytes = TOTAL_BYTES
    assert_not_warns(admissible_declaration_warnings(recipe), "worker.*not admissible")
    recipe.topology.roles[1].resources.memory.peak_bytes = (
        TOTAL_BYTES - ADMISSION_IDLE_USE_BYTES - OS_FLOOR_BYTES + 1
    )
    assert_warns(
        admissible_declaration_warnings(recipe),
        "worker.*not admissible",
    )


def test_tests_judge_admission_only_through_its_owner() -> None:
    # One owner: admissible_declaration_warnings / evaluate_memory_admission. A
    # hand-written peak + reserve bound of 128 GB was a second, stricter copy:
    # it refused the proven 120,370,379,081-byte peak beside an 8 GB reserve.
    # Pinning a declared peak to the loading floor was another: it forbade the
    # honest measured peak (owner 2026-10-27; declaration_problems owns >= for
    # multi-Spark roles).
    recipe = recipe_for("deepseek-v4-flash-vision-exp-sglang-tonyd2wild-dual")
    for role in recipe.topology.roles:
        role.resources.memory.peak_bytes = 120_370_379_081
        role.resources.memory.reserve_bytes = 8_000_000_000
    assert admissible_declaration_warnings(recipe) == []
    # The 128-GB literal and the margin beside an engine budget, in tests.
    restated = re.compile(
        r"\b128(?:_000_000_000|000000000|e9)\b"
        r"|assertLess(?:Equal)?\(\s*(?:envelope|admission)\b"
        r"|peak_bytes\s*==\s*need\.peak_bytes"
        # The margin beside an engine budget has one owner, budget_margin_bytes.
        r"|(?:TOTAL_BYTES|SPARK_MEMORY_BYTES)\)?(?:\s*\*\s*[\d.]+)?\s*\+\s*"
        r"(?:OVERHEAD_BYTES|8_000_000_000|8000000000|8e9)\b"
    )
    # The informational reserve compared or summed, in tests or qualification:
    # admission owns the platform floor and never charges reserve_bytes.
    reserve = re.compile(r"reserve_bytes\W*[<>+]|\+\s*[\w.\[\]\"']*reserve_bytes")
    copies = [
        f"{folder}/{path.name}:{number}"
        for folder in ("tests", "qualification")
        for path in sorted((ROOT / folder).glob("*.py"))
        for number, line in enumerate(path.read_text().splitlines(), 1)
        if reserve.search(line) or (folder == "tests" and restated.search(line))
    ]
    assert not copies, f"restated admission limit: {copies}"


def test_recorded_content_precedes_admission(monkeypatch) -> None:
    from qualification import memory_need

    recipe = recipe_for("deepseek-v4-flash-vision-exp-sglang-tonyd2wild-dual")
    declare_peak(recipe)
    inputs = load_inputs()
    calls = []

    def reject(*args, **kwargs):
        calls.append((args, kwargs))
        return evaluate_memory_admission(1, 0)

    monkeypatch.setattr(memory_need, "evaluate_memory_admission", reject)
    monkeypatch.setattr(memory_need, "unchanged_recipe", lambda *_: True)
    assert_not_warns(
        memory_estimate_warnings(recipe, {}, inputs, document_digest="recorded"),
        "not admissible",
    )
    assert not calls
    monkeypatch.setattr(memory_need, "unchanged_recipe", lambda *_: False)
    assert_warns(
        memory_estimate_warnings(recipe, {}, inputs, document_digest=""),
        "not admissible",
    )
    assert calls


def test_wrapper_cache_alias_counts_toward_actual_rank_demand() -> None:
    recipe = recipe_for("deepseek-v4-flash-vision-exp-sfxnz-vllm-dual")
    initial = per_rank_need(recipe, catalog_models(), load_inputs()).peak_bytes
    declare_peak(recipe, initial)
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
    declare_peak(recipe, 100_000_000_000)
    recipe.runtime.arguments = [
        a for a in recipe.runtime.arguments if a.name != "gpu-memory-utilization"
    ]
    if defect == "oversized":
        recipe.runtime.arguments.append(
            RecipeRuntimeArgument(name="gpu-memory-utilization", value="0.92")
        )
    assert_warns(
        memory_estimate_warnings(
            recipe, catalog_models(), load_inputs(), document_digest=""
        ),
        "startup",
    )


def _creator_startup_variant(slug: str, fraction: str, pool: str | int):
    recipe = documents.recipe(slug)
    values = {
        "gpu-memory-utilization": fraction,
        "kv-cache-memory-bytes": pool,
    }
    for argument in recipe.runtime.arguments:
        if argument.name in values:
            argument.value = values[argument.name]
    return recipe


def test_startup_budget_matches_vllms_free_memory_check() -> None:
    from qualification.memory_need import startup_budget_warnings

    reap = _creator_startup_variant(
        "deepseek-v4-flash-reap-180b-0xsero-vllm-single", "0.88", "6G"
    )
    declare_peak(reap, 120_370_379_081)
    assert startup_budget_warnings(reap) == []

    dual = _creator_startup_variant(
        "deepseek-v4-1-flash-exl3-mia-dual", "0.88", 2_684_354_560
    )
    peak = per_rank_need(dual, catalog_models(), load_inputs()).peak_bytes
    assert peak == 115_984_824_099
    for role in dual.topology.roles:
        role.resources.memory.peak_bytes = peak
    dual = RecipeDefinition.model_validate(dual.model_dump(mode="json"))
    assert startup_budget_warnings(dual) == []
    assert (
        memory_estimate_warnings(
            dual, catalog_models(), load_inputs(), document_digest="", touched=True
        )
        == []
    )


def test_startup_budget_still_refuses_a_fraction_above_the_peak() -> None:
    from qualification.memory_need import startup_budget_warnings

    reap = _creator_startup_variant(
        "deepseek-v4-flash-reap-180b-0xsero-vllm-single", "0.93", "6G"
    )
    declare_peak(reap, 120_370_379_081)
    assert Decimal("0.93") * TOTAL_BYTES > 120_370_379_081
    assert_warns(
        startup_budget_warnings(reap),
        "startup free-memory check",
    )
