"""Catch reserve double charging, guessed host use and divergent consumers."""

import ast
from pathlib import Path

import documents
import pytest
from vonk_forge_contracts import RecipeDefinition, document_sha256
from vonk_forge_contracts.memory_admission import (
    PLATFORM_MEMORY_FLOOR_BYTES,
    evaluate_memory_admission,
)
from vonk_forge_contracts.recipe import RecipeRuntimeArgument

from qualification import memory_need
from qualification.memory_need import vllm_utilization_envelope

ROOT = Path(__file__).resolve().parents[1]


def test_declaration_boundary_and_observed_host_are_distinct() -> None:
    pool = 130_663_235_584
    assert evaluate_memory_admission(
        pool - PLATFORM_MEMORY_FLOOR_BYTES, pool
    ).admissible
    assert not evaluate_memory_admission(
        pool - PLATFORM_MEMORY_FLOOR_BYTES + 1, pool
    ).admissible
    assert evaluate_memory_admission(124_000_000_000, pool).admissible
    assert not evaluate_memory_admission(
        124_000_000_000,
        pool,
        occupied_bytes=pool - 124_000_000_000 - PLATFORM_MEMORY_FLOOR_BYTES + 1,
    ).admissible


@pytest.mark.parametrize(
    "claims",
    [
        {"occupied_bytes": 11},
        {"reserved_bytes": 11},
        {"unmaterialized_bytes": 11},
        {"unknown_residual_bytes": 11},
    ],
)
def test_live_claims_and_release_are_counted_once(claims: dict[str, int]) -> None:
    assert not evaluate_memory_admission(
        80, 100, memory_floor_bytes=10, **claims
    ).admissible
    assert evaluate_memory_admission(
        80, 100, memory_floor_bytes=10, release_bytes=1, **claims
    ).admissible
    assert not evaluate_memory_admission(
        101, 100, memory_floor_bytes=0, release_bytes=100
    ).admissible


def test_recipe_guard_delegates_even_for_historical_content(monkeypatch) -> None:
    recipe = RecipeDefinition.model_validate_json(
        (ROOT / "recipes/d3-mini-vllm-single.json").read_text()
    )
    # Large informational reserve cannot alter admission.
    recipe.topology.roles[0].resources.memory.reserve_bytes = 10**15
    memory_need.require_admissible_declaration(recipe)
    monkeypatch.setattr(
        memory_need,
        "evaluate_memory_admission",
        lambda *a, **kw: evaluate_memory_admission(100, 0),
    )
    with pytest.raises(ValueError, match="not admissible"):
        memory_need.require_memory_need(
            recipe, {}, memory_need.load_inputs(), document_digest=""
        )


def test_unrecorded_declarations_fit_the_idle_target_pool() -> None:
    import json

    inputs = memory_need.load_inputs()
    for path in (ROOT / "recipes").glob("*.json"):
        document = json.loads(path.read_text())
        recipe = RecipeDefinition.model_validate(document)
        if memory_need.unchanged_recipe(recipe, document_sha256(document), inputs):
            continue
        memory_need.require_admissible_declaration(recipe)


def _delegates_without_arithmetic(source: str) -> bool:
    tree = ast.parse(source)
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "require_admissible_declaration"
    )
    return not any(
        isinstance(n, (ast.BinOp, ast.Compare)) for n in ast.walk(function)
    ) and any(
        isinstance(n, ast.Call)
        and isinstance(n.func, ast.Name)
        and n.func.id == "evaluate_memory_admission"
        for n in ast.walk(function)
    )


def test_guard_rejects_a_second_admission_formula() -> None:
    source = (ROOT / "qualification/memory_need.py").read_text()
    assert _delegates_without_arithmetic(source)
    assert not _delegates_without_arithmetic(
        source.replace(
            "not evaluate_memory_admission(\n"
            "                role.resources.memory.peak_bytes,\n"
            "                pool_bytes,\n"
            "                occupied_bytes=ADMISSION_IDLE_USE_BYTES,\n"
            "            ).admissible",
            "role.resources.memory.peak_bytes + role.resources.memory.reserve_bytes > pool_bytes",
        )
    )


@pytest.mark.parametrize("bad", [-1, True, 1.5])
def test_invalid_byte_evidence_is_not_clamped(bad) -> None:
    with pytest.raises(ValueError, match="nonnegative integer"):
        evaluate_memory_admission(1, 100, occupied_bytes=bad)


def _single_at_utilization(value: str) -> RecipeDefinition:
    recipe = documents.recipe("qwen3-6-35b-a3b-nvfp4-r0b0tlab-vllm-single")
    for argument in recipe.runtime.arguments:
        if argument.name == "gpu-memory-utilization":
            argument.value = value
    envelope = vllm_utilization_envelope(recipe)
    assert envelope is not None
    for role in recipe.topology.roles:
        role.resources.memory.peak_bytes = envelope
    return recipe


def test_creator_utilization_088_single_is_admissible() -> None:
    """The creator's 0.88 (122,983,643,710 B) fits an idle Spark beside the 2 GiB floor."""
    recipe = _single_at_utilization("0.88")
    assert vllm_utilization_envelope(recipe) == 121_183_643_710
    memory_need.require_admissible_declaration(recipe)


def test_utilization_090_single_is_admissible() -> None:
    recipe = _single_at_utilization("0.90")
    assert vllm_utilization_envelope(recipe) == 123_796_908_340
    memory_need.require_admissible_declaration(recipe)


@pytest.mark.parametrize(
    ("value", "admissible"),
    [("0.905", True), ("0.9051", False)],
)
def test_single_spark_vllm_utilization_maximum_is_0905(
    value: str, admissible: bool
) -> None:
    recipe = _single_at_utilization(value)
    if admissible:
        memory_need.require_admissible_declaration(recipe)
    else:
        with pytest.raises(ValueError, match="not admissible"):
            memory_need.require_admissible_declaration(recipe)


@pytest.mark.parametrize(
    ("value", "admissible"),
    [("0.8728", True), ("0.8729", False)],
)
def test_multi_spark_vllm_utilization_maximum_is_08728(
    value: str, admissible: bool
) -> None:
    recipe = documents.recipe("deepseek-v4-flash-vllm-mia-dual")
    recipe.runtime.arguments = [
        item
        for item in recipe.runtime.arguments
        if item.name not in {"gpu-memory-utilization", "kv-cache-memory-bytes"}
    ]
    recipe.runtime.arguments.append(
        RecipeRuntimeArgument(name="gpu-memory-utilization", value=value)
    )
    envelope = vllm_utilization_envelope(recipe)
    assert envelope is not None
    assert memory_need.budget_margin_bytes(recipe) == 10_400_000_000
    for role in recipe.topology.roles:
        role.resources.memory.peak_bytes = envelope
    if admissible:
        memory_need.require_admissible_declaration(recipe)
    else:
        with pytest.raises(ValueError, match="not admissible"):
            memory_need.require_admissible_declaration(recipe)


def test_refresh_raises_peaks_to_the_budget_envelope_and_never_lowers() -> None:
    recipe = documents.recipe("deepseek-v4-flash-vllm-mia-dual")
    for role in recipe.topology.roles:
        role.resources.memory.peak_bytes = 100_000_000_000
        role.resources.memory.load_peak_bytes = 101_000_000_000
    raised = memory_need.raise_peaks_to_envelope(recipe)
    assert raised is not None
    envelope = vllm_utilization_envelope(raised)
    assert envelope is not None and envelope > 100_000_000_000
    for role in raised.topology.roles:
        assert role.resources.memory.peak_bytes == envelope
        assert role.resources.memory.load_peak_bytes == envelope + 1_000_000_000
    assert memory_need.raise_peaks_to_envelope(raised) is None
    raised.topology.roles[0].resources.memory.peak_bytes = envelope + 5
    assert memory_need.raise_peaks_to_envelope(raised) is None
    assert raised.topology.roles[0].resources.memory.peak_bytes == envelope + 5
