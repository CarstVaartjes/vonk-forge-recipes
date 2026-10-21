"""Catch reserve double charging, guessed host use and divergent consumers."""

import ast
from pathlib import Path

import pytest
from vonk_forge_contracts import RecipeDefinition, document_sha256
from vonk_forge_contracts.memory_admission import (
    PLATFORM_MEMORY_FLOOR_BYTES,
    evaluate_memory_admission,
)

from qualification import memory_need

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
        124_000_000_000, pool, occupied_bytes=4_515_061_760
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
            "                occupied_bytes=IDLE_USE_BYTES,\n"
            "            ).admissible",
            "role.resources.memory.peak_bytes + role.resources.memory.reserve_bytes > pool_bytes",
        )
    )


@pytest.mark.parametrize("bad", [-1, True, 1.5])
def test_invalid_byte_evidence_is_not_clamped(bad) -> None:
    with pytest.raises(ValueError, match="nonnegative integer"):
        evaluate_memory_admission(1, 100, occupied_bytes=bad)
