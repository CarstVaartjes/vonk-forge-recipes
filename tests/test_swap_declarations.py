"""Split peaks: a load that overshoots the steady peak names the swap it needs."""

import json
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError
from vonk_forge_contracts import RecipeDefinition
from vonk_forge_contracts.recipe import RecipeMemoryResources

from qualification.memory_need import (
    SWAP_DEFAULT_BYTES,
    SWAP_MAXIMUM_BYTES,
    memory_warnings,
    require_admissible_declaration,
    swap_needed_bytes,
    usable_swap_bytes,
)

ROOT = Path(__file__).resolve().parents[1]
SLUG = "agents-a1-nvfp4-r0b0tlab-vllm-single"
GIB = 1024**3
USABLE_MAX = usable_swap_bytes(SWAP_MAXIMUM_BYTES)


def document() -> dict[str, Any]:
    path = ROOT / "recipes" / f"{SLUG}.json"
    return json.loads(path.read_text())


def recipe_with(**memory: int) -> RecipeDefinition:
    value = document()
    for role in value["topology"]["roles"]:
        role["resources"]["memory"].update(memory)
    return RecipeDefinition.model_validate(value)


def peak() -> int:
    return document()["topology"]["roles"][0]["resources"]["memory"]["peak_bytes"]


def declared(overshoot: int, swap: int) -> RecipeDefinition:
    return recipe_with(load_peak_bytes=peak() + overshoot, minimum_swap_bytes=swap)


VECTOR = json.loads((ROOT / "tests/fixtures/swap-v1.json").read_text())


def test_swap_arithmetic_matches_the_shared_vectors() -> None:
    assert VECTOR["source"]["sha256"] and VECTOR["source"]["commit"]
    for pair in VECTOR["usable"]:
        assert usable_swap_bytes(pair["total_bytes"]) == pair["usable_bytes"], pair
    for pair in VECTOR["needed"]:
        assert swap_needed_bytes(pair["load_bytes"]) == pair["swap_bytes"], pair


def test_unset_fields_leave_documents_unchanged() -> None:
    raw = document()
    raw_memory = raw["topology"]["roles"][0]["resources"]["memory"]
    assert "load_peak_bytes" not in raw_memory
    assert "minimum_swap_bytes" not in raw_memory
    parsed = RecipeDefinition.model_validate(raw)
    memory = parsed.topology.roles[0].resources.memory
    assert memory.load_peak_bytes is None and memory.minimum_swap_bytes == 0
    dumped = parsed.model_dump(mode="json")["topology"]["roles"][0]["resources"]
    assert dumped["memory"] == raw_memory


def test_equal_load_and_steady_peaks_are_refused() -> None:
    with pytest.raises(ValidationError, match="must exceed peak_bytes"):
        recipe_with(load_peak_bytes=peak())
    with pytest.raises(ValidationError, match="must exceed peak_bytes"):
        recipe_with(load_peak_bytes=peak() - 1)


def test_minimum_swap_without_a_load_peak_is_refused() -> None:
    with pytest.raises(ValidationError, match="requires load_peak_bytes"):
        recipe_with(minimum_swap_bytes=1)


def test_contract_fields_serialize_only_when_set() -> None:
    memory = RecipeMemoryResources(
        peak_bytes=1, reserve_bytes=0, load_peak_bytes=2, minimum_swap_bytes=3
    )
    assert memory.model_dump(mode="json")["load_peak_bytes"] == 2


def test_overshoot_at_the_usable_maximum_is_admitted_and_one_byte_past_never_fits() -> (
    None
):
    require_admissible_declaration(declared(USABLE_MAX, SWAP_MAXIMUM_BYTES))
    with pytest.raises(ValueError, match="can never fit"):
        require_admissible_declaration(declared(USABLE_MAX + 1, SWAP_MAXIMUM_BYTES))


def test_minimum_swap_must_cover_the_overshoot_exactly() -> None:
    overshoot = 50 * GIB
    needed = swap_needed_bytes(overshoot)
    require_admissible_declaration(declared(overshoot, needed))
    with pytest.raises(ValueError, match="is below"):
        require_admissible_declaration(declared(overshoot, needed - 1))


def test_a_load_peak_without_minimum_swap_is_refused() -> None:
    with pytest.raises(ValueError, match="is below"):
        require_admissible_declaration(recipe_with(load_peak_bytes=peak() + GIB))


def test_minimum_swap_cannot_exceed_the_policy_maximum() -> None:
    require_admissible_declaration(declared(GIB, SWAP_MAXIMUM_BYTES))
    with pytest.raises(ValueError, match="exceeds the .*swap maximum"):
        require_admissible_declaration(declared(GIB, SWAP_MAXIMUM_BYTES + 1))


def test_warning_only_when_needed_swap_passes_the_default() -> None:
    at_default = usable_swap_bytes(SWAP_DEFAULT_BYTES)
    assert not [
        w
        for w in memory_warnings(declared(at_default, SWAP_DEFAULT_BYTES), {}, None)  # type: ignore[arg-type]
        if "of swap" in w
    ]
    over = memory_warnings(
        declared(at_default + 1, swap_needed_bytes(at_default + 1)),
        {},
        None,  # type: ignore[arg-type]
    )
    assert any("of swap" in w for w in over)
