"""The admission maximum caps the vLLM envelope; measured evidence is never raised."""

import ast
from pathlib import Path

import pytest
from catalog_documents import catalog_models
from vonk_forge_contracts import RecipeDefinition
from vonk_forge_contracts.memory_admission import evaluate_memory_admission
from warning_asserts import assert_not_warns, declare_peak

from qualification import documents
from qualification.memory_need import (
    ADMISSION_IDLE_USE_BYTES,
    ADMISSION_MAXIMUM_PEAK_BYTES,
    PLATFORM_MEMORY_FLOOR_BYTES,
    SPARK_IDLE_AVAILABLE_BYTES,
    SPARK_MEMORY_BYTES,
    declarable_vllm_envelope,
    load_inputs,
    memory_estimate_warnings,
    raise_peaks_to_envelope,
    vllm_utilization_envelope,
)

INPUTS = load_inputs()
GLM = "glm-5-3-flash-exl3-k2-0xsero-vllm-single"
PROVEN = "qwen3-6-35b-a3b-fp8-eugr-vllm-single"
LLAMA = "llama-3-1-405b-instruct-awq-int4-nvidia-vllm-four"


def test_maximum_derives_from_the_admission_rule() -> None:
    assert (
        ADMISSION_MAXIMUM_PEAK_BYTES
        == SPARK_IDLE_AVAILABLE_BYTES - PLATFORM_MEMORY_FLOOR_BYTES
        == 124_452_516_352
    )


def test_maximum_is_the_edge_of_the_offline_admission_rule() -> None:
    def admissible(peak: int) -> bool:
        return evaluate_memory_admission(
            peak, SPARK_MEMORY_BYTES, occupied_bytes=ADMISSION_IDLE_USE_BYTES
        ).admissible

    assert admissible(ADMISSION_MAXIMUM_PEAK_BYTES)
    assert not admissible(ADMISSION_MAXIMUM_PEAK_BYTES + 1)


@pytest.mark.parametrize("slug", [GLM, LLAMA])
def test_peak_at_the_admission_maximum_is_not_understated(slug: str) -> None:
    recipe = documents.recipe(slug)
    declare_peak(recipe, ADMISSION_MAXIMUM_PEAK_BYTES)
    warnings = memory_estimate_warnings(
        recipe, catalog_models(), INPUTS, document_digest="", touched=True
    )
    assert_not_warns(warnings, "peak understates (vLLM budget|the vLLM)")


@pytest.mark.parametrize("slug", [GLM, LLAMA])
def test_envelope_never_exceeds_the_admission_maximum(slug: str) -> None:
    recipe = documents.recipe(slug)
    for item in recipe.runtime.arguments:
        if item.name == "gpu-memory-utilization":
            item.value = "0.95"
    uncapped = vllm_utilization_envelope(recipe)
    assert uncapped is not None
    assert uncapped > ADMISSION_MAXIMUM_PEAK_BYTES
    assert declarable_vllm_envelope(recipe) == ADMISSION_MAXIMUM_PEAK_BYTES
    declare_peak(recipe, 1_000_000_000)
    raised = raise_peaks_to_envelope(recipe)
    assert raised is not None
    assert all(
        r.resources.memory.peak_bytes <= ADMISSION_MAXIMUM_PEAK_BYTES
        for r in raised.topology.roles
    )


def with_utilization(recipe: RecipeDefinition, value: str) -> RecipeDefinition:
    moved = recipe.model_copy(deep=True)
    for item in moved.runtime.arguments:
        if item.name == "gpu-memory-utilization":
            item.value = value
    return moved


def test_move_that_raises_the_budget_raises_peaks_to_the_capped_envelope() -> None:
    # Real proven identity: util 0.80 declared, upstream moves it to 0.92.
    previous = documents.recipe(PROVEN)
    moved = with_utilization(previous, "0.92")
    raised = raise_peaks_to_envelope(moved, previous)
    assert raised is not None
    assert {r.resources.memory.peak_bytes for r in raised.topology.roles} == {
        declarable_vllm_envelope(moved)
    }
    assert all(
        r.resources.memory.peak_bytes <= ADMISSION_MAXIMUM_PEAK_BYTES
        for r in raised.topology.roles
    )


def test_move_that_does_not_raise_the_budget_keeps_the_measured_peaks() -> None:
    previous = documents.recipe(PROVEN)
    declare_peak(previous, 1_000_000_000)
    same = previous.model_copy(deep=True)
    assert raise_peaks_to_envelope(same, previous) is None
    lower = with_utilization(previous, "0.5")
    assert raise_peaks_to_envelope(lower, previous) is None
    # Without a pre-move recipe the envelope still applies.
    assert raise_peaks_to_envelope(same) is not None


def test_only_the_admissibility_check_reads_the_uncapped_envelope() -> None:
    allowed = {"declarable_vllm_envelope", "bounded_kv_warnings"}
    root = Path(__file__).resolve().parents[1]
    paths = [
        *(root / "qualification").rglob("*.py"),
        *(root / "tools").iterdir(),
    ]
    assert violations(paths, allowed) == []


def violations(paths: list[Path], allowed: set[str]) -> list[str]:
    found: list[str] = []
    for path in paths:
        try:
            tree = ast.parse(path.read_text())
        except (SyntaxError, UnicodeDecodeError, IsADirectoryError):
            continue
        for scope in ast.walk(tree):
            if not isinstance(scope, ast.FunctionDef) or scope.name in allowed:
                continue
            for node in ast.walk(scope):
                if (
                    isinstance(node, ast.Name)
                    and node.id == "vllm_utilization_envelope"
                    and scope.name != "vllm_utilization_envelope"
                ):
                    found.append(f"{path.name}:{node.lineno} in {scope.name}")
    return sorted(set(found))


def test_the_guard_catches_an_uncapped_caller(tmp_path: Path) -> None:
    bad = tmp_path / "bad.py"
    bad.write_text("def helper(r):\n    return vllm_utilization_envelope(r)\n")
    assert violations([bad], {"ok"}) == ["bad.py:2 in helper"]
