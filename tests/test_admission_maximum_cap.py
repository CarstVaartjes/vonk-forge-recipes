"""The admission maximum caps the vLLM envelope; measured evidence is never raised."""

from types import SimpleNamespace

import pytest
from catalog_documents import catalog_models
from vonk_forge_contracts import QualificationOutcome
from warning_asserts import assert_not_warns, declare_peak

from qualification import documents, memory_need
from qualification.memory_need import (
    ADMISSION_MAXIMUM_PEAK_BYTES,
    PLATFORM_MEMORY_FLOOR_BYTES,
    SPARK_IDLE_AVAILABLE_BYTES,
    declarable_vllm_envelope,
    load_inputs,
    memory_estimate_warnings,
    raise_peaks_to_envelope,
    vllm_utilization_envelope,
)

INPUTS = load_inputs()
GLM = "glm-5-3-flash-exl3-k2-0xsero-vllm-single"
LLAMA = "llama-3-1-405b-instruct-awq-int4-nvidia-vllm-four"


def test_maximum_derives_from_the_admission_rule() -> None:
    assert (
        ADMISSION_MAXIMUM_PEAK_BYTES
        == SPARK_IDLE_AVAILABLE_BYTES - PLATFORM_MEMORY_FLOOR_BYTES
        == 124_452_516_352
    )


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


def evidence_with(outcome: QualificationOutcome | None) -> object:
    records = () if outcome is None else (SimpleNamespace(outcome=outcome),)
    return lambda _key: SimpleNamespace(records=records)


def test_recipe_with_a_passed_record_is_not_raised(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recipe = documents.recipe(GLM)
    declare_peak(recipe, 1_000_000_000)
    monkeypatch.setattr(memory_need, "_evidence", evidence_with(None))
    assert raise_peaks_to_envelope(recipe) is not None
    monkeypatch.setattr(
        memory_need, "_evidence", evidence_with(QualificationOutcome.FAILED)
    )
    assert raise_peaks_to_envelope(recipe) is not None
    monkeypatch.setattr(
        memory_need, "_evidence", evidence_with(QualificationOutcome.PASSED)
    )
    assert raise_peaks_to_envelope(recipe) is None
