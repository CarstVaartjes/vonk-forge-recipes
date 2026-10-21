"""Estimates cannot remove untested catalog recipes from hardware qualification."""

import json
import runpy
from pathlib import Path

import pytest
from catalog_documents import catalog_models
from vonk_forge_contracts import (
    QualificationBlame,
    QualificationCauseCode,
    QualificationOutcome,
    QualificationSummary,
    RecipeDefinition,
    RecipeQualificationEvidence,
    document_sha256,
)

from qualification.memory_need import (
    calculate_need,
    hardware_proven,
    load_inputs,
    memory_variants,
    per_rank_need,
    require_memory_need,
)

ROOT = Path(__file__).resolve().parents[1]
MOSAIC = "glm-5-3-flash-exl3-mosaic-12l-0xsero-sglang-single"


def test_unchanged_untested_document_is_informational() -> None:
    document = json.loads((ROOT / "recipes/d3-edge-vllm-single.json").read_text())
    recipe = RecipeDefinition.model_validate(document)
    assert not hardware_proven(recipe, document_sha256(document))
    assert (
        require_memory_need(
            recipe,
            catalog_models(),
            load_inputs(),
            document_digest=document_sha256(document),
        )
        is None
    )
    # Fault injection: an old digest must never waive a modified launch.
    assert recipe.settings.kind == "decision"
    recipe.settings.context_tokens.value *= 100
    with pytest.raises(ValueError):
        require_memory_need(
            recipe,
            catalog_models(),
            load_inputs(),
            document_digest=document_sha256(document),
        )


def test_catalog_report_warns_without_a_nonfit_verdict(monkeypatch, capsys) -> None:
    main = runpy.run_path(str(ROOT / "tools/check-recipe-memory-need"))["main"]
    monkeypatch.setattr("sys.argv", ["check-recipe-memory-need", "d3-edge-vllm-single"])
    assert main() == 0
    output = capsys.readouterr().out
    assert "WARNING" in output
    assert "does not fit" not in output
    assert "unavailable" in output


def test_mosaic_failure_has_honest_demand_without_retuning_siblings() -> None:
    recipe = RecipeDefinition.model_validate_json(
        (ROOT / "recipes" / f"{MOSAIC}.json").read_text()
    )
    need = calculate_need(recipe, catalog_models(), load_inputs().geometries)
    assert need.total_bytes == 119_029_287_744 > need.limit_bytes == 108_470_788_096
    assert "does-not-fit" in recipe.metadata.tags
    assert all(
        role.resources.memory.peak_bytes >= need.total_bytes
        for role in recipe.topology.roles
    )
    require_memory_need(recipe, catalog_models(), load_inputs(), document_digest="")
    recipe.metadata.tags.remove("does-not-fit")
    with pytest.raises(ValueError, match="memory need"):
        require_memory_need(recipe, catalog_models(), load_inputs(), document_digest="")


def test_calibration_runs_every_hardware_pass_without_exemptions() -> None:
    checked = 0
    for path in (ROOT / "qualification/evidence").rglob("*.json"):
        evidence = RecipeQualificationEvidence.model_validate_json(path.read_text())
        if not any(r.outcome == QualificationOutcome.PASSED for r in evidence.records):
            continue
        recipe = RecipeDefinition.model_validate_json(
            (ROOT / "recipes" / f"{evidence.recipe_key.split('/')[1]}.json").read_text()
        )
        if recipe.runtime.engine not in {"vllm", "sglang"}:
            continue
        for variant in memory_variants(recipe):
            assert per_rank_need(variant, catalog_models()).fits, evidence.recipe_key
        checked += 1
    assert checked > 100


def test_qualification_evidence_is_the_only_pass_source(tmp_path, monkeypatch) -> None:
    import qualification.memory_need as memory

    slug = "gpt-oss-20b-nvidia-vllm-single"
    document = json.loads((ROOT / "recipes" / f"{slug}.json").read_text())
    recipe = RecipeDefinition.model_validate(document)
    path = tmp_path / "qualification/evidence/vonk-forge" / f"{slug}.json"
    path.parent.mkdir(parents=True)
    evidence = RecipeQualificationEvidence.model_validate_json(
        (ROOT / "qualification/evidence/vonk-forge" / f"{slug}.json").read_text()
    )
    passed = evidence.records[-1]
    assert passed.outcome == QualificationOutcome.PASSED
    failed = passed.model_copy(
        update={
            "outcome": QualificationOutcome.FAILED,
            "blame": QualificationBlame.RECIPE,
            "cause_code": QualificationCauseCode.RECIPE_FAILURE,
            "cause_summary": QualificationSummary.RECIPE_FAILED,
            "tested_at": passed.tested_at.replace(year=passed.tested_at.year + 1),
        }
    )
    monkeypatch.setattr(memory, "ROOT", tmp_path)
    assert not hardware_proven(recipe, document_sha256(document))
    path.write_text(evidence.model_dump_json())
    assert hardware_proven(recipe, document_sha256(document))
    path.write_text(
        evidence.model_copy(
            update={"records": (*evidence.records, failed)}
        ).model_dump_json()
    )
    assert not hardware_proven(recipe, document_sha256(document))
    path.unlink()
    assert not hardware_proven(recipe, document_sha256(document))
