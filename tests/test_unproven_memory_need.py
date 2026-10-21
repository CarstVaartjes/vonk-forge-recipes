"""Estimates cannot remove untested catalog recipes from hardware qualification."""

import json
import runpy
import subprocess
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


@pytest.mark.parametrize("use_snapshot", [True, False])
def test_existing_content_needs_no_git(tmp_path, monkeypatch, use_snapshot) -> None:
    import qualification.memory_need as memory

    slug = "d3-edge-vllm-single"
    source = ROOT / "recipes" / f"{slug}.json"
    document = json.loads(source.read_text())
    recipe = RecipeDefinition.model_validate(document)
    digest = document_sha256(document)
    key = f"{recipe.identity.publisher}/{slug}"
    authored = tmp_path / "recipes" / source.name
    authored.parent.mkdir()
    authored.write_text(source.read_text())
    inputs = load_inputs().model_copy(
        update={"historical_documents": {key: [digest]} if use_snapshot else {}}
    )
    if not use_snapshot:
        evidence = RecipeQualificationEvidence.model_validate_json(
            (ROOT / "qualification/evidence" / f"{key}.json").read_text()
        )
        assert evidence.records[0].outcome == QualificationOutcome.FAILED
        assert evidence.records[0].blame == QualificationBlame.PLATFORM
        target = tmp_path / "qualification/evidence" / f"{key}.json"
        target.parent.mkdir(parents=True)
        target.write_text(evidence.model_dump_json())
    monkeypatch.setattr(memory, "ROOT", tmp_path)

    def forbidden(*args, **kwargs):
        pytest.fail("memory validation must not invoke Git or other subprocesses")

    monkeypatch.setattr(subprocess, "run", forbidden)
    assert require_memory_need(recipe, {}, inputs, document_digest=digest) is None
    assert not hardware_proven(recipe, digest)

    # A known digest cannot hide edited bytes, even with the old parsed model.
    assert recipe.settings.kind == "decision"
    changed = recipe.model_copy(deep=True)
    assert changed.settings.kind == "decision"
    changed.settings.context_tokens.value *= 100
    authored.write_text(changed.model_dump_json())
    with pytest.raises(ValueError):
        require_memory_need(recipe, {}, inputs, document_digest=digest)
    with pytest.raises(ValueError):
        require_memory_need(changed, {}, inputs, document_digest=digest)
    with pytest.raises(ValueError):
        require_memory_need(
            changed,
            {},
            inputs,
            document_digest=document_sha256(json.loads(authored.read_text())),
        )

    # Restoring exact content restores the informational treatment immediately.
    authored.write_text(source.read_text())
    assert require_memory_need(recipe, {}, inputs, document_digest=digest) is None


def test_new_recipe_without_git_still_requires_memory_inputs(tmp_path, monkeypatch):
    import qualification.memory_need as memory

    document = json.loads((ROOT / "recipes/d3-edge-vllm-single.json").read_text())
    document["identity"]["slug"] = "new-memory-need-recipe"
    recipe = RecipeDefinition.model_validate(document)
    monkeypatch.setattr(memory, "ROOT", tmp_path)

    def forbidden(*args, **kwargs):
        pytest.fail("new-document validation must not invoke Git")

    monkeypatch.setattr(subprocess, "run", forbidden)
    with pytest.raises(ValueError, match="declare engine context and concurrency"):
        require_memory_need(
            recipe, {}, load_inputs(), document_digest=document_sha256(document)
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
