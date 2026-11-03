"""Estimates cannot remove untested catalog recipes from hardware qualification."""

import json
import runpy
import subprocess
from pathlib import Path

import pytest
from catalog_documents import catalog_models
from documents import recipe as recipe_of
from documents import recipe_display
from vonk_forge_contracts import (
    QualificationBlame,
    QualificationCauseCode,
    QualificationOutcome,
    QualificationSummary,
    RecipeDefinition,
    RecipeQualificationEvidence,
    document_sha256,
)
from warning_asserts import assert_not_warns, assert_warns, declare_peak

from qualification.memory_need import (
    SPARK_MEMORY_BYTES,
    calculate_need,
    envelope_exceptions,
    hardware_proven,
    load_inputs,
    memory_estimate_warnings,
)

ROOT = Path(__file__).resolve().parents[1]
# Read before a test points the module root at a temporary directory.
envelope_exceptions()
MOSAIC = "glm-5-3-flash-exl3-mosaic-12l-0xsero-sglang-single"


def untested_root(tmp_path: Path, monkeypatch, slug: str) -> Path:
    """Authored recipe bytes with an empty evidence corpus, whatever the live one holds."""
    import qualification.memory_need as memory

    (tmp_path / "recipes").mkdir()
    (tmp_path / "recipes" / f"{slug}.json").write_text(
        (ROOT / "recipes" / f"{slug}.json").read_text()
    )
    monkeypatch.setattr(memory, "ROOT", tmp_path)
    return tmp_path


def test_unchanged_untested_document_is_informational(tmp_path, monkeypatch) -> None:
    untested_root(tmp_path, monkeypatch, "d3-edge-vllm-single")
    recipe = recipe_of("d3-edge-vllm-single")
    declare_peak(recipe, 60_000_000_000)
    document = json.loads(recipe.model_dump_json())
    digest = document_sha256(document)
    (tmp_path / "recipes/d3-edge-vllm-single.json").write_text(json.dumps(document))
    key = f"{recipe.identity.publisher}/{recipe.identity.slug}"
    inputs = load_inputs().model_copy(update={"historical_documents": {key: [digest]}})
    assert not hardware_proven(recipe, digest)
    assert_not_warns(
        memory_estimate_warnings(
            recipe,
            catalog_models(),
            inputs,
            document_digest=digest,
        ),
        "not admissible",
    )
    # Fault injection: an old digest must never waive a modified launch.
    for role in recipe.topology.roles:
        role.resources.memory.peak_bytes = SPARK_MEMORY_BYTES
    assert_warns(
        memory_estimate_warnings(
            recipe,
            catalog_models(),
            inputs,
            document_digest=digest,
        ),
        "not admissible",
    )


@pytest.mark.parametrize("use_snapshot", [True, False])
def test_existing_content_needs_no_git(tmp_path, monkeypatch, use_snapshot) -> None:
    import qualification.memory_need as memory

    slug = "d3-edge-vllm-single"
    source = ROOT / "recipes" / f"{slug}.json"
    recipe = recipe_of(slug)
    declare_peak(recipe, 60_000_000_000)
    document = json.loads(recipe.model_dump_json())
    digest = document_sha256(document)
    key = f"{recipe.identity.publisher}/{slug}"
    authored = tmp_path / "recipes" / source.name
    authored.parent.mkdir()
    authored.write_text(json.dumps(document))
    inputs = load_inputs().model_copy(
        update={"historical_documents": {key: [digest]} if use_snapshot else {}}
    )
    if not use_snapshot:
        # Synthetic platform failure for exactly these bytes, never the live corpus.
        template = RecipeQualificationEvidence.model_validate_json(
            (
                ROOT
                / "qualification/evidence/vonk-forge"
                / "gpt-oss-20b-nvidia-vllm-single.json"
            ).read_text()
        )
        failed = template.records[-1].model_copy(
            update={
                "outcome": QualificationOutcome.FAILED,
                "blame": QualificationBlame.PLATFORM,
                "cause_code": QualificationCauseCode.PLATFORM_FAILURE,
                "cause_summary": QualificationSummary.PLATFORM_FAILED,
                "identity": template.records[-1].identity.model_copy(
                    update={"content_sha256": digest}
                ),
            }
        )
        evidence = template.model_copy(update={"recipe_key": key, "records": (failed,)})
        target = tmp_path / "qualification/evidence" / f"{key}.json"
        target.parent.mkdir(parents=True)
        target.write_text(evidence.model_dump_json())
    monkeypatch.setattr(memory, "ROOT", tmp_path)

    def forbidden(*args, **kwargs):
        pytest.fail("memory validation must not invoke Git or other subprocesses")

    monkeypatch.setattr(subprocess, "run", forbidden)
    assert_not_warns(
        memory_estimate_warnings(recipe, {}, inputs, document_digest=digest),
        "peak understates|not admissible|full-context request|cannot hold the selected weights|set absolute|mem-fraction-static|state slots",
    )
    assert not hardware_proven(recipe, digest)

    # A known digest cannot hide edited bytes, even with the old parsed model.
    changed = recipe.model_copy(deep=True)
    for role in changed.topology.roles:
        role.resources.memory.peak_bytes = SPARK_MEMORY_BYTES
    authored.write_text(changed.model_dump_json())
    assert not memory.unchanged_recipe(recipe, digest, inputs)
    assert_warns(
        memory_estimate_warnings(changed, {}, inputs, document_digest=digest),
        "",
    )
    assert_warns(
        memory_estimate_warnings(
            changed,
            {},
            inputs,
            document_digest=document_sha256(json.loads(authored.read_text())),
        ),
        "",
    )

    # Restoring exact content restores the informational treatment immediately.
    authored.write_text(json.dumps(document))
    assert_not_warns(
        memory_estimate_warnings(recipe, {}, inputs, document_digest=digest),
        "peak understates|not admissible|full-context request|cannot hold the selected weights|set absolute|mem-fraction-static|state slots",
    )


def test_new_recipe_without_git_still_requires_memory_inputs(tmp_path, monkeypatch):
    import qualification.memory_need as memory

    document = json.loads(
        (
            ROOT / "recipes/llama-3-1-8b-instruct-nvfp4-nvidia-vllm-single.json"
        ).read_text()
    )
    document["identity"]["slug"] = "new-memory-need-recipe"
    recipe = RecipeDefinition.model_validate(document)
    inputs = load_inputs().model_copy(update={"geometries": {}})
    monkeypatch.setattr(memory, "ROOT", tmp_path)

    def forbidden(*args, **kwargs):
        pytest.fail("new-document validation must not invoke Git")

    monkeypatch.setattr(subprocess, "run", forbidden)
    # Its absolute pool must be shown to hold one request by pinned geometry.
    assert_warns(
        memory_estimate_warnings(
            recipe, catalog_models(), inputs, document_digest=document_sha256(document)
        ),
        "source-backed cache geometry",
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
    declare_peak(recipe)
    need = calculate_need(recipe, catalog_models(), load_inputs())
    assert need.total_bytes == 119_501_146_944 > need.limit_bytes == 108_470_788_096
    assert "does-not-fit" in recipe_display(recipe.identity.slug).tags
    assert all(
        role.resources.memory.peak_bytes >= need.total_bytes
        for role in recipe.topology.roles
    )
    # The estimate is informative: the tag no longer gates the document.
    assert_not_warns(
        memory_estimate_warnings(
            recipe, catalog_models(), load_inputs(), document_digest=""
        ),
        "peak understates|not admissible|full-context request|cannot hold the selected weights|set absolute|mem-fraction-static|state slots",
    )


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
    # A later failure of every proof of this document (resolved or not).
    failures = tuple(
        record.model_copy(
            update={
                "outcome": QualificationOutcome.FAILED,
                "blame": QualificationBlame.RECIPE,
                "cause_code": QualificationCauseCode.RECIPE_FAILURE,
                "cause_summary": QualificationSummary.RECIPE_FAILED,
                "tested_at": record.tested_at.replace(year=record.tested_at.year + 1),
            }
        )
        for record in evidence.records
        if record.identity.content_sha256 == document_sha256(document)
        and record.outcome == QualificationOutcome.PASSED
    )
    monkeypatch.setattr(memory, "ROOT", tmp_path)
    assert not hardware_proven(recipe, document_sha256(document))
    path.write_text(evidence.model_dump_json())
    assert hardware_proven(recipe, document_sha256(document))
    path.write_text(
        evidence.model_copy(
            update={"records": (*evidence.records, *failures)}
        ).model_dump_json()
    )
    assert not hardware_proven(recipe, document_sha256(document))
    path.unlink()
    assert not hardware_proven(recipe, document_sha256(document))


def _reads_live_evidence_for_a_catalog_recipe(function) -> bool:
    import ast

    calls = {
        node.func.id
        if isinstance(node.func, ast.Name)
        else getattr(node.func, "attr", "")
        for node in ast.walk(function)
        if isinstance(node, ast.Call)
    }
    names_catalog_recipe = any(
        isinstance(node, ast.Constant)
        and isinstance(node.value, str)
        and node.value.startswith("recipes/")
        for node in ast.walk(function)
    )
    return (
        "memory_warnings" in calls
        and names_catalog_recipe
        and "untested_root" not in calls
    )


def test_unproven_warnings_never_read_the_live_evidence_corpus() -> None:
    """A catalog recipe's unproven warnings flip as soon as the sweep proves it."""
    import ast

    offenders = [
        f"{path.name}:{node.name}"
        for path in sorted((ROOT / "tests").glob("test_*.py"))
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8")))
        if isinstance(node, ast.FunctionDef)
        and node.name.startswith("test_")
        and _reads_live_evidence_for_a_catalog_recipe(node)
    ]
    assert not offenders, f"isolate the evidence with untested_root(): {offenders}"


def test_live_evidence_guard_catches_an_unisolated_read() -> None:
    import ast

    source = (
        "def test_x():\n"
        "    document = load('recipes/a.json')\n"
        "    memory_warnings(document)\n"
    )
    assert _reads_live_evidence_for_a_catalog_recipe(ast.parse(source).body[0])
    isolated = source.replace(
        "    document", "    untested_root(t, m, 'a')\n    document"
    )
    assert not _reads_live_evidence_for_a_catalog_recipe(ast.parse(isolated).body[0])
