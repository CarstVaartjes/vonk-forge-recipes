"""The one-time Model facts split refuses what it cannot prove (fed bad input).

These tests delete with ``tools/split-model-facts``.
"""

from __future__ import annotations

import json
import runpy
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import vonk_forge_contracts as contracts
from vonk_forge_contracts import (
    ModelDefinition,
    ModelDisplay,
    QualificationEvidenceRecord,
    QualificationIdentity,
    QualificationOutcome,
    QualificationPlatform,
    QualificationRuntimeImage,
    RecipeQualificationEvidence,
)
from vonk_forge_contracts.qualification_status import (
    QUALIFICATION_SUMMARIES,
    QualificationCauseCode,
    QualificationQualityNote,
    QualificationTopology,
)

ROOT = Path(__file__).resolve().parents[1]
TOOL = runpy.run_path(str(ROOT / "tools/split-model-facts"))
EXAMPLES = ROOT / "contracts/src/vonk_forge_contracts/examples"
MOVED = ("modalities", "capabilities", "license")
OLD, NEW, MODEL_OLD, MODEL_NEW = "a" * 64, "b" * 64, "c" * 64, "d" * 64
KEY = "vonk-forge/demo"


def _json(name: str) -> dict[str, Any]:
    return json.loads((EXAMPLES / name).read_text(encoding="utf-8"))


def _proof_inputs(tmp_path: Path) -> dict[str, Any]:
    """One Model as the base revision held it, and as the split writes it."""

    display = _json("model-display.json")
    facts = {name: display.pop(name) for name in MOVED}
    document = _json("model-definition.json")
    old = {**document, **facts}
    path = "models/demo.json"
    (tmp_path / "models").mkdir()
    (tmp_path / path).write_text(json.dumps(old), encoding="utf-8")
    return {
        "new_model": {path: ModelDefinition.model_validate(document)},
        "new_recipe": {},
        "new_display": {path: ModelDisplay.model_validate({**display, **facts})},
        "items": {path: {"kind": "model", "display": display, "facts": facts}},
        "tree": tmp_path,
        "mapping": {},
        "path": path,
    }


def _prove(inputs: dict[str, Any]) -> None:
    TOOL["_prove_only_moved"](
        **{k: v for k, v in inputs.items() if k != "path"},
    )


def test_a_split_that_moved_only_the_facts_is_proved(tmp_path: Path) -> None:
    _prove(_proof_inputs(tmp_path))


def test_more_than_the_moved_fields_changing_is_refused(tmp_path: Path) -> None:
    inputs = _proof_inputs(tmp_path)
    path = tmp_path / inputs["path"]
    old = json.loads(path.read_text(encoding="utf-8"))
    path.write_text(
        json.dumps({**old, "requires_token": not old["requires_token"]}),
        encoding="utf-8",
    )
    with pytest.raises(SystemExit, match="more than the moved fields changed"):
        _prove(inputs)


def test_a_fact_that_was_not_carried_is_refused(tmp_path: Path) -> None:
    inputs = _proof_inputs(tmp_path)
    item = inputs["items"][inputs["path"]]
    item["facts"] = {**item["facts"], "capabilities": ["something-else"]}
    with pytest.raises(SystemExit, match="not carried unchanged"):
        _prove(inputs)


def test_a_display_edited_during_the_split_is_refused(tmp_path: Path) -> None:
    inputs = _proof_inputs(tmp_path)
    item = inputs["items"][inputs["path"]]
    item["display"] = {**item["display"], "description": "A different text."}
    with pytest.raises(SystemExit, match="display changed beyond the moved fields"):
        _prove(inputs)


def _record(
    digest: str, models: str, *, notes: tuple[QualificationQualityNote, ...] = ()
) -> QualificationEvidenceRecord:
    return QualificationEvidenceRecord(
        identity=QualificationIdentity(
            content_sha256=digest,
            source_bundle_sha256="e" * 64,
            runtime_image=QualificationRuntimeImage(
                build_key="f" * 64, image_digest="sha256:" + "1" * 64
            ),
            model_digests=(models,),
            required_sparks=1,
        ),
        outcome=QualificationOutcome.PASSED,
        cause_code=QualificationCauseCode.HARDWARE_PASS,
        cause_summary=QUALIFICATION_SUMMARIES[QualificationCauseCode.HARDWARE_PASS],
        tested_at=datetime(2026, 10, 28, tzinfo=UTC),
        platform=QualificationPlatform(
            controller_source_sha="2" * 40,
            agent_version="1.0.0",
            spark_count=1,
            topology=QualificationTopology.SINGLE,
        ),
        quality_notes=notes,
    )


def _gate_inputs(tmp_path: Path, *, twinned: bool, evidence_on_disk: bool) -> dict:
    records = (_record(OLD, MODEL_OLD),)
    carried = (
        _record(
            NEW,
            MODEL_NEW,
            notes=(QualificationQualityNote.CARRIED_MODEL_FACTS,),
        ),
    )
    evidence = RecipeQualificationEvidence(
        recipe_key=KEY, records=records + (carried if twinned else ())
    )
    root = tmp_path
    (root / "tests/fixtures").mkdir(parents=True)
    (root / "tests/fixtures/proven-recipe-bytes.json").write_text(
        json.dumps({KEY: "0" * 64}), encoding="utf-8"
    )
    if evidence_on_disk:
        path = root / "qualification/evidence" / f"{KEY}.json"
        path.parent.mkdir(parents=True)
        path.write_text(
            RecipeQualificationEvidence(
                recipe_key=KEY, records=records
            ).model_dump_json(indent=2, exclude_none=True),
            encoding="utf-8",
        )
    recipe = SimpleNamespace(
        identity=SimpleNamespace(publisher="vonk-forge", slug="demo"),
        models=[SimpleNamespace(model=SimpleNamespace(content_sha256=MODEL_NEW))],
        topology=SimpleNamespace(node_count=1),
    )
    path = "recipes/demo.json"
    before = {
        "status": "Working",
        "changed": [],
        "reason": None,
        "required_sparks": 1,
        "last_pass": "2026-10-28T00:00:00+00:00",
        "last_failure": None,
    }
    return {
        "contracts": contracts,
        "tree": root,
        "items": {
            path: {
                "old_sha256": OLD,
                "before": [
                    {
                        "bundle": "e" * 64,
                        "image": {
                            "build_key": "f" * 64,
                            "image_digest": "sha256:" + "1" * 64,
                        },
                        "view": before,
                    }
                ],
            }
        },
        "new_recipe": {path: recipe},
        "old_digests": {KEY: OLD},
        "mapping": {OLD: NEW},
        "twinned": {KEY: evidence},
    }


def test_the_gate_passes_when_the_twin_carries_the_proof(tmp_path: Path) -> None:
    inputs = _gate_inputs(tmp_path, twinned=True, evidence_on_disk=True)
    counts = TOOL["_status_gate"](**inputs)
    assert counts["proven_working"] == 1


def test_a_proven_recipe_without_a_resolved_pass_is_refused(tmp_path: Path) -> None:
    inputs = _gate_inputs(tmp_path, twinned=True, evidence_on_disk=False)
    with pytest.raises(SystemExit, match="lack a resolved pass"):
        TOOL["_status_gate"](**inputs)


def test_a_status_that_moves_with_the_split_is_refused(tmp_path: Path) -> None:
    inputs = _gate_inputs(tmp_path, twinned=False, evidence_on_disk=True)
    with pytest.raises(SystemExit, match="published status moved"):
        TOOL["_status_gate"](**inputs)
