"""Evidence retention: newest per class, a fixed point under re-import, refusals safe."""

from __future__ import annotations

import importlib.machinery
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

from test_qualification_status import KEY, A, evidence, record
from test_qualification_status import NOW as T0
from vonk_forge_contracts import (
    QualificationBlame as Blame,
)
from vonk_forge_contracts import (
    QualificationMeasuredRoleMemory as Row,
)
from vonk_forge_contracts import (
    QualificationOutcome as Outcome,
)
from vonk_forge_contracts import (
    RecipeDefinition,
    document_sha256,
)

from spark_sweep import evidence_publisher as publisher
from spark_sweep.qualification import (
    merge_evidence,
    read_evidence,
    retain_evidence,
    retained,
    write_evidence,
)

ROOT = Path(__file__).resolve().parents[1]
LOADER = importlib.machinery.SourceFileLoader(
    "check_evidence_retention", str(ROOT / "tools/check-evidence-retention")
)
SPEC = importlib.util.spec_from_loader(LOADER.name, LOADER)
assert SPEC is not None
guard: Any = importlib.util.module_from_spec(SPEC)
sys.modules[LOADER.name] = guard
LOADER.exec_module(guard)

DOCUMENT = json.loads((ROOT / "recipes/qwen3-5-9b-vllm-single.json").read_text())
DOCUMENT["identity"]["slug"] = "example"
RECIPE = RecipeDefinition.model_validate(DOCUMENT)
CURRENT = document_sha256(DOCUMENT)
NOW = A.model_copy(update={"content_sha256": CURRENT})
OLD = A.model_copy(update={"content_sha256": "9" * 64})
FAIL = {"outcome": Outcome.FAILED, "blame": Blame.RECIPE}
FAULT = {"outcome": Outcome.FAILED, "blame": Blame.PLATFORM}


def day(r):
    return (r.tested_at - T0).days


def history():
    return evidence(
        record(NOW, day=1),
        record(NOW, day=2),  # newer pass of the same identity: the older goes
        record(NOW, day=3, **FAIL),
        record(NOW, day=4, **FAULT),
        record(NOW, day=5, **FAULT),  # newer fault: the older goes
        record(OLD, day=-5),
        record(OLD, day=-4, **FAULT),
    )


def test_each_class_is_kept_and_nothing_else():
    kept = retained(history(), RECIPE, CURRENT)
    assert [(day(r), str(r.outcome), r.blame) for r in kept.records] == [
        (2, "passed", None),  # newest pass
        (3, "failed", Blame.RECIPE),  # newest behavioural and recipe-blamed
        (5, "failed", Blame.PLATFORM),  # newest failure
    ]


def test_the_files_newest_verdicts_stay_when_the_current_content_has_none():
    kept = retained(
        evidence(record(OLD, day=-5), record(OLD, day=-4, **FAIL)), RECIPE, CURRENT
    )
    assert [str(r.outcome) for r in kept.records] == ["passed", "failed"]


def test_the_budget_source_content_and_its_rows_are_kept():
    rows = tuple(
        Row(content_sha256=content, role=role.name, steady_bytes=0, budget_bytes=100)
        for content in ("9" * 64, "8" * 64)
        for role in RECIPE.topology.roles
    )
    old = history().model_copy(update={"measured_memory": rows})
    kept = retained(old, RECIPE, CURRENT)
    assert {m.content_sha256 for m in kept.measured_memory} == {"9" * 64}
    assert (-5, "passed") in [(day(r), str(r.outcome)) for r in kept.records]


def test_reimporting_history_is_a_fixed_point():
    full = history()
    once = retained(full, RECIPE, CURRENT)
    assert retained(once, RECIPE, CURRENT) == once
    for imported in (full, *(evidence(r) for r in full.records)):
        assert retained(merge_evidence(once, imported), RECIPE, CURRENT) == once


def seed(root: Path) -> None:
    (root / "recipes").mkdir()
    (root / "recipes/example.json").write_text(json.dumps(DOCUMENT))


def test_withdrawn_recipe_keeps_nothing_and_others_shrink(tmp_path):
    seed(tmp_path)
    write_evidence(tmp_path, history())
    write_evidence(
        tmp_path,
        evidence(record(OLD)).model_copy(update={"recipe_key": "vonk-forge/gone"}),
    )
    retain_evidence(tmp_path)
    assert not (tmp_path / "qualification/evidence/vonk-forge/gone.json").exists()
    assert len(read_evidence(tmp_path, KEY).records) == 3
    assert retain_evidence(tmp_path) == []


def test_a_refused_newer_pass_leaves_the_older_one(tmp_path):
    seed(tmp_path)
    first, second = record(NOW, day=1), record(NOW, day=2)
    write_evidence(tmp_path, evidence(first, second))
    publisher._drop_refused(tmp_path, {publisher.record_id(KEY, second)})
    retain_evidence(tmp_path)
    assert read_evidence(tmp_path, KEY).records == (first,)


def test_guard_accepts_retained_and_refuses_a_second_record_of_one_identity():
    assert guard.problems(retained(history(), RECIPE, CURRENT)) == []
    assert any(e.startswith("R1") for e in guard.problems(history()))
    many = evidence(
        *(
            record(A.model_copy(update={"content_sha256": f"{n}" * 64}), day=n)
            for n in range(6)
        )
    )
    assert any(e.startswith("R2") for e in guard.problems(many))
    stray = evidence(record(NOW)).model_copy(
        update={
            "measured_memory": (Row(content_sha256="7" * 64, role="r", steady_bytes=0),)
        }
    )
    assert any(e.startswith("R3") for e in guard.problems(stray))
