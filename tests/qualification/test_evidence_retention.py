"""Evidence retention: newest per class, a fixed point under re-import, refusals safe."""

from __future__ import annotations

import importlib.machinery
import importlib.util
import sys
import tempfile
from pathlib import Path
from typing import Any

from repo_root import ROOT
from test_qualification_evidence_publisher import seed_catalog
from test_qualification_status import KEY, A, evidence, record
from test_qualification_status import NOW as T0

from recipe_tooling.qualification.evidence_retention import retain_evidence, retained
from recipe_tooling.qualification.evidence_store import (
    catalog_recipes,
    merge_evidence,
    read_evidence,
    write_evidence,
)
from spark_sweep import evidence_publisher as publisher
from vonk_forge_contracts import (
    QualificationBlame as Blame,
)
from vonk_forge_contracts import (
    QualificationMeasuredRoleMemory as Row,
)
from vonk_forge_contracts import (
    QualificationOutcome as Outcome,
)
from vonk_forge_contracts import derive_status
from vonk_forge_contracts.qualification_status import published_status

LOADER = importlib.machinery.SourceFileLoader(
    "check_evidence_retention", str(ROOT / "tools/check-evidence-retention")
)
SPEC = importlib.util.spec_from_loader(LOADER.name, LOADER)
assert SPEC is not None
guard: Any = importlib.util.module_from_spec(SPEC)
sys.modules[LOADER.name] = guard
LOADER.exec_module(guard)

CATALOG = tempfile.TemporaryDirectory()
CATALOG_ROOT = Path(CATALOG.name)
seed_catalog(CATALOG_ROOT)
RECIPE, CURRENT = catalog_recipes(CATALOG_ROOT)[KEY]
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
    seed_catalog(root)


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
    rows = tuple(
        Row(content_sha256=CURRENT, role=role.name, steady_bytes=0, budget_bytes=100)
        for role in RECIPE.topology.roles
    )
    write_evidence(
        tmp_path, evidence(first, second).model_copy(update={"measured_memory": rows})
    )
    publisher._drop_refused(tmp_path, {publisher.record_id(KEY, second)})
    retain_evidence(tmp_path)
    kept = read_evidence(tmp_path, KEY)
    assert kept.records == (first,)
    assert kept.measured_memory == rows
    # A refused only record leaves rows without a record: retention drops them.
    publisher._drop_refused(tmp_path, {publisher.record_id(KEY, first)})
    assert read_evidence(tmp_path, KEY).records == ()
    write_evidence(tmp_path, evidence().model_copy(update={"measured_memory": rows}))
    retain_evidence(tmp_path)
    assert guard.problems(read_evidence(tmp_path, KEY)) == []


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


def test_the_reducers_read_retained_history_as_the_whole():
    full, kept = history(), retained(history(), RECIPE, CURRENT)
    for identity in (NOW, NOW.model_copy(update={"runtime_image": None})):
        identity = identity.model_copy(update={"required_sparks": 1})
        for fleet in (0, 1, 2):
            assert derive_status(full, identity, fleet, False) == derive_status(
                kept, identity, fleet, False
            )
        assert published_status(full, identity) == published_status(kept, identity)
