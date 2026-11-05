"""One blame rule, proved from the producers: every class a producer can emit is declared,
and the row the sweep writes for it reads back with the same blame in the published status.
"""

from __future__ import annotations

import ast
import contextlib
from pathlib import Path

import pytest
from sweep_fakes import FakeModel, FakeRecipe, make_sweep
from test_qualification_status import Blame as PublishedBlame
from test_qualification_status import Outcome, historical
from vonk_forge_contracts.sweep import SweepFailureClass, SweepWaitCode

from spark_sweep import policy, verdict
from spark_sweep.auth_contract import SmokeAuthClass
from spark_sweep.qualification import from_history
from spark_sweep.verdict import Blame, Disposition

SOURCES = ("run.py", "smoke.py", "smoke_job.py", "prefetch.py")
CALLS = {"classify", "classify_application_failure"}
UNNAMED = "a-class-nobody-named"


def _values(expr: ast.expr, scope: dict, name: str) -> set[str]:
    """The string classes an expression can evaluate to (both arms of a conditional)."""
    if isinstance(expr, ast.IfExp):
        return _values(expr.body, scope, name) | _values(expr.orelse, scope, name)
    value = None
    # A name that cannot be resolved here (a variable, an attribute of a runtime value) is
    # skipped; the scan only counts classes it can read from the source.
    with contextlib.suppress(Exception):
        value = eval(compile(ast.Expression(expr), name, "eval"), scope)
    return {str(value)} if isinstance(value, str) and value else set()


def _call_site_classes(texts: dict[str, str] | None = None) -> set[str]:
    """Every class a classify call or a local ``klass`` in the producers can name (AST scan).

    A call without ``klass`` takes the class ``policy.classify`` gives its literal phase.
    """
    root = Path(policy.__file__).parent
    texts = texts or {name: (root / name).read_text() for name in SOURCES}
    scope = {**vars(policy), **vars(verdict), "SmokeAuthClass": SmokeAuthClass}
    found: set[str] = set()
    for name, text in texts.items():
        for node in ast.walk(ast.parse(text)):
            if isinstance(node, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id == "klass" for t in node.targets
            ):
                found |= _values(node.value, scope, name)
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            called = getattr(func, "attr", None) or getattr(func, "id", None)
            if called not in CALLS:
                continue
            args = [kw.value for kw in node.keywords if kw.arg == "klass"]
            if not args and called == "classify" and len(node.args) >= 4:
                args = [node.args[3]]
            for arg in args:
                found |= _values(arg, scope, name)
            if not args and node.args:
                for phase in _values(node.args[0], scope, name):
                    found.add(policy.classify(phase, "", "").klass)
    return found


def _producer_pairs() -> set[tuple[str, str]]:
    """Every (class, code) a producer can emit."""
    pairs = {(klass, code) for code, klass in policy._CODE_CLASSES.items()}
    classes = (
        set(policy._FIXED_CLASS.values())
        | set(policy._CHILD_PHASE.values())
        | {
            str(member)
            for enum in (SmokeAuthClass, SweepFailureClass)
            for member in enum
        }
        | {str(member) for member in SweepWaitCode}
        | _call_site_classes()
    )
    return pairs | {(klass, "") for klass in classes}


def _unnamed() -> list[tuple[str, str]]:
    return sorted(
        (klass, code)
        for klass, code in _producer_pairs()
        if klass not in verdict.DECLARED_CLASSES
        and code not in policy.RECIPE_FAILURE_CODES
    )


def test_the_producer_scan_finds_the_known_call_sites():
    found = _call_site_classes()
    assert {"smoke-assertion", "recipe-data", "smoke-timeout"} <= found


def test_every_class_a_producer_emits_is_declared():
    assert _unnamed() == []


def test_the_guard_fails_for_a_new_classify_call_site():
    source = {"new.py": "policy.classify('warmup', '', detail)"}
    assert "warmup" in _call_site_classes(source) - verdict.DECLARED_CLASSES


def test_the_scan_reads_the_local_klass_variable_and_the_phase_fallback():
    found = _call_site_classes()
    assert {"smoke-request", "download", "readiness"} <= found


def test_the_guard_fails_for_a_new_phase_nobody_declared(monkeypatch):
    monkeypatch.setitem(policy._CHILD_PHASE, "warm-up", "warmup")
    assert ("warmup", "") in _unnamed()


@pytest.mark.parametrize(
    ("klass", "code", "expected"),
    [
        *[(k, "", Blame.RECIPE) for k in sorted(verdict.FAIL_CLASSES)],
        *[
            (k, "", Blame.PLATFORM)
            for k in sorted(
                verdict.WAIT_CLASSES
                | verdict.OBSERVE_CLASSES
                | verdict.RECORDED_CLASSES
                | verdict.SKIP_CLASSES
            )
        ],
        (UNNAMED, min(policy.RECIPE_FAILURE_CODES), Blame.RECIPE),
        (verdict.UNATTRIBUTED, min(policy.RECIPE_FAILURE_CODES), Blame.PLATFORM),
        ("review", min(policy.REVIEW_FINAL_CODES), Blame.PLATFORM),
    ],
)
def test_each_class_has_its_blame(klass, code, expected):
    assert verdict.blame(klass, code) is expected


def test_an_unnamed_class_is_observed_never_failed():
    failure = policy.classify("start", "", "x", klass=UNNAMED)
    assert verdict.disposition(failure) is Disposition.OBSERVE


def test_the_controllers_final_word_is_recorded_but_is_the_platforms():
    failure = policy.classify("review", min(policy.REVIEW_FINAL_CODES), "x")
    assert verdict.blame(failure.klass, failure.code) is Blame.PLATFORM
    assert verdict.disposition(failure) is Disposition.FAIL


# -- the boundary: the row _record_failure writes, read back by the published status ------


def _record(tmp_path: Path, failure: policy.Failure):
    recipe = FakeRecipe("a", ("1" * 64,), content="a" * 64)
    sweep, _, _ = make_sweep(tmp_path, [recipe], [FakeModel("1" * 64)])
    sweep.refresh_catalog()
    key = next(iter(sweep.recipes))
    sweep._record_failure(key, failure, None)
    return sweep, key


def _verdict_rows(sweep, key):
    return [
        r
        for r in sweep.log.entries()
        if r.get("recipe") == key and r.get("status") in ("failed", "deferred")
    ]


@pytest.mark.parametrize(
    ("klass", "code"), sorted(_producer_pairs() - {("", "")}), ids=str
)
def test_the_sweep_status_and_the_published_blame_agree(tmp_path, klass, code):
    failure = policy.classify("smoke", code, "detail", klass=klass)
    sweep, key = _record(tmp_path, failure)
    rows = _verdict_rows(sweep, key)
    if verdict.disposition(failure) is not Disposition.FAIL:
        assert rows == []  # a wait, a retry or a skip is not a verdict row
        return
    (row,) = rows
    imported = from_history({**historical(), **row, "recipe": historical()["recipe"]})
    assert imported is not None
    recipe_failed = row["status"] == "failed"
    assert recipe_failed == (verdict.blame(failure.klass, failure.code) is Blame.RECIPE)
    assert imported.blame == (
        PublishedBlame.RECIPE if recipe_failed else PublishedBlame.PLATFORM
    )
    assert imported.outcome == Outcome.FAILED


def test_an_unnamed_class_on_one_identity_ends_after_three_identical_outcomes(tmp_path):
    from spark_sweep.verdict import IDENTICAL_EXIT_LIMIT

    recipe = FakeRecipe("a", ("1" * 64,), content="a" * 64)
    sweep, _, _ = make_sweep(tmp_path, [recipe], [FakeModel("1" * 64)])
    sweep.refresh_catalog()
    key = next(iter(sweep.recipes))
    failure = policy.classify("start", "x.code", "boom", klass=UNNAMED)
    for _ in range(IDENTICAL_EXIT_LIMIT - 1):
        sweep._record_failure(key, failure, None)
        assert _verdict_rows(sweep, key) == []
    sweep._record_failure(key, failure, None)
    (row,) = _verdict_rows(sweep, key)
    assert row["evidence"]["identical_exits"] == IDENTICAL_EXIT_LIMIT
    assert row["failure_class"] == UNNAMED and row["code"] == "x.code"
    assert row["status"] == "deferred"  # the platform's, reopened by a release change
    imported = from_history({**historical(), **row, "recipe": historical()["recipe"]})
    assert imported is not None and imported.blame == PublishedBlame.PLATFORM


def test_a_different_unnamed_class_restarts_the_count(tmp_path):
    from spark_sweep.verdict import IDENTICAL_EXIT_LIMIT

    recipe = FakeRecipe("a", ("1" * 64,), content="a" * 64)
    sweep, _, _ = make_sweep(tmp_path, [recipe], [FakeModel("1" * 64)])
    sweep.refresh_catalog()
    key = next(iter(sweep.recipes))
    for i in range(IDENTICAL_EXIT_LIMIT * 2):
        sweep._record_failure(
            key, policy.classify("start", "", "boom", klass=f"{UNNAMED}-{i % 2}"), None
        )
    assert _verdict_rows(sweep, key) == []
