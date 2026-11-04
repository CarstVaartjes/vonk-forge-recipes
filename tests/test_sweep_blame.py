"""One blame rule: the sweep's verdict and the published status read the same function."""

from __future__ import annotations

import pytest
from test_qualification_status import Blame as PublishedBlame
from test_qualification_status import Outcome, historical

from spark_sweep import policy, verdict
from spark_sweep.auth_contract import SmokeAuthClass
from spark_sweep.qualification import from_history
from spark_sweep.verdict import Blame, Disposition

RECIPE_CODE = min(policy.RECIPE_FAILURE_CODES)
FINAL_CODE = min(policy.REVIEW_FINAL_CODES)

# Every class the sweep names, with the blame it must carry (code-less).
EXPECTED = {
    **dict.fromkeys(verdict.FAIL_CLASSES, Blame.RECIPE),
    **dict.fromkeys(verdict.WAIT_CLASSES, Blame.PLATFORM),
    **dict.fromkeys(verdict.OBSERVE_CLASSES, Blame.PLATFORM),
    **dict.fromkeys(verdict.SKIP_CLASSES, Blame.PLATFORM),
    **dict.fromkeys(verdict.RECORDED_CLASSES, Blame.PLATFORM),
    "capacity": Blame.PLATFORM,
}
UNNAMED = "a-class-nobody-named"


@pytest.mark.parametrize("klass", sorted(EXPECTED))
def test_each_named_class_has_its_blame(klass):
    assert verdict.blame(klass, "") is EXPECTED[klass]


def test_every_class_constant_is_covered():
    named = (
        verdict.FAIL_CLASSES
        | verdict.WAIT_CLASSES
        | verdict.OBSERVE_CLASSES
        | verdict.SKIP_CLASSES
        | verdict.RECORDED_CLASSES
    )
    assert named <= set(EXPECTED)
    assert SmokeAuthClass.REFUSAL not in verdict.FAIL_CLASSES


def test_an_unnamed_class_is_observed_never_failed():
    assert verdict.blame(UNNAMED, "") is Blame.UNKNOWN
    failure = policy.classify("start", "", "x", klass=UNNAMED)
    assert verdict.disposition(failure) is Disposition.OBSERVE


@pytest.mark.parametrize("code", [RECIPE_CODE, FINAL_CODE])
def test_a_recipe_code_blames_the_recipe_unless_unattributed(code):
    assert verdict.blame(UNNAMED, code) is Blame.RECIPE
    assert verdict.blame(verdict.UNATTRIBUTED, code) is Blame.PLATFORM


@pytest.mark.parametrize(
    ("klass", "code"),
    [(k, "") for k in sorted(EXPECTED)]
    + [(UNNAMED, ""), (UNNAMED, RECIPE_CODE), (verdict.UNATTRIBUTED, RECIPE_CODE)]
    + [(k, FINAL_CODE) for k in sorted(verdict.WAIT_CLASSES)],
)
def test_published_status_agrees_with_the_sweep(klass, code):
    row = {
        **historical(),
        "status": Outcome.FAILED,
        "failure_class": klass,
        "code": code,
    }
    imported = from_history(row)
    assert imported is not None
    published = (
        PublishedBlame.RECIPE
        if verdict.blame(klass, code) is Blame.RECIPE
        else PublishedBlame.PLATFORM
    )
    assert imported.blame == published
