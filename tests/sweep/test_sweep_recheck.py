"""`--recheck`: an operator-named Working recipe runs once on the deployed release."""

from __future__ import annotations

import pytest
import test_qualification_status as q
from sweep_fakes import FakeModel, FakeRecipe, Gateway, make_sweep

from spark_sweep.lifecycle import RecipeStatus, RetestReason
from spark_sweep.qualification import RECHECK_PLATFORM_TRIES, recheck_due
from vonk_forge_contracts import QualificationBlame as Blame
from vonk_forge_contracts import QualificationOutcome as Outcome

pytest_plugins = ["sweep_bounds"]
pytestmark = pytest.mark.usefixtures("bounded_clock")

RELEASE = q.PLATFORM.controller_source_sha  # the fake Controller's deployed API source
OLDER = q.PLATFORM.model_copy(update={"controller_source_sha": "b" * 40})


@pytest.fixture
def gateway():
    gateway = Gateway()
    yield gateway
    gateway.stop()


def test_a_named_working_recipe_runs_once_on_the_release_then_stays_passed(
    tmp_path, gateway
):
    named = FakeRecipe("named", digests=q.A.model_digests, content=q.A.content_sha256)
    other = FakeRecipe("other", digests=q.A.model_digests, content=q.A.content_sha256)
    sweep, _, _ = make_sweep(
        tmp_path,
        [named, other],
        [FakeModel(q.A.model_digests[0])],
        gateway=gateway,
        recheck=frozenset({named.key.partition("/")[2]}),
    )
    assert q.A.runtime_image is not None
    for recipe in (named, other):
        recipe.image_build_key = q.A.runtime_image.build_key
        recipe.image_digest = q.A.runtime_image.image_digest
    sweep.refresh_catalog()
    sweep.observe_platform()
    for recipe in (named, other):
        older_pass = q.record().model_copy(update={"platform": OLDER})
        sweep.qualification_evidence[recipe.key] = q.evidence(older_pass).model_copy(
            update={"recipe_key": recipe.key}
        )
        sweep.state.entry(recipe.key).update(
            status=RecipeStatus.PASSED, content_sha256=q.A.content_sha256
        )
    sweep.refresh_catalog()
    assert sweep.state.entry(named.key)["status"] == RecipeStatus.PENDING
    assert (
        sweep.state.entry(named.key)["requeued_because"]
        == RetestReason.OPERATOR_RECHECK
    )
    assert sweep.state.entry(other.key)["status"] == RecipeStatus.PASSED
    sweep.run()
    assert not recheck_due(
        sweep.qualification_evidence[named.key], q.A.content_sha256, RELEASE
    )
    sweep.refresh_catalog()  # settled: the same flag asks nothing more
    assert sweep.state.entry(named.key)["status"] == RecipeStatus.PASSED
    assert "requeued_because" not in sweep.state.entry(named.key)


def _named_working(tmp_path, gateway):
    named = FakeRecipe("named", digests=q.A.model_digests, content=q.A.content_sha256)
    sweep, _, clock = make_sweep(
        tmp_path,
        [named],
        [FakeModel(q.A.model_digests[0])],
        gateway=gateway,
        recheck=frozenset({named.key}),
    )
    assert q.A.runtime_image is not None
    named.image_build_key = q.A.runtime_image.build_key
    named.image_digest = q.A.runtime_image.image_digest
    sweep.refresh_catalog()
    sweep.observe_platform()
    return sweep, clock, named.key


def test_a_platform_fault_on_the_release_retries_with_backoff_up_to_the_bound(
    tmp_path, gateway
):
    sweep, clock, key = _named_working(tmp_path, gateway)
    older_pass = q.record().model_copy(update={"platform": OLDER})
    fault = q.record(outcome=Outcome.FAILED, blame=Blame.PLATFORM, day=1)
    sweep.qualification_evidence[key] = q.evidence(older_pass, fault)
    sweep.state.entry(key).update(status=RecipeStatus.DEFERRED, attempts=2)
    sweep.refresh_catalog()
    entry = sweep.state.entry(key)
    assert entry["status"] == RecipeStatus.PENDING
    assert entry["attempts"] == 2  # the try counts; no reset into a reload loop
    assert entry["not_before"] > clock.now()
    faults = [
        q.record(outcome=Outcome.FAILED, blame=Blame.PLATFORM, day=day)
        for day in (1, 2, 3)
    ]
    sweep.qualification_evidence[key] = q.evidence(older_pass, *faults)
    entry.update(status=RecipeStatus.DEFERRED)
    sweep.refresh_catalog()
    # The bound ends the retries: platform faults never count against a Working recipe,
    # and the deferred rows stay in results.jsonl for the investigation.
    assert sweep.state.entry(key)["status"] == RecipeStatus.PASSED


def test_a_pending_recheck_survives_a_relaunch_without_the_flag(tmp_path, gateway):
    sweep, _, key = _named_working(tmp_path, gateway)
    older_pass = q.record().model_copy(update={"platform": OLDER})
    sweep.qualification_evidence[key] = q.evidence(older_pass)
    sweep.state.entry(key).update(status=RecipeStatus.PASSED)
    sweep.refresh_catalog()
    assert sweep.state.entry(key)["status"] == RecipeStatus.PENDING
    sweep.cfg.recheck = frozenset()
    sweep.refresh_catalog()
    assert sweep.state.entry(key)["status"] == RecipeStatus.PENDING


@pytest.mark.parametrize(
    ("record", "release", "due"),
    [
        (q.record(), RELEASE, False),
        (q.record(outcome=Outcome.FAILED, blame=Blame.RECIPE), RELEASE, False),
        (q.record(outcome=Outcome.FAILED, blame=Blame.PLATFORM), RELEASE, True),
        (q.record().model_copy(update={"platform": OLDER}), RELEASE, True),
        (q.record(identity=q.B), RELEASE, True),
        (q.record().model_copy(update={"platform": OLDER}), None, False),
    ],
    ids=[
        "pass-here",
        "recipe-failure-here",
        "platform-fault-here",
        "pass-elsewhere",
        "other-content",
        "release-unknown",
    ],
)
def test_only_a_behavioural_verdict_of_this_content_on_this_release_settles_it(
    record, release, due
):
    assert recheck_due(q.evidence(record), q.A.content_sha256, release) is due


def test_platform_faults_on_the_release_settle_it_only_after_the_bound() -> None:
    faults = [
        q.record(outcome=Outcome.FAILED, blame=Blame.PLATFORM, day=day)
        for day in range(RECHECK_PLATFORM_TRIES)
    ]
    assert recheck_due(q.evidence(*faults[:-1]), q.A.content_sha256, RELEASE)
    assert not recheck_due(q.evidence(*faults), q.A.content_sha256, RELEASE)
