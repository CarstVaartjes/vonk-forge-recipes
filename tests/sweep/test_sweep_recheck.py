"""`--recheck`: an operator-named Working recipe runs once on the deployed release."""

from __future__ import annotations

import pytest
import test_qualification_status as q
from sweep_fakes import FakeModel, FakeRecipe, Gateway, make_sweep

from spark_sweep.lifecycle import RecipeStatus, RetestReason
from spark_sweep.qualification import recheck_due
from vonk_forge_contracts import QualificationBlame as Blame
from vonk_forge_contracts import QualificationOutcome as Outcome

pytest_plugins = ["sweep_bounds"]
pytestmark = pytest.mark.usefixtures("bounded_clock")

RELEASE = q.PLATFORM.controller_source_sha  # the fake Controller's deployed API source
OLDER = q.PLATFORM.model_copy(update={"controller_source_sha": "b" * 40})
KEY = "vonk-forge/named"


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
        {named.key},
        named.key,
        q.A.content_sha256,
        sweep.qualification_evidence[named.key],
        RELEASE,
    )
    sweep.refresh_catalog()  # settled: the same flag asks nothing more
    assert sweep.state.entry(named.key)["status"] == RecipeStatus.PASSED
    assert "requeued_because" not in sweep.state.entry(named.key)


def _named_working(tmp_path, gateway, recheck):
    named = FakeRecipe("named", digests=q.A.model_digests, content=q.A.content_sha256)
    sweep, _, _ = make_sweep(
        tmp_path,
        [named],
        [FakeModel(q.A.model_digests[0])],
        gateway=gateway,
        recheck=recheck,
    )
    assert q.A.runtime_image is not None
    named.image_build_key = q.A.runtime_image.build_key
    named.image_digest = q.A.runtime_image.image_digest
    sweep.refresh_catalog()
    sweep.observe_platform()
    return sweep, named.key


def test_a_recheck_that_ended_deferred_stays_deferred_while_named(tmp_path, gateway):
    sweep, key = _named_working(tmp_path, gateway, frozenset({KEY}))
    older_pass = q.record().model_copy(update={"platform": OLDER})
    fault = q.record(outcome=Outcome.FAILED, blame=Blame.PLATFORM, day=1)
    sweep.qualification_evidence[key] = q.evidence(older_pass, fault).model_copy(
        update={"recipe_key": key}
    )
    sweep.state.entry(key).update(
        status=RecipeStatus.DEFERRED, requeued_because=RetestReason.OPERATOR_RECHECK
    )
    sweep.refresh_catalog()
    assert sweep.state.entry(key)["status"] == RecipeStatus.DEFERRED  # settled, visible
    sweep.cfg.recheck = frozenset()
    sweep.refresh_catalog()
    assert (
        sweep.state.entry(key)["status"] == RecipeStatus.PASSED
    )  # Working by evidence


def test_a_pending_recheck_survives_a_relaunch_without_the_flag(tmp_path, gateway):
    sweep, key = _named_working(tmp_path, gateway, frozenset({KEY}))
    older_pass = q.evidence(
        q.record().model_copy(update={"platform": OLDER})
    ).model_copy(update={"recipe_key": key})
    sweep.qualification_evidence[key] = older_pass
    sweep.state.data["qualification_evidence"][key] = older_pass.model_dump(mode="json")
    sweep.state.entry(key).update(status=RecipeStatus.PASSED)
    sweep.refresh_catalog()
    assert sweep.state.entry(key)["status"] == RecipeStatus.PENDING
    sweep.state.save()
    relaunched, _ = _named_working(tmp_path, gateway, frozenset())
    relaunched.refresh_catalog()
    assert relaunched.state.entry(key)["status"] == RecipeStatus.PENDING


@pytest.mark.parametrize(
    ("record", "release", "due"),
    [
        (q.record(), RELEASE, False),
        (q.record(outcome=Outcome.FAILED, blame=Blame.RECIPE), RELEASE, False),
        (q.record(outcome=Outcome.FAILED, blame=Blame.PLATFORM), RELEASE, False),
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
def test_any_record_of_this_content_on_this_release_settles_it(record, release, due):
    evidence = q.evidence(record)
    assert recheck_due({KEY}, KEY, q.A.content_sha256, evidence, release) is due
    assert not recheck_due({"other"}, KEY, q.A.content_sha256, evidence, release)
