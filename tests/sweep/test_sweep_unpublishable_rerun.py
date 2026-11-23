"""End to end: a local pass the catalog would not publish gets a lane and publishes Working."""

from __future__ import annotations

import pytest
import test_qualification_status as q
from sweep_fakes import FakeModel, FakeRecipe, Gateway, make_sweep
from vonk_forge_contracts import QualificationStatus as Status
from vonk_forge_contracts.qualification_status import published_status

from spark_sweep.lifecycle import RecipeStatus, RetestReason
from spark_sweep.qualification import current_identity

pytest_plugins = ["sweep_bounds"]
pytestmark = pytest.mark.usefixtures("bounded_clock")


@pytest.fixture
def gateway():
    gateway = Gateway()
    yield gateway
    gateway.stop()


def test_the_rerun_gets_a_lane_records_the_catalog_image_and_publishes_working(
    tmp_path, gateway
):
    recipe = FakeRecipe(
        "example", digests=q.A.model_digests, content=q.A.content_sha256
    )
    sweep, _, _ = make_sweep(
        tmp_path, [recipe], [FakeModel(q.A.model_digests[0])], gateway=gateway
    )
    assert q.A.runtime_image is not None
    recipe.image_build_key = q.A.runtime_image.build_key
    recipe.image_digest = q.A.runtime_image.image_digest
    sweep.refresh_catalog()
    sweep.observe_platform()
    key = recipe.key
    sweep.qualification_evidence[key] = q.evidence(
        q.record().model_copy(update={"platform": q.NO_CONTROLLER})
    )
    sweep.state.entry(key).update(
        status=RecipeStatus.PASSED, content_sha256=q.A.content_sha256
    )
    sweep.refresh_catalog()
    assert sweep.state.entry(key)["requeued_because"] == RetestReason.UNPUBLISHABLE_PASS
    sweep.run()
    assert sweep.state.entry(key)["status"] == RecipeStatus.PASSED
    published = published_status(
        sweep.qualification_evidence[key], current_identity(sweep.recipes[key])
    )
    assert published.status == Status.WORKING
    last = published.last_pass
    assert last is not None and last.platform.resolved
    assert last.identity.runtime_image is not None
