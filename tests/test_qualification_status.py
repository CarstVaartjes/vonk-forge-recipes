"""Hardware proof survives rechecks, reverts and infrastructure faults."""

import json
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from pydantic import ValidationError
from vonk_forge_contracts import (
    PublishedQualificationStatusView,
    derive_status,
    published_qualification_status_json_schema,
    published_status,
)
from vonk_forge_contracts import (
    QualificationBlame as Blame,
)
from vonk_forge_contracts import (
    QualificationCauseCode as Code,
)
from vonk_forge_contracts import (
    QualificationChange as Change,
)
from vonk_forge_contracts import (
    QualificationEvidenceRecord as Record,
)
from vonk_forge_contracts import (
    QualificationIdentity as Identity,
)
from vonk_forge_contracts import (
    QualificationOutcome as Outcome,
)
from vonk_forge_contracts import (
    QualificationPlatform as Platform,
)
from vonk_forge_contracts import (
    QualificationRuntimeImage as Image,
)
from vonk_forge_contracts import (
    QualificationStatus as Status,
)
from vonk_forge_contracts import (
    QualificationSummary as Summary,
)
from vonk_forge_contracts import (
    QualificationTopology as Topology,
)
from vonk_forge_contracts import (
    QualificationUnverifiedReason as Reason,
)
from vonk_forge_contracts import (
    RecipeQualificationEvidence as Evidence,
)

from spark_sweep.qualification import (
    backfill,
    current_identity,
    from_history,
    read_evidence,
    scheduling_priority,
)

ROOT = Path(__file__).resolve().parents[1]
NOW = datetime(2026, 10, 21, tzinfo=UTC)
KEY = "vonk-forge/example"
A = Identity(
    content_sha256="a" * 64,
    runtime_image=Image(build_key="b" * 64, image_digest="sha256:" + "c" * 64),
    model_digests=("d" * 64,),
)
B = A.model_copy(update={"content_sha256": "e" * 64})
PLATFORM = Platform(
    controller_source_sha="a" * 40,
    controller_version="2.0.0",
    agent_version="2.0.0",
    spark_count=1,
    topology=Topology.SINGLE,
)


def record(identity=A, outcome=Outcome.PASSED, blame=None, day=0):
    code, summary = {
        None: (Code.HARDWARE_PASS, Summary.PASSED),
        Blame.RECIPE: (Code.RECIPE_FAILURE, Summary.RECIPE_FAILED),
        Blame.PLATFORM: (Code.PLATFORM_FAILURE, Summary.PLATFORM_FAILED),
    }[blame]
    return Record(
        identity=identity,
        outcome=outcome,
        blame=blame,
        cause_code=code,
        cause_summary=summary,
        tested_at=NOW + timedelta(days=day),
        platform=PLATFORM,
    )


def evidence(*records):
    return Evidence(recipe_key=KEY, records=records)


@pytest.mark.parametrize(
    "records,current,fleet,testing,status",
    [
        ((), A, 2, False, Status.UNTESTED),
        ((record(),), A, 2, False, Status.WORKING),
        ((record(),), B, 2, False, Status.RECHECK),
        (
            (record(outcome=Outcome.FAILED, blame=Blame.RECIPE),),
            B,
            2,
            False,
            Status.FIX_TO_VERIFY,
        ),
        (
            (record(outcome=Outcome.FAILED, blame=Blame.RECIPE),),
            A,
            2,
            False,
            Status.FAILED_RECIPE,
        ),
        (
            (record(outcome=Outcome.FAILED, blame=Blame.PLATFORM),),
            A,
            2,
            False,
            Status.BLOCKED_PLATFORM,
        ),
        ((), A.model_copy(update={"required_sparks": 4}), 2, False, Status.CANNOT_TEST),
        ((record(),), B, 2, True, Status.TESTING),
    ],
)
def test_status_table(records, current, fleet, testing, status):
    view = derive_status(evidence(*records), current, fleet, testing)
    assert view.status == status
    if records and records[0].outcome == Outcome.PASSED:
        assert view.last_pass == records[0]


def test_restore_on_revert_and_newest_exact_hardware_verdict_wins():
    passed = record()
    failed_other = record(B, Outcome.FAILED, Blame.RECIPE, 1)
    history = evidence(failed_other, passed)
    assert derive_status(history, A, 0, False).status == Status.WORKING
    assert derive_status(history, B, 2, False).status == Status.FAILED_RECIPE
    failed_same = record(A, Outcome.FAILED, Blame.RECIPE, 2)
    assert (
        derive_status(evidence(passed, failed_same), A, 2, False).status
        == Status.FAILED_RECIPE
    )
    repassed = record(day=3)
    assert (
        derive_status(evidence(repassed, failed_same, passed), A, 2, False).status
        == Status.WORKING
    )


def test_platform_failure_never_invalidates_proof_or_creates_recipe_blame():
    failed = record(outcome=Outcome.FAILED, blame=Blame.PLATFORM, day=1)
    history = evidence(record(), failed)
    view = derive_status(history, A, 2, False)
    assert view.status == Status.WORKING
    assert view.last_failure == failed
    assert derive_status(history, B, 2, False).status == Status.RECHECK
    assert published_status(evidence(failed), A).status == Status.NOT_YET_VERIFIED
    assert published_status(evidence(failed), A).reason == Reason.PLATFORM_ISSUE


def test_changes_are_precise_and_prior_proof_visible_in_lane():
    current = B.model_copy(
        update={
            "runtime_image": Image(
                build_key="f" * 64, image_digest="sha256:" + "0" * 64
            ),
            "model_digests": ("1" * 64,),
        }
    )
    view = derive_status(evidence(record()), current, 2, True)
    assert view.status == Status.TESTING
    assert view.changed == (Change.RECIPE, Change.IMAGE, Change.MODEL)
    assert view.last_pass is not None


def test_incomplete_history_is_not_published_as_exact_proof():
    unresolved = A.model_copy(update={"runtime_image": None})
    view = published_status(evidence(record(unresolved)), unresolved)
    assert view.status == Status.NOT_YET_VERIFIED
    assert view.reason == Reason.INCOMPLETE_PROVENANCE
    unknown_agent = record().model_copy(
        update={"platform": PLATFORM.model_copy(update={"agent_version": None})}
    )
    assert (
        published_status(evidence(unknown_agent), A).status == Status.NOT_YET_VERIFIED
    )
    assert published_status(evidence(record()), A).status == Status.WORKING
    assert published_status(evidence(record()), B).status == Status.RECHECK
    assert published_status(evidence(), A).reason == Reason.NEVER_TESTED


def test_ordering():
    statuses = [
        Status.RECHECK,
        Status.BLOCKED_PLATFORM,
        Status.UNTESTED,
        Status.FIX_TO_VERIFY,
    ]
    views = [
        derive_status(evidence(), A, 2, False).model_copy(update={"status": s})
        for s in statuses
    ]
    assert [v.status for v in sorted(views, key=scheduling_priority)] == list(
        reversed(statuses)
    )


@pytest.mark.parametrize(
    "field",
    [
        "spark_names",
        "hostname",
        "ip",
        "operation_id",
        "run_id",
        "request_id",
        "path",
        "nas",
        "key",
        "error",
    ],
)
def test_public_evidence_forbids_private_fields_at_every_object(field):
    payload = record().model_dump(mode="json")
    for target in (
        payload,
        payload["identity"],
        payload["identity"]["runtime_image"],
        payload["platform"],
    ):
        target[field] = "private"
        with pytest.raises(ValidationError):
            Record.model_validate(payload)
        del target[field]
    with pytest.raises(ValidationError):
        Record.model_validate(
            {**payload, "cause_summary": "private-host /internal/path 10.0.0.1"}
        )
    schema = json.loads(json.dumps(published_qualification_status_json_schema()))
    assert schema["additionalProperties"] is False
    assert all(
        v.get("additionalProperties") is False
        for v in schema["$defs"].values()
        if v.get("type") == "object"
    )
    assert Status.TESTING.value not in schema["properties"]["status"]["enum"]
    with pytest.raises(ValidationError):
        PublishedQualificationStatusView.model_validate(
            {"status": Status.TESTING, "required_sparks": 1}
        )


def historical():
    return {
        "recipe": KEY,
        "step": "smoke",
        "status": Outcome.PASSED.value,
        "recorded_at": NOW.isoformat(),
        "content_sha256": A.content_sha256,
        "verdict_identity": A.model_dump(mode="json"),
        "controller_release": PLATFORM.controller_source_sha,
        "controller_version": PLATFORM.controller_version,
        "agent_version": PLATFORM.agent_version,
        "node_ids": ["private-node"],
        "run_id": "private-run",
        "error": "10.0.0.1 /private/path",
    }


def test_backfill_rotations_idempotent_read_only_and_sanitized(tmp_path):
    old, current = tmp_path / "results.jsonl.old", tmp_path / "results.jsonl"
    row = historical()
    old.write_text(json.dumps(row) + "\n")
    row["recorded_at"] = (NOW + timedelta(days=1)).isoformat()
    current.write_text(json.dumps(row) + "\n{torn")
    before = {p: p.read_bytes() for p in (old, current)}
    assert backfill([old, current], tmp_path).changed_files == 1
    seeded = read_evidence(tmp_path, KEY)
    assert len(seeded.records) == 2
    assert backfill([old, current], tmp_path).changed_files == 0
    assert before == {p: p.read_bytes() for p in (old, current)}
    text = seeded.model_dump_json()
    assert not any(
        private in text
        for private in ("private-node", "private-run", "10.0.0.1", "/private/path")
    )
    legacy = historical()
    del legacy["verdict_identity"]
    imported = from_history(legacy)
    assert imported is not None and imported.identity.runtime_image is None
    assert not imported.identity.models_resolved


def test_historical_infrastructure_failures_are_never_recipe_failures():
    row = {**historical(), "status": Outcome.FAILED, "failure_class": "capacity"}
    imported = from_history(row)
    assert imported is not None and imported.blame == Blame.PLATFORM
    row.update(failure_class="smoke-assertion")
    imported = from_history(row)
    assert imported is not None and imported.blame == Blame.RECIPE


def test_sweep_restore_and_retry_failed_use_evidence(tmp_path):
    from sweep_fakes import FakeModel, FakeRecipe, make_sweep

    from spark_sweep.catalog_contract import RuntimeImageIdentity
    from spark_sweep.lifecycle import RecipeStatus

    fake = FakeRecipe("example")
    sweep, _, _ = make_sweep(tmp_path, [fake], [FakeModel("m1")])
    sweep.refresh_catalog()
    assert A.runtime_image is not None
    current = replace(
        sweep.recipes[KEY],
        content_sha256=A.content_sha256,
        model_digests=A.model_digests,
        runtime_image=RuntimeImageIdentity.model_validate(A.runtime_image.model_dump()),
    )
    sweep.recipes[KEY] = current
    assert current_identity(current) == A
    sweep.qualification_evidence[KEY] = evidence(
        record(), record(B, Outcome.FAILED, Blame.RECIPE, 1)
    )
    entry = sweep.state.entry(KEY)
    entry.update(status=RecipeStatus.FAILED, content_sha256=B.content_sha256)
    assert sweep.requeue_failed() == 0
    sweep._reconcile_entry(current, 2)
    assert entry["status"] == RecipeStatus.PASSED
    assert entry["content_sha256"] == A.content_sha256
    assert sweep.qualification_view(current).status == Status.WORKING


def test_published_catalog_matches_shared_derivation(built_catalog):
    catalog, _ = built_catalog
    for entry in catalog["recipes"]:
        document = entry["document"]
        key = f"{document['identity']['publisher']}/{document['identity']['slug']}"
        identity = Identity(
            content_sha256=entry["content_sha256"],
            model_digests=tuple(
                sorted({b["model"]["content_sha256"] for b in document["models"]})
            ),
            required_sparks=document["topology"]["node_count"],
        )
        expected = published_status(read_evidence(ROOT, key), identity)
        assert entry["qualification"] == expected.model_dump(
            mode="json", exclude_none=True
        )
        if expected.status != Status.NOT_YET_VERIFIED:
            assert (
                expected.status
                == derive_status(
                    read_evidence(ROOT, key), identity, identity.required_sparks, False
                ).status
            )


def test_catalog_derives_after_resolving_release_image(
    tmp_path, monkeypatch, built_catalog
):
    import runpy

    from vonk_forge_contracts import document_sha256

    from spark_sweep.qualification import write_evidence

    tool = runpy.run_path(str(ROOT / "tools/build-catalog-index"))
    build = tool["build"]
    globals_ = build.__globals__
    document = built_catalog[0]["recipes"][0]["document"]
    digest = document_sha256(document)
    key = f"{document['identity']['publisher']}/{document['identity']['slug']}"
    recipe_dir = tmp_path / "recipes"
    recipe_dir.mkdir()
    (recipe_dir / "example.json").write_text(json.dumps(document))
    image = Image(build_key="b" * 64, image_digest="sha256:" + "c" * 64)
    identity = Identity(
        content_sha256=digest,
        runtime_image=image,
        model_digests=tuple(
            sorted({b["model"]["content_sha256"] for b in document["models"]})
        ),
        required_sparks=document["topology"]["node_count"],
    )
    write_evidence(tmp_path, Evidence(recipe_key=key, records=(record(identity),)))
    monkeypatch.setitem(globals_, "ROOT", tmp_path)
    monkeypatch.setitem(globals_, "_catalog_entity_documents", dict)
    monkeypatch.setitem(
        globals_,
        "recipe_package",
        lambda *a, **kw: (b"package", {"path": "packages/example.tar.gz"}),
    )
    monkeypatch.setitem(globals_, "_commit_time", lambda *a: NOW.isoformat())
    images = {
        digest: {
            "build_key": image.build_key,
            "reference": "ghcr.io/public/image@" + image.image_digest,
        }
    }
    first = build(
        package_dir=tmp_path / "packages",
        source_commit="a" * 40,
        prebuilt_images=images,
    )
    assert first["recipes"][0]["qualification"]["status"] == Status.WORKING
    images[digest]["reference"] = "ghcr.io/public/image@sha256:" + "f" * 64
    second = build(
        package_dir=tmp_path / "packages",
        source_commit="a" * 40,
        prebuilt_images=images,
    )
    published = second["recipes"][0]["qualification"]
    assert published["status"] == Status.RECHECK
    assert published["changed"] == [Change.IMAGE]
    assert (
        published["last_pass"]["platform"]["controller_source_sha"]
        == PLATFORM.controller_source_sha
    )


def test_saved_report_recomputes_from_inputs_and_keeps_prior_proof():
    from vonk_forge_contracts import QualificationDerivationInput

    from spark_sweep.report import _qualification_text, saved_qualification

    snapshot = QualificationDerivationInput(
        evidence=evidence(record()), current_identity=B, fleet_size=2, testing_now=True
    )
    payload = snapshot.model_dump(mode="json")
    assert "status" not in payload
    views = saved_qualification({KEY: payload})
    assert views[KEY]["status"] == Status.TESTING
    assert "last pass" in _qualification_text(views[KEY])
    assert views[KEY]["last_pass"]["identity"]["content_sha256"] == A.content_sha256


def test_public_topology_requirement_is_not_the_publishers_fleet():
    current = A.model_copy(update={"required_sparks": 4})
    local = derive_status(evidence(), current, 2, False)
    assert local.status == Status.CANNOT_TEST
    public = published_status(evidence(), current)
    assert public.status == Status.NOT_YET_VERIFIED
    assert public.reason == Reason.NEEDS_SPARKS and public.required_sparks == 4
    with pytest.raises(ValidationError, match="required Spark count"):
        record(current)


def test_fix_to_verify_precedes_untested_even_with_a_limit(tmp_path):
    from sweep_fakes import FakeModel, FakeRecipe, make_sweep

    from spark_sweep.catalog_contract import RuntimeImageIdentity

    untested, fixed = FakeRecipe("untested"), FakeRecipe("fixed")
    sweep, _, _ = make_sweep(tmp_path, [untested, fixed], [FakeModel("m1")])
    sweep.refresh_catalog()
    assert A.runtime_image is not None
    for key, recipe in list(sweep.recipes.items()):
        sweep.recipes[key] = replace(
            recipe,
            content_sha256=B.content_sha256,
            model_digests=A.model_digests,
            runtime_image=RuntimeImageIdentity.model_validate(
                A.runtime_image.model_dump()
            ),
        )
    sweep.qualification_evidence[fixed.key] = Evidence(
        recipe_key=fixed.key, records=(record(A, Outcome.FAILED, Blame.RECIPE),)
    )
    sweep.cfg.limit = 1
    assert [r.key for r in sweep.pending()] == [fixed.key]


def test_authorization_faults_are_platform_evidence_and_preserve_recipe_proof():
    from spark_sweep.auth_contract import SmokeAuthClass

    row = {
        **historical(),
        "status": Outcome.FAILED,
        "failure_class": SmokeAuthClass.REFUSAL,
    }
    imported = from_history(row)
    assert imported is not None and imported.blame == Blame.PLATFORM
    assert (
        derive_status(evidence(record(), imported), A, 2, False).status
        == Status.WORKING
    )


def test_platform_faults_do_not_count_against_recipes_in_report_headline():
    from spark_sweep.lifecycle import RecipeStatus
    from spark_sweep.report import build_report, render_report

    view = derive_status(
        evidence(record(A, Outcome.FAILED, Blame.PLATFORM)), A, 2, False
    )
    report = build_report(
        {KEY: {"status": RecipeStatus.FAILED}}, {KEY: view.model_dump(mode="json")}
    )
    assert render_report(report).splitlines()[2] == f"{Status.BLOCKED_PLATFORM.value} 1"


def test_elapsed_time_never_retries_a_failure_with_platform_blame(
    tmp_path, monkeypatch
):
    from sweep_fakes import FakeModel, FakeRecipe, make_sweep

    from spark_sweep.identity import content_identity
    from spark_sweep.lifecycle import RecipeStatus
    from spark_sweep.run import RETEST_SECONDS

    fake = FakeRecipe("example")
    sweep, _, clock = make_sweep(tmp_path, [fake], [FakeModel("m1")], watch_seconds=60)
    sweep.refresh_catalog()
    recipe = sweep.recipes[KEY]
    sweep.state.entry(KEY).update(
        status=RecipeStatus.FAILED,
        content_sha256=recipe.content_sha256,
        verdict_identity=content_identity(recipe).model_dump(mode="json"),
        finished_at=clock.now() - RETEST_SECONDS,
    )
    view = derive_status(
        evidence(record(A, Outcome.FAILED, Blame.PLATFORM)), A, 2, False
    )
    monkeypatch.setattr(sweep, "qualification_view", lambda recipe: view)
    sweep.tick()
    assert sweep.state.status(KEY) == RecipeStatus.FAILED
    assert not sweep.state.data["loads"]
    assert not sweep.state.entry(KEY).get("recovery_history")
