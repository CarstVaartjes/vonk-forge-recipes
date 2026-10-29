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
    required_sparks=1,
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


@pytest.mark.parametrize("ok", [True, False])
def test_only_missed_quality_checks_become_public_notes(ok):
    row = historical()
    row["result"] = {"quality": [{"ok": ok, "got": "private model answer"}]}
    imported = from_history(row)
    assert imported is not None
    assert bool(imported.quality_notes) is not ok
    assert "private model answer" not in imported.model_dump_json()


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
            source_bundle_sha256=entry["source_bundle_sha256"],
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
            assert identity.required_sparks is not None
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
        source_bundle_sha256="1" * 64,
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
        lambda *a, **kw: (
            b"package",
            {"path": "packages/example.tar.gz", "source_bundle_sha256": "1" * 64},
        ),
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


@pytest.mark.parametrize("field", ["controller_version", "agent_version"])
@pytest.mark.parametrize(
    "unsafe",
    [
        "192.0.2.10",
        "2001:db8::1",
        "spark-1",
        "nas.private.example",
        "/private/state",
        "1.2.3+spark-1.internal",
        "1.2.3+request-123",
    ],
)
def test_public_versions_cannot_be_used_to_carry_infrastructure_details(field, unsafe):
    from jsonschema import Draft202012Validator
    from jsonschema import ValidationError as SchemaValidationError

    with pytest.raises(ValidationError):
        Platform.model_validate({**PLATFORM.model_dump(), field: unsafe})
    payload = json.loads(
        published_status(evidence(record()), A).model_dump_json(exclude_none=True)
    )
    payload["last_pass"]["platform"][field] = unsafe
    with pytest.raises(SchemaValidationError):
        Draft202012Validator(published_qualification_status_json_schema()).validate(
            payload
        )


def test_missing_historical_model_bindings_are_unknown_not_an_empty_model_set():
    row = historical()
    del row["verdict_identity"]["model_digests"]
    imported = from_history(row)
    assert imported is not None
    assert not imported.identity.models_resolved


def test_missing_historical_topology_stays_unknown_and_roundtrips():
    row = {**historical(), "status": Outcome.FAILED}
    del row["node_ids"]
    imported = from_history(row)
    assert imported is not None
    assert imported.identity.required_sparks is None
    assert imported.platform.spark_count is None and imported.platform.topology is None
    restored = Record.model_validate_json(imported.model_dump_json(exclude_none=True))
    assert restored == imported
    with pytest.raises(ValueError, match="Spark count"):
        derive_status(
            evidence(), A.model_copy(update={"required_sparks": None}), 2, False
        )


@pytest.mark.parametrize(
    "suffix",
    [
        record(day=1),
        record(A, Outcome.FAILED, Blame.PLATFORM, day=1),
        record(B, Outcome.FAILED, Blame.RECIPE, day=1),
        record(B, day=1),
    ],
)
def test_incomplete_later_telemetry_never_erases_complete_working_proof(suffix):
    incomplete = suffix.model_copy(
        update={"platform": PLATFORM.model_copy(update={"agent_version": None})}
    )
    proven = record()
    view = published_status(evidence(proven, incomplete), A)
    assert view.status == Status.WORKING
    assert view.last_pass == proven


def test_partial_pass_after_recipe_failure_cannot_revive_an_invalidated_public_proof():
    failed = record(A, Outcome.FAILED, Blame.RECIPE, day=1)
    partial = record(day=2).model_copy(
        update={"platform": PLATFORM.model_copy(update={"agent_version": None})}
    )
    history = evidence(record(), failed, partial)
    assert published_status(history, A).status == Status.NOT_YET_VERIFIED
    assert (
        published_status(evidence(*history.records, record(day=3)), A).status
        == Status.WORKING
    )


def evidence_sweep(tmp_path, names=("example",), **kwargs):
    from sweep_fakes import FakeModel, FakeRecipe, make_sweep

    fakes = [
        FakeRecipe(name, digests=A.model_digests, content=A.content_sha256)
        for name in names
    ]
    sweep, fleet, clock = make_sweep(
        tmp_path, fakes, [FakeModel(A.model_digests[0])], **kwargs
    )
    sweep.refresh_catalog()
    # Use the same fully resolved identity in the fake Controller and the evidence.
    assert A.runtime_image is not None
    for fake in fakes:
        fake.image_build_key = A.runtime_image.build_key
        fake.image_digest = A.runtime_image.image_digest
    sweep.refresh_catalog()
    return sweep, fleet, clock


@pytest.mark.parametrize(
    "changed", ["document", "model", "image", "build", "library", "unchanged"]
)
@pytest.mark.parametrize(
    "outcome,blame",
    [
        (Outcome.PASSED, None),
        (Outcome.FAILED, Blame.RECIPE),
        (Outcome.FAILED, Blame.PLATFORM),
    ],
)
def test_evidence_identity_drives_requeue(tmp_path, changed, outcome, blame):
    from spark_sweep.lifecycle import RecipeStatus

    sweep, _, _ = evidence_sweep(tmp_path)
    current = sweep.recipes[KEY]
    assert current_identity(current) == A
    if changed == "document":
        current = replace(current, content_sha256=B.content_sha256)
    elif changed == "model":
        current = replace(current, model_digests=("f" * 64,))
    elif changed in ("image", "build"):
        image = current.runtime_image
        assert image is not None
        current = replace(
            current,
            runtime_image=image.model_copy(
                update={
                    "image_digest" if changed == "image" else "build_key": "sha256:"
                    + "f" * 64
                    if changed == "image"
                    else "f" * 64
                }
            ),
        )
    elif changed == "library":
        current = replace(current, library_commit="new-publication")
    sweep.qualification_evidence[KEY] = evidence(record(A, outcome, blame))
    entry = sweep.state.entry(KEY)
    entry.update(
        status=RecipeStatus.PASSED if outcome == Outcome.PASSED else RecipeStatus.FAILED
    )
    sweep._reconcile_entry(current, 2)
    assert (entry["status"] == RecipeStatus.PENDING) == (
        changed not in ("library", "unchanged")
    )
    view = sweep.qualification_view(current)
    assert view is not None
    if outcome == Outcome.PASSED:
        assert view.last_pass == record()


def test_evidence_revalidation_of_newest_cached_pass_is_last_even_with_limit(tmp_path):
    from spark_sweep import policy
    from spark_sweep.lifecycle import RecipeStatus

    sweep, _, _ = evidence_sweep(tmp_path, ("passed", "untested", "failed", "deferred"))
    for slug, records in (
        ("passed", (record(),)),
        ("failed", (record(A, Outcome.FAILED, Blame.RECIPE),)),
        ("deferred", (record(A, Outcome.FAILED, Blame.PLATFORM),)),
    ):
        key = f"vonk-forge/{slug}"
        current = replace(sweep.recipes[key], content_sha256=B.content_sha256)
        if slug == "passed":
            current = replace(current, updated_at="2026-10-21", artifact_bytes=1)
        sweep.recipes[key] = current
        sweep.qualification_evidence[key] = Evidence(recipe_key=key, records=records)
        sweep.state.entry(key)["status"] = (
            RecipeStatus.PASSED if slug == "passed" else RecipeStatus.FAILED
        )
        sweep._reconcile_entry(current, 2)
    pending = sweep.pending()
    assert pending[-1].slug == "passed"
    plans = [policy.GroupPlan(r.model_set, (r.key,), r.artifact_bytes) for r in pending]
    ordered = policy.order_queue(
        plans, sweep.recipes, cached=set(A.model_digests), sizes={}
    )
    ordered.sort(
        key=lambda key: scheduling_priority(
            sweep.qualification_view(sweep.recipes[key])
        )
    )
    assert ordered[-1] == "vonk-forge/passed"
    assert set(ordered[:-1]) == {
        "vonk-forge/" + s for s in ("untested", "failed", "deferred")
    }
    sweep.cfg.limit = 1
    assert [r.slug for r in sweep.pending()] == ["failed"]
    # With no other failure to verify, untested work still wins the one slot.
    sweep.recipes.pop("vonk-forge/failed")
    assert [r.slug for r in sweep.pending()] == ["untested"]


@pytest.mark.parametrize("active", [False, True])
def test_saved_verdict_adoption_restores_without_rerun_and_survives_interrupted_save(
    tmp_path, active
):
    from sweep_fakes import FakeModel, FakeRecipe, make_sweep

    from spark_sweep.identity import content_identity
    from spark_sweep.lifecycle import RecipeStatus

    sweep, fleet, _ = evidence_sweep(tmp_path)
    accepted = sweep.recipes[KEY]
    saved = sweep.state.entry(KEY)
    saved.update(
        status=RecipeStatus.PENDING,
        previous={
            "status": RecipeStatus.PASSED,
            "content_sha256": A.content_sha256,
            "verdict_identity": content_identity(accepted).model_dump(mode="json"),
            "node_count": 1,
            "finished_at": NOW.timestamp(),
        },
    )
    if active:
        sweep.state.slots[KEY] = {
            "phase": "loading",
            "request_key": "saved-request",
            "alias": "saved-alias",
            "node_ids": ["spk_a"],
            "started_at": 1,
        }
    sweep.state.data.pop("qualification_evidence", None)
    sweep.state.save()
    before = sweep.state.path.read_bytes()

    def restart():
        again, _, _ = make_sweep(
            tmp_path, [FakeRecipe("example")], [FakeModel("m1")], fleet=fleet
        )
        again.refresh_catalog()
        return again

    adopted = restart()
    # Loading and deriving neither writes the source state nor submits a load.
    assert sweep.state.path.read_bytes() == before
    assert adopted.state.status(KEY) == (
        RecipeStatus.PENDING if active else RecipeStatus.PASSED
    )
    assert not adopted.log.entries()
    interrupted = restart()
    assert interrupted.qualification_evidence == adopted.qualification_evidence
    adopted.state.save()
    persisted = restart()
    assert persisted.qualification_evidence == adopted.qualification_evidence
    assert not [c for _, c in fleet.calls if c[:2] == ("profile", "load")]


@pytest.mark.parametrize("changed", ["document", "model"])
def test_adopted_unknown_image_pass_keeps_proof_but_cannot_hide_known_changes(
    tmp_path, changed
):
    from spark_sweep.lifecycle import RecipeStatus
    from spark_sweep.state import State

    sweep, _, clock = evidence_sweep(tmp_path)
    historical_identity = A.model_copy(update={"runtime_image": None})
    if changed == "document":
        historical_identity = historical_identity.model_copy(
            update={"content_sha256": B.content_sha256}
        )
    else:
        historical_identity = historical_identity.model_copy(
            update={"model_digests": ("f" * 64,)}
        )
    sweep.state.entry(KEY).update(
        status=RecipeStatus.PASSED,
        content_sha256=historical_identity.content_sha256,
        verdict_identity=historical_identity.model_dump(mode="json"),
    )
    sweep.state.data.pop("qualification_evidence", None)
    sweep.state.save()
    sweep.state = State.load(sweep.state.path, clock.now)
    sweep.qualification_evidence[KEY] = Evidence.model_validate(
        sweep.state.data["qualification_evidence"][KEY]
    )
    sweep._reconcile_entry(sweep.recipes[KEY], 2)
    assert sweep.state.status(KEY) == RecipeStatus.PENDING
    view = sweep.qualification_view(sweep.recipes[KEY])
    assert view is not None and view.last_pass is not None
    assert view.last_pass.identity == historical_identity.model_copy(
        update={"required_sparks": None}
    )
    assert view.last_pass.identity.runtime_image is None
    assert (Change.RECIPE if changed == "document" else Change.MODEL) in view.changed


def test_adoption_is_once_and_retry_failed_never_requeues_proven_pass(tmp_path):
    from spark_sweep.lifecycle import RecipeStatus
    from spark_sweep.state import State

    sweep, _, clock = evidence_sweep(tmp_path)
    sweep.state.data["qualification_evidence"] = {
        KEY: evidence(record()).model_dump(mode="json")
    }
    # Obsolete bookkeeping must not be imported on subsequent loads.
    sweep.state.entry(KEY).update(
        status=RecipeStatus.FAILED, content_sha256=B.content_sha256
    )
    sweep.state.save()
    sweep.state = State.load(sweep.state.path, clock.now)
    sweep.qualification_evidence[KEY] = Evidence.model_validate(
        sweep.state.data["qualification_evidence"][KEY]
    )
    assert sweep.qualification_evidence[KEY].records == (record(),)
    assert sweep.requeue_failed() == 0
    sweep._reconcile_entry(sweep.recipes[KEY], 2)
    assert sweep.state.status(KEY) == RecipeStatus.PASSED


@pytest.mark.parametrize("blame", [Blame.RECIPE, Blame.PLATFORM])
def test_retry_failed_uses_adopted_failure_evidence(tmp_path, blame):
    from spark_sweep.lifecycle import RecipeStatus

    sweep, _, _ = evidence_sweep(tmp_path)
    sweep.qualification_evidence[KEY] = evidence(record(A, Outcome.FAILED, blame))
    sweep.state.entry(KEY)["status"] = (
        RecipeStatus.FAILED if blame == Blame.RECIPE else RecipeStatus.DEFERRED
    )
    assert sweep.requeue_failed() == 1
    assert sweep.state.status(KEY) == RecipeStatus.PENDING


def test_superseded_verdict_paths_cannot_return():
    import ast

    for name in ("identity.py", "lifecycle.py", "policy.py", "run.py"):
        tree = ast.parse((ROOT / "spark_sweep" / name).read_text())
        defined = {
            n.name
            for n in ast.walk(tree)
            if isinstance(n, (ast.ClassDef, ast.FunctionDef))
        }
        assert not defined & {
            "legacy_image_unknown",
            "requeue_reason",
            "PassReconciliationRecord",
            "LegacyPassedVerdict",
        }
    run = ast.parse((ROOT / "spark_sweep/run.py").read_text())
    for name in ("_reconcile_entry", "requeue_failed"):
        function = next(
            n
            for n in ast.walk(run)
            if isinstance(n, ast.FunctionDef) and n.name == name
        )
        calls = {
            n.func.attr
            for n in ast.walk(function)
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
        }
        assert "qualification_view" in calls
        assert not {n.id for n in ast.walk(function) if isinstance(n, ast.Name)} & {
            "verdict_matches",
            "recorded_identity",
        }


def test_adoption_never_dates_a_sparse_previous_pass_after_a_known_failure(tmp_path):
    from spark_sweep.lifecycle import RecipeStatus
    from spark_sweep.state import State

    sweep, _, clock = evidence_sweep(tmp_path)
    saved = historical()
    sweep.state.entry(KEY).update(
        status=RecipeStatus.FAILED,
        content_sha256=A.content_sha256,
        verdict_identity=A.model_dump(mode="json"),
        failure_class="smoke-assertion",
        node_count=1,
        finished_at=NOW.timestamp(),
        previous={**saved, "node_count": 1},
    )
    sweep.state.data.pop("qualification_evidence", None)
    clock.t = (NOW + timedelta(days=1)).timestamp()
    sweep.state.save()
    sweep.state = State.load(sweep.state.path, clock.now)
    sweep.qualification_evidence[KEY] = Evidence.model_validate(
        sweep.state.data["qualification_evidence"][KEY]
    )
    view = sweep.qualification_view(sweep.recipes[KEY])
    assert view is not None and view.status == Status.FAILED_RECIPE
    assert view.last_pass is not None


def test_repository_evidence_only_names_current_catalog_recipes():
    from spark_sweep.qualification import catalog_recipe_keys

    keys = catalog_recipe_keys(ROOT)
    for path in (ROOT / "qualification/evidence").rglob("*.json"):
        assert (
            path.relative_to(ROOT / "qualification/evidence").with_suffix("").as_posix()
            in keys
        )


@pytest.mark.parametrize(
    "outcome,blame",
    [
        (Outcome.PASSED, None),
        (Outcome.FAILED, Blame.RECIPE),
        (Outcome.FAILED, Blame.PLATFORM),
    ],
)
def test_unknown_image_library_publication_is_not_a_qualification_change(
    tmp_path, outcome, blame
):
    from spark_sweep.lifecycle import RecipeStatus

    sweep, _, _ = evidence_sweep(tmp_path)
    current = replace(
        sweep.recipes[KEY], runtime_image=None, library_commit="new-library"
    )
    unknown = A.model_copy(update={"runtime_image": None})
    sweep.qualification_evidence[KEY] = evidence(record(unknown, outcome, blame))
    status = RecipeStatus.PASSED if outcome == Outcome.PASSED else RecipeStatus.FAILED
    sweep.state.entry(KEY)["status"] = status
    sweep._reconcile_entry(current, 2)
    assert sweep.state.status(KEY) == status
    view = sweep.qualification_view(current)
    assert view is not None and not view.changed
    if view.last_pass is not None:
        assert view.last_pass.identity.runtime_image is None


@pytest.mark.parametrize("schema", [1, 2])
def test_one_load_adopts_schema_and_verdicts_without_losing_accepted_effects(
    tmp_path, monkeypatch, schema
):
    from vonk_forge_contracts.sweep import SweepAdoptionReason

    from spark_sweep import qualification
    from spark_sweep import state as state_module
    from spark_sweep.lifecycle import RecipeStatus

    path = tmp_path / "state.json"
    saved = {
        "schema": schema,
        "nonce": "adoption-test",
        "updated_at": NOW.timestamp(),
        "recipes": {
            KEY: {
                "status": RecipeStatus.PENDING,
                "previous": {
                    "status": RecipeStatus.PASSED,
                    "content_sha256": A.content_sha256,
                    "verdict_identity": A.model_dump(mode="json"),
                    "node_count": 1,
                    "finished_at": NOW.timestamp(),
                },
            },
        },
        "loads": {"accepted": {"request_key": "accepted", "deadline": 123}},
        "downloads": {KEY: {"request_key": "download", "deadline": 456}},
    }
    if schema == 1:
        saved["load"] = saved["loads"].pop("accepted")
        saved["recipes"]["vonk-forge/obsolete"] = {
            "status": RecipeStatus.FAILED,
            "failure_class": "timeout",
            "attempts": 7,
        }
    path.write_text(json.dumps(saved))
    before = path.read_bytes()
    adopter = state_module._adopt
    evidence_adopter = qualification.adopt_state_evidence
    calls = []

    def adopt(*args):
        calls.append("state")
        return adopter(*args)

    def adopt_evidence(*args):
        calls.append("evidence")
        return evidence_adopter(*args)

    monkeypatch.setattr(state_module, "_adopt", adopt)
    monkeypatch.setattr(qualification, "adopt_state_evidence", adopt_evidence)
    adopted = state_module.State.load(path, clock=lambda: NOW.timestamp())
    assert calls == ["state", "evidence"]
    assert path.read_bytes() == before
    assert adopted.data["schema"] == state_module.SCHEMA
    assert adopted.data["loads"] == {
        "accepted": {"request_key": "accepted", "deadline": 123}
    }
    assert adopted.downloads == saved["downloads"]
    imported = Evidence.model_validate(adopted.data["qualification_evidence"][KEY])
    assert derive_status(imported, A, 2, False).status == Status.WORKING
    if schema == 1:
        assert adopted.entry("vonk-forge/obsolete")["status"] == RecipeStatus.PENDING
        assert adopted.adoptions[0].reason == SweepAdoptionReason.LOAD_RULES
        assert adopted.adoptions[0].previous["attempts"] == 7
    # A crash before save repeats the same adoption without manufacturing proof.
    interrupted = state_module.State.load(path, clock=lambda: NOW.timestamp())
    assert interrupted.data == adopted.data
    adopted.save()
    calls.clear()
    reloaded = state_module.State.load(path, clock=lambda: NOW.timestamp())
    assert calls == ["state"]
    assert reloaded.data == adopted.data
    assert not reloaded.adoptions


@pytest.mark.parametrize("field", ["unchanged", "recipe", "image", "model", "topology"])
def test_missing_historical_spark_count_preserves_only_matching_content(field):
    historical = A.model_copy(update={"required_sparks": None})
    saved = record().model_copy(update={"identity": historical})
    current = {
        "unchanged": A,
        "recipe": B,
        "image": A.model_copy(update={"runtime_image": None}),
        "model": A.model_copy(update={"model_digests": ("f" * 64,)}),
        # A real topology change changes the recipe content digest too.
        "topology": B.model_copy(update={"required_sparks": 2}),
    }[field]
    view = derive_status(evidence(saved), current, 2, False)
    assert (view.status == Status.WORKING) == (field == "unchanged")
    assert view.last_pass == saved
    # Missing historical provenance never becomes public certification.
    assert published_status(evidence(saved), current).status != Status.WORKING


def test_explicit_historical_topology_mismatch_cannot_restore_working():
    saved = record()
    current = A.model_copy(update={"required_sparks": 2})
    assert derive_status(evidence(saved), current, 2, False).status == Status.RECHECK


def test_sparse_topology_failure_invalidates_older_public_proof():
    failed = record(A, Outcome.FAILED, Blame.RECIPE, day=1).model_copy(
        update={"identity": A.model_copy(update={"required_sparks": None})}
    )
    partial = record(day=2).model_copy(
        update={"platform": PLATFORM.model_copy(update={"agent_version": None})}
    )
    history = evidence(record(), failed, partial)
    assert derive_status(history, A, 2, False).status == Status.WORKING
    assert published_status(history, A).status == Status.NOT_YET_VERIFIED


def test_real_projected_library_and_evidence_restore_requeued_pass(tmp_path):
    from sweep_fakes import make_sweep

    from spark_sweep.identity import content_identity, recorded_identity
    from spark_sweep.lifecycle import RecipeStatus, RetestReason
    from spark_sweep.state import State
    from spark_sweep.vonkctl import Reply

    # Identity-only slices of the owner's five-row Controller GET, plus a d3 GET
    # row from the same projected API. No document-shaped compatibility reader.
    page = json.loads(
        (ROOT / "tests/fixtures/sweep-projected-library-2026-10-21.json").read_text()
    )
    sweep, fleet, _ = make_sweep(tmp_path, [], [])
    assert sweep.recipe_list.step(reply=Reply((), 0, page, ""))
    assert sweep.recipe_list.invalid_rows == 0
    for row in page["recipes"]:
        recipe = sweep.recipes[row["selector"]]
        identity = current_identity(recipe)
        assert identity.content_sha256 == row["identity"]["content_sha256"]
        assert identity.model_digests == tuple(
            sorted({model["content_sha256"] for model in row["projected"]["models"]})
        )
        assert identity.required_sparks == row["projected"]["topology"]["node_count"]
        assert identity.runtime_image is not None
        assert identity.runtime_image.build_key == row["prebuilt_image"]["build_key"]
        assert (
            identity.runtime_image.image_digest
            == row["prebuilt_image"]["reference"].partition("@")[2]
        )
        assert content_identity(recipe).model_digests == identity.model_digests

    recipe = sweep.recipes["vonk-forge/d3-vllm-single"]
    proof = read_evidence(ROOT, recipe.key)
    # Pin the actual incident measurement; later evidence publication must not
    # turn this historical boundary regression into a live catalog assertion.
    passed = next(
        record
        for record in proof.records
        if "83d67a9b089b3b92274634e0877cd52257b6845085b085682509abf1f5cd40e0"
        in record.evidence_refs
    )
    proof = proof.model_copy(update={"records": (passed,)})
    sweep.qualification_evidence[recipe.key] = proof
    assert passed.identity.runtime_image is None  # actual historical provenance
    assert passed.identity.model_digests == recipe.model_digests
    assert passed.identity.content_sha256 == recipe.content_sha256
    assert passed.identity.required_sparks == recipe.node_count
    entry = sweep.state.entry(recipe.key)
    entry.update(
        status=RecipeStatus.PENDING,
        revalidate=True,
        requeued_because=RetestReason.PASSED_OLDER_REVISION,
        attempts=0,
    )
    sweep.state.save()
    sweep.state = State.load(sweep.state.path)
    entry = sweep.state.entry(recipe.key)
    sweep._reconcile_entry(recipe, 2)
    assert entry["status"] == RecipeStatus.PASSED
    assert entry["attempts"] == 0
    assert entry["revalidate"] is False
    assert "requeued_because" not in entry
    view = sweep.qualification_view(recipe)
    assert view.status == Status.WORKING
    assert view.last_pass == passed
    restored = recorded_identity(entry)
    assert restored is not None and restored.runtime_image is None
    assert (
        published_status(proof, current_identity(recipe)).status
        == Status.NOT_YET_VERIFIED
    )
    assert not fleet.calls  # no inference, loads, downloads or remote reads
    assert not (tmp_path / "results.jsonl").exists()  # no manufactured measurement


@pytest.mark.parametrize(
    "changed", ["unchanged", "recipe", "model", "topology", "unknown-models"]
)
def test_unrecorded_image_is_not_a_change_or_invented_provenance(changed):
    saved_identity = A.model_copy(update={"runtime_image": None})
    saved = record(saved_identity)
    current = {
        "unchanged": A,
        "recipe": B,
        "model": A.model_copy(update={"model_digests": ("f" * 64,)}),
        "topology": A.model_copy(update={"required_sparks": 2}),
        "unknown-models": A.model_copy(update={"models_resolved": False}),
    }[changed]
    view = derive_status(evidence(saved), current, 2, False)
    assert (view.status == Status.WORKING) == (changed == "unchanged")
    assert view.last_pass == saved
    assert Change.IMAGE not in view.changed
    assert published_status(evidence(saved), current).status != Status.WORKING


def test_unbound_pass_cannot_hide_a_recorded_image_change():
    unbound = record(A.model_copy(update={"runtime_image": None}))
    pinned = record(day=1)
    current = A.model_copy(
        update={
            "runtime_image": Image(
                build_key="d" * 64, image_digest="sha256:" + "e" * 64
            )
        }
    )
    view = derive_status(evidence(unbound, pinned), current, 2, False)
    assert view.status == Status.RECHECK
    assert view.changed == (Change.IMAGE,)
    # Exact pin revert restores the original measurement, not the unbound pass.
    restored = derive_status(evidence(unbound, pinned), A, 2, False)
    assert restored.status == Status.WORKING
    assert restored.last_pass == pinned


def test_adapter_only_fix_invalidates_a_failure_without_a_prebuilt_image():
    # A recipe built on the Spark has no published image; source still changes
    # what it builds. Reusing the document-only verdict would suppress retests.
    previous = A.model_copy(
        update={"runtime_image": None, "source_bundle_sha256": "1" * 64}
    )
    current = previous.model_copy(update={"source_bundle_sha256": "2" * 64})
    failed = record(previous, Outcome.FAILED, Blame.RECIPE)
    view = derive_status(evidence(failed), current, 1, False)
    assert view.status == Status.FIX_TO_VERIFY
    assert view.changed == (Change.SOURCE,)
    assert (
        derive_status(evidence(failed), previous, 1, False).status
        == Status.FAILED_RECIPE
    )


def test_pass_recorded_before_source_digests_stays_working():
    # Evidence from before the index carried build-context digests has no
    # source provenance; missing history is unknown, never a changed pin.
    legacy = A.model_copy(update={"source_bundle_sha256": None})
    current = A.model_copy(update={"source_bundle_sha256": "1" * 64})
    passed = record(legacy)
    view = derive_status(evidence(passed), current, 2, False)
    assert view.status == Status.WORKING
    assert view.changed == ()
    # Once a digest is recorded, a later adapter-only change is a change.
    bound = record(current)
    moved = current.model_copy(update={"source_bundle_sha256": "2" * 64})
    changed = derive_status(evidence(bound), moved, 2, False)
    assert changed.status == Status.RECHECK
    assert changed.changed == (Change.SOURCE,)


def test_a_same_content_deferral_is_not_requeued_as_an_older_revision(
    tmp_path, monkeypatch
):
    from spark_sweep import identity
    from spark_sweep.lifecycle import RecipeStatus

    sweep, _, _ = evidence_sweep(tmp_path)
    failed = record(A, Outcome.FAILED, Blame.PLATFORM)
    sweep.qualification_evidence[KEY] = evidence(failed)
    sweep.state.entry(KEY).update(status=RecipeStatus.DEFERRED)
    # The stored record has no fingerprint; today's identity of the same content has one.
    monkeypatch.setattr(identity, "_fingerprint_identity", lambda recipe: "f" * 64)
    sweep._reconcile_entry(sweep.recipes[KEY], 2)
    assert "requeued_because" not in sweep.state.entry(KEY)
