"""Verdicts and cache reuse certify the accepted runtime, independently of recipe JSON."""

from __future__ import annotations

import ast
import json
from dataclasses import replace
from pathlib import Path

import pytest
from sweep_fakes import FakeModel, FakeRecipe, Gateway, make_sweep
from test_sweep_policy import recipe

from spark_sweep import policy
from spark_sweep.catalog import parse_recipe
from spark_sweep.lifecycle import RecipeStatus, RetestReason

pytest_plugins = ["sweep_bounds"]
pytestmark = pytest.mark.usefixtures("bounded_clock")


@pytest.fixture
def gateway():
    gateway = Gateway()
    yield gateway
    gateway.stop()


def test_failed_verdict_requeues_when_only_the_published_image_changes(
    tmp_path: Path, gateway: Gateway
) -> None:
    fake = FakeRecipe(
        "runtime",
        fail_load="runtime exited",
        fail_phase="start",
        fail_code="recipe.runtime_exit",
    )
    sweep, fleet, _ = make_sweep(tmp_path, [fake], [FakeModel("m1")], gateway=gateway)
    original_row = fleet._recipe_row
    image = {
        "build_key": "a" * 64,
        "reference": "ghcr.io/test/runtime@sha256:" + "b" * 64,
    }

    def row(r):
        return {**original_row(r), "prebuilt_image": dict(image)}

    fleet._recipe_row = row
    sweep.run()
    entry = sweep.state.entry(fake.key)
    assert entry["status"] == RecipeStatus.FAILED
    again, _, _ = make_sweep(tmp_path, [fake], [FakeModel("m1")], fleet=fleet)
    again.refresh_catalog()
    assert again.state.entry(fake.key)["status"] == RecipeStatus.FAILED
    image["reference"] = "ghcr.io/test/runtime@sha256:" + "c" * 64
    fake.fail_load = None
    again.refresh_catalog()
    entry = again.state.entry(fake.key)
    assert entry["status"] == RecipeStatus.PENDING
    assert entry["requeued_because"] == RetestReason.FAILED_OLDER_REVISION
    assert again.run() == 0
    entry = again.state.entry(fake.key)
    assert entry["status"] == RecipeStatus.PASSED
    assert entry["content_sha256"] == "c1"
    assert (
        entry["verdict_identity"]["runtime_image"]["image_digest"]
        == "sha256:" + "c" * 64
    )
    rows = [
        json.loads(line)
        for line in (tmp_path / "results.jsonl").read_text().splitlines()
    ]
    passed = next(row for row in rows if row["status"] == RecipeStatus.PASSED)
    assert passed["verdict_identity"] == entry["verdict_identity"]
    previous = entry["previous"]["verdict_identity"]
    assert previous["runtime_image"]["image_digest"] == "sha256:" + "b" * 64
    report = json.loads((tmp_path / "report.json").read_text())
    assert report["recipes"][0]["verdict_identity"] == entry["verdict_identity"]
    status = json.loads((tmp_path / "status.json").read_text())
    assert status["verdict_identities"][fake.key] == entry["verdict_identity"]
    assert "sha256:" + "c" * 64 in (tmp_path / "report.md").read_text()


@pytest.mark.parametrize(
    "changed", ["digest", "build", "model", "document", "library", "unchanged"]
)
@pytest.mark.parametrize(
    "status", [RecipeStatus.FAILED, RecipeStatus.DEFERRED, RecipeStatus.PASSED]
)
def test_full_identity_drives_requeue(changed, status):
    from spark_sweep.catalog_contract import RuntimeImageIdentity
    from spark_sweep.identity import content_identity

    accepted = recipe(
        "runtime",
        runtime_image=RuntimeImageIdentity(
            build_key="a" * 64, image_digest="sha256:" + "b" * 64
        ),
    )
    assert accepted.runtime_image is not None
    current = accepted
    if changed == "digest":
        current = replace(
            accepted,
            runtime_image=accepted.runtime_image.model_copy(
                update={"image_digest": "sha256:" + "c" * 64}
            ),
        )
    elif changed == "build":
        current = replace(
            accepted,
            runtime_image=accepted.runtime_image.model_copy(
                update={"build_key": "d" * 64}
            ),
        )
    elif changed == "model":
        current = replace(accepted, model_digests=("another-model",))
    elif changed == "document":
        current = replace(accepted, content_sha256="c2")
    elif changed == "library":
        current = replace(accepted, library_commit="new-library")
    entry = {
        "status": status,
        "content_sha256": accepted.content_sha256,
        "verdict_identity": content_identity(accepted).model_dump(mode="json"),
    }
    assert (policy.requeue_reason(entry, current) is not None) == (
        changed not in {"library", "unchanged"}
    )


@pytest.mark.parametrize(
    "status", [RecipeStatus.PASSED, RecipeStatus.FAILED, RecipeStatus.DEFERRED]
)
@pytest.mark.parametrize("revalidate", [False, True])
@pytest.mark.parametrize("identity_form", ["absent", "no-image-field", "unknown-image"])
def test_legacy_unknown_images_preserve_passes_and_retry_unsuccessful_verdicts(
    tmp_path, status, revalidate, identity_form
):
    fake = FakeRecipe("legacy")
    sweep, _, _ = make_sweep(
        tmp_path, [fake], [FakeModel("m1")], revalidate_passes=revalidate
    )
    sweep.refresh_catalog()
    entry = sweep.state.entry(fake.key)
    entry.update(status=status, content_sha256="c1")
    if identity_form != "absent":
        from spark_sweep.identity import content_identity

        identity = content_identity(sweep.recipes[fake.key]).model_dump(mode="json")
        identity.pop("runtime_image")
        if identity_form == "unknown-image":
            identity["runtime_image"] = None
        entry["verdict_identity"] = identity
    sweep._reconcile_entry(sweep.recipes[fake.key], 2)
    expected = (
        RecipeStatus.PASSED if status == RecipeStatus.PASSED else RecipeStatus.PENDING
    )
    assert entry["status"] == expected
    if status == RecipeStatus.PASSED:
        from spark_sweep.identity import content_identity

        # Unknown historical image provenance must never be filled from today's catalog.
        assert not (entry.get("verdict_identity") or {}).get("runtime_image")
        sweep.state.save()
        again, _, _ = make_sweep(tmp_path, [fake], [FakeModel("m1")])
        again.refresh_catalog()
        assert again.state.entry(fake.key)["status"] == RecipeStatus.PASSED


def test_requeue_and_removal_decisions_use_the_shared_identity_helper():
    root = Path(__file__).resolve().parents[1]
    for filename, name in [("policy.py", "requeue_reason"), ("removal.py", "tick")]:
        tree = ast.parse((root / "spark_sweep" / filename).read_text())
        function = next(
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef) and node.name == name
        )
        assert any(
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "verdict_matches"
            for node in ast.walk(function)
        )
        assert not any(
            isinstance(node, ast.Compare)
            and any(
                isinstance(part, ast.Constant) and part.value == "content_sha256"
                for part in ast.walk(node)
            )
            for node in ast.walk(function)
        )


def test_image_projection_reads_signed_pin_and_ignores_registry_alias():
    row = {
        "selector": "vonk-forge/runtime",
        "document": {},
        "prebuilt_image": {
            "build_key": "a" * 64,
            "reference": "ghcr.io/test/runtime@sha256:" + "b" * 64,
        },
    }
    first = parse_recipe(row)
    assert first is not None and first.runtime_image is not None
    assert first.runtime_image.build_key == "a" * 64
    row = {
        **row,
        "prebuilt_image": {
            "build_key": "a" * 64,
            "reference": "mirror.test/runtime@sha256:" + "b" * 64,
        },
    }
    second = parse_recipe(row)
    assert second is not None and second.runtime_image == first.runtime_image


def test_changed_content_is_ordered_newest_first():
    newest = recipe("z-newest", updated_at="2026-10-21", artifact_bytes=80)
    older = recipe("a-older", updated_at="2026-10-20", artifact_bytes=1)
    recipes = {item.key: item for item in [older, newest]}
    plans = [policy.GroupPlan(frozenset({"m"}), tuple(recipes), 0)]
    assert policy.order_queue(
        plans, recipes, cached={"m"}, sizes={}, refreshed=set(recipes)
    ) == [newest.key, older.key]


def test_controller_without_image_pin_invalidates_on_new_library_publication():
    from spark_sweep.identity import content_identity

    row = {"selector": "vonk-forge/runtime", "document": {}}
    accepted = parse_recipe(row, "old-library")
    same = parse_recipe(row, "old-library")
    changed = parse_recipe(row, "new-library")
    assert accepted is not None and same is not None and changed is not None
    entry = {
        "status": RecipeStatus.FAILED,
        "verdict_identity": content_identity(accepted).model_dump(mode="json"),
    }
    assert policy.requeue_reason(entry, same) is None
    assert policy.requeue_reason(entry, changed) == RetestReason.FAILED_OLDER_REVISION
    assert entry["verdict_identity"]["runtime_image"] is None


def test_evidence_export_keeps_verdicts_for_different_images(tmp_path):
    from spark_sweep.catalog_contract import RuntimeImageIdentity
    from spark_sweep.identity import content_identity
    from spark_sweep.report import export_evidence

    first = recipe(
        "runtime",
        content_sha256="e" * 64,
        model_digests=("f" * 64,),
        runtime_image=RuntimeImageIdentity(
            build_key="a" * 64, image_digest="sha256:" + "b" * 64
        ),
    )
    second = replace(
        first,
        runtime_image=RuntimeImageIdentity(
            build_key="c" * 64, image_digest="sha256:" + "d" * 64
        ),
    )
    rows = [
        {
            "recipe": item.key,
            "step": "smoke",
            "recorded_at": "2026-10-21T00:00:00Z",
            "status": RecipeStatus.PASSED,
            "content_sha256": item.content_sha256,
            "verdict_identity": content_identity(item).model_dump(mode="json"),
        }
        for item in (first, second)
    ]
    written = export_evidence(rows, tmp_path, [first.key])
    assert len(written) == 1
    from vonk_forge_contracts import RecipeQualificationEvidence

    assert (
        len(
            RecipeQualificationEvidence.model_validate_json(
                written[0].read_text()
            ).records
        )
        == 2
    )


@pytest.mark.parametrize(
    "status", [RecipeStatus.PASSED, RecipeStatus.FAILED, RecipeStatus.DEFERRED]
)
@pytest.mark.parametrize("revalidate", [False, True])
def test_library_commit_only_refresh_preserves_passes_by_default(
    tmp_path, status, revalidate
):
    from spark_sweep.identity import content_identity

    fake = FakeRecipe("runtime")
    sweep, _, _ = make_sweep(
        tmp_path, [fake], [FakeModel("m1")], revalidate_library_passes=revalidate
    )
    sweep.refresh_catalog()
    accepted = replace(sweep.recipes[fake.key], runtime_image=None)
    entry = sweep.state.entry(fake.key)
    entry.update(
        status=status,
        content_sha256=accepted.content_sha256,
        verdict_identity=content_identity(accepted).model_dump(mode="json"),
    )
    current = replace(accepted, library_commit="new-library")
    sweep._reconcile_entry(current, 2)
    assert entry["status"] == (
        RecipeStatus.PASSED
        if status == RecipeStatus.PASSED and not revalidate
        else RecipeStatus.PENDING
    )


def test_library_pass_revalidation_requires_the_explicit_cli_flag():
    from spark_sweep.cli import _config, _parser

    parser = _parser()
    assert not _config(parser.parse_args(["run"])).revalidate_library_passes
    assert _config(
        parser.parse_args(["run", "--revalidate-passes"])
    ).revalidate_library_passes
