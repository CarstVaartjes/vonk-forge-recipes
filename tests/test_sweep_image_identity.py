"""Verdicts and cache reuse certify the accepted runtime, independently of recipe JSON."""

from __future__ import annotations

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
        digests=("d" * 64,),
        content="e" * 64,
        fail_load="runtime_process_exited: exit_cause=bad_arguments exception=runtime exited",
        fail_phase="start",
    )
    sweep, fleet, _ = make_sweep(
        tmp_path, [fake], [FakeModel("d" * 64)], gateway=gateway
    )
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
    again, _, _ = make_sweep(tmp_path, [fake], [FakeModel("d" * 64)], fleet=fleet)
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
    assert entry["content_sha256"] == "e" * 64
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


def test_image_projection_reads_signed_pin_and_ignores_registry_alias():
    row = {
        "selector": "vonk-forge/runtime",
        "projected": {},
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
