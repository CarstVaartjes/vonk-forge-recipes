"""Fault injection at publication: missing builds remain pending; stale pins must fail."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
COMMIT = "1" * 40
CONTENT = "2" * 64
KEY = "3" * 64
IMAGE = {"reference": "ghcr.io/o/r@sha256:" + "4" * 64, "build_key": KEY}


def test_publication_cannot_succeed_without_its_image_artifact(tmp_path: Path) -> None:
    # Execute the actual release-build shell with harmless successful builders.
    # On main this prints a warning, omits --prebuilt-images and exits zero.
    workflow = (ROOT / ".github/workflows/publish.yml").read_text()
    step = workflow.split(
        "      - name: Build release assets from the release commit\n"
    )[1]
    script = textwrap.dedent(
        step.split("        run: |\n")[1].split("\n      - name:")[0]
    )
    result = subprocess.run(
        ["bash", "-c", script],
        cwd=ROOT,
        env={
            **os.environ,
            "RUNNER_TEMP": str(tmp_path),
            "RELEASE_TEST_PYTHON": "/usr/bin/true",
            "GITHUB_SHA": COMMIT,
        },
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0, result.stdout


@pytest.mark.parametrize(
    "fault",
    [
        None,
        "missing",
        "stale-key",
        "stale-digest",
        "wrong-commit",
        "wrong-content",
        "unresolved",
        "pending",
        "excluded",
        "resolved-wrong-key",
        "invalid-digest",
        "refused",
    ],
)
def test_release_must_pin_current_build_and_resolved_digest(
    tmp_path: Path, fault: str | None
) -> None:
    recipe = {
        "slug": "test",
        "source_path": "recipes/test.json",
        "content_sha256": CONTENT,
    }
    plan = {
        "source_commit": COMMIT,
        "images": [{"build_key": KEY, "recipes": [recipe]}],
        "skipped": [],
    }
    index = {
        "source_commit": COMMIT,
        "recipes": [{**recipe, "prebuilt_image": dict(IMAGE)}],
    }
    images = {CONTENT: IMAGE}
    exclusions = {}
    row = index["recipes"][0]
    if fault in {"missing", "excluded"}:
        row.pop("prebuilt_image")
    if fault in {"pending", "excluded"}:
        row.pop("prebuilt_image", None)
        images = {}
    if fault == "excluded":
        exclusions = {"test": "CI cannot build this image"}
    if fault == "resolved-wrong-key":
        images = {CONTENT: {**IMAGE, "build_key": "9" * 64}}
    if fault == "invalid-digest":
        images = {CONTENT: {**IMAGE, "reference": "ghcr.io/o/r:mutable"}}
    if fault == "refused":
        plan["skipped"] = [{"slug": "refused", "reason": "source policy"}]
    if fault == "stale-key":
        row["prebuilt_image"]["build_key"] = "5" * 64
    if fault == "stale-digest":
        row["prebuilt_image"]["reference"] = "ghcr.io/o/r@sha256:" + "6" * 64
    if fault == "wrong-commit":
        plan["source_commit"] = "7" * 40
    if fault == "wrong-content":
        row["content_sha256"] = "8" * 64
    if fault == "unresolved":
        images = {}
    for name, value in [
        ("plan", plan),
        ("index", index),
        ("images", images),
        ("prebuilt-image-exclusions", exclusions),
    ]:
        (tmp_path / f"{name}.json").write_text(json.dumps(value))
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "tools/check-published-images"),
            "--plan",
            "plan.json",
            "--index",
            "index.json",
            "--images",
            "images.json",
            "--source-commit",
            COMMIT,
        ],
        cwd=tmp_path,
        check=False,
        capture_output=True,
        text=True,
    )
    assert (result.returncode == 0) == (fault in {None, "pending", "excluded"}), (
        result.stderr
    )
    if fault in {"pending", "excluded"}:
        assert f"Image pending: test: {KEY}" in result.stdout


def test_publication_accepts_failed_builds_but_requires_verified_artifacts() -> None:
    workflow = (ROOT / ".github/workflows/publish.yml").read_text()
    publish = workflow.split("\n  publish:\n")[1]
    assert "needs.images.result" not in publish
    assert "!cancelled()" in publish
    artifact = publish.split("- name: Download the pinned prebuilt images")[1].split(
        "- name:"
    )[0]
    assert "continue-on-error" not in artifact
    assert "tools/check-published-images" in publish
    assert '--include "$RUNNER_TEMP/plan/plan.json"' in publish
    assert 'tee -a "$GITHUB_STEP_SUMMARY"' in publish
