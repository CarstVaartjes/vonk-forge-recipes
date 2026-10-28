"""Fault injection at publication: missing builds remain pending; stale pins must fail."""

from __future__ import annotations

import importlib.machinery
import importlib.util
import json
import os
import subprocess
import sys
import textwrap
import urllib.error
from pathlib import Path
from typing import Any

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
    row = index["recipes"][0]
    if fault == "missing":
        row.pop("prebuilt_image")
    if fault == "pending":
        row.pop("prebuilt_image")
        images = {}
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
    result = _check(tmp_path, plan, index, images)
    assert (result.returncode == 0) == (fault in {None, "pending"}), result.stderr
    if fault == "pending":
        assert f"Image pending: test: {KEY}" in result.stdout


def _check(
    tmp_path: Path, plan: Any, index: Any, images: Any
) -> subprocess.CompletedProcess[str]:
    for name, value in [("plan", plan), ("index", index), ("images", images)]:
        (tmp_path / f"{name}.json").write_text(json.dumps(value))
    return subprocess.run(
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


def _prebuilt_images() -> Any:
    loader = importlib.machinery.SourceFileLoader(
        "prebuilt_images_guard", str(ROOT / "tools/prebuilt-images")
    )
    spec = importlib.util.spec_from_loader(loader.name, loader)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    loader.exec_module(module)
    return module


@pytest.mark.parametrize("noise", ["not-found", "rate-limited"])
def test_lookup_noise_ships_the_recipe_pending_and_a_later_publish_pins_it(
    tmp_path: Path, noise: str
) -> None:
    # The image exists, but resolve sees no digest: ghcr answers 401/403/404 (the
    # lookup returns None) or keeps answering 429 (resolve logs it and records
    # None). Publication must not block the catalog; the next one pins it.
    tool = _prebuilt_images()
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
    digest = IMAGE["reference"].split("@", 1)[1]

    def noisy(_owner: str, _name: str, _tag: str) -> str | None:
        if noise == "rate-limited":
            raise urllib.error.HTTPError("https://ghcr.io", 429, "Too Many", None, None)  # type: ignore[arg-type]
        return None

    def publish(lookup: Any, attempt: Path) -> subprocess.CompletedProcess[str]:
        # The index pins exactly what resolve found (build-catalog-index).
        images, _missing = tool.resolve(plan, owner="o", lookup=lookup)
        row = {**recipe, "prebuilt_image": images.get(CONTENT)}
        attempt.mkdir()
        return _check(
            attempt, plan, {"source_commit": COMMIT, "recipes": [row]}, images
        )

    first = publish(noisy, tmp_path / "first")
    assert first.returncode == 0, first.stderr
    assert f"Image pending: test: {KEY}" in first.stdout
    later = publish(lambda _o, _n, _t: digest, tmp_path / "later")
    assert later.returncode == 0, later.stderr
    assert "Image pending" not in later.stdout
    pinned = json.loads((tmp_path / "later/index.json").read_text())["recipes"][0]
    assert pinned["prebuilt_image"]["reference"].endswith("@" + digest)


def test_publication_requires_image_proof_and_verified_artifacts() -> None:
    workflow = (ROOT / ".github/workflows/publish.yml").read_text()
    publish = workflow.split("\n  publish:\n")[1]
    assert "needs.images.result == 'success'" in publish
    assert "!cancelled()" in publish
    artifact = publish.split("- name: Download the pinned prebuilt images")[1].split(
        "- name:"
    )[0]
    assert "continue-on-error" not in artifact
    assert "tools/check-published-images" in publish
    assert '--include "$RUNNER_TEMP/plan/plan.json"' in publish
    assert 'tee -a "$GITHUB_STEP_SUMMARY"' in publish
