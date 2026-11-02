"""One-time proof for the display split: every recipe keeps its status.

Deleted with ``tools/split-display``. It compares the status computed from the
base revision's files with the status computed from this checkout's, for every
recipe, with the identity of ``tools/build-catalog-index`` (document digest,
Model digests, Spark count; the runtime image and source bundle of the recorded
run, which the split does not touch).
"""

from __future__ import annotations

import json
import subprocess
import tarfile
from io import BytesIO
from pathlib import Path

import pytest
from vonk_forge_contracts import (
    QualificationIdentity,
    QualificationOutcome,
    RecipeDefinition,
    RecipeQualificationEvidence,
    derive_status,
    document_sha256,
    published_status,
    read_recipe,
)

from spark_sweep.qualification import read_evidence
from spark_sweep.run import retest_reason

ROOT = Path(__file__).resolve().parents[1]


def _base_tree(tmp_path: Path) -> Path:
    try:
        listing = subprocess.run(
            ["git", "ls-tree", "--name-only", "origin/main", "display/"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        if listing.strip():
            pytest.skip("base already split")
        archive = subprocess.run(
            ["git", "archive", "origin/main", "recipes", "qualification/evidence"],
            cwd=ROOT,
            check=True,
            capture_output=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        pytest.skip("origin/main is unavailable")
    with tarfile.open(fileobj=BytesIO(archive)) as tar:
        tar.extractall(tmp_path, filter="data")
    return tmp_path


def _identity(
    recipe: RecipeDefinition, digest: str, evidence: RecipeQualificationEvidence
) -> QualificationIdentity:
    models = tuple(sorted({b.model.content_sha256 for b in recipe.models}))
    recorded = [
        r
        for r in evidence.records
        if r.identity.content_sha256 == digest
        and r.identity.model_digests == models
        and r.identity.resolved
    ]
    # The run a Working verdict stands on, when there is one.
    passed = [r for r in recorded if r.outcome == QualificationOutcome.PASSED]
    reference = (passed or recorded)[-1].identity if recorded else None
    return QualificationIdentity(
        content_sha256=digest,
        source_bundle_sha256=reference.source_bundle_sha256 if reference else None,
        runtime_image=reference.runtime_image if reference else None,
        model_digests=models,
        required_sparks=recipe.topology.node_count,
    )


def _digest_free(value: object) -> object:
    """A status view without the digests the split changes by design."""

    if isinstance(value, dict):
        return {
            k: _digest_free(v)
            for k, v in value.items()
            if k not in {"content_sha256", "model_digests", "quality_notes"}
        }
    if isinstance(value, list):
        return [_digest_free(v) for v in value]
    return value


def test_every_recipe_keeps_its_status_across_the_split(tmp_path: Path) -> None:
    base = _base_tree(tmp_path)
    compared = working = 0
    for path in sorted((ROOT / "recipes").glob("*.json")):
        old_bytes = json.loads((base / "recipes" / path.name).read_text())
        new_bytes = json.loads(path.read_text())
        before, after = read_recipe(old_bytes), read_recipe(new_bytes)
        key = f"{after.identity.publisher}/{after.identity.slug}"
        old_evidence = read_evidence(base, key)
        new_evidence = read_evidence(ROOT, key)
        old_id = _identity(before, document_sha256(old_bytes), old_evidence)
        new_id = _identity(after, document_sha256(new_bytes), new_evidence)
        assert old_id.required_sparks == new_id.required_sparks
        views = (
            lambda e, i: published_status(e, i).model_dump(mode="json"),
            lambda e, i: derive_status(e, i, i.required_sparks or 1, False).model_dump(
                mode="json"
            ),
        )
        twinned = any(
            r.identity.content_sha256 == old_id.content_sha256
            for r in old_evidence.records
        )
        for view in views:
            old = _digest_free(view(old_evidence, old_id))
            new = _digest_free(view(new_evidence, new_id))
            assert isinstance(old, dict) and isinstance(new, dict)
            if not twinned:
                # Evidence only on older digests names the Models of its run, whose
                # digests changed by design: only that hint may differ.
                for shown in (old, new):
                    shown["changed"] = [c for c in shown["changed"] if c != "model"]
            # The whole view, not only the status: the hint of what changed and
            # the proof or failure it stands on are the same across the split.
            assert old == new, key
            working += old["status"] == "Working"
        old_view = derive_status(
            old_evidence, old_id, old_id.required_sparks or 1, False
        )
        new_view = derive_status(
            new_evidence, new_id, new_id.required_sparks or 1, False
        )
        # Which record the view stands on is the same record, seen against the
        # same identity: it names the current document exactly when it did.
        for record in ("last_pass", "last_failure"):
            old_record, new_record = (
                getattr(old_view, record),
                getattr(new_view, record),
            )
            assert (old_record is None) == (new_record is None), key
            if old_record is not None and new_record is not None:
                assert (
                    old_record.identity.content_sha256 == old_id.content_sha256
                ) == (new_record.identity.content_sha256 == new_id.content_sha256), (
                    key,
                    record,
                )
        # The sweep requeues from this decision alone.
        assert retest_reason(old_view, old_id) == retest_reason(new_view, new_id), key
        compared += 1
    assert compared == len(list((base / "recipes").glob("*.json")))
    assert working
