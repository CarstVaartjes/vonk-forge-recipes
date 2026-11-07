"""Repository evidence boundary shared by backfill, sweep and publication."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field
from vonk_forge_contracts import (
    QualificationBlame,
    QualificationCauseCode,
    QualificationDerivationInput,
    QualificationEvidenceRecord,
    QualificationIdentity,
    QualificationMeasuredRoleMemory,
    QualificationOutcome,
    QualificationPlatform,
    QualificationQualityNote,
    QualificationRuntimeImage,
    QualificationStatus,
    QualificationStatusView,
    QualificationSummary,
    QualificationTopology,
    RecipeQualificationEvidence,
    derive_status,
    read_recipe,
)
from vonk_forge_contracts.qualification_status import (
    QUALIFICATION_SUMMARIES,
    QualificationEvidenceStep,
    QualificationSourceFailureClass,
    published_status,
)

from qualification import documents

from .catalog import Recipe, never_measured
from .identity import current_identity
from .lifecycle import RecipeStatus
from .state import write_atomic


class HistoricalIdentity(BaseModel):
    content_sha256: str
    source_bundle_sha256: str | None = None
    runtime_image: QualificationRuntimeImage | None = None
    model_digests: tuple[str, ...] | None = None


class HistoricalQualityCheck(BaseModel):
    ok: bool


class HistoricalSmokeResult(BaseModel):
    quality: tuple[HistoricalQualityCheck, ...] = ()


class HistoricalResult(BaseModel):
    """Input projection only. Infrastructure fields are counted, never copied."""

    recipe: str
    status: RecipeStatus
    recorded_at: datetime
    content_sha256: str
    recipe_revision_id: str | None = None
    # The revision the review bound to the run; absent in older rows.
    ran_revision: str | None = None
    verdict_identity: HistoricalIdentity | None = None
    model_digests: tuple[str, ...] | None = None
    controller_release: str | None = None
    controller_version: str | None = None
    node_count: int | None = Field(default=None, ge=1)
    node_ids: tuple[str, ...] = ()
    quality: tuple[HistoricalQualityCheck, ...] = ()
    result: HistoricalSmokeResult = Field(default_factory=HistoricalSmokeResult)
    failure_class: str | None = None
    code: str | None = None
    # What a pass measured, as evidence rows keyed (content_sha256, role).
    measured_memory: tuple[QualificationMeasuredRoleMemory, ...] = ()


def evidence_path(root: Path, recipe_key: str) -> Path:
    # Validate before constructing a path; authored keys cannot traverse directories.
    key = RecipeQualificationEvidence(recipe_key=recipe_key).recipe_key
    return root / "qualification" / "evidence" / f"{key}.json"


def read_evidence(root: Path, recipe_key: str) -> RecipeQualificationEvidence:
    path = evidence_path(root, recipe_key)
    if not path.exists():
        return RecipeQualificationEvidence(recipe_key=recipe_key)
    evidence = RecipeQualificationEvidence.model_validate_json(path.read_text())
    if evidence.recipe_key != recipe_key:
        raise ValueError(f"evidence key does not match {path.name}")
    return evidence


def merge_evidence(
    left: RecipeQualificationEvidence, right: RecipeQualificationEvidence
) -> RecipeQualificationEvidence:
    if left.recipe_key != right.recipe_key:
        raise ValueError("cannot merge different recipes")
    unique = {r.model_dump_json(): r for r in (*left.records, *right.records)}
    ordered = sorted(unique.values(), key=lambda r: (r.tested_at, r.model_dump_json()))
    rows: dict[tuple[str, str], QualificationMeasuredRoleMemory] = {}
    for row in (*left.measured_memory, *right.measured_memory):
        rows[row.key] = (
            merge_measurement(rows[row.key], row) if row.key in rows else row
        )
    return left.model_copy(
        update={
            "records": tuple(ordered),
            "measured_memory": tuple(rows[key] for key in sorted(rows)),
        }
    )


def merge_measurement(
    left: QualificationMeasuredRoleMemory, right: QualificationMeasuredRoleMemory
) -> QualificationMeasuredRoleMemory:
    """One (content, role) measured twice: the larger use, the smaller budget.

    Both are the conservative side for a reader asking whether a declaration
    covers what hardware used under no smaller budget; no budget is smallest.
    """
    a, b = left.budget_bytes, right.budget_bytes
    return left.model_copy(
        update={
            "steady_bytes": max(left.steady_bytes, right.steady_bytes),
            "budget_bytes": None if a is None or b is None else min(a, b),
        }
    )


def history_evidence(value: Mapping[str, Any]) -> RecipeQualificationEvidence | None:
    """One history row as evidence: its verdict record and, for a pass, its measurements."""
    record = from_history(value)
    if record is None:
        return None
    row = HistoricalResult.model_validate(value)
    return RecipeQualificationEvidence(
        recipe_key=row.recipe,
        records=(record,),
        measured_memory=tuple(sorted(row.measured_memory, key=lambda m: m.key))
        if record.outcome == QualificationOutcome.PASSED
        else (),
    )


def serving_rows(
    root: Path, evidence: RecipeQualificationEvidence
) -> RecipeQualificationEvidence:
    """The evidence a log or state may add to the repo: a job recipe carries no measured rows."""
    slug = evidence.recipe_key.partition("/")[2]
    if (
        not evidence.measured_memory
        or not (root / "recipes" / f"{slug}.json").is_file()
    ):
        return evidence
    if not never_measured(documents.recipe(slug, root)):
        return evidence
    return evidence.model_copy(update={"measured_memory": ()})


def write_evidence(root: Path, evidence: RecipeQualificationEvidence) -> bool:
    path = evidence_path(root, evidence.recipe_key)
    payload = evidence.model_dump_json(indent=2, exclude_none=True) + "\n"
    if path.exists() and path.read_text() == payload:
        return False
    write_atomic(path, payload)
    return True


def from_history(value: Mapping[str, Any]) -> QualificationEvidenceRecord | None:
    """Convert only actual verdicts; never copy free-form source strings to public data.

    Missing historical identity/context stays missing. In particular an image
    cannot be inferred from today's library or from the recipe's base image.
    """
    if value.get("step") != QualificationEvidenceStep.SMOKE:
        return None  # a cleanup/stop receipt is never hardware qualification
    if value.get("status") not in (
        RecipeStatus.PASSED,
        RecipeStatus.FAILED,
        RecipeStatus.DEFERRED,
    ):
        return None
    row = HistoricalResult.model_validate(value)
    if row.ran_revision and row.ran_revision != row.recipe_revision_id:
        return None  # the run executed other bytes than the identity this row names
    recorded = row.verdict_identity
    models = recorded.model_digests if recorded else row.model_digests
    identity = QualificationIdentity(
        content_sha256=recorded.content_sha256 if recorded else row.content_sha256,
        runtime_image=recorded.runtime_image if recorded else None,
        source_bundle_sha256=recorded.source_bundle_sha256 if recorded else None,
        model_digests=tuple(sorted(set(models or ()))),
        models_resolved=models is not None,
        required_sparks=row.node_count or len(row.node_ids) or None,
    )
    from .verdict import Blame
    from .verdict import blame as sweep_blame

    passed = row.status == RecipeStatus.PASSED
    blame = (
        None
        if passed
        else (
            QualificationBlame.RECIPE
            if row.status == RecipeStatus.FAILED
            and sweep_blame(row.failure_class or "", row.code) is Blame.RECIPE
            else QualificationBlame.PLATFORM
        )
    )
    causes = {
        None: (QualificationCauseCode.HARDWARE_PASS, QualificationSummary.PASSED),
        QualificationBlame.RECIPE: (
            QualificationCauseCode.RECIPE_FAILURE,
            QualificationSummary.RECIPE_FAILED,
        ),
        QualificationBlame.PLATFORM: (
            QualificationCauseCode.PLATFORM_FAILURE,
            QualificationSummary.PLATFORM_FAILED,
        ),
    }
    code, summary = causes[blame]
    if blame == QualificationBlame.RECIPE:
        try:
            candidate = QualificationCauseCode(row.code)
        except ValueError:
            fallback: dict[str, QualificationCauseCode] = {
                QualificationSourceFailureClass.SERVING_ASSERTION: QualificationCauseCode.SERVING_ASSERTION,
                QualificationSourceFailureClass.JOB_FAILED: QualificationCauseCode.JOB_FAILURE,
                QualificationSourceFailureClass.RECIPE_DATA: QualificationCauseCode.INVALID_DEFINITION,
            }
            candidate = fallback.get(
                row.failure_class or "", QualificationCauseCode.RECIPE_FAILURE
            )
        if candidate not in (
            QualificationCauseCode.HARDWARE_PASS,
            QualificationCauseCode.PLATFORM_FAILURE,
        ):
            code, summary = candidate, QUALIFICATION_SUMMARIES[candidate]
    # The digest links to source bytes without publishing an internal location/id.
    reference = hashlib.sha256(
        json.dumps(dict(value), sort_keys=True, default=str).encode()
    ).hexdigest()
    return QualificationEvidenceRecord(
        identity=identity,
        outcome=QualificationOutcome.PASSED if passed else QualificationOutcome.FAILED,
        blame=blame,
        cause_code=code,
        cause_summary=summary,
        tested_at=row.recorded_at,
        platform=QualificationPlatform(
            controller_source_sha=row.controller_release,
            controller_version=row.controller_version,
            spark_count=identity.required_sparks,
            topology=(
                QualificationTopology.SINGLE
                if identity.required_sparks == 1
                else QualificationTopology.MULTI
                if identity.required_sparks is not None
                else None
            ),
        ),
        evidence_refs=(reference,),
        quality_notes=(QualificationQualityNote.ANSWER_VARIANCE,)
        if any(not check.ok for check in (*row.quality, *row.result.quality))
        else (),
    )


def adopt_state_evidence(
    data: Mapping[str, Any],
) -> dict[str, RecipeQualificationEvidence]:
    """One load-time adoption of saved verdicts; never bind them to today's catalog.

    Sparse requeue snapshots retain unknown provenance. The saved completion time
    (or, if absent, the state's observation time) orders them; this is not a new
    hardware measurement. Subsequent loads consume only the adopted evidence.
    """
    adopted = {
        key: QualificationDerivationInput.model_validate(value).evidence
        for key, value in data.get("qualification_inputs", {}).items()
    }
    for key, entry in data.get("recipes", {}).items():
        for saved in (entry.get("previous", {}), entry):
            if saved.get("status") not in (
                RecipeStatus.PASSED,
                RecipeStatus.FAILED,
                RecipeStatus.DEFERRED,
            ):
                continue
            # A sparse previous snapshot predates the current result. Never give
            # it the state's newer timestamp and thereby override a later failure.
            timestamp = saved.get("finished_at") or (
                data.get("updated_at", 0) if saved is entry else 0
            )
            try:
                record = from_history(
                    {
                        **saved,
                        "recipe": key,
                        "step": QualificationEvidenceStep.SMOKE,
                        "recorded_at": datetime.fromtimestamp(
                            timestamp, UTC
                        ).isoformat(),
                    }
                )
            except (ValueError, TypeError, OverflowError):
                continue  # unusable provenance cannot certify a recipe
            if record is not None:
                evidence = RecipeQualificationEvidence(
                    recipe_key=key, records=(record,)
                )
                adopted[key] = merge_evidence(
                    adopted.get(key, RecipeQualificationEvidence(recipe_key=key)),
                    evidence,
                )
    return adopted


class BackfillResult(BaseModel):
    read: int = 0
    imported: int = 0
    rejected: int = 0
    changed_files: int = 0
    unpublished: int = 0  # resolved sweep records the repo lacked before this batch


def read_state_evidence(path: Path) -> dict[str, RecipeQualificationEvidence]:
    """The sweep's merged evidence from its saved state, read only.

    State.load already adopted the saved verdicts, so only the persisted
    `qualification_evidence` is read. Never use State.load here (its recovery
    path mutates source state). An entry that does not validate is skipped; an
    unreadable file prints its cause and publishes history as before.
    """
    try:
        raw = json.loads(path.read_text(encoding="utf-8")).get(
            "qualification_evidence", {}
        )
    except (OSError, ValueError, AttributeError) as error:
        print(f"evidence: sweep state unreadable: {type(error).__name__}", flush=True)
        return {}
    found = {}
    for key, value in raw.items():
        try:
            found[key] = RecipeQualificationEvidence.model_validate(value)
        except ValueError:
            continue
    return found


def resolved_missing(
    local: RecipeQualificationEvidence, repo: RecipeQualificationEvidence
) -> tuple[QualificationEvidenceRecord, ...]:
    """The one rule: resolved local records (same identity and outcome) the repo lacks."""
    known = {(r.identity.model_dump_json(), r.outcome) for r in repo.records}
    return tuple(
        r
        for r in local.records
        if r.identity.models_resolved
        and (r.identity.model_dump_json(), r.outcome) not in known
    )


def backfill(
    paths: Iterable[Path],
    root: Path,
    extra: Mapping[str, RecipeQualificationEvidence] | None = None,
) -> BackfillResult:
    """Append history rows and the sweep's merged evidence (`extra`) to the repo."""
    result = BackfillResult()
    grouped: dict[str, RecipeQualificationEvidence] = {}
    catalog = catalog_recipe_keys(root) if extra else set()
    for key, evidence in (extra or {}).items():
        if key not in catalog:
            continue  # withdrawn recipes are pruned, never counted
        previous = read_evidence(root, key)
        lacking = resolved_missing(evidence, previous)
        result.unpublished += len(lacking)
        if lacking:  # resolved records only; nothing is rewritten, only appended
            grouped[key] = merge_evidence(
                previous,
                serving_rows(root, evidence.model_copy(update={"records": lacking})),
            )
    for path in paths:
        # Read only. Never use State.load: its recovery path mutates source state.
        for line in path.read_text().splitlines():
            result.read += 1
            try:
                evidence = history_evidence(json.loads(line))
                if evidence is None:
                    continue
                key = evidence.recipe_key
            except (ValueError, TypeError, AttributeError):
                result.rejected += 1
                continue
            previous = grouped.get(key) or read_evidence(root, key)
            grouped[key] = merge_evidence(previous, serving_rows(root, evidence))
            result.imported += 1
    for evidence in grouped.values():
        result.changed_files += int(write_evidence(root, evidence))
    return result


def local_status(
    evidence: RecipeQualificationEvidence,
    recipe: Recipe,
    fleet_size: int,
    testing_now: bool = False,
    identity: QualificationIdentity | None = None,
) -> QualificationStatusView:
    return derive_status(
        evidence, identity or current_identity(recipe), fleet_size, testing_now
    )


STATUS_ORDER = {
    QualificationStatus.FIX_TO_VERIFY: 0,
    QualificationStatus.UNTESTED: 1,
    QualificationStatus.BLOCKED_PLATFORM: 2,
    QualificationStatus.RECHECK: 3,
}


def scheduling_priority(view: QualificationStatusView) -> int:
    return STATUS_ORDER.get(view.status, 4)


def catalog_recipe_keys(root: Path) -> set[str]:
    directory = root / "recipes"
    if not directory.is_dir():
        raise ValueError("evidence pruning requires the authored recipe catalog")
    keys = set()
    for path in directory.glob("*.json"):
        recipe = read_recipe(json.loads(path.read_text()))
        keys.add(f"{recipe.identity.publisher}/{recipe.identity.slug}")
    return keys


def prune_withdrawn_evidence(root: Path) -> list[Path]:
    """A complete authored catalog owns the current evidence set; history stays in git."""
    keys = catalog_recipe_keys(root)
    removed = []
    for path in (root / "qualification/evidence").rglob("*.json"):
        key = (
            path.relative_to(root / "qualification/evidence").with_suffix("").as_posix()
        )
        if key not in keys:
            path.unlink()
            removed.append(path)
    return removed


def unpublishable_pass(
    view: QualificationStatusView | None,
    evidence: RecipeQualificationEvidence | None,
    identity: QualificationIdentity,
    *,
    release_known: bool,
) -> bool:
    """A local Working pass the catalog would not publish as Working, though a run could fix it.

    The published derivation is the contracts package's (the one the catalog uses), asked
    only for a locally Working recipe. A re-run can publish only when the catalog pins a
    runtime image and the Controller release is readable (a run recorded now would
    otherwise lack it); otherwise the pass is left alone.
    """
    if (
        view is None
        or evidence is None
        or view.status != QualificationStatus.WORKING
        or not release_known
        or identity.runtime_image is None
    ):
        return False
    try:
        published = published_status(evidence, identity)
    except (ValueError, TypeError):
        return False
    return published.status != QualificationStatus.WORKING


def pass_marker(recipe: Recipe, identity: QualificationIdentity) -> str:
    image = identity.runtime_image
    return f"{recipe.content_sha256}:{image.image_digest if image else ''}"


def rerun_settled(
    evidence: RecipeQualificationEvidence,
    requeue: object,
    marker: str,
    release: str | None,
) -> bool:
    """The re-run asked for under this marker recorded a verdict that stands.

    Until then the request stands (a pending requeue is never undone). A behavioural record
    (a pass, or a failure blamed on the recipe) made after the request on a known Controller
    release settles it. A record blamed on the platform says nothing about the recipe: it
    settles the request only for the Controller release it carries, so the recipe is asked
    again once a different release is deployed. A record without a release does not settle it.
    """
    if not isinstance(requeue, Mapping) or requeue.get("marker") != marker:
        return False
    asked = datetime.fromtimestamp(float(requeue["at"]), UTC)
    for r in evidence.records:
        carried = r.platform.controller_source_sha or r.platform.controller_version
        if r.tested_at <= asked or not carried:
            continue
        if r.blame != QualificationBlame.PLATFORM or carried == release:
            return True
    return False
