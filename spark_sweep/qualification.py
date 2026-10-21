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
)

from .auth_contract import SmokeAuthClass
from .catalog import Recipe
from .identity import current_identity
from .lifecycle import RecipeStatus
from .state import write_atomic


class HistoricalIdentity(BaseModel):
    content_sha256: str
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
    verdict_identity: HistoricalIdentity | None = None
    model_digests: tuple[str, ...] | None = None
    controller_release: str | None = None
    controller_version: str | None = None
    agent_version: str | None = None
    node_count: int | None = Field(default=None, ge=1)
    node_ids: tuple[str, ...] = ()
    quality: tuple[HistoricalQualityCheck, ...] = ()
    result: HistoricalSmokeResult = Field(default_factory=HistoricalSmokeResult)
    failure_class: str | None = None
    code: str | None = None


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
    return left.model_copy(update={"records": tuple(ordered)})


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
    recorded = row.verdict_identity
    models = recorded.model_digests if recorded else row.model_digests
    identity = QualificationIdentity(
        content_sha256=recorded.content_sha256 if recorded else row.content_sha256,
        runtime_image=recorded.runtime_image if recorded else None,
        model_digests=tuple(sorted(set(models or ()))),
        models_resolved=models is not None,
        required_sparks=row.node_count or len(row.node_ids) or None,
    )
    from .policy import RECIPE_FAILURE_CODES
    from .verdict import FAIL_CLASSES

    passed = row.status == RecipeStatus.PASSED
    blame = (
        None
        if passed
        else (
            QualificationBlame.RECIPE
            if row.status == RecipeStatus.FAILED
            and (
                row.code in RECIPE_FAILURE_CODES
                or (
                    row.failure_class in FAIL_CLASSES
                    and row.failure_class != SmokeAuthClass.REFUSAL
                )
            )
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
            agent_version=row.agent_version,
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


def backfill(paths: Iterable[Path], root: Path) -> BackfillResult:
    result = BackfillResult()
    grouped: dict[str, RecipeQualificationEvidence] = {}
    for path in paths:
        # Read only. Never use State.load: its recovery path mutates source state.
        for line in path.read_text().splitlines():
            result.read += 1
            try:
                raw = json.loads(line)
                record = from_history(raw)
                if record is None:
                    continue
                key = HistoricalResult.model_validate(raw).recipe
                evidence = RecipeQualificationEvidence(
                    recipe_key=key, records=(record,)
                )
            except (ValueError, TypeError, AttributeError):
                result.rejected += 1
                continue
            previous = grouped.get(key) or read_evidence(root, key)
            grouped[key] = merge_evidence(previous, evidence)
            result.imported += 1
    for evidence in grouped.values():
        result.changed_files += int(write_evidence(root, evidence))
    return result


def local_status(
    evidence: RecipeQualificationEvidence,
    recipe: Recipe,
    fleet_size: int,
    testing_now: bool = False,
) -> QualificationStatusView:
    return derive_status(evidence, current_identity(recipe), fleet_size, testing_now)


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
