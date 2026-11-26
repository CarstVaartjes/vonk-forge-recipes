"""Status derivation: the reducers over public hardware evidence, and the sweep-only enums.

Producer-side code. The wire models (``RecipeQualificationEvidence``,
``QualificationStatusView`` and the published views) stay in
``vonk_forge_contracts.qualification_status``; a consumer reads their result and
never derives a status.
"""

from __future__ import annotations

from enum import StrEnum

from vonk_forge_contracts import (
    PublishedQualificationStatusView,
    QualificationBlame,
    QualificationChange,
    QualificationEvidenceRecord,
    QualificationIdentity,
    QualificationOutcome,
    QualificationStatus,
    QualificationStatusView,
    QualificationUnverifiedReason,
    RecipeQualificationEvidence,
)


class QualificationEvidenceStep(StrEnum):
    SMOKE = "smoke"


class QualificationSourceFailureClass(StrEnum):
    SERVING_ASSERTION = "smoke-assertion"
    JOB_FAILED = "job-failed"
    RECIPE_DATA = "recipe-data"


class QualificationGlbProfile(StrEnum):
    """What a job recipe's ``glb-structure`` assertion requires of its GLB."""

    TRIANGLE_MESH = "triangle-mesh"
    TEXTURED = "textured"
    TEXTURED_PBR = "textured-pbr"
    SKINNED = "skinned"


def _changes(
    previous: QualificationIdentity, current: QualificationIdentity
) -> tuple[QualificationChange, ...]:
    return tuple(
        change
        for change, differs in (
            (
                QualificationChange.RECIPE,
                previous.content_sha256 != current.content_sha256,
            ),
            (
                QualificationChange.IMAGE,
                previous.runtime_image is not None
                and previous.runtime_image != current.runtime_image,
            ),
            (
                QualificationChange.SOURCE,
                previous.source_bundle_sha256 is not None
                and previous.source_bundle_sha256 != current.source_bundle_sha256,
            ),
            (
                QualificationChange.MODEL,
                (previous.model_digests, previous.models_resolved)
                != (current.model_digests, current.models_resolved),
            ),
        )
        if differs
    )


def _same_content_identity(
    previous: QualificationIdentity, current: QualificationIdentity
) -> bool:
    # Compare recorded content, never turn missing historical provenance into a
    # changed pin. Publication separately requires fully resolved exact proof.
    return (
        previous.content_sha256 == current.content_sha256
        and (
            previous.source_bundle_sha256 is None
            or previous.source_bundle_sha256 == current.source_bundle_sha256
        )
        and previous.model_digests == current.model_digests
        and previous.models_resolved == current.models_resolved
        and previous.required_sparks in (None, current.required_sparks)
        and (
            previous.runtime_image is None
            or previous.runtime_image == current.runtime_image
        )
    )


def derive_status(
    evidence: RecipeQualificationEvidence,
    current_identity: QualificationIdentity,
    fleet_size: int,
    *,
    testing_now: bool,
) -> QualificationStatusView:
    """Latest hardware verdict on an identity wins; platform faults never erase proof.

    Exact historical proof restores on a revert, regardless of later results on
    other identities. Simultaneous contradictory results conservatively prefer
    the recipe failure. Input order never determines the outcome.
    """
    required_sparks = current_identity.required_sparks
    if required_sparks is None:
        raise ValueError("current identity requires the recipe's Spark count")
    records = sorted(
        evidence.records,
        key=lambda r: (
            r.tested_at,
            # On a tie in time the record of the current document is the later
            # one: a run recorded under two digests of the same execution never
            # lets the older digest decide.
            r.identity.content_sha256 == current_identity.content_sha256,
            r.outcome == QualificationOutcome.FAILED,
            r.model_dump_json(),
        ),
    )
    passes = [r for r in records if r.outcome == QualificationOutcome.PASSED]
    failures = [r for r in records if r.outcome == QualificationOutcome.FAILED]
    behavioural = [r for r in records if r.blame != QualificationBlame.PLATFORM]
    # Once this content has image-bound evidence, an older unbound pass cannot
    # certify a different image. Unbound failures still invalidate local proof.
    image_bound = any(
        r.identity.runtime_image is not None
        and _same_content_identity(
            r.identity.model_copy(update={"runtime_image": None}), current_identity
        )
        for r in behavioural
    )
    exact = [
        r
        for r in behavioural
        if _same_content_identity(r.identity, current_identity)
        and not (
            image_bound
            and r.identity.runtime_image is None
            and r.outcome == QualificationOutcome.PASSED
        )
    ]
    last_pass = passes[-1] if passes else None
    last_failure = failures[-1] if failures else None
    changed: tuple[QualificationChange, ...] = ()
    reason = None
    if exact:
        status = (
            QualificationStatus.WORKING
            if exact[-1].outcome == QualificationOutcome.PASSED
            else QualificationStatus.FAILED_RECIPE
        )
        if status == QualificationStatus.WORKING:
            last_pass = exact[-1]
        else:
            last_failure = exact[-1]
    elif any(_same_content_identity(r.identity, current_identity) for r in failures):
        status = QualificationStatus.BLOCKED_PLATFORM
        reason = QualificationUnverifiedReason.PLATFORM_ISSUE
        last_failure = [
            r for r in failures if _same_content_identity(r.identity, current_identity)
        ][-1]
        if last_pass:
            changed = _changes(last_pass.identity, current_identity)
    elif behavioural:
        previous = behavioural[-1]
        changed = _changes(previous.identity, current_identity)
        if previous.outcome == QualificationOutcome.FAILED:
            last_failure = previous
        status = (
            QualificationStatus.RECHECK
            if previous.outcome == QualificationOutcome.PASSED
            else QualificationStatus.FIX_TO_VERIFY
        )
    elif failures:
        status = QualificationStatus.BLOCKED_PLATFORM
        reason = QualificationUnverifiedReason.PLATFORM_ISSUE
    else:
        status = QualificationStatus.UNTESTED
        reason = QualificationUnverifiedReason.NEVER_TESTED
    if fleet_size < required_sparks and not exact:
        status = QualificationStatus.CANNOT_TEST
        reason = QualificationUnverifiedReason.NEEDS_SPARKS
    if testing_now:
        status = QualificationStatus.TESTING
    return QualificationStatusView(
        status=status,
        changed=changed,
        last_pass=last_pass,
        last_failure=last_failure,
        reason=reason,
        required_sparks=required_sparks,
    )


def published_status(
    evidence: RecipeQualificationEvidence, current_identity: QualificationIdentity
) -> PublishedQualificationStatusView:
    """Release-relative view: no lane state or publisher fleet leaks into the catalog."""
    required_sparks = current_identity.required_sparks
    if required_sparks is None:
        raise ValueError("current identity requires the recipe's Spark count")
    view = derive_status(evidence, current_identity, required_sparks, testing_now=False)
    certified: list[QualificationEvidenceRecord] = []
    if view.status == QualificationStatus.WORKING:
        # A later successful test with incomplete telemetry does not erase a
        # complete proof. A recipe failure on this identity invalidates older
        # proofs; a partial pass after that failure cannot revive them publicly.
        failures = [
            r.tested_at
            for r in evidence.records
            if _same_content_identity(r.identity, current_identity)
            and r.blame == QualificationBlame.RECIPE
        ]
        failed_at = max(failures) if failures else None
        certified = [
            r
            for r in evidence.records
            if r.identity == current_identity
            and r.outcome == QualificationOutcome.PASSED
            and r.identity.resolved
            and r.platform.resolved
            and (failed_at is None or r.tested_at > failed_at)
        ]
        if certified:
            view = view.model_copy(
                update={
                    "last_pass": max(
                        certified, key=lambda r: (r.tested_at, r.model_dump_json())
                    )
                }
            )
    reason = view.reason
    if view.status == QualificationStatus.UNTESTED and required_sparks > 1:
        # Describe the required topology without publishing our enrollment.
        reason = QualificationUnverifiedReason.NEEDS_SPARKS
    if view.status in (
        QualificationStatus.WORKING,
        QualificationStatus.FAILED_RECIPE,
    ) and (
        not current_identity.resolved
        or (view.status == QualificationStatus.WORKING and not certified)
        or (
            view.status == QualificationStatus.FAILED_RECIPE
            and view.last_failure is not None
            and not view.last_failure.platform.resolved
        )
    ):
        reason = QualificationUnverifiedReason.INCOMPLETE_PROVENANCE
    if view.status == QualificationStatus.BLOCKED_PLATFORM and view.last_pass:
        # A platform fault on the current identity after a pass elsewhere: the
        # recipe worked before and is re-tested, never "Not yet verified".
        view = view.model_copy(
            update={"status": QualificationStatus.RECHECK, "reason": None}
        )
        reason = None
    if (
        view.status
        in (
            QualificationStatus.UNTESTED,
            QualificationStatus.BLOCKED_PLATFORM,
            QualificationStatus.CANNOT_TEST,
            QualificationStatus.TESTING,
        )
        or reason == QualificationUnverifiedReason.INCOMPLETE_PROVENANCE
    ):
        view = view.model_copy(
            update={"status": QualificationStatus.NOT_YET_VERIFIED, "reason": reason}
        )
    return PublishedQualificationStatusView.model_validate(
        view.model_copy(update={"reason": reason}).model_dump()
    )
