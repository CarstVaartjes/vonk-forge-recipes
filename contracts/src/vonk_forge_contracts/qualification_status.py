"""Public hardware evidence and pure, identity-based qualification views.

No opaque messages or infrastructure identifiers cross this boundary. Summaries
and codes are closed vocabularies; references identify public fixes or evidence
bytes, never a host, request, operation or local artifact path.
"""

from enum import StrEnum
from typing import Annotated, Literal, Self

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, model_validator

from ._schema_version import QualificationEvidenceSchemaVersion

Digest = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
SourceSHA = Annotated[str, Field(pattern=r"^[0-9a-f]{40}$")]
# Public Forge tags, numbered prereleases and source builds. Arbitrary build
# labels can contain hostnames; they are not public platform version evidence.
Version = Annotated[
    str,
    Field(
        max_length=96,
        pattern=r"^v?[0-9]+\.[0-9]+\.[0-9]+(?:(?:~|-)(?:dev|alpha|beta|rc)[.-][0-9]+)?(?:\+g[0-9a-f]{7,40})?$",
    ),
]
RecipeKey = Annotated[str, Field(pattern=r"^[a-z0-9][a-z0-9-]*/[a-z0-9][a-z0-9-]*$")]


class PublicQualificationModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class QualificationStatus(StrEnum):
    WORKING = "Working"
    RECHECK = "Re-check: worked before"
    FIX_TO_VERIFY = "Fix to verify"
    FAILED_RECIPE = "Failed: recipe"
    BLOCKED_PLATFORM = "Blocked: platform"
    UNTESTED = "Untested"
    CANNOT_TEST = "Can't test here"
    TESTING = "Testing now"
    NOT_YET_VERIFIED = "Not yet verified"


class QualificationEvidenceStep(StrEnum):
    SMOKE = "smoke"


class QualificationOutcome(StrEnum):
    PASSED = "passed"
    FAILED = "failed"


class QualificationBlame(StrEnum):
    RECIPE = "recipe"
    PLATFORM = "platform"


class QualificationCauseCode(StrEnum):
    HARDWARE_PASS = "qualification.hardware_pass"
    RECIPE_FAILURE = "qualification.recipe_failure"
    PLATFORM_FAILURE = "qualification.platform_failure"
    RUNTIME_EXIT = "recipe.runtime_exit"
    RUNTIME_OOM = "recipe.runtime_oom"
    MODEL_INTEGRITY = "model.digest_mismatch"
    RECIPE_INTEGRITY = "recipe.digest_mismatch"
    INVALID_DEFINITION = "recipe.invalid_definition"
    INVALID_DOCUMENT = "recipe.invalid_document"
    INVALID_OPTION = "recipe.invalid_option"
    BUILD_POLICY = "dockerfile.heredoc_forbidden"
    SERVING_ASSERTION = "qualification.serving_assertion"
    JOB_FAILURE = "qualification.job_failure"


class QualificationSummary(StrEnum):
    PASSED = "Hardware checks passed"
    RECIPE_FAILED = "Recipe checks failed; needs fix"
    PLATFORM_FAILED = "Platform issue under fix"
    RUNTIME_EXIT = "Recipe runtime exited; needs fix"
    RUNTIME_OOM = "Recipe runtime exceeded available memory; needs fix"
    MODEL_INTEGRITY = "Model content integrity check failed; needs fix"
    INVALID_RECIPE = "Recipe definition or build policy check failed; needs fix"
    SERVING_ASSERTION = "Serving checks failed; needs fix"
    JOB_FAILURE = "Artifact job failed; needs fix"


class QualificationSourceFailureClass(StrEnum):
    SERVING_ASSERTION = "smoke-assertion"
    JOB_FAILED = "job-failed"
    RECIPE_DATA = "recipe-data"


QUALIFICATION_SUMMARIES = {
    QualificationCauseCode.HARDWARE_PASS: QualificationSummary.PASSED,
    QualificationCauseCode.RECIPE_FAILURE: QualificationSummary.RECIPE_FAILED,
    QualificationCauseCode.PLATFORM_FAILURE: QualificationSummary.PLATFORM_FAILED,
    QualificationCauseCode.RUNTIME_EXIT: QualificationSummary.RUNTIME_EXIT,
    QualificationCauseCode.RUNTIME_OOM: QualificationSummary.RUNTIME_OOM,
    QualificationCauseCode.MODEL_INTEGRITY: QualificationSummary.MODEL_INTEGRITY,
    QualificationCauseCode.RECIPE_INTEGRITY: QualificationSummary.MODEL_INTEGRITY,
    QualificationCauseCode.INVALID_DEFINITION: QualificationSummary.INVALID_RECIPE,
    QualificationCauseCode.INVALID_DOCUMENT: QualificationSummary.INVALID_RECIPE,
    QualificationCauseCode.INVALID_OPTION: QualificationSummary.INVALID_RECIPE,
    QualificationCauseCode.BUILD_POLICY: QualificationSummary.INVALID_RECIPE,
    QualificationCauseCode.SERVING_ASSERTION: QualificationSummary.SERVING_ASSERTION,
    QualificationCauseCode.JOB_FAILURE: QualificationSummary.JOB_FAILURE,
}


class QualificationQualityNote(StrEnum):
    ANSWER_VARIANCE = "Model output differed from the expected answer"


class QualificationChange(StrEnum):
    RECIPE = "recipe"
    IMAGE = "image"
    MODEL = "model"


class QualificationUnverifiedReason(StrEnum):
    NEVER_TESTED = "never tested"
    NEEDS_SPARKS = "needs more Sparks"
    PLATFORM_ISSUE = "platform issue under fix"
    INCOMPLETE_PROVENANCE = "historical platform or runtime identity unavailable"


class QualificationHardware(StrEnum):
    GB10 = "GB10"


class QualificationTopology(StrEnum):
    SINGLE = "single"
    MULTI = "multi"


class QualificationRuntimeImage(PublicQualificationModel):
    build_key: Digest
    image_digest: Annotated[str, Field(pattern=r"^sha256:[0-9a-f]{64}$")]


class QualificationIdentity(PublicQualificationModel):
    content_sha256: Digest
    runtime_image: QualificationRuntimeImage | None = None
    model_digests: tuple[Digest, ...] = ()
    required_sparks: int = Field(default=1, ge=1)
    # Unknown historical model bindings must not equal a proven empty model set.
    models_resolved: bool = True

    @model_validator(mode="after")
    def canonical_models(self) -> Self:
        if self.model_digests != tuple(sorted(set(self.model_digests))):
            raise ValueError("model digests must be sorted and unique")
        return self

    @property
    def resolved(self) -> bool:
        return self.runtime_image is not None and self.models_resolved


class QualificationPlatform(PublicQualificationModel):
    controller_source_sha: SourceSHA | None = None
    controller_version: Version | None = None
    agent_version: Version | None = None
    hardware_class: QualificationHardware = QualificationHardware.GB10
    spark_count: int = Field(ge=1)
    topology: QualificationTopology

    @model_validator(mode="after")
    def check_topology(self) -> Self:
        expected = (
            QualificationTopology.SINGLE
            if self.spark_count == 1
            else QualificationTopology.MULTI
        )
        if self.topology != expected:
            raise ValueError("topology must match Spark count")
        return self

    @property
    def resolved(self) -> bool:
        return bool(
            (self.controller_source_sha or self.controller_version)
            and self.agent_version
        )


class QualificationFixReference(PublicQualificationModel):
    # Always in the public Vonk repositories; no arbitrary URL or free text.
    issue_number: int | None = Field(default=None, gt=0)
    source_sha: SourceSHA | None = None

    @model_validator(mode="after")
    def nonempty(self) -> Self:
        if self.issue_number is None and self.source_sha is None:
            raise ValueError("a fix must identify an issue or source commit")
        return self


class QualificationEvidenceRecord(PublicQualificationModel):
    identity: QualificationIdentity
    outcome: QualificationOutcome
    blame: QualificationBlame | None = None
    cause_code: QualificationCauseCode
    cause_summary: QualificationSummary
    evidence_refs: tuple[Digest, ...] = ()
    tested_at: AwareDatetime
    platform: QualificationPlatform
    fix_reference: QualificationFixReference | None = None
    quality_notes: tuple[QualificationQualityNote, ...] = ()

    @model_validator(mode="after")
    def consistent_verdict(self) -> Self:
        if (
            self.outcome == QualificationOutcome.PASSED
            and self.platform.spark_count != self.identity.required_sparks
        ):
            raise ValueError(
                "a hardware pass must use the recipe's required Spark count"
            )
        if (self.outcome == QualificationOutcome.PASSED) != (self.blame is None):
            raise ValueError("passes have no blame; failures must identify blame")
        if self.cause_summary != QUALIFICATION_SUMMARIES[self.cause_code]:
            raise ValueError("summary must match the cause code")
        if (
            self.blame is None
            and self.cause_code != QualificationCauseCode.HARDWARE_PASS
        ):
            raise ValueError("a pass must identify successful hardware checks")
        if (
            self.blame == QualificationBlame.PLATFORM
            and self.cause_code != QualificationCauseCode.PLATFORM_FAILURE
        ):
            raise ValueError("platform failures cannot carry recipe blame")
        if self.blame == QualificationBlame.RECIPE and self.cause_code in (
            QualificationCauseCode.HARDWARE_PASS,
            QualificationCauseCode.PLATFORM_FAILURE,
        ):
            raise ValueError("recipe failures must identify a recipe cause")
        return self


class RecipeQualificationEvidence(PublicQualificationModel):
    schema_version: QualificationEvidenceSchemaVersion = 1
    recipe_key: RecipeKey
    records: tuple[QualificationEvidenceRecord, ...] = ()


class QualificationDerivationInput(PublicQualificationModel):
    """Local report inputs, persisted without a derived status label."""

    evidence: RecipeQualificationEvidence
    current_identity: QualificationIdentity
    fleet_size: int = Field(ge=0)
    testing_now: bool


class QualificationStatusView(PublicQualificationModel):
    status: QualificationStatus
    changed: tuple[QualificationChange, ...] = ()
    last_pass: QualificationEvidenceRecord | None = None
    last_failure: QualificationEvidenceRecord | None = None
    reason: QualificationUnverifiedReason | None = None
    required_sparks: int = Field(ge=1)


class PublishedQualificationStatusView(PublicQualificationModel):
    status: Literal[
        QualificationStatus.WORKING,
        QualificationStatus.RECHECK,
        QualificationStatus.FIX_TO_VERIFY,
        QualificationStatus.FAILED_RECIPE,
        QualificationStatus.NOT_YET_VERIFIED,
    ]
    changed: tuple[QualificationChange, ...] = ()
    last_pass: QualificationEvidenceRecord | None = None
    last_failure: QualificationEvidenceRecord | None = None
    reason: QualificationUnverifiedReason | None = None
    required_sparks: int = Field(ge=1)


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
                previous.runtime_image != current.runtime_image,
            ),
            (
                QualificationChange.MODEL,
                (previous.model_digests, previous.models_resolved)
                != (current.model_digests, current.models_resolved),
            ),
        )
        if differs
    )


def derive_status(
    evidence: RecipeQualificationEvidence,
    current_identity: QualificationIdentity,
    fleet_size: int,
    testing_now: bool,
) -> QualificationStatusView:
    """Latest hardware verdict on an identity wins; platform faults never erase proof.

    Exact historical proof restores on a revert, regardless of later results on
    other identities. Simultaneous contradictory results conservatively prefer
    the recipe failure. Input order never determines the outcome.
    """
    records = sorted(
        evidence.records,
        key=lambda r: (
            r.tested_at,
            r.outcome == QualificationOutcome.FAILED,
            r.model_dump_json(),
        ),
    )
    passes = [r for r in records if r.outcome == QualificationOutcome.PASSED]
    failures = [r for r in records if r.outcome == QualificationOutcome.FAILED]
    behavioural = [r for r in records if r.blame != QualificationBlame.PLATFORM]
    exact = [r for r in behavioural if r.identity == current_identity]
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
    elif any(r.identity == current_identity for r in failures):
        status = QualificationStatus.BLOCKED_PLATFORM
        reason = QualificationUnverifiedReason.PLATFORM_ISSUE
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
    if fleet_size < current_identity.required_sparks and not exact:
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
        required_sparks=current_identity.required_sparks,
    )


def published_status(
    evidence: RecipeQualificationEvidence, current_identity: QualificationIdentity
) -> PublishedQualificationStatusView:
    """Release-relative view: no lane state or publisher fleet leaks into the catalog."""
    view = derive_status(
        evidence, current_identity, current_identity.required_sparks, False
    )
    reason = view.reason
    if (
        view.status == QualificationStatus.UNTESTED
        and current_identity.required_sparks > 1
    ):
        # Describe the required topology without publishing our enrollment.
        reason = QualificationUnverifiedReason.NEEDS_SPARKS
    if view.status in (
        QualificationStatus.WORKING,
        QualificationStatus.FAILED_RECIPE,
    ) and (
        not current_identity.resolved
        or (
            view.status == QualificationStatus.WORKING
            and view.last_pass is not None
            and not view.last_pass.platform.resolved
        )
        or (
            view.status == QualificationStatus.FAILED_RECIPE
            and view.last_failure is not None
            and not view.last_failure.platform.resolved
        )
    ):
        reason = QualificationUnverifiedReason.INCOMPLETE_PROVENANCE
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


def qualification_evidence_json_schema() -> dict[str, object]:
    return RecipeQualificationEvidence.model_json_schema()


def qualification_status_json_schema() -> dict[str, object]:
    return QualificationStatusView.model_json_schema()


def published_qualification_status_json_schema() -> dict[str, object]:
    return PublishedQualificationStatusView.model_json_schema()
