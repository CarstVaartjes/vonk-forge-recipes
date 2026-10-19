"""Typed sweep projections of the installed CLI's validated removal contract.

The Controller owns cache_removal_review.CacheRemovalReview and
model_cache_contract.ModelCacheOperatorResponse. Use the sweep's existing
schema-checked lifecycle enum and typed result/failure
evidence to end or re-plan a request. No new Controller vocabulary is duplicated.
"""

from enum import StrEnum
from typing import Annotated, Literal

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, JsonValue

Digest = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]


class RemovalOutcome(StrEnum):
    OWNER = "owner-profile"
    WAIT = "waiting"
    RETRY = "retry"
    RECEIPT = "receipt"
    EXHAUSTED = "exhausted"


class RecipeVerdict(StrEnum):
    PENDING = "pending"
    DEFERRED = "deferred"
    SKIPPED = "skipped"
    PASSED = "passed"
    FAILED = "failed"


class Finding(BaseModel):
    detail: str
    retryable: bool = False


class RemovalTarget(BaseModel):
    node_id: str
    observation_available: bool
    objects: list[JsonValue]
    installations: list[JsonValue]


class RemovalEffects(BaseModel):
    nas_remove: bool
    targets: list[RemovalTarget]


class RemovalCheckpoint(BaseModel):
    deadline_at: AwareDatetime
    effects: RemovalEffects
    completed_installations: list[str] = Field(default_factory=list)
    confirmed_nodes: list[str] = Field(default_factory=list)


class RemovalReview(BaseModel):
    action: Literal["remove"] = "remove"
    resource_kind: Literal["model"]
    target_identity: Digest
    review_digest: Digest
    fleet: RemovalEffects
    blockers: list[Finding]


class RemovalResult(BaseModel):
    removed_entries: list[Digest]
    reclaimed_bytes: int


class RemovalReceipt(BaseModel):
    action: Literal["remove"]
    request_key: str
    model_content_sha256: Digest
    operation_id: str
    review_digest: Digest
    fleet: RemovalCheckpoint
    state: str
    result: RemovalResult | None = None
    failure: Finding | None = None


class ProfileAssignment(BaseModel):
    recipe_selector: str
    model_variant: str | None = None


class ProfileModel(BaseModel):
    content_sha256: Digest | None = None


class AssignmentView(BaseModel):
    recipe_selector: str
    model: ProfileModel


class ProfileDefinition(BaseModel):
    assignments: list[ProfileAssignment] = Field(default_factory=list)
    labels: dict[str, str] = Field(default_factory=dict)


class OwnerProfile(BaseModel):
    number: int
    definition: ProfileDefinition | None
    assignments: list[AssignmentView] = Field(default_factory=list)


class ProfileList(BaseModel):
    profiles: list[OwnerProfile]


class RemovalRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    outcome: RemovalOutcome = RemovalOutcome.WAIT
    attempt_count: int = 0
    observations: int = 0
    cycle: int = 0
    needed_again: bool = False
    consumers: dict[str, str] = Field(default_factory=dict)
    request_key: str | None = None
    deadline: float = 0
    next_check: float = 0
    message: str = ""
    review: dict[str, JsonValue] | None = None
    receipt: dict[str, JsonValue] | None = None
    last_response: JsonValue = None
    history: list[dict[str, JsonValue]] = Field(default_factory=list)
