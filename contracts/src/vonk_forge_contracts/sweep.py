"""Typed local sweep reconciliation records; no Controller success is inferred."""

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field


class CleanupEndReason(StrEnum):
    TARGET_ABSENT = "sweep.cleanup_target_absent"


class CleanupEnd(BaseModel):
    model_config = ConfigDict(extra="forbid")

    reason: CleanupEndReason
    observed_at: float


class CleanupObservationBudget(BaseModel):
    model_config = ConfigDict(extra="forbid")

    attempts: int = 0
    next_check: float = 0.0


class SweepSlotPhase(StrEnum):
    LOADING = "loading"


class SweepStorageEvidence(BaseModel):
    required_free_bytes: int = Field(ge=0)
    available_bytes: int = Field(ge=0)
    reclaimable_bytes: int = Field(ge=0)


class SweepAdmissionBlocker(BaseModel):
    node_ids: list[str]
    detail: str
    storage: SweepStorageEvidence | None = None


class SweepAdmissionEffect(BaseModel):
    """Only presence is needed; an accepted effect prevents local retirement."""


class SweepPendingAdmission(BaseModel):
    admission_pending: bool
    effects: list[SweepAdmissionEffect]
    blockers: list[SweepAdmissionBlocker]

    @property
    def storage_wait_nodes(self) -> set[str]:
        return {
            node
            for blocker in self.blockers
            if blocker.storage is not None
            and blocker.storage.required_free_bytes > blocker.storage.available_bytes
            for node in blocker.node_ids
        }


class LoadedRunObservation(BaseModel):
    run_id: str


class FleetNodeObservation(BaseModel):
    id: str
    connection: "FleetConnectionObservation"
    loaded: list[LoadedRunObservation]


class FleetOnlineState(StrEnum):
    ONLINE = "online"


class FleetConnectionObservation(BaseModel):
    # External wire vocabulary is validated by the installed CLI. Only ONLINE
    # certifies absence; other values remain observed, without authorizing it.
    online_state: str


class CleanupFleetObservation(BaseModel):
    nodes: list[FleetNodeObservation]


class ReviewedCleanupStop(BaseModel):
    run_id: str
    node_ids: list[str]


class InfrastructureIncident(BaseModel):
    """Durable local retry history; escalation never invents a Controller ending."""

    model_config = ConfigDict(extra="forbid")

    message: str = ""
    since: float
    count: int = 0
    until: float = 0.0
    dependency: str | None = None
    escalate_at: float | None = None
