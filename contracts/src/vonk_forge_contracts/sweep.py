"""Typed local sweep reconciliation records; no Controller success is inferred."""

from enum import StrEnum

from pydantic import BaseModel, ConfigDict


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
