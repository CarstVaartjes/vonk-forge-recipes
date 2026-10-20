"""Canonical local wait evidence; observation time never impersonates progress."""

from enum import StrEnum

from pydantic import BaseModel, ConfigDict


class WaitReason(StrEnum):
    CONTROLLER_HELD = "sweep.controller_held"
    OBSERVATION_UNAVAILABLE = "sweep.observation_unavailable"
    SEMANTIC_OUTPUT = "sweep.semantic_output"


class WaitOwner(StrEnum):
    CONTROLLER = "controller"
    SMOKE = "smoke"


class NextCheck(StrEnum):
    OBSERVE_REQUEST = "observe-request"
    RECONCILE_REQUEST = "reconcile-request"
    OBSERVE_OUTPUT = "observe-output"


class WaitEvidence(BaseModel):
    model_config = ConfigDict(extra="forbid")

    reason: WaitReason
    owner: WaitOwner
    detail: str
    first_seen_at: float
    last_progress_at: float
    observed_at: float
    next_check_at: float
    next_check: NextCheck
    reconcile_at: float
    last_reconciled_at: float | None = None
