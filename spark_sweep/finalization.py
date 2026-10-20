"""Durable ownership of the sweep's requested final fleet state."""

from enum import StrEnum

from pydantic import BaseModel, ConfigDict


class FinalAction(StrEnum):
    STOP = "stop"
    RESTORE = "restore-owner"


class Finalization(BaseModel):
    model_config = ConfigDict(extra="forbid")

    action: FinalAction
    profile: int
    request_key: str
    since: float
    attempts: int = 0
    next_check: float = 0.0
