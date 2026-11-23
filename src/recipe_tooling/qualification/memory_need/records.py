"""The records of a memory estimate."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from pydantic import BaseModel, ConfigDict

from recipe_tooling.qualification.memory_need.constants import (
    LIMIT_BYTES,
    OVERHEAD_BYTES,
)


@dataclass(frozen=True)
class MemoryNeed:
    weights_bytes: int
    auxiliary_weights_bytes: int
    kv_bytes: int
    state_bytes: int
    resident_expansion_bytes: int = 0
    overhead_bytes: int = OVERHEAD_BYTES
    limit_bytes: int = LIMIT_BYTES

    @property
    def total_bytes(self) -> int:
        return (
            self.weights_bytes
            + self.auxiliary_weights_bytes
            + self.kv_bytes
            + self.state_bytes
            + self.resident_expansion_bytes
            + self.overhead_bytes
        )


class EnvelopeException(BaseModel):
    """An owner-approved envelope exception for one exact recipe content."""

    model_config = ConfigDict(extra="forbid")
    content_sha256: str
    reason: str
    owner_decision: str
    date: str


class MissingEstimateInput(ValueError):
    """The estimate lacks sourced data (memory-inputs.json); an agent can add it.

    Any other error from an estimate rule is a creator flag or a value that
    disagrees, which stays advisory.
    """


class EstimateFindingKind(StrEnum):
    """What a finding asks for: data (actionable) or only a look (advisory)."""

    UNAVAILABLE = "unavailable"
    DISAGREEMENT = "disagreement"


@dataclass(frozen=True)
class EstimateFinding:
    kind: EstimateFindingKind
    text: str


@dataclass(frozen=True)
class PerRankMemoryNeed:
    peak_bytes: int
    safe_limit_bytes: int
    node_count: int

    @property
    def fits(self) -> bool:
        return self.peak_bytes <= self.safe_limit_bytes

    @property
    def verdict(self) -> str:
        return f"{'fits' if self.fits else 'does not fit'} {self.node_count} Sparks"
