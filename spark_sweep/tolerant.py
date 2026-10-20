"""Optional observations have diagnostics, never authority over mandatory work.

Only the read/validation belongs inside this boundary. Security checks, durable
writes and destructive actions stay outside it. Interrupts always propagate.
"""

from collections.abc import Callable
from enum import StrEnum

from pydantic import BaseModel, ConfigDict


class InputReason(StrEnum):
    INVALID_ROW = "sweep.invalid_row"
    INVALID_OPTIONAL = "sweep.invalid_optional"
    EVIDENCE_UNAVAILABLE = "sweep.evidence_unavailable"
    HOUSEKEEPING_UNAVAILABLE = "refresh.housekeeping_unavailable"
    PUBLICATION_UNAVAILABLE = "refresh.publication_unavailable"
    ASSESSMENT_UNAVAILABLE = "refresh.assessment_unavailable"


class InputDiagnostic(BaseModel):
    model_config = ConfigDict(frozen=True)

    source: str
    reason: InputReason
    detail: str


def read_optional[T](
    source: str,
    read: Callable[[], T],
    diagnose: Callable[[InputDiagnostic], None],
    reason: InputReason = InputReason.INVALID_OPTIONAL,
) -> T | None:
    try:
        return read()
    except Exception as error:  # noqa: BLE001 - only optional observations enter this boundary
        diagnose(
            InputDiagnostic(source=source, reason=reason, detail=str(error)[:1000])
        )
        return None
