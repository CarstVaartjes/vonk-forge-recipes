"""Controller lifecycle vocabulary, checked against the published OpenAPI fixture.

The recipe-authoring contracts do not export Controller lifecycle states.
Unknown observations stay active: observe again instead of inventing a failure.
"""

from enum import StrEnum


class LifecycleState(StrEnum):
    """Consumer vocabulary from the published Controller Pydantic schema."""

    QUEUED = "queued"
    RUNNING = "running"
    OBSERVING = "observing"
    BACKOFF = "backoff"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"
    SUPERSEDED = "superseded"
    NEEDS_OPERATOR = "needs-operator"


STATES = frozenset(LifecycleState)
TERMINAL = frozenset(
    {
        LifecycleState.SUCCEEDED,
        LifecycleState.FAILED,
        LifecycleState.CANCELLED,
        LifecycleState.SUPERSEDED,
    }
)
WAITING = frozenset({LifecycleState.NEEDS_OPERATOR})
ACTIVE = STATES - TERMINAL - WAITING


def active(state: object) -> bool:
    return not isinstance(state, str) or (
        state not in TERMINAL and state not in WAITING
    )
