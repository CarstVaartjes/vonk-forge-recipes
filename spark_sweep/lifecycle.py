"""Controller lifecycle vocabulary, checked against the published OpenAPI fixture.

The recipe-authoring contracts do not export Controller lifecycle states.
Unknown observations stay active: observe again instead of inventing a failure.
"""

from typing import Literal, get_args

LifecycleState = Literal[
    "queued",
    "running",
    "observing",
    "backoff",
    "succeeded",
    "failed",
    "cancelled",
    "superseded",
    "needs-operator",
]
STATES = frozenset(get_args(LifecycleState))
TERMINAL = frozenset({"succeeded", "failed", "cancelled", "superseded"})
WAITING = frozenset({"needs-operator"})
ACTIVE = STATES - TERMINAL - WAITING


def active(state: object) -> bool:
    return not isinstance(state, str) or (
        state not in TERMINAL and state not in WAITING
    )
