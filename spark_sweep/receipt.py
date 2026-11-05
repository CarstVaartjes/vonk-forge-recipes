"""The one reader of a terminal download receipt: its failure becomes the sweep's outcome.

A receipt's lifecycle word is not its whole story: the Controller ends an exhausted image
preparation as ``cancelled`` (keeping the model download) with a failure, and a superseded
one as ``cancelled`` with a recovery action. ``tests/test_sweep_receipt_reader.py`` keeps
every other module from reading a receipt's ``failure`` directly.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .lifecycle import LifecycleState
from .policy import Failure, classify, excerpt

#: An unnamed class: ``disposition`` observes it and ``Sweep._bound_identical_exit`` ends a
#: third identical outcome on the same content identity.
CANCELLED_FAILURE = "download-cancelled"


def receipt_failure(document: Mapping[str, Any]) -> Mapping[str, Any] | None:
    """The failure a receipt (or one of its children) carries, if any."""
    failure = document.get("failure")
    return failure if isinstance(failure, Mapping) else None


def terminal_outcome(document: Mapping[str, Any]) -> Failure | None:
    """What a terminal download receipt came to, or None when it is no outcome.

    FAILED is an outcome. CANCELLED is one only when it carries a failure code, says
    ``retryable: false`` and offers no recovery action: a superseded or operator cancel names
    a recovery action (``download_again``, ``retry``) and stays a plain retry.
    """
    state = document.get("state")
    failure_doc = receipt_failure(document) or {}
    code = str(failure_doc.get("code", ""))
    actions = [str(a) for a in failure_doc.get("recovery_actions") or []]
    if state == LifecycleState.CANCELLED and not (
        code and failure_doc.get("retryable") is False and not actions
    ):
        return None
    if state not in (LifecycleState.FAILED, LifecycleState.CANCELLED):
        return None
    detail = str(failure_doc.get("detail", "") or document.get("detail", ""))
    return classify(
        "download",
        code,
        detail,
        klass=CANCELLED_FAILURE if state == LifecycleState.CANCELLED else "",
        evidence={
            "operation_id": document.get("id"),
            "state": state,
            "reason": excerpt(failure_doc.get("detail") or detail, 500),
            "recovery_actions": actions,
            "children": [
                {
                    k: child.get(k)
                    for k in ("kind", "state", "id")
                    if child.get(k) is not None
                }
                for child in document.get("children", [])
                if isinstance(child, dict)
                and child.get("state") == LifecycleState.FAILED
            ][:8],
            "failure": excerpt(failure_doc, 1000),
        },
    )
