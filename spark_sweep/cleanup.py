"""Consumer decisions over canonical profile effect projections and receipts.

The Controller and installed CLI own the strict wire models. This module only
matches an observed cleanup to its previously accepted identity; it never
constructs another workload authority or infers a stop from an absent run.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

from pydantic import ValidationError
from vonk_forge_contracts.sweep import ReviewedCleanupStop

STATES = frozenset(
    {"not-issued", "pending", "succeeded", "failed", "cancelled", "unknown"}
)


def stop_signature(effect: Mapping[str, Any]) -> str:
    """Compare the full canonical stop, including any topology-bound partial scope."""
    # Preserve every canonical field. Only this declared nullable optional field
    # treats an omitted JSON member and explicit null as the same value.
    full = dict(effect)
    full["profile_stop_scope"] = effect.get("profile_stop_scope")
    return json.dumps(full, sort_keys=True, separators=(",", ":"))


def stop_effects(document: Mapping[str, Any]) -> list[dict[str, Any]] | None:
    """Read the one current typed projection; missing capability is unknown."""
    progress = document.get("progress")
    rows = progress.get("effects") if isinstance(progress, Mapping) else None
    if not isinstance(rows, list):
        return None
    result: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in rows:
        if not isinstance(row, Mapping):
            return None
        if row.get("kind") != "stop":
            continue
        effect = row.get("stop_effect")
        nodes = row.get("node_ids")
        if (
            not isinstance(effect, Mapping)
            or effect.get("action") != "stop"
            or effect.get("run_id") != row.get("target_id")
            or nodes != effect.get("node_ids")
            or not isinstance(nodes, list)
            or not nodes
            or any(not isinstance(node, str) or not node for node in nodes)
            or len(nodes) != len(set(nodes))
            or row.get("state") not in STATES
            or type(row.get("queue_index")) is not int
            or row["queue_index"] < 0
            or not isinstance(row.get("effect_id"), str)
            or row["effect_id"] in seen
            or not isinstance(row.get("request_key"), str)
            or not isinstance(row.get("application_id"), str)
            or not isinstance(row.get("plan_digest"), str)
            or len(row["plan_digest"]) != 64
            or any(c not in "0123456789abcdef" for c in row["plan_digest"])
            or type(row.get("workload_intent_ordinal")) is not int
            or row["workload_intent_ordinal"] < 1
        ):
            return None
        seen.add(row["effect_id"])
        result.append(dict(row))
    return result


def same_effect(previous: Mapping[str, Any], current: Mapping[str, Any]) -> bool:
    for key in (
        "effect_id",
        "application_id",
        "plan_digest",
        "workload_intent_ordinal",
        "queue_index",
        "kind",
        "target_id",
        "node_ids",
        "request_key",
    ):
        if previous.get(key) != current.get(key):
            return False
    if stop_signature(previous["stop_effect"]) != stop_signature(
        current["stop_effect"]
    ):
        return False
    old = previous.get("operation_id")
    new = current.get("operation_id")
    return old is None or old == new or old == current.get("original_operation_id")


def stopped(row: Mapping[str, Any]) -> bool:
    """Only the canonical successful exact-run stop receipt releases occupancy."""
    if row.get("state") != "succeeded" or not row.get("operation_id"):
        return False
    result = row.get("result")
    operation = result.get("run_switch") if isinstance(result, Mapping) else None
    phases = operation.get("phase_results") if isinstance(operation, Mapping) else None
    if (
        not isinstance(result, Mapping)
        or result.get("run_switch_operation_id") != row.get("operation_id")
        or not isinstance(phases, list)
    ):
        return False
    return any(
        isinstance(phase, Mapping)
        and phase.get("phase") == "stop"
        and phase.get("run_id") == row.get("target_id")
        for phase in phases
    )


def adopted(review: Mapping[str, Any], row: Mapping[str, Any]) -> bool:
    """A newer whole-fleet preview explicitly binds this exact pending cleanup."""
    effects = review.get("effects")
    links = effects.get("adopted") if isinstance(effects, Mapping) else None
    if not isinstance(links, list) or not row.get("operation_id"):
        return False
    for link in links:
        if not isinstance(link, Mapping) or any(
            link.get(key) != row.get(key)
            for key in ("application_id", "plan_digest", "workload_intent_ordinal")
        ):
            continue
        nodes = link.get("node_ids")
        stops = link.get("stops")
        if (
            not isinstance(nodes, list)
            or any(not isinstance(node, str) for node in nodes)
            or not set(row["node_ids"]) <= set(nodes)
        ):
            continue
        if not isinstance(stops, list):
            continue
        for stop in stops:
            if not isinstance(stop, Mapping) or not isinstance(
                stop.get("effect"), Mapping
            ):
                continue
            if (
                stop.get("queue_index") == row["queue_index"]
                and stop.get("operation_id") == row["operation_id"]
                and stop.get("request_key") == row["request_key"]
                and stop_signature(stop["effect"]) == stop_signature(row["stop_effect"])
            ):
                return True
    return False


def reviewed_stop(signature: str) -> ReviewedCleanupStop | None:
    """Recover the reviewed target identities from the durable canonical signature."""
    try:
        return ReviewedCleanupStop.model_validate_json(signature)
    except ValidationError:
        return None
