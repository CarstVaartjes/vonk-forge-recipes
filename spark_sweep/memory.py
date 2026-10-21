"""What a serving run holds beside its declared envelope: a quality note, never a verdict."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from pydantic import BaseModel, ValidationError

from qualification.memory_need import OS_FLOOR_BYTES


class _Sample(BaseModel):
    memory_total_bytes: int
    memory_available_bytes: int


class _Telemetry(BaseModel):
    sample: _Sample


class _Loaded(BaseModel):
    alias: str


class _Node(BaseModel):
    id: str
    loaded: list[_Loaded] = []
    telemetry: _Telemetry | None = None


class _Fleet(BaseModel):
    nodes: list[_Node]


def observe(
    document: Any, node_ids: Iterable[str], alias: str, declared_bytes: int
) -> dict[str, Any] | None:
    """Each lane Spark's use (total less available) and whether the run was alone there.

    Node telemetry is the run's own use only when the run is alone on that Spark.
    """
    try:
        fleet = _Fleet.model_validate(document)
    except ValidationError:
        return None
    wanted = set(node_ids)
    observed = [
        {
            "node_id": node.id,
            "used_bytes": sample.memory_total_bytes - sample.memory_available_bytes,
            "alone": {item.alias for item in node.loaded} <= {alias},
        }
        for node in fleet.nodes
        if node.id in wanted and node.telemetry is not None
        for sample in [node.telemetry.sample]
        if 0 <= sample.memory_available_bytes <= sample.memory_total_bytes
        and sample.memory_total_bytes > 0
    ]
    if not observed:
        return None
    return {"declared_bytes": declared_bytes, "nodes": observed}


def notes(memory: Mapping[str, Any] | None) -> list[dict[str, Any]]:
    """A run alone on a Spark that holds more than its declared envelope."""
    if not memory or memory["declared_bytes"] <= 0:
        return []
    return [
        {
            "case": "memory-envelope",
            "expected": f"at most {memory['declared_bytes']} bytes (declared)",
            "got": f"{node['used_bytes']} bytes in use on {node['node_id']} "
            f"(run alone; OS included, {OS_FLOOR_BYTES} allowed for it)",
            "ok": False,
        }
        for node in memory["nodes"]
        if node["alone"]
        and node["used_bytes"] > memory["declared_bytes"] + OS_FLOOR_BYTES
    ]
