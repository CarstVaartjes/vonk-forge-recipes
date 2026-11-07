"""What a serving run holds: a quality note beside its envelope, and the evidence it measured.

Neither is a verdict: a run that cannot be measured still passes, unmeasured.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from pydantic import BaseModel, ValidationError
from vonk_forge_contracts import QualificationMeasuredRoleMemory, RecipeDefinition

from qualification.memory_need import OS_FLOOR_BYTES, holds_within, vllm_budget_bytes


class _Sample(BaseModel):
    memory_total_bytes: int
    memory_available_bytes: int


class _Swap(BaseModel):
    total_bytes: int | None = None
    free_bytes: int | None = None


class _HostSettings(BaseModel):
    swap: _Swap | None = None


class _Inventory(BaseModel):
    host_settings: _HostSettings | None = None


class _Telemetry(BaseModel):
    sample: _Sample


class _Loaded(BaseModel):
    alias: str


class _Node(BaseModel):
    id: str
    loaded: list[_Loaded] = []
    telemetry: _Telemetry | None = None
    inventory: _Inventory | None = None


class _Fleet(BaseModel):
    nodes: list[_Node]


def _swap(node: _Node) -> tuple[int | None, int | None]:
    """The node's swap total and used bytes; ``None`` where it did not report them."""
    swap = (
        node.inventory.host_settings.swap
        if node.inventory and node.inventory.host_settings
        else None
    )
    if swap is None or swap.total_bytes is None:
        return None, None
    if swap.free_bytes is None:
        return swap.total_bytes, None
    return swap.total_bytes, max(0, swap.total_bytes - swap.free_bytes)


def observe(
    document: Any,
    node_ids: Iterable[str],
    alias: str,
    declared_bytes: int,
    load_peak_bytes: int | None = None,
    minimum_swap_bytes: int = 0,
) -> dict[str, Any] | None:
    """Each lane Spark's use (total less available) and whether the run was alone there.

    Node telemetry is the run's own use only when the run is alone on that Spark.
    The swap each Spark reported at that moment and the recipe's declared load peak and
    minimum swap travel with it as evidence.
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
            "swap_total_bytes": swap_total,
            "swap_used_bytes": swap_used,
        }
        for node in fleet.nodes
        if node.id in wanted and node.telemetry is not None
        for sample in [node.telemetry.sample]
        for swap_total, swap_used in [_swap(node)]
        if 0 <= sample.memory_available_bytes <= sample.memory_total_bytes
        and sample.memory_total_bytes > 0
    ]
    if not observed:
        return None
    return {
        "declared_bytes": declared_bytes,
        "load_peak_bytes": load_peak_bytes,
        "minimum_swap_bytes": minimum_swap_bytes,
        "nodes": observed,
    }


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
        and not holds_within(memory["declared_bytes"], node["used_bytes"])
    ]


class _Placement(BaseModel):
    node_id: str
    role: str


class _Assignment(BaseModel):
    alias: str | None = None
    nodes: list[_Placement]
    option_choices: dict[str, str] = {}


class _IntendedProfile(BaseModel):
    assignments: list[_Assignment]


class _Progress(BaseModel):
    intended_profile: _IntendedProfile


class _Application(BaseModel):
    progress: _Progress


class _DetailIdentity(BaseModel):
    content_sha256: str


class _Detail(BaseModel):
    identity: _DetailIdentity
    document: dict[str, Any]


def intended_assignment(application: Any, alias: str) -> _Assignment:
    """The Controller application's assignment for ``alias``: its Sparks and option choices.

    Raises ``StopIteration`` when the alias is not assigned and a validation error when
    the document is not an application.
    """
    assignments = _Application.model_validate(
        application
    ).progress.intended_profile.assignments
    return next(item for item in assignments if item.alias == alias)


def ran_recipe(detail: Any, content_sha256: str) -> RecipeDefinition | None:
    """The recipe document ``vonkctl recipe detail --technical`` reports, if it is what ran.

    Bound by the Controller's own content identity; another content is ``None``.
    """
    try:
        parsed = _Detail.model_validate(detail)
        if parsed.identity.content_sha256 != content_sha256:
            return None
        return RecipeDefinition.model_validate(parsed.document)
    except ValidationError:
        return None


def measurement(
    observed: Mapping[str, Any] | None,
    application: Any,
    alias: str,
    recipe: RecipeDefinition | None,
    content_sha256: str,
) -> tuple[QualificationMeasuredRoleMemory, ...]:
    """The evidence rows this pass measured, keyed (content, role); none when unclear.

    ``observed`` is :func:`observe`'s sample; the Controller ``application`` document
    places each Spark in a role for the run's ``alias`` and names its option
    choices, so the budget is the one of the launch that ran. Only Sparks the run
    had to itself count, only roles the recipe declares; a role keeps the largest
    use over its Sparks. Measuring never costs a verdict: anything it cannot tell
    apart or compute is "not measured".
    """
    if not observed or recipe is None:
        return ()
    try:
        assignment = intended_assignment(application, alias)
        launch = recipe.with_option_choices(assignment.option_choices)
        budget = vllm_budget_bytes(launch)
    except (StopIteration, ValueError, TypeError, ArithmeticError):
        return ()
    declared = {role.name for role in launch.topology.roles}
    role_of = {item.node_id: item.role for item in assignment.nodes}
    steady: dict[str, int] = {}
    for node in observed["nodes"]:
        role = role_of.get(node["node_id"])
        if role in declared and node["alone"]:
            steady[role] = max(steady.get(role, 0), node["used_bytes"])
    try:
        return tuple(
            QualificationMeasuredRoleMemory(
                content_sha256=content_sha256,
                role=role,
                steady_bytes=used,
                budget_bytes=budget,
            )
            for role, used in sorted(steady.items())
        )
    except ValueError:
        return ()
