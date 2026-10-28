"""Every scheduling decision of the sweep, as pure functions.

Nothing here talks to the fleet or the clock, so each rule is unit-testable:

* ``plan_groups``      which model to download and test next (value per byte);
* ``order_queue``      test order (cached first, shared models back to back);
* ``place_*``        single/dual placement in queue order;
* ``load_timeout``     adaptive first-start timeouts learned per engine;
* ``classify`` and ``cluster``     failure classes, retry rules and root-cause clusters;
* ``RateTracker``      the measured download rate and the ETA built on it.
"""

from __future__ import annotations

import hashlib
import re
import statistics
from collections import defaultdict
from collections.abc import Callable, Collection, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from pydantic import ValidationError
from vonk_forge_contracts import QualificationCauseCode
from vonk_forge_contracts.sweep import (
    SweepFailureClass,
    WorkloadExitCause,
    WorkloadExitFacts,
    WorkloadFailureCode,
    WorkloadMemoryTrigger,
)

from .auth_contract import SmokeAuthClass
from .catalog import Model, Recipe
from .lifecycle import LifecycleState

Boost = Callable[[Recipe], float]


def no_boost(_recipe: Recipe) -> float:
    return 1.0


def keyword_boost(keywords: Sequence[str], factor: float = 4.0) -> Boost:
    """Boost recipes whose selector or title contains an owner-priority keyword."""
    lowered = [word.lower() for word in keywords if word]

    def boost(recipe: Recipe) -> float:
        text = f"{recipe.key} {recipe.title}".lower()
        return factor if any(word in text for word in lowered) else 1.0

    return boost


def recent_boost(now: float, days: float, factor: float = 1.5) -> Boost:
    """Boost recipes whose newest revision is at most ``days`` old."""

    def boost(recipe: Recipe) -> float:
        try:
            published = datetime.fromisoformat(recipe.updated_at).timestamp()
        except ValueError:
            return 1.0
        return factor if 0 <= now - published <= days * 86400 else 1.0

    return boost


def make_boost(
    keywords: Sequence[str],
    seed: Collection[str],
    now: float,
    recent_days: float = 3.0,
) -> Boost:
    """Owner-priority families, recently updated recipes, and the operator's seed list."""
    keyword = keyword_boost(keywords)
    recent = recent_boost(now, recent_days)

    def boost(recipe: Recipe) -> float:
        seeded = recipe.slug in seed or recipe.key in seed
        return keyword(recipe) * recent(recipe) * (1000.0 if seeded else 1.0)

    return boost


# --------------------------------------------------------------------------
# Value per byte: which model next
# --------------------------------------------------------------------------


def build_sizes(
    recipes: Iterable[Recipe], models: Mapping[str, Model]
) -> dict[str, int]:
    """Bytes per model digest: the Model's download size, else the recipes' declared artifact bytes."""
    sizes = {digest: model.bytes for digest, model in models.items() if model.bytes > 0}
    for recipe in recipes:
        if len(recipe.model_digests) == 1 and recipe.model_digests[0] not in sizes:
            sizes[recipe.model_digests[0]] = recipe.artifact_bytes
    return sizes


def present_models(models: Mapping[str, Model]) -> set[str]:
    """Models already on the NAS or being fetched: they are never requested again."""
    return {
        digest
        for digest, model in models.items()
        if model.local in ("cached", "preparing")
    }


@dataclass(frozen=True)
class GroupPlan:
    digests: frozenset[str]
    recipes: tuple[str, ...]
    new_bytes: (
        int  # model bytes that must still be downloaded when this group is reached
    )

    @property
    def cached(self) -> bool:
        return self.new_bytes == 0


def plan_groups(
    pending: Sequence[Recipe],
    sizes: Mapping[str, int],
    present: Collection[str],
    boost: Boost = no_boost,
) -> list[GroupPlan]:
    """Greedy order: cached models first (small first), then the most recipes per new byte.

    A recipe is unlocked once every model it needs is present, so recipes that
    share part of a model set cost nothing extra once the shared part is in.
    Picking a group makes its models present for the groups ranked after it.
    """
    unclaimed: dict[str, Recipe] = {r.key: r for r in pending}
    multi = [r for r in pending if len(r.model_set) > 1]
    have = set(present)
    plans: list[GroupPlan] = []

    def cost(digests: frozenset[str]) -> int:
        return sum(sizes.get(digest, 0) for digest in digests - have)

    for _ in range(len(unclaimed)):
        if not unclaimed:
            break
        own_by_set: dict[frozenset[str], list[Recipe]] = defaultdict(list)
        for recipe in unclaimed.values():
            own_by_set[recipe.model_set].append(recipe)
        best_key: tuple[Any, ...] | None = None
        best_set: frozenset[str] | None = None
        for digests, own in own_by_set.items():
            reachable = have | digests
            # Only multi-model recipes can be unlocked by someone else's group.
            extra = [
                r
                for r in multi
                if r.key in unclaimed
                and r.model_set != digests
                and r.model_set <= reachable
                and not r.model_set <= have
            ]
            value = len(own) + len(extra)
            factor = max(boost(r) for r in own)
            new = cost(digests)
            tie = tuple(sorted(digests))
            if new == 0:
                total = sum(sizes.get(d, 0) for d in digests)
                key: tuple[Any, ...] = (0, -factor, total, -value, tie)
            else:
                key = (1, -(value / new) * factor, new, -value, tie)
            if best_key is None or key < best_key:
                best_key, best_set = key, digests
        assert best_set is not None
        new_bytes = cost(best_set)
        reachable = have | best_set
        claimed = [r.key for r in own_by_set[best_set]] + [
            r.key
            for r in multi
            if r.key in unclaimed
            and r.model_set != best_set
            and r.model_set <= reachable
            and not r.model_set <= have
        ]
        for key_ in claimed:
            del unclaimed[key_]
        have |= best_set
        plans.append(GroupPlan(best_set, tuple(claimed), new_bytes))
    return plans


def order_queue(
    plans: Sequence[GroupPlan],
    recipes: Mapping[str, Recipe],
    *,
    cached: Collection[str],
    sizes: Mapping[str, int],
    revalidate: Collection[str] = (),
    deprioritised: Collection[str] = (),
    variants_last: bool = False,
    refreshed: Collection[str] = (),
) -> list[str]:
    """Cached singles, cached duals, then downloads; smallest models first.

    Explicit cache facts avoid treating greedy plans' projected reuse as NAS cache.
    Engine/image locality breaks ties between equally sized recipes.
    """
    cached_models = set(cached)
    head: list[str] = []
    tail: list[str] = []
    for plan in plans:
        members = sorted(
            (recipes[key] for key in plan.recipes if key in recipes),
            key=lambda r: (r.engine, r.base_image, r.key),
        )
        keys = [r.key for r in members]
        head += keys[:1] if variants_last else keys
        tail += keys[1:] if variants_last else []
    ordered = head + tail

    def tier(key: str) -> int:
        return int(key in deprioritised)

    positions = {key: index for index, key in enumerate(ordered)}

    def priority(key: str) -> tuple[int, int, int, int, int]:
        recipe = recipes[key]
        ready = recipe.model_set <= cached_models
        category = (0 if recipe.node_count == 1 else 1) if ready else 2
        size = sum(sizes.get(d, 0) for d in recipe.model_set)
        return tier(key), category, int(key in tail), size, positions[key]

    newest = sorted(
        (key for key in ordered if key in refreshed),
        key=lambda key: (recipes[key].updated_at, key),
        reverse=True,
    )
    ordered = newest + sorted(
        (key for key in ordered if key not in refreshed), key=priority
    )
    # Refresh recency and cache locality must never put pass revalidation ahead
    # of recipes still needing successful evidence, including deferred models.
    return sorted(ordered, key=lambda key: key in revalidate)


# --------------------------------------------------------------------------
# Placement: one recipe per Spark, in queue order
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class SparkBin:
    id: str
    name: str
    capacity: int
    used: int = 0
    ports: frozenset[int] = frozenset()
    aliases: frozenset[str] = frozenset()
    model_digests: frozenset[str] = frozenset()
    # An exact installed revision hints at runtime reuse; base images do not.
    installed_revisions: frozenset[str] = frozenset()

    disk_blocked_recipes: frozenset[str] = frozenset()

    def holds_models(self, recipe: Recipe) -> bool:
        return bool(recipe.model_set) and recipe.model_set <= self.model_digests


@dataclass(frozen=True)
class Placement:
    recipe: Recipe
    spark_ids: tuple[str, ...]


def never_fits(recipe: Recipe, capacity: int, reserve: int) -> bool:
    """Declared memory beyond an empty Spark: not testable on this fleet."""
    return recipe.memory_bytes > 0 and recipe.memory_bytes + reserve > capacity


def _fits(recipe: Recipe, spark: SparkBin, reserve: int, alias: str) -> bool:
    if recipe.key in spark.disk_blocked_recipes:
        return False
    if spark.used or spark.aliases:
        return False  # a single recipe owns its Spark for the entire attempt
    if alias in spark.aliases:
        return False
    if spark.ports & set(recipe.ports):
        return False
    if recipe.memory_bytes == 0:  # unknown demand: only on an empty Spark
        return spark.used == 0
    return spark.used + recipe.memory_bytes + reserve <= spark.capacity


def place_singles(
    candidates: Sequence[Recipe],
    bins: Sequence[SparkBin],
    reserve: int,
    alias_of: Callable[[Recipe], str],
) -> list[Placement]:
    """One single recipe per free Spark, in queue order.

    Memory fit is advisory; the Controller's review remains authoritative.
    A busy Spark never takes a second recipe, even with spare declared memory.
    The first fitting recipe never waits for locality. Later variants may yield
    once within this call, then a fallback pass fills any remaining free Spark.
    """
    # A queue window containing only duals or oversized singles must not hide
    # a ready single that fits the free Spark farther down the queue.
    singles = [recipe for recipe in candidates if recipe.node_count == 1]
    state = {b.id: b for b in bins}
    placed: list[Placement] = []
    claimed: set[str] = set()
    for locality_pass in (True, False):
        for recipe in singles:
            if recipe.key in claimed:
                continue
            alias = alias_of(recipe)
            options = [b for b in state.values() if _fits(recipe, b, reserve, alias)]
            if not options:
                continue

            if (
                locality_pass
                and placed
                and not any(b.holds_models(recipe) for b in options)
                and any(
                    b.holds_models(recipe) and _fits(recipe, b, reserve, alias)
                    for b in bins
                )
            ):
                continue  # its owner was assigned this round; offer the other lane later work
            spark = min(
                options,
                key=lambda b: (
                    not b.holds_models(recipe),
                    recipe.revision_id not in b.installed_revisions,
                    b.used,
                    b.id,
                ),
            )
            del state[spark.id]  # one owner, including recipes with unknown/zero memory
            claimed.add(recipe.key)
            placed.append(Placement(recipe, (spark.id,)))
    placed.sort(key=lambda p: candidates.index(p.recipe))
    return placed


def place_dual(
    candidates: Sequence[Recipe], bins: Sequence[SparkBin], reserve: int
) -> Placement | None:
    """A two-Spark recipe needs every Spark empty."""
    if len(bins) < 2 or any(b.used for b in bins):
        return None
    for recipe in candidates:
        if recipe.node_count == 2 and all(
            _fits(recipe, b, reserve, "") for b in bins[:2]
        ):
            return Placement(recipe, tuple(b.id for b in bins[:2]))
    return None


# --------------------------------------------------------------------------
# Adaptive timeouts
# --------------------------------------------------------------------------

GIB = 1024**3


@dataclass(frozen=True)
class TimeoutPolicy:
    first_start_seconds: float = 3600.0  # no samples yet: allow a first-start compile
    multiplier: float = 3.0
    floor_seconds: float = 1200.0
    cap_seconds: float = 7200.0
    min_expected_seconds: float = 300.0


DEFAULT_TIMEOUTS = TimeoutPolicy()


def expected_load_seconds(
    samples: Mapping[str, Sequence[Sequence[float]]], engine: str, model_bytes: int
) -> float | None:
    """Seconds per GiB learned from this engine's completed loads, else from all engines."""

    def per_gib(rows: Iterable[Sequence[float]]) -> list[float]:
        return [seconds / max(size / GIB, 1.0) for size, seconds in rows if seconds > 0]

    rates = per_gib(samples.get(engine, ()))
    if not rates:
        rates = per_gib(row for rows in samples.values() for row in rows)
    if not rates:
        return None
    return statistics.median(rates) * max(model_bytes / GIB, 1.0)


def load_timeout(
    samples: Mapping[str, Sequence[Sequence[float]]],
    engine: str,
    model_bytes: int,
    policy: TimeoutPolicy = DEFAULT_TIMEOUTS,
) -> float:
    expected = expected_load_seconds(samples, engine, model_bytes)
    if expected is None:
        return policy.first_start_seconds
    expected = max(expected, policy.min_expected_seconds)
    return min(
        policy.cap_seconds, max(policy.floor_seconds, expected * policy.multiplier)
    )


def learn(
    samples: dict[str, list[list[float]]],
    engine: str,
    model_bytes: int,
    seconds: float,
    keep: int = 20,
) -> None:
    rows = samples.setdefault(engine, [])
    rows.append([float(model_bytes), float(seconds)])
    del rows[:-keep]


# --------------------------------------------------------------------------
# Distribution: bytes moving to the Spark are watched for progress, not timed
# --------------------------------------------------------------------------

_COPYING = re.compile(r"cop(y|ying)|transfer|distribut|download|sync", re.IGNORECASE)
# Bumped when the load-timeout rules change; results from older rules may be retried once.
LOAD_RULES = 3
# Bumped when the smoke assertions change what fails a recipe. 2: a model's answer
# (the exact value) is a quality note, only a functional miss fails a recipe.
# 4: an answer's format (a bare integer, one word) is a quality note too.
SMOKE_RULES = 4


@dataclass(frozen=True)
class Distribution:
    copying: bool
    phase: str | None
    # Changes whenever the transfer moves: completed bytes and the Controller's last-progress time.
    signature: tuple[int, str] | None
    completed_bytes: int


def distribution(
    document: Mapping[str, Any], *, previous_bytes: int | None = None
) -> Distribution:
    """Is this profile application still moving bytes, and has it moved since last looked?"""
    child = (document.get("progress") or {}).get("child_progress") or {}
    operation = child.get("operation") or {}
    phase = child.get("phase")
    op_phase = str(operation.get("phase") or "")
    done = max(int(operation.get("completed_bytes") or 0), int(child.get("bytes") or 0))
    copying = bool(_COPYING.search(op_phase)) or (
        previous_bytes is not None and done > previous_bytes
    )
    signature = (done, str(operation.get("last_progress_at") or ""))
    return Distribution(
        copying,
        op_phase or (str(phase) if phase else None),
        signature if copying else None,
        done,
    )


def activity(document: Mapping[str, Any]) -> tuple[Any, ...]:
    """Forward movement of a load: it changes only when the load really advances.

    Phase and sub-phase, completed steps, bytes and items, and the Controller's last-progress
    time (it is only restamped when bytes, items or phase advance: operation_progress.py
    sample_progress). Never retries: attempt counters, retry times, the backoff view state and
    label flap while a load is stuck, and a poll time moves on every observation; counting
    them would keep a dead load alive forever.
    """
    progress = document.get("progress") or {}
    child = progress.get("child_progress") or {}
    operation = child.get("operation") or {}
    return (
        progress.get("completed_steps"),
        child.get("phase"),
        child.get("bytes"),
        operation.get("phase"),
        operation.get("completed_bytes"),
        operation.get("completed_items"),
        operation.get("last_progress_at"),
    )


def forward_marks(document: Mapping[str, Any]) -> tuple[int | None, ...]:
    """The load's counters that only move forward while it really advances.

    The Controller max-merges ``child_progress.bytes`` (run_switch_progress.py) and only
    raises ``completed_steps``; ``operation`` is the child's per-attempt snapshot, which a
    retry resets, so only its item count is kept and only as a high-water mark
    (``verdict.progress_verdict``). Phases and progress times are not counters: they flap.
    """
    progress = document.get("progress") or {}
    child = progress.get("child_progress") or {}
    operation = child.get("operation") or {}
    return (
        child.get("bytes"),
        progress.get("completed_steps"),
        operation.get("completed_items"),
    )


# --------------------------------------------------------------------------
# Failure classes, retry, clusters
# --------------------------------------------------------------------------

_CHILD_PHASE = {
    "model-download": "download",
    "container-download": "download",
    "container-build": "build",
    "target-copy": "install",
    "transfer": "install",
    "runtime-install": "install",
    "prepare": "install",
    "start": "start",
    "final-verify": "start",
    "final_verify": "start",
    "verify": "start",
}
# The one table of recipe-data codes, for CLI errors and Controller failures alike:
# attribution follows typed codes; platform detail text is evidence only.
RECIPE_FAILURE_CODES: dict[str, str] = {
    QualificationCauseCode.INVALID_DEFINITION: "build-policy",
    QualificationCauseCode.INVALID_DOCUMENT: "build-policy",
    QualificationCauseCode.INVALID_OPTION: "build-policy",
    QualificationCauseCode.RECIPE_INTEGRITY: "model-integrity",
    QualificationCauseCode.MODEL_INTEGRITY: "model-integrity",
    QualificationCauseCode.BUILD_POLICY: "build-policy",
    QualificationCauseCode.RUNTIME_OOM: "oom",
    QualificationCauseCode.RUNTIME_EXIT: "start",
}
_CODE_CLASSES = {
    **RECIPE_FAILURE_CODES,
    "run-switch.resource.insufficient_capacity": "capacity",
    "storage.insufficient_capacity": "capacity",
    "controller.transport_timeout": "network",
}
_FIXED_CLASS = {"timeout": "timeout", "readiness": "readiness"}
# Only typed expiry/revocation is renewable; route and other auth denials are final.
CLIENT_CREDENTIAL = SmokeAuthClass.CREDENTIAL
MODEL_LEVEL_CLASSES = frozenset({"model-integrity"})


@dataclass(frozen=True)
class Failure:
    phase: (
        str  # download | review | install | build | start | readiness | smoke | timeout
    )
    klass: str
    code: str
    detail: str
    signature: str
    cluster: str
    # What was seen when it failed (case, HTTP status, response excerpt, operation id, ...).
    # ``classify`` always fills it: a failure nobody can diagnose later is a bug.
    evidence: Mapping[str, Any] = field(default_factory=dict, compare=False)

    def describe(self) -> str:
        """The result row's error text: the detail plus the evidence that explains it."""
        extra = []
        for name in ("case", "http_status", "operation_id"):
            if self.evidence.get(name) not in (None, ""):
                extra.append(f"{name}={self.evidence[name]}")
        body = self.evidence.get("body")
        if body:
            extra.append("body=" + " ".join(str(body).split())[:300])
        return " ".join([self.detail, *(f"[{e}]" for e in extra)])[:1024]

    @property
    def model_level(self) -> bool:
        return self.klass in MODEL_LEVEL_CLASSES


_NOT_REFS = frozenset({"code", "detail", "message", "severity"})


def _leaves(value: Any, skip: frozenset[str] = frozenset()) -> list[str]:
    """Every string in a nested document (the identifiers a blocker or failure carries)."""
    if isinstance(value, str):
        return [value]
    if isinstance(value, Mapping):
        return [
            leaf
            for name, item in value.items()
            if name not in skip
            for leaf in _leaves(item)
        ]
    if isinstance(value, (list, tuple)):
        return [leaf for item in value for leaf in _leaves(item)]
    return []


def admission_blockers(document: Mapping[str, Any]) -> list[dict[str, Any]]:
    """What holds a profile application back: error blockers and a phase that keeps being retried.

    A warning is a brief contention hold and never counts; the Controller raises ``phase-retry``
    only once a phase has been repeated. ``refs`` carries what else the blocker names (its
    assignment, alias, node or recipe), whatever the field is called.
    """
    held = (
        (document.get("progress") or {}).get("blockers")
        or document.get("blockers")
        or []
    )
    return [
        {
            "code": str(item.get("code", "")),
            "detail": str(item.get("detail", ""))[:300],
            "refs": _leaves(item, _NOT_REFS)[:20],
        }
        for item in held
        if isinstance(item, Mapping)
        and (
            item.get("severity") == "error"
            or item.get("code") == "run-switch.phase-retry"
        )
    ]


def _mentions(text: str, name: str) -> bool:
    return len(name) >= 3 and (
        re.search(rf"(?<![\w.-]){re.escape(name)}(?![\w.-])", text) is not None
    )


@dataclass(frozen=True)
class AssignmentFailure:
    """One failed assignment of an application, resolved to the lane it belongs to.

    ``FleetProfileAssignmentFailure`` names an assignment by id only; the alias and nodes come
    from ``progress.intended_profile``. ``from_effect`` marks the failed ``run`` effect of a
    lane adopted from an earlier application, which is that lane's own result.
    """

    reason: str
    assignment_id: str | None = None
    alias: str | None = None
    node_ids: tuple[str, ...] = ()
    failure_code: str | None = None
    from_effect: bool = False

    def as_state(self) -> dict[str, Any]:
        """The JSON row persisted with the settled application and read by ``blamed``."""
        return {
            "reason": self.reason,
            "assignment_id": self.assignment_id,
            "alias": self.alias,
            "node_ids": list(self.node_ids),
            "failure_code": self.failure_code,
            "from_effect": self.from_effect,
        }


def assignment_failures(document: Mapping[str, Any]) -> list[AssignmentFailure]:
    """The application's failures resolved to their assignments by id (never by text or node)."""
    progress = document.get("progress")
    if not isinstance(progress, Mapping):
        return []
    owners = {
        a.get("id"): a
        for a in (progress.get("intended_profile") or {}).get("assignments") or []
        if isinstance(a, Mapping)
    }

    def resolved(
        assignment_id: object, reason: str, **fields: Any
    ) -> AssignmentFailure:
        owner = owners.get(assignment_id) or {}
        return AssignmentFailure(
            reason=reason,
            assignment_id=assignment_id if isinstance(assignment_id, str) else None,
            alias=owner.get("alias"),
            node_ids=tuple(
                n["node_id"]
                for n in owner.get("nodes") or []
                if isinstance(n, Mapping) and isinstance(n.get("node_id"), str)
            ),
            **fields,
        )

    failures = [
        resolved(f.get("assignment_id"), str(f.get("reason") or ""))
        for f in (progress.get("switch_adapter") or {}).get("assignment_failures") or []
        if isinstance(f, Mapping)
    ]
    named = {f.assignment_id for f in failures}
    for effect in progress.get("effects") or []:
        if (
            not isinstance(effect, Mapping)
            or effect.get("kind") != "run"
            or effect.get("state") != "failed"
            or effect.get("application_id") == document.get("id")
            or effect.get("target_id")
            not in owners  # a run effect's target is its assignment
            or effect["target_id"] in named
        ):
            continue
        code = ((effect.get("result") or {}).get("run_switch") or {}).get(
            "failure_code"
        )
        members = (effect.get("progress") or {}).get("members") or []
        error = next(
            (
                str(m["error"])
                for m in members
                if isinstance(m, Mapping) and m.get("error")
            ),
            None,
        )
        failures.append(
            resolved(
                effect["target_id"],
                error or code or "the adopted run failed",
                failure_code=code if isinstance(code, str) else None,
                from_effect=True,
            )
        )
    return failures


def blamed(
    items: Iterable[Mapping[str, Any]], lanes: Mapping[str, Mapping[str, Sequence[str]]]
) -> set[str]:
    """The lanes the blockers or assignment failures name; empty when they name none.

    ``lanes`` maps a recipe key to ``names`` (its alias, key and slug: a match is the recipe)
    and ``places`` (its node ids and Spark names: a match only says where it was placed).
    A recipe named outright wins; a Spark is only evidence when nothing names a recipe.
    """
    named: set[str] = set()
    placed: set[str] = set()
    for item in items:
        refs = {r for r in (item.get("refs") or _leaves(item, _NOT_REFS))}
        text = " ".join(
            str(item.get(f, "")) for f in ("detail", "message", "reason", "error")
        )
        for key, lane in lanes.items():
            if any(n in refs or _mentions(text, n) for n in lane.get("names", ())):
                named.add(key)
            if any(n in refs or _mentions(text, n) for n in lane.get("places", ())):
                placed.add(key)
    return named or placed


def child_phase(name: str | None) -> str | None:
    return _CHILD_PHASE.get(name or "")


def _normalise(detail: str) -> str:
    text = re.sub(
        r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}",
        "<id>",
        detail,
        flags=re.IGNORECASE,
    )
    text = re.sub(r"\b[0-9a-f]{12,}\b", "<hex>", text, flags=re.IGNORECASE)
    text = re.sub(r"(/[\w.\-]+){2,}", "<path>", text)
    text = re.sub(r"\d+", "N", text)
    return " ".join(text.split())[:120].lower()


EVIDENCE_BODY_LIMIT = 2048
_BEARER = re.compile(r"(?i)(bearer\s+)[A-Za-z0-9._~+/=-]+")
_SECRETS = re.compile(
    r"(?i)((?:authorization|api[_-]?key|access[_-]?token|token|secret|password)"
    r"[\"']?\s*[:=]\s*[\"']?)[^\s\"',}]+|\bsk-[A-Za-z0-9_-]{8,}"
)


def redact(text: str, secrets: Iterable[str] = ()) -> str:
    """Remove bearer tokens, key-looking values and the given literal secrets."""
    for secret in secrets:
        if secret:
            text = text.replace(secret, "<redacted>")
    text = _BEARER.sub(r"\1<redacted>", text)
    return _SECRETS.sub(lambda m: (m.group(1) or "") + "<redacted>", text)


def excerpt(
    text: Any, limit: int = EVIDENCE_BODY_LIMIT, secrets: Iterable[str] = ()
) -> str:
    """A redacted, bounded piece of a response body or value."""
    return redact(str(text), secrets)[:limit]


def classify_application_failure(
    phase: str, code: str, detail: str, *, evidence: Mapping[str, Any] | None = None
) -> Failure:
    """Attribute fixed Controller exit facts only at the workload start boundary.

    Assignment reasons carry the helper's token summary, not a recipe code.
    Exception text can itself contain exit_cause=...; it is never a fact.
    Explicit infrastructure codes and installation/admission failures stay platform-owned.
    """
    if phase not in {"start", "readiness"} or code not in WorkloadFailureCode:
        return classify(phase, code, detail, evidence=evidence)
    facts_text = detail.split(" exception=", 1)[0]
    facts = re.findall(r"(?:^|\s)(exit_cause|trigger)=([a-z_]+)(?=\s|$)", facts_text)
    if not facts and WorkloadFailureCode.RUNTIME_PROCESS_EXITED not in facts_text:
        return classify(phase, code, detail, evidence=evidence)
    try:
        if len({name for name, _ in facts}) != len(facts):
            raise ValueError("ambiguous exit facts")
        exit_facts = WorkloadExitFacts.model_validate(dict(facts))
    except (ValidationError, ValueError):
        exit_facts = None
    recipe_code = None
    if exit_facts is not None:
        if exit_facts.exit_cause in {
            WorkloadExitCause.BAD_ARGUMENTS,
            WorkloadExitCause.PYTHON_EXCEPTION,
            WorkloadExitCause.MISSING_FILE,
            WorkloadExitCause.UNSUPPORTED_GPU_ARCH,
            WorkloadExitCause.IMAGE_ARCH_MISMATCH,
        }:
            recipe_code = QualificationCauseCode.RUNTIME_EXIT
        elif exit_facts.exit_cause == WorkloadExitCause.OOM_KILLED or (
            exit_facts.exit_cause == WorkloadExitCause.HOST_MEMORY_EXHAUSTED
            and exit_facts.trigger == WorkloadMemoryTrigger.NVIDIA_ALLOCATION_FAILURE
        ):
            recipe_code = QualificationCauseCode.RUNTIME_OOM
    return classify(
        phase,
        recipe_code or code,
        detail,
        klass="" if recipe_code else SweepFailureClass.PLATFORM_EXIT,
        evidence={**(evidence or {}), "controller_code": code},
    )


def classify(
    phase: str,
    code: str = "",
    detail: str = "",
    klass: str = "",
    evidence: Mapping[str, Any] | None = None,
) -> Failure:
    """Name a failure by where it happened and what it said; same cause, same cluster.

    ``klass`` fixes the class when the caller already knows it (an assertion
    failure quotes model output, which must not be pattern-matched).
    """
    if (
        phase in _FIXED_CLASS
    ):  # our own wording, not the platform's: never pattern-match it
        klass = klass or _FIXED_CLASS[phase]
    klass = klass or _CODE_CLASSES.get(code, "")
    if phase == "smoke" and klass == "network":
        klass = "smoke-timeout"
    if not klass:
        klass = {"smoke": "smoke", "review": "fit"}.get(phase, phase)
    signature = f"{phase}|{klass}|{code or '-'}|{_normalise(detail)}"
    cluster = hashlib.sha1(signature.encode()).hexdigest()[:10]
    proof: dict[str, Any] = {
        k: v for k, v in (evidence or {}).items() if v not in (None, "", [], {})
    }
    proof.setdefault("phase", phase)
    proof.setdefault("code", code or "-")
    proof.setdefault("detail", excerpt(detail, 500))
    return Failure(phase, klass, code, detail[:500], signature, cluster, proof)


def evidence_of(entry: Mapping[str, Any]) -> dict[str, Any]:
    """The structured evidence of a result row."""
    value = entry.get("evidence")
    if isinstance(value, Mapping):
        return dict(value)
    return {}


@dataclass(frozen=True)
class Cluster:
    id: str
    signature: str
    recipes: tuple[str, ...]


def cluster(entries: Mapping[str, Mapping[str, Any]]) -> list[Cluster]:
    """Group failed recipes by root-cause signature, biggest cluster first."""
    groups: dict[str, list[str]] = defaultdict(list)
    signatures: dict[str, str] = {}
    for key, entry in entries.items():
        if entry.get("status") == "failed" and entry.get("cluster"):
            groups[entry["cluster"]].append(key)
            signatures[entry["cluster"]] = str(entry.get("signature", ""))
    return sorted(
        (
            Cluster(cid, signatures[cid], tuple(sorted(keys)))
            for cid, keys in groups.items()
        ),
        key=lambda c: (-len(c.recipes), c.id),
    )


# --------------------------------------------------------------------------
# Throughput
# --------------------------------------------------------------------------


@dataclass
class RateTracker:
    """Smoothed bytes per second of the downloads that were actually moving."""

    alpha: float = 0.2
    rate: float = 0.0
    samples: int = 0
    history: list[float] = field(default_factory=list)

    def add(self, bytes_per_second: float) -> None:
        if bytes_per_second <= 0:
            return
        self.rate = (
            bytes_per_second
            if not self.samples
            else self.alpha * bytes_per_second + (1 - self.alpha) * self.rate
        )
        self.samples += 1
        self.history = (self.history + [bytes_per_second])[-50:]

    def eta_seconds(self, remaining_bytes: int) -> float | None:
        return remaining_bytes / self.rate if self.rate > 0 else None


@dataclass(frozen=True)
class Supersession:
    """An application the Controller replaced: never a recipe failure."""

    successor: str | None


def supersession(document: Mapping[str, Any]) -> Supersession | None:
    """Was this application replaced (by an automatic retry or a later intent)?

    ``superseded_by`` names the successor to adopt; without one the intent is requeued.
    """
    if document.get("state") != LifecycleState.SUPERSEDED:
        return None
    successor = document.get("superseded_by")
    return Supersession(successor if isinstance(successor, str) and successor else None)
