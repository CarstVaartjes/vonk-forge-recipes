"""Every scheduling decision of the sweep, as pure functions.

Nothing here talks to the fleet or the clock, so each rule is unit-testable:

* ``plan_groups``      which model to download and test next (value per byte);
* ``order_queue``      test order (cached first, shared models back to back);
* ``choose_mode`` and ``place_*``  single/dual windows and first-fit-decreasing packing;
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

from .catalog import Model, Recipe

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

    while unclaimed:
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
    revalidate: Collection[str] = (),
    deprioritised: Collection[str] = (),
    variants_last: bool = False,
) -> list[str]:
    """Test order: group by group; within a group by engine and base image (cache reuse)."""
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
        if key in deprioritised:
            return 2
        return 1 if key in revalidate else 0

    return sorted(ordered, key=lambda key: (tier(key), ordered.index(key)))


# --------------------------------------------------------------------------
# Placement: windows, bins, first-fit-decreasing
# --------------------------------------------------------------------------


def choose_mode(
    previous: str, singles_ready: int, duals_ready: int, min_dual_batch: int = 2
) -> str:
    """Run duals as a window (both Sparks drained once), not interleaved with singles."""
    if duals_ready == 0:
        return "single"
    if singles_ready == 0 or previous == "dual":
        return "dual"
    return "dual" if duals_ready >= min_dual_batch else "single"


@dataclass(frozen=True)
class SparkBin:
    id: str
    name: str
    capacity: int
    used: int = 0
    ports: frozenset[int] = frozenset()
    aliases: frozenset[str] = frozenset()


@dataclass(frozen=True)
class Placement:
    recipe: Recipe
    spark_ids: tuple[str, ...]


def never_fits(recipe: Recipe, capacity: int, reserve: int) -> bool:
    """Declared memory beyond an empty Spark: not testable on this fleet."""
    return recipe.memory_bytes > 0 and recipe.memory_bytes + reserve > capacity


def _fits(recipe: Recipe, spark: SparkBin, reserve: int, alias: str) -> bool:
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
    """First-fit-decreasing over the head of the queue; every Spark is a lane.

    ``bins`` carry what already runs. A recipe goes to the emptiest Spark it
    fits (declared ``memory_bytes`` plus reserve, disjoint ports and aliases),
    so two lanes fill before any Spark takes a second recipe. ``profile load
    --review`` still has the last word.
    """
    window = list(candidates[: max(4, 2 * len(bins))])
    window.sort(key=lambda r: -r.memory_bytes)  # stable: queue order breaks ties
    state = {b.id: b for b in bins}
    placed: list[Placement] = []
    for recipe in window:
        if recipe.node_count != 1:
            continue
        alias = alias_of(recipe)
        options = [b for b in state.values() if _fits(recipe, b, reserve, alias)]
        if not options:
            continue
        spark = min(options, key=lambda b: (b.used, b.id))
        state[spark.id] = SparkBin(
            spark.id,
            spark.name,
            spark.capacity,
            spark.used + recipe.memory_bytes,
            spark.ports | frozenset(recipe.ports),
            spark.aliases | {alias},
        )
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
            recipe.memory_bytes + reserve <= b.capacity for b in bins[:2]
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
_RULES: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "oom",
        re.compile(
            r"out of memory|\boom\b|exit(ed)? (with )?(code|status) 137|oom-?kill",
            re.IGNORECASE,
        ),
    ),
    (
        "build-policy",
        re.compile(r"dockerfile\.|heredoc|build_source|source policy", re.IGNORECASE),
    ),
    (
        "capacity",
        re.compile(
            r"storage\.|insufficient|resource\.|no space|free_space|shortfall|disk full",
            re.IGNORECASE,
        ),
    ),
    (
        "model-integrity",
        re.compile(
            r"digest mismatch|sha256 mismatch|checksum|corrupt|safetensors|unsupported (model )?architecture"
            r"|invalid model|incompatible checkpoint|state_dict|no such model file",
            re.IGNORECASE,
        ),
    ),
    (
        "network",
        re.compile(
            r"timed out|timeout|connection (reset|refused|aborted)|temporar|unavailable|\b50[234]\b|\b429\b"
            r"|rate limit|name resolution|dns|ssl|network|eof occurred|broken pipe",
            re.IGNORECASE,
        ),
    ),
)
_FIXED_CLASS = {"timeout": "timeout", "readiness": "readiness"}
# Failures that say something about the recipe or its model, not the platform: a new Controller
# release does not change them. Everything else (install, start, publication, review, admission,
# an unclassified download or build error) may have been the platform's.
RECIPE_SIDE_CLASSES = frozenset(
    {
        "model-integrity",
        "build-policy",
        "oom",
        "network",
        "timeout",
        "smoke",
        "smoke-assertion",
        "smoke-request",
        "smoke-timeout",
    }
)
TRANSIENT_CLASSES = frozenset({"network", "smoke-timeout"})
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

    @property
    def transient(self) -> bool:
        return self.klass in TRANSIENT_CLASSES

    @property
    def model_level(self) -> bool:
        return self.klass in MODEL_LEVEL_CLASSES


def platform_side(failure_class: str | None) -> bool:
    """Could a platform fix have changed this failure? Unknown classes count as the platform's."""
    return failure_class not in RECIPE_SIDE_CLASSES


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


# The assignment fields that carry the recipe's own data; the rest of a profile
# edit (name, Sparks, state) is what the sweep chose.
RECIPE_PROFILE_FIELDS = frozenset(
    {"recipe_selector", "option_choices", "model_variant"}
)
_ASSIGNMENT_FIELD = re.compile(r"assignments(?:\[\d+\]|\.\d+)\.([a-z_]+)")
_PROFILE_FIELD = re.compile(r"(?:\$|\bbody)\.([a-z_]+)")


def refused_profile_field(text: str) -> str | None:
    """The profile field a contract refusal names (``$.assignments[0].assignment_name``), if any."""
    match = _ASSIGNMENT_FIELD.search(text) or _PROFILE_FIELD.search(text)
    return match.group(1) if match else None


def classify(phase: str, code: str = "", detail: str = "", klass: str = "") -> Failure:
    """Name a failure by where it happened and what it said; same cause, same cluster.

    ``klass`` fixes the class when the caller already knows it (an assertion
    failure quotes model output, which must not be pattern-matched).
    """
    text = f"{code} {detail}"
    if (
        phase in _FIXED_CLASS
    ):  # our own wording, not the platform's: never pattern-match it
        klass = klass or _FIXED_CLASS[phase]
    klass = klass or next(
        (name for name, pattern in _RULES if pattern.search(text)), ""
    )
    if phase == "smoke" and klass == "network":
        klass = "smoke-timeout"
    if not klass:
        klass = {"smoke": "smoke", "review": "fit"}.get(phase, phase)
    signature = f"{phase}|{klass}|{code or '-'}|{_normalise(detail)}"
    cluster = hashlib.sha1(signature.encode()).hexdigest()[:10]
    return Failure(phase, klass, code, detail[:500], signature, cluster)


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
# Fixes feed back
# --------------------------------------------------------------------------


def requeue_reason(entry: Mapping[str, Any] | None, recipe: Recipe) -> str | None:
    """Why a recorded result no longer stands: the recipe document changed under it."""
    if not entry or entry.get("status") not in ("passed", "failed"):
        return None
    if entry.get("content_sha256") and entry["content_sha256"] != recipe.content_sha256:
        return (
            "failed-on-older-revision"
            if entry["status"] == "failed"
            else "passed-on-older-revision"
        )
    return None


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
