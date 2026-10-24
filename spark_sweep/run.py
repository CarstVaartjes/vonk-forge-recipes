"""The sweep itself: lanes over one sweep profile, fed by the prefetcher.

Profiles cover the whole fleet (an unassigned Spark is idle and its workload is
stopped by a load), so the lanes are *assignments of one sweep profile*, not
one profile per lane. A lane swap edits that profile and loads it; the planner
keeps every unchanged assignment (``--review`` is checked to prove it).
Unchanged running and cleanup effects are carried by canonical adoption links,
so an independent lane can progress while another exact child remains pending.

``Sweep.tick`` is a small state machine, so the whole sweep is deterministic
under a fake clock and a fake ``vonkctl``:

    refresh catalog -> owner guard -> advance slots/receipts -> schedule -> prefetch -> status
"""

from __future__ import annotations

import itertools
import json
import signal
import threading
import time
import traceback
from collections.abc import Callable, Iterable, Mapping, Sequence
from concurrent.futures import Future
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

from pydantic import ValidationError
from vonk_forge_contracts import (
    QualificationStatus,
    QualificationStatusView,
    RecipeQualificationEvidence,
)
from vonk_forge_contracts.sweep import (
    CleanupEnd,
    CleanupEndReason,
    CleanupFleetObservation,
    CleanupObservationBudget,
    FleetOnlineState,
    InfrastructureIncident,
    ReviewInfrastructureSource,
    SweepSlotPhase,
    SweepStorageReview,
    SweepStorageWait,
    SweepWait,
    SweepWaitReason,
)

from qualification.memory_need import SPARK_MEMORY_BYTES

from . import cleanup, memory, policy, recovery, verdict
from .auth_contract import SmokeAuthClass
from .catalog import (
    Fleet,
    Listing,
    Model,
    Recipe,
    as_int,
    dig,
    fetch_fleet,
    model_listing,
    recipe_listing,
    serving_run,
)
from .definitions import Definitions
from .evidence_publisher import EvidencePublisher
from .finalization import FinalAction, Finalization
from .identity import (
    AcceptedRequest,
    ControllerWaitHold,
    IdentityReason,
    VerdictIdentity,
    content_identity,
    verdict_matches,
)
from .lifecycle import ACTIVE as ACTIVE_APP
from .lifecycle import TERMINAL as SETTLED
from .lifecycle import (
    LifecycleState,
    RecipeStatus,
    ReconciliationStatus,
    ReleaseAuthority,
    RetestReason,
    SkipReasonKind,
)
from .owner import OwnerGuard, OwnerStatus
from .prefetch import SWEEP_LABEL, PrefetchConfig, Prefetcher
from .profile_alias import profile_alias, unique_profile_alias
from .qualification import (
    current_identity,
    from_history,
    local_status,
    merge_evidence,
    read_evidence,
    scheduling_priority,
)
from .removal import ModelRemover
from .removal_contract import RemovalWait
from .report import write_status
from .smoke import (
    HttpConfig,
    SmokeResult,
    smoke_readiness,
    smoke_service,
)
from .smoke_job import job_budget_seconds, smoke_job
from .state import ResultsLog, State, StateLock
from .tolerant import InputReason, read_optional
from .verdict import Disposition, Progress
from .vonkctl import Reply, Vonkctl, VonkctlError, request_key

WAIT_EVENT_SECONDS = 60.0  # a wait says what it waits for at least this often
# Clearing the fleet and reviewing again, for a capacity refusal (an OBSERVE outcome),
# before it counts against the recipe.
MAX_REVIEW_FREEING = 2
TAKEOVER_RETRY_SECONDS = 30.0
SKEW_CHECK_INTERVAL = 300.0
INFRA_BASE = 30.0
INFRA_CAP = 600.0
RETEST_SECONDS = 24 * 3600.0
ENDPOINT_RECHECK_SECONDS = 20.0
# A results row carries the Controller document it ended on, bounded.
DOCUMENT_EVIDENCE_LIMIT = 65536


@dataclass(frozen=True)
class ProfilePruneOutcome:
    waiting_on: str | None = None


class SweepProfileOwnershipError(RuntimeError):
    """Destructive profile edits may not adopt another requester's assignments."""


class DaemonExecutor:
    """Smoke tests on daemon threads: a request that hangs never keeps an interrupted sweep alive."""

    def submit(self, fn: Callable[..., Any], *args: Any) -> Future[Any]:
        future: Future[Any] = Future()

        def work() -> None:
            if not future.set_running_or_notify_cancel():
                return
            try:
                future.set_result(fn(*args))
            except BaseException as error:  # noqa: BLE001 - delivered through the future
                future.set_exception(error)

        threading.Thread(target=work, daemon=True, name="smoke").start()
        return future

    def shutdown(self, **_: Any) -> None:
        return None


def _interrupt(_signum: int, _frame: Any) -> None:
    raise KeyboardInterrupt


def _install_signal_handlers() -> dict[int, Any]:
    """SIGINT and SIGTERM raise KeyboardInterrupt, even when started with SIGINT ignored.

    A process started in the background of a non-interactive shell inherits SIGINT as
    ignored, and Python then installs no handler of its own: Ctrl-C and ``kill -INT``
    would do nothing for as long as it ran.
    """
    if threading.current_thread() is not threading.main_thread():
        return {}
    previous: dict[int, Any] = {}
    for number in (signal.SIGINT, signal.SIGTERM):
        previous[number] = signal.signal(number, _interrupt)
    return previous


def _restore_signal_handlers(previous: Mapping[int, Any]) -> None:
    for number, handler in previous.items():
        signal.signal(number, handler)


class Clock(Protocol):
    def now(self) -> float: ...
    def sleep(self, seconds: float) -> None: ...


class RealClock:
    def now(self) -> float:
        return time.time()

    def sleep(self, seconds: float) -> None:
        time.sleep(seconds)


FLEET_SKIP = SkipReasonKind.FLEET
#: Skipped because the recipe left the complete catalog; undone when it comes back.
CATALOG_SKIP = SkipReasonKind.CATALOG
#: Skipped because the platform cannot load this kind of recipe through a fleet profile yet:
#: the sweep's own decision, neither a recipe verdict nor a platform fault. It heals by itself
#: once ``_platform_cannot_load`` no longer holds for the recipe.
PLATFORM_SKIP = SkipReasonKind.PLATFORM
JOB_SKIP_CODE = "platform.job_recipes_not_loadable"


def _fleet_skip(entry: Mapping[str, Any]) -> bool:
    """A skip whose typed reason depends on the fleet."""
    return (
        entry.get("status") == RecipeStatus.SKIPPED
        and entry.get("reason_kind") == FLEET_SKIP
    )


@dataclass
class SweepConfig:
    state_dir: Path
    authority_id: str
    sweep_profile: int = 10
    pin_profile: int | None = 13
    sparks: tuple[str, ...] = ()
    reserve_bytes: int = 8_000_000_000
    default_spark_memory: int = SPARK_MEMORY_BYTES
    poll_seconds: float = 10.0
    catalog_seconds: float = 60.0
    status_seconds: float = 15.0
    idle_alert_seconds: float = 600.0
    owner_hold_seconds: float = 1800.0
    owner_poll_seconds: float = 15.0
    retry_delay: float = 60.0
    defer_delay: float = 120.0
    http: HttpConfig = field(default_factory=HttpConfig)
    prefetch: PrefetchConfig = field(default_factory=PrefetchConfig)
    timeouts: policy.TimeoutPolicy = field(default_factory=policy.TimeoutPolicy)
    boost: tuple[str, ...] = ("glm-5-3",)
    recent_days: float = 3.0
    seed: frozenset[str] = frozenset()
    only: tuple[str, ...] = ()
    limit: int = 0
    variants_last: bool = False
    revalidate_passes: bool = True
    restore_owner: int | None = None
    stop_at_end: bool = True
    watch_seconds: float = 0.0
    allow_version_skew: bool = False
    release_seconds: float = 300.0
    retry_failed: bool = False
    publish_evidence: bool = False
    # The stall window while bytes are copied to a Spark (the learned window covers the rest).
    copy_stall_seconds: float = 600.0
    # Preparation waits use this shorter bound, capped by the lane budget.
    # Expiry re-plans without spending an attempt.
    blocked_release_seconds: float = 300.0
    # The deployed Controller loads job recipes as profile members (vonk-forge #1491) and
    # reports a held job run healthy (#1494). Off: job recipes are skipped before any lane.
    job_recipes_loadable: bool = False


#: The traceback frames a sweep crash keeps as evidence: where it raised, and its callers.
CRASH_FRAMES = 3


def crash_evidence(error: BaseException) -> dict[str, Any]:
    """What a crash of the sweep's own code says: type, message, filename, last frames."""
    frames = [
        line.strip()
        for line in traceback.format_tb(error.__traceback__)[-CRASH_FRAMES:]
    ]
    return {
        "type": type(error).__name__,
        "message": str(error)[:500],
        "filename": str(getattr(error, "filename", None) or "") or None,
        "frames": [frame[:400] for frame in frames],
    }


def alias_for(recipe: Recipe, definitions: Definitions) -> str:
    """The name the lane serves under: the reviewed service alias, else the slug.

    Always a valid profile alias, so a spelling the contract refuses (capitals,
    a ``/``) never reaches the Controller.
    """
    return profile_alias(definitions.alias(recipe.key) or recipe.slug)


class Sweep:
    def __init__(
        self,
        config: SweepConfig,
        vk: Vonkctl,
        state: State,
        log: ResultsLog,
        definitions: Definitions,
        clock: Clock | None = None,
        executor: Any = None,
    ) -> None:
        self.cfg = config
        self.vk = vk
        self.state = state
        self.log = log
        self.defs = definitions
        self.clock = clock or RealClock()
        self.executor = executor or DaemonExecutor()
        self.futures: dict[str, Future[SmokeResult]] = {}
        self.smoke_stops: dict[str, threading.Event] = {}
        self.catalog_executor = DaemonExecutor()
        self.catalog_future: Future[Reply] | None = None
        self.catalog_active: Listing[Recipe] | Listing[Model] | None = None
        # The cached listings: scheduling reads these, a refresh updates them page by page.
        self.recipe_list = recipe_listing(vk)
        self.model_list = model_listing(vk)
        self.recipes: dict[str, Recipe] = self.recipe_list.rows
        self.models: dict[str, Model] = self.model_list.rows
        self.fleet = Fleet((), ())
        self.release_at = self.clock.now()  # check_client has just read it
        self.status_at = -1e18
        self.owner_status = OwnerStatus()
        # Visible application alarms never reserve unrelated lanes.
        self.stall_reason = ""
        self.stalled_applications: set[str] = set()
        self.idle_since: float | None = None
        self.blind = False  # this tick could not observe the Controller
        self.unobserved_loads: set[str] = set()
        # This pass's free Sparks whose placement review did not adopt a loading lane.
        self.adoption_holds: dict[str, str] = {}
        self._skew_checked_at = float("-inf")
        self.library_commit = ""
        self.rate = policy.RateTracker()
        self.rate.rate = float(state.data["rate"].get("ema", 0.0))
        self.rate.samples = int(state.data["rate"].get("samples", 0))
        self.guard = OwnerGuard(vk, state, config.owner_hold_seconds)
        self.model_remover = ModelRemover(vk, state)
        # This is a projection of live catalog/owner observations, never a
        # persisted authorization or a gate to carry into a new process.
        self.state.data.pop("model_removal_wait", None)
        self.prefetcher = Prefetcher(
            vk,
            state,
            config.prefetch,
            self.rate,
            self.clock.now,
            self._download_failed,
            self._download_done,
        )
        self.prefetcher.on_infra = self.note_infra
        self.prefetcher.on_ok = self.clear_infra
        self.boost = self._build_boost()
        self.queue: list[str] = []
        self.sizes: dict[str, int] = {}
        self.last_prefetch: dict[str, Any] = {}
        self.cached_models: set[str] = set()
        self.idle_ticks = 0
        self._assigned: dict[str, str] = {}
        self._wait_event_at = -1e18
        # The latest Controller document of each load (by request key): evidence for its rows.
        self.documents: dict[str, Mapping[str, Any]] = {}
        self.retry_failed_pending = config.retry_failed
        self.qualification_evidence = {
            key: merge_evidence(
                read_evidence(Path(__file__).resolve().parents[1], key),
                RecipeQualificationEvidence.model_validate(value),
            )
            for key, value in state.data.get("qualification_evidence", {}).items()
        }
        self.evidence_publisher = EvidencePublisher(
            log.path,
            config.state_dir / "evidence-publisher",
            Path(__file__).resolve().parents[1] / "tools/sync-qualification-evidence",
        )
        for row in log.entries():
            self._remember_qualification(row)

    def _remember_qualification(self, row: Mapping[str, Any]) -> None:
        try:
            record = from_history(row)
            if record is None:
                return
            key = str(row["recipe"])
            evidence = RecipeQualificationEvidence(recipe_key=key, records=(record,))
            previous = self.qualification_evidence.get(key) or read_evidence(
                Path(__file__).resolve().parents[1], key
            )
            merged = merge_evidence(previous, evidence)
            self.qualification_evidence[key] = merged
            self.state.data.setdefault("qualification_evidence", {})[key] = (
                merged.model_dump(mode="json")
            )
        except (ValueError, TypeError, OSError) as error:
            self.state.event(
                f"qualification evidence unavailable: {type(error).__name__}"
            )

    def qualification_view(self, recipe: Recipe) -> QualificationStatusView | None:
        try:
            evidence = self.qualification_evidence.get(recipe.key) or read_evidence(
                Path(__file__).resolve().parents[1], recipe.key
            )
            return local_status(
                evidence,
                recipe,
                len(self.enrolled()),
                recipe.key in self.state.slots
                and self.state.slots[recipe.key].get("phase")
                != SweepSlotPhase.FINISHED,
            )
        except (ValueError, TypeError, OSError):
            return None  # malformed observed identity cannot certify public proof

    # ------------------------------------------------------------------ setup

    def _build_boost(self) -> policy.Boost:
        return policy.make_boost(
            self.cfg.boost, self.cfg.seed, self.clock.now(), self.cfg.recent_days
        )

    def preflight(self) -> None:
        """Read-only facts before anything is changed: fleet, owner baseline, owner profile export."""
        self.fleet = fetch_fleet(self.vk, self.cfg.default_spark_memory)
        if not any(s.online for s in self.fleet.sparks):
            raise VonkctlError(
                ("fleet",), None, "no Spark is online; observing enrollment"
            )
        usable = self.sparks()
        if usable and not self.prefetcher.config.pin_spark:
            self.prefetcher.config = replace(
                self.prefetcher.config, pin_spark=usable[0].name
            )
        self.claim_profile(self.cfg.sweep_profile, "Hardware sweep")
        if self.cfg.prefetch.pin_profile is not None:
            self.claim_profile(
                self.cfg.prefetch.pin_profile, "Hardware sweep pins (never loaded)"
            )
        if not self.state.data["owner"]["export"]:
            for number in sorted(OWNER_PROFILES_READ):
                reply = self.vk.run("profile", "export", profile=number)
                if reply.ok and isinstance(reply.document, dict):
                    self.state.data["owner"]["export"][str(number)] = reply.document
        self.owner_status = self.guard.check(self.clock.now())
        self.state.save()

    def claim_profile(self, number: int, name: str) -> None:
        """Use a profile only if it is empty or already ours: never adopt someone's saved profile."""
        reply = self.vk.run("profile", "export", profile=number)
        if not reply.ok or not isinstance(reply.document, dict):
            raise VonkctlError(
                reply.argv, reply, "profile ownership is unreadable; retry observation"
            )
        definition = reply.document
        labels = definition.get("labels") or {}
        if labels.get(SWEEP_LABEL[0]) == SWEEP_LABEL[1]:
            return
        if definition.get("assignments"):
            raise SweepProfileOwnershipError(
                f"profile {number} already holds assignments that are not the sweep's; "
                "choose another number"
            )
        self.vk.call(
            "profile",
            "configure",
            "--name",
            name,
            "--label",
            "=".join(SWEEP_LABEL),
            "--yes",
            profile=number,
        )

    # ---------------------------------------------------------------- catalog

    def enrolled(self) -> list[Any]:
        """The Sparks the fleet listing enrolls (the chosen ones), online or not: the topology."""
        if self.cfg.sparks:
            picked = [self.fleet.by_ref(ref) for ref in self.cfg.sparks]
            return [s for s in picked if s is not None]
        return list(self.fleet.sparks)

    def sparks(self) -> list[Any]:
        """The lanes that can take a placement now: enrolled and online."""
        return [s for s in self.enrolled() if s.online]

    def lanes_short(self, recipe: Recipe) -> int:
        """How many more online lanes a recipe waits for (0: it can be placed now)."""
        return max(0, recipe.node_count - len(self.sparks()))

    @property
    def catalog_loaded(self) -> bool:
        """Nothing is done, or stuck, before the recipe library has been read once."""
        return self.recipe_list.passes > 0

    def advance_catalog(self, now: float) -> None:
        """One independent in-flight read; only the owner thread merges its reply.

        A library timeout never holds lane observation or status publication. Retry
        age and next check survive restart; cursors and cached rows keep their owner.
        """
        budget = 1 if self.recipe_list.passes and self.model_list.passes else 6
        for _ in range(budget):
            if self.catalog_future is not None:
                if not self.catalog_future.done():
                    return
                active = self.catalog_active
                future, self.catalog_future = self.catalog_future, None
                self.catalog_active = None
                assert active is not None
                try:
                    completed = active.step(now, reply=future.result())
                    self.state.data["catalog_diagnostics"] = [
                        diagnostic.model_dump(mode="json")
                        for listing in (self.recipe_list, self.model_list)
                        for diagnostic in listing.diagnostics
                    ]
                except Exception as error:  # noqa: BLE001 - independent observation owner
                    self.note_infra("catalog", f"{type(error).__name__}: {error}")
                else:
                    if active.last_error:
                        self.note_infra("catalog", active.last_error)
                        self.state.event(
                            f"library page failed, retrying: {active.last_error[:120]}"
                        )
                    else:
                        self.clear_infra("catalog")
                        self.library_commit = self.recipe_list.commit
                        self._reconcile_all()
                        if completed and not any(
                            item.in_pass for item in (self.recipe_list, self.model_list)
                        ):
                            # Give removal the complete read before opening another
                            # pass. Slow alternating reads otherwise keep in_pass true
                            # at every removal check, even when every read succeeds.
                            return
            incident = self.state.data["infra"].get("catalog", {})
            if now < float(incident.get("until", 0)):
                return
            active = next(
                (item for item in (self.recipe_list, self.model_list) if item.in_pass),
                None,
            )
            if active is None:
                listings = (self.recipe_list, self.model_list)
                if not any(
                    now - item.pass_done_at >= self.cfg.catalog_seconds
                    for item in listings
                ):
                    return
                # One refresh round reads both libraries before yielding the
                # complete view. Neither library can restart ahead of the other.
                for item in listings:
                    item.begin(now)
                active = self.recipe_list
            self.catalog_active = active
            self.catalog_future = self.catalog_executor.submit(active.read_page)
            if not self.catalog_future.done():
                return  # never wait for the worker, including during cold start

    def refresh_catalog(self, max_steps: int = 200) -> None:
        """Read both libraries completely now (tests, and anyone who needs a fresh view)."""
        now = self.clock.now()
        for item in (self.recipe_list, self.model_list):
            item.begin(now)
            for _ in range(max_steps):
                if item.step(now):
                    break
        self.library_commit = self.recipe_list.commit
        self._reconcile_all()

    def _reconcile_all(self) -> None:
        # Fleet capacity is read fresh every time: a snapshot taken during a restart, a takeover or
        # a dual load in flight must never decide a recipe's fate for longer than one pass.
        try:
            self.fleet = fetch_fleet(self.vk, self.cfg.default_spark_memory)
        except VonkctlError as error:
            self.state.event(f"fleet unreadable: {str(error)[:120]}")
            if not self.fleet.sparks:
                return
        if self.recipe_list.passes and not self.recipe_list.in_pass:
            # A complete authoritative catalog pass can retire disappeared work.
            for key in set(self.state.recipes) - self.recipes.keys():
                slot = self.state.slots.get(key)
                if slot and slot["phase"] != "finished":
                    future = self.futures.pop(key, None)
                    if future:
                        future.cancel()
                    self._record_failure(
                        key,
                        policy.classify(
                            "catalog",
                            "catalog.recipe_removed",
                            "recipe is absent from the complete catalog",
                        ),
                        None,
                    )
                elif key not in self.state.slots:
                    if self.state.status(key) == "pending":
                        self.record_result(
                            key,
                            "catalog",
                            "skipped",
                            reason="recipe is absent from the complete catalog",
                            code="catalog.recipe_removed",
                        )
                    self.state.recipes.pop(key, None)
                record = self.state.downloads.pop(key, None)
                if record and record.get("state") not in SETTLED:
                    # A deterministic answer, not an infrastructure problem: no backoff.
                    self.state.event(f"download dropped: {key}: catalog.recipe_removed")
        self.sizes = policy.build_sizes(self.recipes.values(), self.models)
        enrolled = self.fleet_capacity()
        if not enrolled:
            return  # a listing without Sparks decides nothing about the topology
        for recipe in list(self.recipes.values()):
            self._reconcile_entry(recipe, enrolled)

    def fleet_capacity(self) -> int:
        """Sparks the fleet enrolls, online or not: never how many are online or free now."""
        return len(self.enrolled())

    def _fleet_evidence(self) -> dict[str, Any]:
        """The fleet listing's facts behind a fleet-derived decision (P6)."""
        return {
            "fleet": [
                {
                    "id": s.id,
                    "name": s.name,
                    "online": s.online,
                    "offline_reason": s.offline_reason or None,
                    "last_seen_at": s.last_seen_at or None,
                    "loaded": [
                        {"alias": p.alias, "run_id": p.run_id, "run_state": p.run_state}
                        for p in self.fleet.presences
                        if p.node_id == s.id
                    ],
                }
                for s in self.enrolled()
            ]
        }

    def observe_lanes(self) -> None:
        """An enrolled Spark keeps its lane: going offline is a wait, written once per change."""
        seen = self.state.data.setdefault("lane_state", {})
        for spark in self.enrolled():
            state = "online" if spark.online else "offline"
            before = seen.get(spark.id)
            seen[spark.id] = state
            if before == state or (before is None and spark.online):
                continue
            reason = (
                f"lane {spark.name} offline ({spark.offline_reason}, last seen "
                f"{spark.last_seen_at or 'unknown'}): kept, waiting for the Spark"
                if not spark.online
                else f"lane {spark.name} online again"
            )
            self.record_result(
                None,
                "lane",
                "waiting" if not spark.online else "available",
                reason=reason,
                spark=spark.name,
                spark_id=spark.id,
                waiting_recipes=sum(1 for r in self.pending() if self.lanes_short(r)),
                document=self._fleet_evidence(),
            )
            self.state.event(reason)

    def heal_fleet_skips(self) -> None:
        """Fleet-dependent skips are derived, not verdicts: re-evaluate them against the fleet now."""
        if not any(_fleet_skip(e) for e in self.state.recipes.values()):
            return
        try:
            self.fleet = fetch_fleet(self.vk, self.cfg.default_spark_memory)
        except VonkctlError as error:
            self.state.event(f"fleet unreadable: {str(error)[:120]}")
            return
        enrolled = self.fleet_capacity()
        for key, entry in self.state.recipes.items():
            recipe = self.recipes.get(key)
            if recipe is None or not _fleet_skip(entry):
                continue
            if recipe.node_count <= enrolled:
                self._reconcile_entry(recipe, enrolled)

    def _platform_cannot_load(self, recipe: Recipe) -> bool:
        """Job recipes, until the deployed Controller can load them (``--job-recipes``).

        Without vonk-forge #1491 the run-switch builds a serving start payload for a job
        plan, which never validates; without #1494 a held job run reads degraded. Either
        way the lane would end in a false verdict, so the sweep decides before admission.
        """
        return recipe.is_job and not self.cfg.job_recipes_loadable

    def _selected(self, recipe: Recipe) -> bool:
        return not self.cfg.only or any(word in recipe.key for word in self.cfg.only)

    def _reconcile_entry(self, recipe: Recipe, enrolled_sparks: int) -> None:
        """Testable is the enrolled topology; fewer lanes online now is a wait, never a skip."""
        if not self._selected(recipe):
            return
        entry = self.state.entry(recipe.key)
        slot = self.state.slots.get(recipe.key)
        if slot is not None and slot.get("phase") != "finished":
            return
        view = self.qualification_view(recipe)
        if (
            view is not None
            and view.status == QualificationStatus.WORKING
            and view.last_pass is not None
            and (slot is None or slot.get("phase") == SweepSlotPhase.FINISHED)
        ):
            entry.update(
                status=RecipeStatus.PASSED,
                content_sha256=recipe.content_sha256,
                # Restore the measured identity, not today's unmeasured image pin.
                verdict_identity=VerdictIdentity.model_validate(
                    view.last_pass.identity.model_dump(
                        exclude={"required_sparks", "models_resolved"}
                    )
                ).model_dump(mode="json"),
                revalidate=False,
            )
            for stale in (
                "reason",
                "reason_kind",
                "failure_class",
                "error",
                "code",
                "requeued_because",
            ):
                entry.pop(stale, None)
            return
        # A recipe that left the catalog is decided in _reconcile_all; here it is back.
        if entry.get("reason_kind") == CATALOG_SKIP:
            entry["status"] = "pending"
            entry.pop("reason", None)
            entry.pop("reason_kind", None)
            self.record_result(
                recipe.key, "reconcile", "pending", reason="back in the catalog"
            )
            self.state.event(f"{recipe.key} is back in the catalog: pending")
        if entry.get("reason_kind") == PLATFORM_SKIP and not self._platform_cannot_load(
            recipe
        ):
            entry["status"] = "pending"
            entry.pop("reason", None)
            entry.pop("reason_kind", None)
            self.record_result(
                recipe.key, "reconcile", "pending", reason="loadable on this platform"
            )
            self.state.event(f"{recipe.key} is loadable on this platform: pending")
        if self._platform_cannot_load(recipe):
            slot = self.state.slots.get(recipe.key)
            if entry.get("status") == "pending" and (
                slot is None or slot.get("phase") == "finished"
            ):
                reason = "job recipes are not loadable through a fleet profile on this platform yet"
                self.record_result(
                    recipe.key,
                    "reconcile",
                    "skipped",
                    reason=reason,
                    code=JOB_SKIP_CODE,
                )
                self.state.event(f"skipped {recipe.key}: {JOB_SKIP_CODE}")
                entry.update(
                    status="skipped",
                    reason=reason,
                    reason_kind=PLATFORM_SKIP,
                    content_sha256=recipe.content_sha256,
                )
            return  # never requeued while the platform cannot load it
        if recipe.node_count > enrolled_sparks and (
            entry.get("status") == "pending" or _fleet_skip(entry)
        ):
            reason = f"not testable on this fleet: needs {recipe.node_count} Sparks, {enrolled_sparks} enrolled"
            if entry.get("status") != "skipped" or entry.get("reason") != reason:
                self.record_result(
                    recipe.key,
                    "reconcile",
                    "skipped",
                    reason=reason,
                    document=self._fleet_evidence(),
                )
            entry.update(
                status="skipped",
                reason=reason,
                reason_kind=FLEET_SKIP,
                content_sha256=recipe.content_sha256,
            )
            return
        if _fleet_skip(entry) and recipe.node_count <= enrolled_sparks:
            entry["status"] = "pending"
            entry.pop("reason", None)
            entry.pop("reason_kind", None)
            short = self.lanes_short(recipe)
            why = f"testable on this fleet: needs {recipe.node_count} Sparks, {enrolled_sparks} enrolled"
            if short:
                why += f"; waiting for {short} more lane{'s' if short > 1 else ''}"
            self.record_result(
                recipe.key,
                "reconcile",
                "pending",
                reason=why,
                document=self._fleet_evidence(),
            )
            self.state.event(f"{recipe.key} is {why}: pending")
        if recipe.node_count > enrolled_sparks:
            return  # a Spark leaving the fleet must not rewrite a recorded result
        why = None
        if view is not None:
            why = {
                QualificationStatus.RECHECK: RetestReason.PASSED_OLDER_REVISION,
                QualificationStatus.FIX_TO_VERIFY: RetestReason.FAILED_OLDER_REVISION,
            }.get(view.status)
            if (
                view.status == QualificationStatus.BLOCKED_PLATFORM
                and view.last_failure is not None
                and view.last_failure.identity != current_identity(recipe)
            ):
                why = RetestReason.DEFERRED_OLDER_REVISION
        if entry.get("status") == RecipeStatus.PENDING:
            return
        if why is None:
            return
        if why == RetestReason.PASSED_OLDER_REVISION and not self.cfg.revalidate_passes:
            return
        entry["previous"] = {
            k: entry.get(k)
            for k in (
                "status",
                "content_sha256",
                "verdict_identity",
                "revision_id",
                "cluster",
            )
        }
        # The evidence about to be cleared is kept in the log first.
        self.record_result(
            recipe.key,
            "requeue",
            "requeued",
            reason=why,
            previous={
                k: entry.get(k)
                for k in (
                    "status",
                    "phase",
                    "failure_class",
                    "code",
                    "error",
                    "evidence",
                    "cluster",
                    "content_sha256",
                    "verdict_identity",
                    "revision_id",
                )
            },
        )
        entry.update(
            status=RecipeStatus.PENDING,
            attempts=0,
            revalidate=why == RetestReason.PASSED_OLDER_REVISION,
            requeued_because=why,
        )
        for stale in (
            "phase",
            "failure_class",
            "signature",
            "cluster",
            "error",
            "quality",
            "review_freeing",
        ):
            entry.pop(stale, None)
        self.state.event(f"requeued {recipe.key}: {why}")

    def pending(self) -> list[Recipe]:
        out: list[Recipe] = []
        for recipe in self.recipes.values():
            entry = self.state.recipes.get(recipe.key)
            if (
                entry is None
                or entry.get("status") != "pending"
                or recipe.key in self.state.slots
            ):
                continue
            if not self._selected(recipe) or self._platform_cannot_load(recipe):
                continue
            out.append(recipe)
        out.sort(
            key=lambda recipe: (
                scheduling_priority(view)
                if (view := self.qualification_view(recipe)) is not None
                else scheduling_priority(
                    QualificationStatusView(
                        status=QualificationStatus.UNTESTED,
                        required_sparks=recipe.node_count,
                    )
                )
            )
        )
        if self.cfg.limit:
            done = sum(
                1
                for e in self.state.recipes.values()
                if e.get("status") in ("passed", "failed")
            )
            out = out[: max(0, self.cfg.limit - done)]
        return out

    # ------------------------------------------------------------------ tick

    def waiting(self, what: str, since: float) -> None:
        """Called on every poll of a wait: save state and status, say what is awaited and since when.

        A wait is never silent: the state file's ``updated_at`` and the status page move on every
        poll. A wait never expires into a verdict.
        """
        now = self.clock.now()
        self.state.data["waiting"] = {"what": what, "since": since, "at": now}
        if now - self._wait_event_at >= WAIT_EVENT_SECONDS:
            self._wait_event_at = now
            self.state.event(f"waiting for {what} since {int((now - since) // 60)} min")
        self._save_quietly()
        try:
            write_status(self)
        except Exception as error:  # noqa: BLE001 - a status page never stops a wait
            self.note_infra("status", f"{type(error).__name__}: {error}")
        else:
            self.clear_infra("status")

    @property
    def paused(self) -> bool:
        """Only fresh owner safety observations stop every lane."""
        return self.owner_status.paused

    @property
    def pause_reason(self) -> str:
        return self.owner_status.reason

    def observe_stalls(self) -> None:
        """Alert on stalled work without making an item's stall a fleet-wide hold.

        Read from the followed applications' ``progress.stalled`` blocker and from
        ``/api/platform.stalled_applications``. Both are additive: absent means no stall.
        A condition the Controller states, re-read every pass, never a timed backoff.
        """
        stalled = [
            f"{load.get('app_id')} {load['stalled']}"
            for load in self.state.data["loads"].values()
            if load.get("stalled")
        ]
        listed = self.state.data["platform"].get("stalled_applications")
        self.stalled_applications = {
            item["application_id"]
            for item in listed or []
            if isinstance(item, Mapping) and isinstance(item.get("application_id"), str)
        }
        stalled += [
            f"{item.get('application_id')} {item.get('detail') or ''}".strip()
            for item in listed or []
            if isinstance(item, Mapping)
        ]
        reason = (
            "Controller reports stalled work: " + "; ".join(dict.fromkeys(stalled))
            if stalled
            else ""
        )
        if reason != self.stall_reason:
            self.state.event(
                f"lane alert: {reason}"
                if reason
                else "resumed: the Controller reports no stalled work"
            )
        self.stall_reason = reason

    def done_waiting(self) -> None:
        self.state.data.pop("waiting", None)
        self._wait_event_at = -1e18

    def tick(self) -> None:
        """One pass. A Controller that stops answering costs a pass, never the sweep."""
        try:
            self._tick()
            self.clear_infra("tick")
        except VonkctlError as error:
            self.state.event(f"vonkctl failed, will retry: {str(error)[:150]}")
            self._save_quietly()
        except Exception as error:  # noqa: BLE001 - an unattended sweep never dies of one bad pass
            self.note_infra("tick", f"{type(error).__name__}: {error}")
            self.state.event(
                f"unexpected error in a pass, will retry: {type(error).__name__}: {str(error)[:150]}"
            )
            self._save_quietly()
        finally:
            self.done_waiting()

    def _save_quietly(self) -> None:
        try:
            self.state.save()
        except Exception as error:  # noqa: BLE001 - the next pass saves again
            self.note_infra("state-save", f"{type(error).__name__}: {error}")
        else:
            self.clear_infra("state-save")

    def _startup(self, step: Callable[[], None]) -> bool:
        """One try of a startup step that only reads the fleet: a failing vonkctl waits, never exits.

        The caller tries again on its next pass; the wait has no deadline and backs off.
        """
        try:
            step()
        except (VonkctlError, OSError) as error:
            self.note_infra("startup", f"{type(error).__name__}: {error}")
            item = self.state.data["infra"]["startup"]
            item["until"] = self.clock.now() + self.cfg.poll_seconds
            self.waiting(f"startup {step.__name__}: {error}", float(item["since"]))
            self.clock.sleep(self.cfg.poll_seconds)
            return False
        self.clear_infra("startup")
        self.done_waiting()
        return True

    def _observe_owner(self, now: float) -> None:
        self.owner_status = self.guard.check(now)
        if self.owner_status.new_activity:
            self.state.data["took_over"] = False  # their load replaced what ours ran
            self.state.data["owner"]["seen_at"] = now
            self.preempt("an owner profile was loaded")
        if self.owner_status.paused:
            # Readable owner activity holds; unreadable observations have a per-profile budget.
            self.state.data["owner"].setdefault("paused_at", now)
        else:
            self.state.data["owner"].pop("paused_at", None)

    def _tick(self) -> None:
        # Placement/cleanup re-derive these observations only when asked this pass.
        self.state.data.pop("profile_edit_wait", None)
        self.state.data.pop("dual_reservation", None)
        self.adoption_holds.clear()
        now = self.clock.now()
        self.advance_catalog(now)
        if self.retry_failed_pending and self.catalog_loaded:
            self.requeue_failed()
            self.retry_failed_pending = False
        self.observe_lanes()
        self.check_release(now)
        if self.cfg.watch_seconds > 0:
            for key, entry in list(self.state.recipes.items()):
                if (
                    entry.get("status") == RecipeStatus.FAILED
                    and (
                        key not in self.recipes
                        or (view := self.qualification_view(self.recipes[key])) is None
                        or view.status != QualificationStatus.BLOCKED_PLATFORM
                    )
                    and now >= float(entry.get("finished_at", now)) + RETEST_SECONDS
                    and self._selected_key(key)
                ):
                    self._schedule_retest(key, RetestReason.SCHEDULED)
        self._observe_owner(now)
        self.observe_platform()
        self.advance_slots(now)
        self.observe_stalls()
        self.heal_fleet_skips()
        self.enqueue_finished_cleanup()
        # Cold-start paging may not yet have seen the owner's seed/priority recipe.
        # Existing receipts still advance above; new work uses the first complete view.
        pending = self.pending() if self.catalog_loaded else []
        self.cached_models = self._cached_models()
        self.prefetcher.library_fresh_since = self.recipe_list.complete_started_at
        present = self._present()
        plans = self.prefetcher.plan_queue(pending, self.sizes, present, self.boost)
        self.queue = policy.order_queue(
            plans,
            self.recipes,
            cached=self.cached_models,
            sizes=self.sizes,
            revalidate={
                k for k, e in self.state.recipes.items() if e.get("revalidate")
            },
            deprioritised=self._deprioritised(pending),
            variants_last=self.cfg.variants_last,
            refreshed={
                k
                for k, e in self.state.recipes.items()
                if e.get("requeued_because")
                in (
                    RetestReason.PASSED_OLDER_REVISION,
                    RetestReason.FAILED_OLDER_REVISION,
                    RetestReason.DEFERRED_OLDER_REVISION,
                )
            },
        )
        self.queue.sort(
            key=lambda key: (
                scheduling_priority(view)
                if (view := self.qualification_view(self.recipes[key])) is not None
                else 1
            )
        )
        if self.cfg.publish_evidence and self.qualification_evidence:
            self.evidence_publisher.tick(now)
        if not self.paused:
            self.schedule(self.clock.now())
            if not self.state.data.get("profile_edit_wait"):
                self.cleanup_finished()
        # Previously confirmed immutable stop receipts may feed a fresh canonical
        # admission before optional progress reads of an offline borrowed child.
        self.observe_cleanup()
        # Ready cached placements and their durable request receipts come first.
        # Download observation, preparation and pin maintenance may wait on the API.
        self.last_prefetch = self.prefetcher.tick(
            self.recipes,
            self.models,
            self.pending() if self.catalog_loaded else [],
            self.sizes,
            present,
            self.boost,
            self.cached_models,
        )
        self.remove_finished_models()
        self.check_idle(self.clock.now())
        self.state.data["rate"] = {"ema": self.rate.rate, "samples": self.rate.samples}
        if now - self.status_at >= self.cfg.status_seconds:
            self.status_at = now
            write_status(self)
        self.state.save()

    def remove_finished_models(self, *, final: bool = False) -> None:
        now = self.clock.now()
        causes = []
        for listing in (self.recipe_list, self.model_list):
            detail = ""
            if listing.last_error:
                detail = listing.last_error
            elif listing.invalid_rows:
                detail = f"{listing.invalid_rows} invalid rows"
                if listing.diagnostics:
                    detail += f": {listing.diagnostics[0].detail}"
            elif listing.in_pass:
                detail = (
                    f"read in progress; {len(listing.seen)} rows read, "
                    f"cursor {listing.cursor or 'first page'}, "
                    f"pass age {max(0, now - listing.pass_started_at):.0f}s"
                )
            elif not listing.passes:
                detail = "no complete read yet"
            if detail:
                causes.append(f"{listing.noun} library: {detail}")
        if self.paused:
            causes.insert(0, f"owner pause: {self.pause_reason}")
        if causes:
            incident = self.state.data["infra"].get("catalog", {})
            self._removal_wait(
                SweepWaitReason.OPERATOR
                if self.paused
                else SweepWaitReason.OBSERVATION,
                "; ".join(causes),
                max(now + self.cfg.poll_seconds, float(incident.get("until", 0))),
            )
            return
        own = {self.cfg.sweep_profile}
        if self.cfg.prefetch.pin_profile is not None:
            own.add(self.cfg.prefetch.pin_profile)
        try:
            self.model_remover.tick(self.recipes, own, self.clock.now(), final=final)
            self.state.data.pop("model_removal_wait", None)
        except (ValueError, OSError, VonkctlError) as error:
            self._removal_wait(
                SweepWaitReason.OBSERVATION, str(error), now + self.cfg.poll_seconds
            )
            self.state.event(
                f"model removal observation failed; sweep continues: {error}"
            )

    def _removal_wait(
        self, reason: SweepWaitReason, detail: str, next_check: float
    ) -> None:
        previous = self.state.data.get("model_removal_wait")
        wait = RemovalWait(
            reason=reason,
            detail=detail,
            since=self.clock.now(),
            next_check=next_check,
        )
        if previous:
            prior = RemovalWait.model_validate(previous)
            if prior.reason == reason:
                wait.since = prior.since
        self.state.data["model_removal_wait"] = wait.model_dump(mode="json")

    def _cached_models(self) -> set[str]:
        """Models on the NAS: the library says so, a download of ours finished, or the library's own
        assessment of a recipe says its exact assets are ready. Never fleet fit or readiness."""
        cached = {d for d, m in self.models.items() if m.local == "cached"}
        # Completion receipts bridge lagging catalog reads; a later complete
        # model pass can prove eviction. History must not reserve absent bytes.
        latest: dict[str, float] = {}
        for record in self.state.downloads.values():
            if record.get("state") != LifecycleState.SUCCEEDED:
                continue
            accepted = AcceptedRequest.read(record)
            if accepted is not None:
                for digest in accepted.recipe.model_digests:
                    latest[digest] = max(
                        latest.get(digest, 0.0), float(record.get("done_at") or 0.0)
                    )
        cached |= {
            digest
            for digest in self.state.data["models_done"]
            if not (
                digest in self.models
                and self.models[digest].local == "not_cached"
                and self.model_list.complete_started_at > latest.get(digest, 0.0)
            )
        }
        for recipe in self.recipes.values():
            if recipe.cache_ready:
                cached |= set(recipe.model_digests)
        return cached

    def _present(self) -> set[str]:
        """Models on the NAS or arriving."""
        return policy.present_models(self.models) | self.cached_models

    def _download_done(self, key: str) -> None:
        accepted = AcceptedRequest.read(self.state.downloads.get(key))
        if accepted is not None:
            done = set(self.state.data["models_done"]) | set(
                accepted.recipe.model_digests
            )
            self.state.data["models_done"] = sorted(done)

    def _deprioritised(self, pending: Iterable[Recipe]) -> set[str]:
        failures = self.state.data["model_failures"]
        return {r.key for r in pending if any(d in failures for d in r.model_digests)}

    def done(self) -> bool:
        if not self.catalog_loaded or self.state.slots:
            return False
        if self._cleanup_nodes():
            return False
        if any(r.get("state") in ACTIVE_APP for r in self.state.downloads.values()):
            return False
        return not any(self._open(k) for k in self.recipes)

    def _open(self, key: str) -> bool:
        entry = self.state.recipes.get(key)
        return bool(
            entry
            and entry.get("status") == "pending"
            and self._selected(self.recipes[key])
            and not self._platform_cannot_load(self.recipes[key])
        )

    def adopt_in_flight(self) -> None:
        """A restarted sweep carries on with a load it submitted itself instead of redoing it.

        The application keeps running on the Controller whatever happens to this process; the
        saved lane slots and the saved ``load`` pick it up on the next pass, and the clearing
        step is skipped while a slot exists. The time this process was away is not load time
        and not a stalled copy.
        """
        now = self.clock.now()
        adopted = []
        for key, slot in self.state.slots.items():
            if slot["phase"] != "loading":
                continue
            gap = now - float(slot.get("last_tick", slot["started_at"]))
            if gap > 2 * self.cfg.poll_seconds:
                for name in ("watch", "ready_watch", "smoke_watch"):
                    watch = slot.get(name)
                    if isinstance(watch, dict) and "progressed_at" in watch:
                        watch["progressed_at"] = float(watch["progressed_at"]) + gap
            slot["last_tick"] = now
            adopted.append(key)
        if adopted:
            self.state.event(
                "adopted an in-flight load of "
                + ", ".join(adopted)
                + " (time away not counted)"
            )

    def start_takeover(self) -> None:
        """``run --yes`` means the sweep owns the fleet: clear it once, now, so reviews see idle Sparks.

        While an owner load runs the clearing waits for it (``schedule`` clears once the owner
        is done); the wait never ends the sweep.
        """
        if self.owner_status.paused:
            self.state.event(
                f"waiting for an owner load before clearing: {self.owner_status.reason}"
            )
            return
        self._take_over(self.clock.now())

    def check_client(self) -> None:
        """Observe deployment and let the signed CLI updater select a compatible client."""
        version = self.vk.run("--version")
        if version.ok and isinstance(version.document, dict):
            self.state.data["client"] = {
                k: version.document.get(k)
                for k in ("version", "source_sha", "control_contract_sha256")
            }
        self.observe_platform()
        check = self.vk.run("update", timeout=60)
        document = check.document if isinstance(check.document, dict) else {}
        if not check.ok or "update_available" not in document:
            self.state.event(
                "could not compare vonkctl with the signed publication: "
                f"{check.error_text[:120]}"
            )
            return
        self.observe_publication(document)
        if document["update_available"] is not True:
            self.requeue_for_changed_owner()
            return
        current = dig(document, "current", "version", default="unknown")
        accepted = document.get("accepted_version", "unknown")
        message = f"vonkctl {current} differs from the signed publication {accepted}"
        if self.cfg.allow_version_skew:
            self.state.event(f"WARNING: {message}; continuing as asked")
            return
        # The installed updater owns signed artifact verification and the actual
        # deployed API compatibility check. No publication/source-label shortcut.
        upgraded = self.vk.run("update", "--apply", timeout=120)
        update_document = (
            upgraded.document if isinstance(upgraded.document, Mapping) else {}
        )
        if upgraded.ok and update_document.get("updated") is not True:
            self.state.event(
                f"{message}; signed client update pending: "
                f"{update_document.get('compatibility') or 'not applied'}"
            )
            return
        if upgraded.ok:
            self.state.event(f"{message}; compatible signed client update applied")
            version = self.vk.run("--version")
            if version.ok and isinstance(version.document, dict):
                self.state.data["client"] = {
                    k: version.document.get(k)
                    for k in ("version", "source_sha", "control_contract_sha256")
                }
            self.clear_infra("client-update")
            self.requeue_for_changed_owner()
        else:
            self.note_infra("client-update", f"{message}; update will be retried")

    @property
    def release_sha(self) -> str | None:
        """Current deployed API source, never an old publication proxy."""
        identity = self._component_observations().get("api")
        return identity.source_sha if identity is not None else None

    def _component_observations(
        self,
    ) -> dict[recovery.Owner, recovery.Fingerprint | None]:
        return recovery.fingerprints(
            self.state.data["platform"],
            self.state.data.get("client", {}),
            self.clock.now(),
        )

    def observe_platform(self) -> None:
        reply = self.vk.run("platform", timeout=30)
        document = (
            reply.document if reply.ok and isinstance(reply.document, dict) else {}
        )
        self.state.data["platform"] = document
        identity = self._component_observations().get("api")
        release = self.state.data["release"]
        observed = {
            "sha": identity.source_sha if identity else None,
            "control_contract_sha256": identity.contract_sha256 if identity else None,
            "authority": ReleaseAuthority.DEPLOYED_CONTROLLER,
        }
        if any(release.get(k) != value for k, value in observed.items()):
            self.state.data["release_history"].append(
                {**observed, "seen_at": self.clock.now()}
            )
            self.state.event(
                "deployed Controller observed: " + (observed["sha"] or "unknown")
            )
        release.clear()
        release.update(**observed, seen_at=self.clock.now())
        # Unknown provenance does not pause otherwise valid canonical scheduling.
        self.state.data["platform_issue"] = (
            None
            if identity is not None
            else "deployed component provenance unavailable"
        )
        self.requeue_for_changed_owner()

    def observe_publication(self, document: Mapping[str, Any]) -> None:
        """Record signed publication facts without changing fault eligibility."""
        publication = {
            "source_sha": document.get("accepted_source_sha"),
            "version": document.get("accepted_version"),
        }
        previous = self.state.data["publication"]
        if any(previous.get(k) != value for k, value in publication.items()):
            self.state.data["publication_history"].append(
                {**publication, "seen_at": self.clock.now()}
            )
        previous.update(**publication, seen_at=self.clock.now())

    def check_release(self, now: float) -> None:
        if now - self.release_at < self.cfg.release_seconds:
            return
        self.release_at = now
        self.check_client()

    def requeue_for_changed_owner(self) -> None:
        observations = self._component_observations()
        for key, entry in self.state.recipes.items():
            if entry.get("status") not in (
                "failed",
                "deferred",
            ) or not self._selected_key(key):
                continue
            basis = recovery.changed_owner(entry, observations)
            if basis is not None:
                self._schedule_retest(key, RetestReason.OWNER_CHANGED, basis=basis)

    def _schedule_retest(
        self, key: str, reason: str, *, basis: Mapping[str, str] | None = None
    ) -> None:
        """Schedule an automatic retest without erasing its cause or retry history."""
        if key in self.recipes:
            view = self.qualification_view(self.recipes[key])
            if view is not None and view.status == QualificationStatus.WORKING:
                return
        entry = self.state.entry(key)
        previous = {
            k: entry.get(k)
            for k in (
                "status",
                "attempts",
                "phase",
                "failure_class",
                "code",
                "error",
                "evidence",
                "evidence_bundle",
                "finished_at",
                "release",
                "recovery_basis",
                "release_authority",
                "defer_until",
                "not_before",
                "content_sha256",
                "revision_id",
            )
        }
        event = {"reason": reason, "at": self.clock.now(), "previous": previous}
        if basis is not None:
            event["basis"] = dict(basis)
        history = entry.setdefault("recovery_history", [])
        history.append(event)
        del history[:-60]
        # Exact existing request/download identities, attempts and cooldowns stay.
        entry.update(status="pending", retest=event, requeued_because=reason)
        self.record_result(
            key,
            "retest",
            "scheduled",
            reason=reason,
            previous=previous,
            recovery_basis=basis,
            controller_release=self.release_sha,
        )
        self.state.event(f"scheduled retest {key}: {reason}")

    def requeue_uncertain(self) -> None:
        """Reopen outcomes whose evidence no longer justifies a terminal result."""
        for key, entry in self.state.recipes.items():
            if (
                entry.get("status") == RecipeStatus.DEFERRED
                and entry.get("failure_class") in verdict.OBSERVE_CLASSES
                and self._selected_key(key)
            ):
                # A saved count-exhausted uncertainty still has an owner on restart.
                # Keep its diagnostic count and original download identity.
                attempts = int(entry.get("attempts", 0))
                self._requeue_failed(key, verdict.RecoveryReason.UNCERTAIN)
                due = self.clock.now() + verdict.retry_delay(
                    self.cfg.retry_delay, attempts
                )
                entry.update(attempts=attempts, not_before=due)
                record = self.state.downloads.get(key)
                if record is not None:
                    record["retry_at"] = due

    def requeue_failed(self) -> int:
        """``--retry-failed``: every failed recipe (within ``--only``), whatever the cause."""
        keys = [
            recipe.key
            for recipe in self.recipes.values()
            if self._selected(recipe)
            and (view := self.qualification_view(recipe)) is not None
            and view.status
            in (
                QualificationStatus.FAILED_RECIPE,
                QualificationStatus.FIX_TO_VERIFY,
                QualificationStatus.BLOCKED_PLATFORM,
            )
            and self.state.status(recipe.key) != RecipeStatus.PENDING
            and (
                recipe.key not in self.state.slots
                or self.state.slots[recipe.key].get("phase") == SweepSlotPhase.FINISHED
            )
        ]
        for key in keys:
            self._requeue_failed(key, RetestReason.OPERATOR)
        return len(keys)

    def _selected_key(self, key: str) -> bool:
        return not self.cfg.only or any(word in key for word in self.cfg.only)

    def _requeue_failed(self, key: str, why: str) -> None:
        """Back to pending; the failure stays in results.jsonl, which is only ever appended to."""
        if key in self.recipes:
            view = self.qualification_view(self.recipes[key])
            if view is not None and view.status == QualificationStatus.WORKING:
                return
        entry = self.state.entry(key)
        previous = {
            name: entry.get(name)
            for name in (
                "status",
                "phase",
                "failure_class",
                "cluster",
                "release",
                "error",
                "content_sha256",
            )
        }
        self.record_result(
            key,
            "requeue",
            "requeued",
            reason=why,
            controller_release=self.release_sha,
            previous=previous,
        )
        entry["previous"] = {"status": entry.get("status"), **previous}
        entry.update(status="pending", attempts=0, requeued_because=why)
        for stale in (
            "phase",
            "failure_class",
            "code",
            "error",
            "signature",
            "cluster",
            "evidence",
            "evidence_bundle",
            "inherited_from",
            "not_before",
            "defer_until",
            "review_freeing",
            "release",
            "quality",
        ):
            entry.pop(stale, None)
        recipe = self.recipes.get(key)
        for digest in recipe.model_digests if recipe else ():
            self.state.data["model_failures"].pop(digest, None)
        record = self.state.downloads.get(key)
        if record is not None and record.get("state") in ("failed", "cancelled"):
            if record.get("operation_id") or record.get("request_key"):
                # Explicit operator intent qualifies a retest, not a terminal
                # receipt. Keep the accepted parent unchanged until fresh exact
                # failure proof permits the canonical parent-bound retry.
                if why == RetestReason.OPERATOR:
                    entry["retest"] = {
                        "reason": why,
                        "at": self.clock.now(),
                        "previous": previous,
                    }
            else:
                record["state"] = "retired"  # genuinely unaccepted failure
        self.state.event(f"requeued {key}: {why}")

    def run(self) -> int:
        """Run the sweep. One sweep per state directory; SIGINT and SIGTERM end it cleanly."""
        lock = StateLock(self.cfg.state_dir)
        lock.acquire()
        previous = _install_signal_handlers()
        try:
            return self._run()
        finally:
            _restore_signal_handlers(previous)
            lock.release()

    def _run(self) -> int:
        # Adoption parses no remote state and writes nothing during read-only
        # loads. Publish its evidence only under this run's exclusive lock.
        for row in self.state.adoptions:
            if not self.record_result(
                row.recipe,
                "requeue",
                ReconciliationStatus.REQUEUED,
                reason=row.reason,
                previous=row.previous,
            ):
                return 1
        if self.state.adoptions:
            self.state.save()
            self.state.adoptions.clear()
        self.state.data.pop("end", None)
        # Retired sources may be forgotten, but restarting does not prove that a
        # review/adoption dependency recovered. Its first sighting and budget survive.
        self.state.data["infra"] = {
            source: item
            for source, item in self.state.data["infra"].items()
            if source in ("review", "catalog")
            or source.startswith(f"{ReviewInfrastructureSource.TRANSPORT}:")
        }
        stuck = 0
        try:
            self.check_client()
            if self.state.data.get("finalization"):
                self._leave_fleet()
            self.requeue_uncertain()
            started = False
            for _ in itertools.count():
                try:
                    if not started:
                        if not self._startup(self.preflight):
                            continue
                        self.adopt_in_flight()
                        started = True
                        try:
                            self.start_takeover()
                        except (VonkctlError, OSError) as error:
                            # Eager clearing is useful, but is never a startup gate.
                            self.note_infra("clearing", str(error))
                    self.tick()
                    if self.done():
                        self.state.data.pop("pending_idle", None)
                        self.state.data["alerts"] = []
                        if self.cfg.watch_seconds <= 0:
                            break
                        self.clock.sleep(self.cfg.watch_seconds)
                        self.recipe_list.pass_done_at = (
                            self.model_list.pass_done_at
                        ) = -1e18
                        continue
                    idle = self._nothing_moves()
                    self.clear_infra("loop")
                    stuck = stuck + 1 if idle else 0
                    if not idle:
                        self.state.data.pop("pending_idle", None)
                    if stuck >= 30:
                        self.state.event(
                            "IDLE: pending work has no runnable lane; observing and retrying"
                        )
                        self.state.data["pending_idle"] = (
                            "IDLE: pending work; sweep remains active"
                        )
                        self.state.data["alerts"] = [self.state.data["pending_idle"]]
                        self._save_quietly()
                        stuck = 0
                except SweepProfileOwnershipError:
                    raise  # authorization boundary: never adopt foreign saved assignments
                except Exception as error:  # noqa: BLE001 - only a stop or a signal ends the loop
                    self.note_infra("loop", f"{type(error).__name__}: {error}")
                    self.state.event(
                        f"unexpected error in the loop, continuing: {type(error).__name__}: {str(error)[:150]}"
                    )
                self.clock.sleep(self.cfg.poll_seconds)
        except KeyboardInterrupt:
            self.interrupt()
            self.finish(interrupted=True)
            return 130
        self.finish()
        return 0

    def _model_preparing(self, recipe: Recipe) -> bool:
        return any(
            (m := self.models.get(d)) is not None and m.local == "preparing"
            for d in recipe.model_digests
        )

    def _nothing_moves(self) -> bool:
        if not self.catalog_loaded:
            return False  # the Controller has not answered yet: keep trying
        busy = bool(self.state.slots) or any(
            r.get("state") in ACTIVE_APP
            or (r.get("retry_at") and r.get("state") == "failed")
            for r in self.state.downloads.values()
        )
        external = any(
            self.recipes[k].local == "preparing"
            or self._model_preparing(self.recipes[k])
            for k in self.queue
            if k in self.recipes
        )  # downloads started by someone else still deliver
        waiting = (
            external
            or self.paused
            or self.clock.now()
            < max(
                self.prefetcher.pressure_until,
                self.prefetcher.pause_until,
            )
        )
        return not busy and not waiting and not self.ready_candidates()

    # --------------------------------------------------------------- failures

    def _download_failed(
        self, key: str, failure: policy.Failure, op_id: str | None, model_level: bool
    ) -> None:
        accepted = AcceptedRequest.read(self.state.downloads.get(key))
        recipe = accepted.recipe if accepted else None
        self._record_failure(key, failure, op_id, from_download=True)
        if (
            model_level
            and failure.code in policy.RECIPE_FAILURE_CODES
            and recipe is not None
        ):
            for digest in recipe.model_digests:
                self.state.data["model_failures"][digest] = failure.cluster

    def _unplaced_recipe(self, key: str) -> Recipe | None:
        """Only pre-placement failures may describe the currently requested document."""
        return self.recipes.get(key)

    def _record_failure(
        self,
        key: str,
        failure: policy.Failure,
        op_id: str | None,
        *,
        inherited_from: str | None = None,
        from_download: bool = False,
        document: Mapping[str, Any] | None = None,
    ) -> None:
        """Record what an attempt came to; only an OBSERVE outcome spends an attempt.

        WAIT (the platform or a dependency is not ready) puts the recipe back without an
        attempt; OBSERVE retains ownership with capped backoff; FAIL records at once. Every outcome
        writes a results row with its evidence.
        """
        now = self.clock.now()
        entry = self.state.entry(key)
        slot = None if from_download else self.state.slots.get(key)
        accepted = AcceptedRequest.read(
            self.state.downloads.get(key) if from_download else slot
        )
        recipe = (
            accepted.recipe
            if accepted
            else (None if slot or from_download else self._unplaced_recipe(key))
        )
        outcome = verdict.disposition(failure)
        if slot is not None and accepted is None and outcome is not Disposition.SKIP:
            self.requeue(key, IdentityReason.UNBOUND)
            return
        if (
            slot is not None
            and failure.phase != "review"
            and self.owner_overlapped(slot)
        ):
            self.requeue(
                key,
                f"{failure.phase} failure overlapped an owner profile load",
                failure=failure,
            )
            return
        if slot is not None:
            slot["phase"] = "finished"
        if outcome is Disposition.WAIT:
            entry.update(
                status="pending",
                not_before=now + self.cfg.retry_delay,
                last_failure=failure.signature,
            )
            self.record_result(
                key,
                failure.phase,
                "waiting",
                reason=failure.describe(),
                disposition=outcome.value,
                failure_class=failure.klass,
                code=failure.code,
                evidence=dict(failure.evidence),
                slot=slot,
                document=document,
            )
            self.state.event(f"wait {key}: {failure.klass} ({failure.phase})")
            return
        if outcome is Disposition.SKIP:
            entry.update(
                status="skipped",
                reason=failure.describe(),
                reason_kind=CATALOG_SKIP,
            )
            self.record_result(
                key,
                failure.phase,
                "skipped",
                reason=failure.describe(),
                disposition=outcome.value,
                failure_class=failure.klass,
                code=failure.code,
                evidence=dict(failure.evidence),
                slot=slot,
                document=document,
            )
            self.state.event(f"skipped {key}: {failure.describe()[:160]}")
            return
        entry["attempts"] = int(entry.get("attempts", 0)) + 1
        if outcome is Disposition.OBSERVE:
            entry.update(
                status="pending",
                not_before=now
                + verdict.retry_delay(self.cfg.retry_delay, entry["attempts"]),
                last_failure=failure.signature,
            )
            self.record_result(
                key,
                failure.phase,
                "retry",
                reason=failure.describe(),
                disposition=outcome.value,
                failure_class=failure.klass,
                code=failure.code,
                evidence=dict(failure.evidence),
                slot=slot,
                document=document,
            )
            self.state.event(f"retry {key}: {failure.klass} ({failure.phase})")
            return
        entry["evidence_seq"] = int(entry.get("evidence_seq", 0)) + 1
        bundle = (
            read_optional(
                f"evidence/{key}",
                lambda: self._evidence(key, op_id, entry["evidence_seq"]),
                lambda diagnostic: entry.update(
                    diagnostics=[diagnostic.model_dump(mode="json")]
                ),
                InputReason.EVIDENCE_UNAVAILABLE,
            )
            if inherited_from is None
            else None
        )
        proof = dict(failure.evidence)
        if op_id:
            proof.setdefault("operation_id", op_id)
            proof["evidence_command"] = f"vonkctl fleet evidence {op_id}"
        attributable = failure.klass != verdict.UNATTRIBUTED and (
            failure.code in policy.RECIPE_FAILURE_CODES
            or failure.klass in verdict.FAIL_CLASSES
        )
        status = "failed" if attributable else "deferred"
        entry.update(
            status=status,
            phase=failure.phase,
            failure_class=failure.klass,
            code=failure.code,
            error=failure.describe(),
            evidence=proof,
            signature=failure.signature,
            cluster=failure.cluster,
            finished_at=now,
            release=self.release_sha,
            release_authority=ReleaseAuthority.DEPLOYED_CONTROLLER
            if self.release_sha
            else None,
            recovery_basis=recovery.failure_basis(
                failure.code, self._component_observations()
            ),
            rules=policy.LOAD_RULES,
            smoke_rules=policy.SMOKE_RULES,
            content_sha256=recipe.content_sha256
            if recipe
            else entry.get("content_sha256"),
            revision_id=recipe.revision_id if recipe else entry.get("revision_id"),
            verdict_identity=content_identity(recipe).model_dump(mode="json")
            if recipe
            else entry.get("verdict_identity"),
        )
        if inherited_from:
            entry["inherited_from"] = inherited_from
        if bundle:
            entry["evidence_bundle"] = bundle
        if failure.model_level and attributable and recipe is not None:
            for digest in recipe.model_digests:
                self.state.data["model_failures"][digest] = failure.cluster
        self.record_result(
            key,
            "smoke",
            status,
            error=f"{failure.phase}/{failure.klass}: {failure.describe()}"[:1536],
            disposition=outcome.value,
            slot=slot,
            document=document,
        )
        self.state.event(f"{status.upper()} {key}: {failure.phase}/{failure.klass}")
        self._reconcile_completed(key)

    def _reconcile_completed(self, key: str) -> None:
        # Re-plan now: a one-shot sweep may exit before another catalog tick.
        recipe = self.recipes.get(key)
        if recipe is not None:
            self._reconcile_entry(recipe, self.fleet_capacity())

    def record_result(
        self,
        key: str | None,
        step: str,
        status: str,
        *,
        slot: Mapping[str, Any] | None = None,
        document: Mapping[str, Any] | None = None,
        **fields: Any,
    ) -> bool:
        """The one writer of ``results.jsonl``: every lane exit leaves a row.

        The row carries the recipe's facts and the Controller document its load was last
        seen in, so the evidence of a wait, a retry or a requeue is never only in the
        bounded event ring. A lane's own row (``key`` None) names its Spark instead.
        """
        row: dict[str, Any] = {"batch": "sweep", "step": step, "status": status}
        if key is not None:
            entry = self.state.entry(key)
            slot = slot if slot is not None else self.state.slots.get(key)
            if document is None and slot is not None:
                document = self.documents.get(str(slot.get("request_key", "")))
            row.update(recipe=key, **self._facts(key, entry, slot))
        row.update({k: v for k, v in fields.items() if v is not None})
        if document:
            row["controller_document"] = _bounded_document(document)
        try:
            recorded = self.log.append(**row)
            self._remember_qualification(recorded)
        except OSError as error:
            # The log is the evidence; a full disk is an infrastructure fact to show.
            self.note_infra("results-log", f"{type(error).__name__}: {error}")
            return False
        else:
            self.clear_infra("results-log")
            return True

    def _facts(
        self, key: str, entry: Mapping[str, Any], slot: Mapping[str, Any] | None
    ) -> dict[str, Any]:
        accepted = AcceptedRequest.read(
            slot if slot is not None else self.state.downloads.get(key)
        )
        if (
            slot is None
            and accepted is not None
            and not verdict_matches(entry, accepted.recipe)
        ):
            accepted = None
        recipe = accepted.recipe if accepted else None
        facts: dict[str, Any] = {
            "phase": entry.get("phase"),
            "failure_class": entry.get("failure_class"),
            "code": entry.get("code"),
            "signature": entry.get("signature"),
            "cluster": entry.get("cluster"),
            "attempt": entry.get("attempts"),
            "evidence": entry.get("evidence"),
            "evidence_bundle": entry.get("evidence_bundle"),
            "diagnostics": entry.get("diagnostics"),
            "content_sha256": (recipe.content_sha256 if recipe else None)
            if slot
            else entry.get("content_sha256"),
            "recipe_revision_id": (recipe.revision_id if recipe else None)
            if slot
            else entry.get("revision_id"),
            "library_commit": accepted.library_commit if accepted else None,
            "controller_release": entry.get("release"),
            "controller_version": self.state.data["release"].get("version"),
            "node_count": recipe.node_count if recipe else None,
            # Agent identity is not inferred from Controller/client versions.
            "agent_version": self.state.data["platform"].get("agent_version"),
            "engine": recipe.engine if recipe else None,
            "timings": entry.get("timings"),
            "verdict_identity": content_identity(recipe).model_dump(mode="json")
            if slot and recipe is not None
            else entry.get("verdict_identity"),
        }
        if slot:
            facts["node_ids"] = list(slot.get("node_ids", []))
            facts["sparks"] = list(slot.get("spark_names", []))
            facts["run_id"] = slot.get("run_id")
        if accepted is not None:
            facts["request_key"] = accepted.request_key
            facts["model_digests"] = list(accepted.recipe.model_digests)
        return {k: v for k, v in facts.items() if v is not None}

    def _evidence(self, key: str, op_id: str | None, attempt: int) -> str | None:
        if not op_id:
            return None
        directory = self.cfg.state_dir / "evidence"
        directory.mkdir(parents=True, exist_ok=True)
        target = directory / f"{key.partition('/')[2]}-{attempt}.json"
        reply = self.vk.run("fleet", "evidence", op_id, "--output", str(target))
        if not reply.ok:
            raise ValueError(reply.error_text)
        return str(target)

    def _record_pass(self, key: str, slot: dict[str, Any], result: SmokeResult) -> None:
        now = self.clock.now()
        entry = self.state.entry(key)
        accepted = AcceptedRequest.read(slot)
        if accepted is None:
            self.requeue(key, IdentityReason.UNBOUND)
            return
        recipe = accepted.recipe
        download = self.state.downloads.get(key, {})
        downloaded = AcceptedRequest.read(download)
        if downloaded is None or content_identity(
            downloaded.recipe
        ) != content_identity(recipe):
            download = {}
        timings = {
            "download_s": round(
                float(download["done_at"]) - float(download["started_at"]), 1
            )
            if download.get("done_at") and download.get("started_at")
            else None,
            "load_s": round(float(slot["ready_at"]) - float(slot["started_at"]), 1),
            "smoke_s": result.seconds,
            "total_s": round(now - float(slot["started_at"]), 1),
        }
        entry.update(
            status="passed",
            attempts=int(entry.get("attempts", 0)) + 1,
            finished_at=now,
            release=self.release_sha,
            release_authority=ReleaseAuthority.DEPLOYED_CONTROLLER
            if self.release_sha
            else None,
            content_sha256=recipe.content_sha256,
            revision_id=recipe.revision_id,
            verdict_identity=content_identity(recipe).model_dump(mode="json"),
            timings=timings,
            smoke=result.kind,
            perf=result.perf,
            smoke_rules=policy.SMOKE_RULES,
            revalidate=False,
        )
        fleet = self.vk.run("fleet")
        observed = memory.observe(
            fleet.document if fleet.ok else None,
            slot["node_ids"],
            str(slot["alias"]),
            recipe.memory_bytes,
        )
        quality = result.quality_notes + memory.notes(observed)
        if quality:
            entry["quality"] = quality
        else:
            entry.pop("quality", None)
        for stale in (
            "phase",
            "failure_class",
            "signature",
            "cluster",
            "error",
            "evidence",
            "evidence_bundle",
            "inherited_from",
        ):
            entry.pop(stale, None)
        slot["phase"] = "finished"
        self.record_result(
            key,
            "smoke",
            "passed",
            result={"endpoint_alias": slot["alias"], **result.record()},
            memory=observed,
            slot=slot,
        )
        notes = len(quality)
        self.state.event(
            f"passed {key}"
            + (f" ({notes} quality note{'s' if notes != 1 else ''})" if notes else "")
        )

        self._reconcile_completed(key)

    # ------------------------------------------------------------------ slots

    def poll_application(self) -> bool:
        """Refresh the load's application; False when the Controller could not be observed."""
        observed = True
        fleet_blind = self.blind
        for load in list(self.state.data["loads"].values()):
            self.blind = fleet_blind
            fresh = self._poll_application(load)
            if not fresh:
                self.unobserved_loads.add(load["request_key"])
            observed = fresh and observed
        self.blind = fleet_blind
        return observed

    def _poll_application(self, load: dict[str, Any]) -> bool:
        args = ["progress"]
        if load.get("app_id"):
            args += ["--application", load["app_id"]]
        else:
            args += ["--request-key", load["request_key"]]
        reply = self.vk.run("profile", *args, profile=self.cfg.sweep_profile)
        document = reply.document
        if reply.is_not_found and not load.get("app_id"):
            # An exact fresh lookup proved there is no receipt. Replay the same request.
            try:
                document = self._submit_load(
                    self.cfg.sweep_profile, load["request_key"], "placement"
                )
            except VonkctlError as error:
                self._blind(str(error), error.reply)
                return False
        elif reply.is_not_found:
            # A known application the Controller does not find: ask by our own request key,
            # which names the same intent, before concluding anything.
            by_key = self.vk.run(
                "profile",
                "progress",
                "--request-key",
                load["request_key"],
                profile=self.cfg.sweep_profile,
            )
            if by_key.is_not_found:
                self._load_absent(load, by_key)
                return True
            if not by_key.ok:
                self._blind(by_key.error_text, by_key)
                return False
            document = by_key.document
        elif not reply.ok:
            self._blind(reply.error_text, reply)
            return False
        if not isinstance(document, dict) or not isinstance(document.get("id"), str):
            self._blind("application observation is unreadable")
            return False
        # This receipt resolves submission uncertainty, never another review's wait.
        self.clear_infra("load-submit")
        owner = self._own_load_request(document["id"])
        if owner not in (None, load["request_key"]):
            self._carry_loading_slots(owner)
            # Only lanes absent from the successor's canonical adoption requeue.
            self.state.data.setdefault("apps", {})[load["request_key"]] = {
                "state": LifecycleState.SUPERSEDED,
                "reason": f"superseded by this sweep's own load {owner}",
            }
            self.state.data["loads"].pop(load["request_key"], None)
            return True
        self.documents[load["request_key"]] = document
        load["admission_unbound"] = verdict.admission_unbound(document)
        load["observed_at"] = self.clock.now()
        load["app_id"] = document["id"]
        status = str(document.get("state", ""))
        load["state"] = status
        moving = policy.distribution(
            document,
            previous_bytes=int(load["copy_bytes"]) if "copy_bytes" in load else None,
        )
        load["copy_bytes"] = moving.completed_bytes
        load["copying"] = moving.copying
        load["phase"] = moving.phase
        load["blockers"] = policy.admission_blockers(document)
        signal = verdict.application_signal(document)
        # Capture known deployed content before a long wait ages its provenance.
        if (
            signal.wait is not None
            and signal.wait.controller is not None
            and not load.get("wait_platform")
        ):
            load["wait_platform"] = {
                owner: fingerprint.record()
                for owner, fingerprint in self._component_observations().items()
                if owner in ("api", "worker") and fingerprint is not None
            }
        load["signal"] = {
            "signature": list(signal.signature),
            "wait": signal.wait.model_dump(mode="json") if signal.wait else None,
            "marks": list(signal.marks),
        }
        load["stalled"] = verdict.stalled_detail(document)
        failure = document.get("failure") or {}
        failure_code = str(failure.get("code") or document.get("code") or "")
        replaced = policy.supersession(document) if status in SETTLED else None
        if replaced is not None and replaced.successor is not None:
            successor = self._own_load_request(replaced.successor)
            if successor is not None:
                self._carry_loading_slots(successor)
        if (
            replaced is not None
            and replaced.successor not in (None, document["id"])
            and self._own_load_request(replaced.successor)
            in (None, load["request_key"])
        ):
            # The Controller continued this application as another one: follow it.
            # Our own successors carry adopted lanes on their existing receipt.
            self.state.event(
                f"application {document['id']} was superseded: following {replaced.successor}"
            )
            load["app_id"] = replaced.successor
            load["state"] = "queued"
            load["copying"] = False
            # Unobserved, the successor may still be pending admission.
            load.pop("admission_unbound", None)
            return True
        if status in SETTLED:
            child = dig(document, "progress", "child_progress", "phase")
            self.state.data.setdefault("apps", {})[load["request_key"]] = {
                "state": status,
                "reason": document.get("status_reason"),
                "failure_code": failure_code,
                "adopted_requests": load.get("adopted_requests", []),
                "operation_id": document.get("current_operation_id") or document["id"],
                "child_phase": child,
                "cause": dig(document, "cancellation", "cause"),
                "failures": dig(
                    document,
                    "progress",
                    "switch_adapter",
                    "assignment_failures",
                    default=[],
                ),
            }
            self.state.data["loads"].pop(load["request_key"], None)
        return True

    def _carry_loading_slots(self, request: str) -> None:
        """Observe unchanged lanes through the latest accepted whole-fleet load.

        Admission already bound each adopted request's exact application, plan,
        assignment and Spark scope in `_adopted_requests`. Keep its accepted
        recipe content and deadline; only the aggregate observer changes.
        """
        successor = self.state.data["loads"].get(request) or self.state.data.get(
            "apps", {}
        ).get(request)
        if successor is None:
            return
        adopted = set(successor.get("adopted_requests") or [])
        for slot in self.state.slots.values():
            if (
                slot["phase"] == SweepSlotPhase.LOADING
                and slot["request_key"] in adopted
            ):
                slot["request_key"] = request

    def _own_load_request(self, app_id: str) -> str | None:
        """The request key of this sweep's own load whose application ``app_id`` is."""
        for item in self.state.data.get("own_loads") or []:
            if isinstance(item, dict) and item.get("application_id") == app_id:
                request = item.get("request_key")
                return request if isinstance(request, str) else None
        return None

    def _load_absent(self, load: dict[str, Any], reply: Reply) -> None:
        """The Controller has no application for this load, by id or by request key.

        That is its answer, not a wait: the lanes are requeued without blame or attempt,
        each with a results row carrying the not-found document.
        """
        evidence = reply.document if isinstance(reply.document, Mapping) else None
        self.state.data["loads"].pop(load["request_key"], None)
        for key, slot in list(self.state.slots.items()):
            if (
                slot.get("request_key") == load["request_key"]
                and slot["phase"] == "loading"
            ):
                self.requeue(
                    key,
                    f"application.absent: the Controller has no application {load.get('app_id')}",
                    document=evidence,
                )
                self.state.entry(key)["not_before"] = (
                    self.clock.now() + self.cfg.retry_delay
                )

    def _blind(self, message: str, reply: Any = None) -> None:
        """The sweep cannot observe the Controller (outage, 502/503, protocol or version skew)."""
        self.blind = True
        self.note_infra("observe", message)
        if "protocol_invalid" in message or "OpenAPI" in message:
            self._diagnose_skew()

    def _diagnose_skew(self) -> None:
        """protocol_invalid means vonkctl is older than the Controller: say so, loudly, once in a while."""
        now = self.clock.now()
        if now < self._skew_checked_at + SKEW_CHECK_INTERVAL:
            return
        self._skew_checked_at = now
        check = self.vk.run("update", timeout=60)  # read-only: never --apply
        document = check.document if isinstance(check.document, dict) else {}
        current = dig(document, "current", "version", default="unknown")
        accepted = document.get("accepted_version", "unknown")
        self.state.event(
            "WARNING: the Controller answers in a protocol this vonkctl does not speak "
            f"(vonkctl {current}, accepted release {accepted}): the CLIENT MUST BE UPDATED, "
            "run `vonkctl update --apply`. Not a recipe failure: loads are paused, not timed."
        )

    def advance_slots(self, now: float) -> None:
        self.blind = False
        self.unobserved_loads.clear()
        if not self.state.slots:
            self.poll_application()
            if not self.blind and not self.unobserved_loads:
                self.clear_infra("observe")
            return
        try:
            self.fleet = fetch_fleet(self.vk, self.cfg.default_spark_memory)
        except VonkctlError as error:
            # An unreadable fleet is unknown: nothing is judged on it this pass.
            self._blind(str(error), error.reply)
        self.poll_application()
        if not self.blind and not self.unobserved_loads:
            self.clear_infra("observe")
        for key, slot in list(self.state.slots.items()):
            if slot["phase"] == "loading":
                self._advance_loading(key, slot, now)
            elif slot["phase"] == "smoking":
                self._advance_smoking(key, slot)

    def _load_window(self, slot: Mapping[str, Any]) -> float:
        """The lane's stall window: learned per engine, or the copy window while bytes move."""
        if slot.get("copying"):
            return float(self.cfg.copy_stall_seconds)
        return float(slot["deadline"]) - float(slot["started_at"])

    def _observe_load(self, slot: dict[str, Any], now: float) -> Progress:
        """What the slot's load is doing: PROGRESSING, WAITING or STALLED (``progress_verdict``).

        Moving copies retain their inactivity window. Dependency waits use the lane
        budget from admission, restarted only when a forward-only counter rises; changing
        reasons, retries and unreadable observations cannot extend it.
        """
        load = self.state.data["loads"].get(slot["request_key"])
        elapsed = now - float(slot.get("last_tick", slot["started_at"]))
        slot["last_tick"] = now
        watch = slot.setdefault("watch", {})
        unreadable = self.blind or slot["request_key"] in self.unobserved_loads
        if unreadable or load is None or not isinstance(load.get("signal"), Mapping):
            if unreadable:
                slot["blind_seconds"] = float(slot.get("blind_seconds", 0)) + elapsed
            return verdict.progress_verdict(
                None,
                watch,
                window=float(slot["deadline"]) - float(slot["started_at"]),
                now=now,
                started_at=float(slot["started_at"]),
            )
        copying = bool(load.get("copying"))
        slot["copying"] = copying
        if copying:
            slot["copy_seconds"] = float(slot.get("copy_seconds", 0)) + elapsed
        blockers = None if copying else load.get("blockers")
        if blockers:
            slot["blocked_seconds"] = float(slot.get("blocked_seconds", 0)) + elapsed
            slot.setdefault("blocked_since", now)
            slot["blocked"] = blockers
        else:
            slot.pop("blocked_since", None)
            slot.pop("blocked", None)
        if load.get("phase"):
            slot["load_phase"] = load["phase"]
        signal = verdict.Signal(
            tuple(load["signal"].get("signature") or ()),
            SweepWait.model_validate(load["signal"]["wait"])
            if load["signal"].get("wait")
            else None,
            tuple(load["signal"].get("marks") or ()),
        )
        if (
            copying
            and load.get("state") in ACTIVE_APP
            and (signal.wait is None or signal.wait.controller is None)
        ):
            # Distribution retries still own the same copy. Only changed bytes
            # or phases restart its window; a retry timestamp cannot keep a lane
            # forever. The existing stalled-copy path releases and observes it.
            signal = replace(signal, wait=None)
        if signal.waiting_on is not None:
            slot["waited_seconds"] = float(slot.get("waited_seconds", 0)) + elapsed
        window = (
            float(slot["deadline"]) - float(slot["started_at"])
            if signal.wait
            else self._load_window(slot)
        )
        if signal.wait and verdict.preparation_blockers(load.get("blockers") or []):
            window = min(window, self.cfg.blocked_release_seconds)
        return verdict.progress_verdict(
            signal, watch, window=window, now=now, started_at=float(slot["started_at"])
        )

    def _observe_readiness(self, slot: dict[str, Any], now: float) -> Progress:
        """After the load succeeded: the run turning healthy and its route being published.

        A healthy run whose route is not published yet waits on the Controller's route
        publication; a run that is missing or unhealthy is the recipe's own work and stalls
        after the lane's window without any change. The publication wait has no forward
        counter, so a new run or route state never renews it.
        """
        prior = set(slot.get("prior_runs", ()))
        found = [
            p
            for p in self.fleet.presences
            if p.alias == slot["alias"]
            and p.node_id in slot["node_ids"]
            and p.run_id not in prior
        ]
        signature = tuple(
            sorted(
                (p.node_id, p.run_id, p.run_state, p.healthy, p.route_state)
                for p in found
            )
        )
        publishing = bool(found) and all(
            p.healthy and p.run_state == "running" for p in found
        )
        return verdict.progress_verdict(
            verdict.Signal(
                signature,
                SweepWait(reason=SweepWaitReason.ROUTE_PUBLICATION)
                if publishing
                else None,
            ),
            slot.setdefault("ready_watch", {}),
            window=float(slot["deadline"]) - float(slot["started_at"]),
            now=now,
            started_at=float(slot["started_at"]),
        )

    def _advance_loading(self, key: str, slot: dict[str, Any], now: float) -> None:
        accepted = AcceptedRequest.read(slot)
        if accepted is None:
            self.requeue(key, IdentityReason.UNBOUND)
            return
        recipe = accepted.recipe
        if self.blind or slot["request_key"] in self.unobserved_loads:
            if self._observe_load(slot, now) is Progress.WAIT_EXPIRED:
                self._expire_wait(key, slot, slot["watch"])
            return
        run_id = serving_run(
            self.fleet,
            slot["alias"],
            slot["node_ids"],
            recipe.node_count,
            slot.get("prior_runs", ()),
            job=recipe.is_job,
        )
        if run_id:
            slot.pop("waiting_on", None)
            slot.update(
                phase="smoking", run_id=run_id, ready_at=now, smoke_started_at=now
            )
            self.clear_infra("load-stall")
            policy.learn(
                self.state.data["learned"],
                recipe.engine,
                slot["model_bytes"],
                now - slot["started_at"] - float(slot.get("copy_seconds", 0)),
            )
            self._submit_smoke(key, slot)
            return
        app = self.state.data.get("apps", {}).get(slot["request_key"])
        if app is not None:
            if app["state"] == LifecycleState.SUPERSEDED:
                self.requeue(key, "load superseded by another application")
                return
            if app["state"] == "cancelled":
                # A cancel is an ending somebody requested (the operator, the sweep itself,
                # a superseding load), never the recipe's outcome. A child the cancel could
                # not confirm keeps its own lifecycle: the next load's admission waits on it.
                self.requeue(
                    key,
                    f"application cancelled (cause {app.get('cause') or 'unknown'}): "
                    "a cancel is no verdict on the recipe",
                    status_reason=app.get("reason"),
                    operation_id=app.get("operation_id"),
                )
                self.state.entry(key)["not_before"] = now + self.cfg.retry_delay
                return
            if app["state"] == "succeeded":
                progress = self._observe_readiness(slot, now)
                if progress is Progress.WAIT_EXPIRED:
                    self._expire_wait(key, slot, slot["ready_watch"])
                elif progress is Progress.STALLED:
                    silent = verdict.silent_seconds(slot["ready_watch"], now)
                    self._fail_slot(
                        key,
                        slot,
                        policy.classify(
                            "readiness",
                            "run.not_published",
                            "the load succeeded but the run never became healthy and "
                            f"{'held' if recipe.is_job else 'published'} "
                            f"(no change for {round(silent)}s)",
                        ),
                        app,
                    )
                return
            group = self._lane_group(slot)
            unattributed = False
            if len(group) > 1:
                failures = [
                    f for f in app.get("failures") or [] if isinstance(f, Mapping)
                ]
                blamed = policy.blamed(failures, self._lanes(group))
                culprits, collateral = self._split(group, blamed)
                self._requeue_collateral(collateral, culprits, "its application failed")
                unattributed = not blamed
                group = culprits
            else:
                group = [key]
            phase = policy.child_phase(app.get("child_phase")) or "start"
            detail = str(app.get("reason") or app["state"])
            for member in group:
                own_failure = next(
                    (
                        item
                        for item in app.get("failures") or []
                        if isinstance(item, Mapping)
                        and member in policy.blamed([item], self._lanes(group))
                    ),
                    None,
                )
                failure = policy.classify_application_failure(
                    phase,
                    str(own_failure["code"])
                    if own_failure is not None
                    and isinstance(own_failure.get("code"), str)
                    else app.get("failure_code") or f"application.{app['state']}",
                    str(
                        own_failure.get("reason") or own_failure.get("detail") or detail
                    )
                    if own_failure is not None
                    else detail,
                    evidence={"assignment_failures": app.get("failures") or []},
                )
                if unattributed:
                    # Nothing the Controller said names this lane: an uncertain outcome,
                    # observed again (one attempt each), never a verdict on one of them.
                    failure = replace(failure, klass=verdict.UNATTRIBUTED)
                self._fail_slot(member, self.state.slots[member], failure, app)
            return
        load = self.state.data["loads"].get(slot["request_key"]) or {}
        if load.get("stalled"):
            self._controller_stalled(key, slot, load)
            return
        load.pop("stall_released", None)
        slot.pop("waiting_on", None)
        progress = self._observe_load(slot, now)
        if progress is Progress.WAIT_EXPIRED:
            self._expire_wait(key, slot, slot["watch"])
            return
        if progress is Progress.STALLED:
            self._stalled(key, slot, now)

    def _controller_stalled(
        self, key: str, slot: dict[str, Any], load: dict[str, Any]
    ) -> None:
        """The Controller reports the lane's application stalled: a platform problem.

        Observe the original receipt within the lane budget. Expiry re-plans locally
        without asserting a remote ending or spending a recipe attempt.
        """
        declared = (load.get("signal") or {}).get("wait")
        wait = SweepWait.model_validate(declared) if declared is not None else None
        if self._wait_slot(
            key,
            slot,
            wait
            if wait is not None and wait.controller is not None
            else SweepWait(
                reason=SweepWaitReason.CONTROLLER_STALLED,
                detail=f"application {load.get('app_id')}: {load['stalled']}",
            ),
            watch_name="watch",
            started_at=float(slot["started_at"]),
        ):
            return
        if load.get("stall_released"):
            return
        load["stall_released"] = True
        detail = str(load["stalled"])
        self.record_result(
            key,
            "load",
            LifecycleState.OBSERVING,
            reason=f"the Controller reports the load stalled: {detail}",
            slot=slot,
            disposition=Disposition.OBSERVE.value,
            failure_class=verdict.PLATFORM_STALLED,
            code=verdict.STALLED_CODE,
            evidence={"application_id": load.get("app_id")},
        )

    def _stalled(self, key: str, slot: dict[str, Any], now: float) -> None:
        """The load's own progress was silent for a whole window: the one verdict that ends it."""
        silent = round(verdict.silent_seconds(slot.get("watch", {}), now))
        if slot.get("copying"):
            self._timeout(
                key,
                slot,
                policy.classify(
                    "install",
                    "copy.stalled",
                    f"the copy to the Spark made no progress for {silent}s "
                    f"(phase {slot.get('load_phase')})",
                    "copy-stalled",
                ),
            )
            return
        self._timeout(key, slot)

    def _lane_group(self, slot: Mapping[str, Any]) -> list[str]:
        """The lanes still loading in the same application as this one."""
        return [
            k
            for k, s in self.state.slots.items()
            if s["phase"] == "loading"
            and s["request_key"] == slot["request_key"]
            and not serving_run(
                self.fleet,
                s["alias"],
                s["node_ids"],
                self.recipes[k].node_count,
                s.get("prior_runs", ()),
            )  # a lane that already serves goes on to its smoke
        ]

    def _lanes(self, group: Sequence[str]) -> dict[str, dict[str, list[str]]]:
        lanes: dict[str, dict[str, list[str]]] = {}
        for key in group:
            slot = self.state.slots[key]
            lanes[key] = {
                "names": [slot["alias"], key, key.rsplit("/", 1)[-1]],
                "places": [*slot["node_ids"], *slot.get("spark_names", [])],
            }
        return lanes

    def _split(
        self, group: Sequence[str], blamed: set[str]
    ) -> tuple[list[str], list[str]]:
        """Who of a load that ended carries the failure, and who is collateral.

        Only the lanes the Controller's failures name are the recipes' own. When nothing names
        a lane of a shared load, every lane is observed again (an uncertain outcome that
        spends one attempt each), so a recipe that really fails is still recorded and none
        is blamed for another's failure on one load.
        """
        culprits = [k for k in group if k in blamed]
        if culprits:
            return culprits, [k for k in group if k not in blamed]
        return list(group), []

    def _requeue_collateral(
        self, collateral: Sequence[str], culprits: Sequence[str], cause: str
    ) -> None:
        for key in collateral:
            why = (
                "collateral of " + ", ".join(culprits)
                if culprits
                else f"collateral: {cause}, and no blocker names a recipe"
            )
            self.requeue(key, why)

    def _timeout(
        self, key: str, slot: dict[str, Any], failure: policy.Failure | None = None
    ) -> None:
        load = self.state.data["loads"].get(slot["request_key"])
        others = [
            k
            for k, s in self.state.slots.items()
            if k != key
            and s["phase"] == "loading"
            and s.get("request_key") == slot["request_key"]
        ]
        if load and load.get("app_id") and not others:
            self._cancel_load(load)
        seconds = round(verdict.silent_seconds(slot.get("watch", {}), self.clock.now()))
        self._fail_slot(
            key,
            slot,
            failure
            or policy.classify(
                "timeout",
                "load.timeout",
                f"not serving and no progress for {seconds}s "
                f"(stall window {round(slot['deadline'] - slot['started_at'])}s, "
                f"{round(float(slot.get('copy_seconds', 0)))}s of copying not counted; "
                f"{round(float(slot.get('waited_seconds', 0)))}s of waiting on the "
                f"Controller not counted; last phase {slot.get('load_phase')})",
            ),
            load or {},
        )

    def _fail_slot(
        self,
        key: str,
        slot: dict[str, Any],
        failure: policy.Failure,
        app: Mapping[str, Any],
    ) -> None:
        context = {
            "application_state": app.get("state"),
            "reason": app.get("reason"),
            "child_phase": app.get("child_phase"),
            "assignment_failures": policy.excerpt(app.get("failures") or "", 500)
            or None,
            "application_id": app.get("app_id"),
        }
        failure = replace(
            failure,
            evidence={
                **{k: v for k, v in context.items() if v},
                **failure.evidence,
            },
        )
        self._record_failure(key, failure, app.get("operation_id") or app.get("app_id"))

    def owner_overlapped(self, slot: Mapping[str, Any]) -> bool:
        """An owner load happened during this slot's life: the Sparks were taken from under it."""
        return float(self.state.data["owner"].get("seen_at", 0)) >= float(
            slot["started_at"]
        )

    def requeue(
        self,
        key: str,
        why: str,
        *,
        failure: policy.Failure | None = None,
        document: Mapping[str, Any] | None = None,
        disposition: str | None = None,
        **evidence: Any,
    ) -> None:
        """Back to pending without blame or attempt: the recipe did not get a fair run.

        Every requeue leaves a results row with the Controller document it was last seen in.
        """
        entry = self.state.entry(key)
        slot = self.state.slots.get(key)
        self.record_result(
            key,
            "requeue",
            "requeued",
            reason=why,
            slot=slot,
            document=document,
            disposition=disposition,
            failure_class=failure.klass if failure else None,
            code=failure.code if failure else None,
            evidence={**(dict(failure.evidence) if failure else {}), **evidence}
            or None,
        )
        entry["status"] = "pending"
        if slot is not None:
            slot["phase"] = "finished"
        self.state.event(f"requeued {key}: {why}")

    def _expire_wait(
        self, key: str, slot: dict[str, Any], watch: Mapping[str, Any]
    ) -> None:
        expiry = verdict.wait_expiry(watch, self.clock.now())
        stop = self.smoke_stops.pop(key, None)
        if stop is not None:
            stop.set()
        future = self.futures.pop(key, None)
        if future is not None:
            future.cancel()
        # Keep the original receipt for observation; the next canonical review
        # supersedes changed intent. Local expiry is never a remote terminal claim.
        load = self.state.data["loads"].get(slot["request_key"])
        if load is not None:
            load["replanned_pending"] = True
        accepted = AcceptedRequest.read(slot)
        if (
            expiry.wait.controller is not None
            and expiry.wait.controller.identity is not None
            and accepted is not None
            and load is not None
            and load.get("app_id")
        ):
            hold = ControllerWaitHold(
                wait=expiry.wait,
                identity=current_identity(accepted.recipe),
                application_id=load["app_id"],
                next_check=self.clock.now() + self.cfg.defer_delay,
                platform=load.get("wait_platform")
                or {
                    owner: fingerprint
                    for owner, fingerprint in self._component_observations().items()
                    if owner in ("api", "worker") and fingerprint is not None
                },
            )
            self.state.entry(key)["controller_wait"] = hold.model_dump(mode="json")
            self.state.event(
                f"placement waits for Controller dependency: {hold.wait.description}; "
                f"observing {hold.application_id} again at {hold.next_check}"
            )
        failure = policy.classify(
            slot["phase"],
            expiry.code,
            f"wait budget exhausted: {expiry.wait.description}",
            klass=expiry.code,
            evidence=expiry.model_dump(mode="json"),
        )
        group = [
            k
            for k, lane in self.state.slots.items()
            if lane.get("request_key") == slot["request_key"]
            and lane["phase"] == "loading"
        ]
        self.requeue(
            key, failure.detail, failure=failure, disposition=Disposition.WAIT.value
        )
        self.state.entry(key)["wait_deferral"] = expiry.model_dump(mode="json")
        # requeue marked this slot finished; include it when attributing a shared hold.
        lanes = self._lanes(sorted(set(group) | {key}))
        held = policy.blamed(load.get("blockers") or [], lanes) if load else set()
        delay = 0.0 if held and key not in held else self.cfg.defer_delay
        self.state.entry(key)["not_before"] = self.clock.now() + delay

    def _wait_slot(
        self,
        key: str,
        slot: dict[str, Any],
        wait: SweepWait,
        *,
        watch_name: str = "smoke_watch",
        started_at: float | None = None,
    ) -> bool:
        watch = slot.setdefault(watch_name, {})
        since = (
            float(
                slot.get("smoke_started_at", slot.get("ready_at", slot["started_at"]))
            )
            if started_at is None
            else started_at
        )
        window = float(
            slot.get(
                "smoke_budget_s", float(slot["deadline"]) - float(slot["started_at"])
            )
        )
        progress = verdict.bounded_wait(
            wait, watch, window=window, now=self.clock.now(), started_at=since
        )
        if progress is Progress.WAIT_EXPIRED:
            self._expire_wait(key, slot, watch)
            return True
        return False

    def _submit_smoke(self, key: str, slot: dict[str, Any]) -> None:
        accepted = AcceptedRequest.read(slot)
        if accepted is None:
            self.requeue(key, IdentityReason.UNBOUND)
            return
        recipe = accepted.recipe
        slot.setdefault("smoke_started_at", slot.get("ready_at", self.clock.now()))
        if recipe.is_job:
            # A job run has no endpoint: its smoke is one artifact job against the run.
            spec = self.defs.artifact(key)
            if spec is not None:
                # Each smoke enforces its own budget from its start: the lane shows that one.
                slot.update(
                    smoke_budget_s=job_budget_seconds(spec),
                )
            self.futures[key] = self.executor.submit(
                smoke_job,
                recipe,
                self.defs,
                self.vk,
                slot["run_id"],
                slot["request_key"],
            )
            return
        if not recipe.is_openai:
            self.futures[key] = _done_future(smoke_readiness(slot["run_id"]))
            return
        try:
            endpoints = self.vk.call(
                "profile", "endpoint", slot["alias"], profile=self.cfg.sweep_profile
            )
        except VonkctlError as error:
            if verdict.reply_disposition(error.reply) is Disposition.WAIT:
                if self._wait_slot(
                    key,
                    slot,
                    SweepWait(
                        reason=SweepWaitReason.ENDPOINT_PUBLICATION, detail=str(error)
                    ),
                ):
                    return
                self.note_infra("endpoint", str(error))
                slot["smoke_not_before"] = self.clock.now() + ENDPOINT_RECHECK_SECONDS
                return  # the run is up: look the endpoint up again on the next pass
            self._fail_slot(
                key, slot, policy.classify("readiness", "endpoint", str(error)), {}
            )
            return
        self.clear_infra("endpoint")
        item = next(
            (
                item
                for item in dig(endpoints, "assignments", default=[])
                if isinstance(item, dict) and item.get("alias") == slot["alias"]
            ),
            {},
        )
        base = dig(item, "endpoint", "api_base")
        if not isinstance(base, str):
            if self._wait_slot(
                key,
                slot,
                SweepWait(
                    reason=SweepWaitReason.ENDPOINT_PUBLICATION,
                    detail=f"assignment state {item.get('state')}",
                ),
            ):
                return
            slot["smoke_not_before"] = self.clock.now() + ENDPOINT_RECHECK_SECONDS
            return
        slot.pop("waiting_on", None)
        stop = threading.Event()
        self.smoke_stops[key] = stop
        self.futures[key] = self.executor.submit(
            smoke_service,
            recipe,
            self.defs,
            base,
            slot["alias"],
            replace(self.cfg.http, cancelled=stop),
        )

    def _advance_smoking(self, key: str, slot: dict[str, Any]) -> None:
        future = self.futures.get(key)
        watch = slot.get("smoke_watch") or {}
        wait = (
            SweepWait.model_validate(watch["wait"])
            if watch.get("wait")
            else SweepWait(reason=SweepWaitReason.SMOKE_RESPONSE)
        )
        if (future is None or not future.done()) and self._wait_slot(key, slot, wait):
            return
        if (
            future is None
        ):  # resumed after a restart: the smoke is idempotent, run it again
            if self.clock.now() >= float(slot.get("smoke_not_before", 0)):
                self._submit_smoke(key, slot)
            return
        if not future.done():
            self._wait_slot(key, slot, SweepWait(reason=SweepWaitReason.SMOKE_RESPONSE))
            return
        del self.futures[key]
        self.smoke_stops.pop(key, None)
        try:
            result = future.result()
        except Exception as error:  # noqa: BLE001 - a bug in one smoke must not stop an unattended sweep
            self._smoke_crashed(key, error)
            return
        if result.ok:
            self.clear_infra("client-key")
            self.clear_infra("smoke-tool")
            self._record_pass(key, slot, result)
            return
        assert result.failure is not None
        failure = result.failure
        if failure.klass == SmokeAuthClass.REFUSAL:
            self.clear_infra("client-key")
        if verdict.disposition(failure) is Disposition.WAIT:
            self._smoke_wait(key, slot, failure)
            return
        self._record_failure(key, failure, slot.get("operation_id"))

    def _smoke_crashed(self, key: str, error: BaseException) -> None:
        """The sweep's own smoke code raised: a tool fault, never a verdict on the recipe.

        The exception's type, message (an ``OSError`` keeps its filename) and last frames
        go into the row and the event log. The recipe is requeued without spending an
        attempt, and this recipe backs off (``note_infra``) until a smoke passes again, so a
        crash that repeats costs backed-off loads, never the recipe.
        """
        evidence = crash_evidence(error)
        failure = policy.classify(
            "smoke",
            "smoke.crash",
            f"{evidence['type']}: {evidence['message']}",
            verdict.SWEEP_CRASH,
            evidence,
        )
        self.state.event(
            f"smoke of {key} crashed in the sweep (a tool fault, not the recipe's): "
            f"{failure.detail[:200]} at {evidence['frames'][-1][:200] if evidence['frames'] else '?'}"
        )
        self.note_infra("smoke-tool", f"the sweep's smoke crashed: {failure.detail}")
        self.requeue(
            key,
            f"the sweep's smoke crashed: {failure.detail}",
            failure=failure,
            disposition=Disposition.OBSERVE.value,
        )
        self.state.entry(key)["not_before"] = self.state.data["infra"]["smoke-tool"][
            "until"
        ]

    def _smoke_wait(
        self, key: str, slot: dict[str, Any], failure: policy.Failure
    ) -> None:
        """Retry busy responses and typed key expiry without spending an attempt.

        Both reasons use the smoke phase budget, even during retry backoff.
        Expiry releases the lane without spending a recipe attempt.
        """
        now = self.clock.now()
        if failure.klass == policy.CLIENT_CREDENTIAL:
            self.note_infra(
                "client-key",
                "the gateway refused the sweep's client key "
                f"({failure.describe()}); renew it in the --client-key-file",
            )
            until = float(self.state.data["infra"]["client-key"]["until"])
            wait = SweepWait(
                reason=SweepWaitReason.CLIENT_KEY, detail=failure.describe()
            )
        else:
            until = now + self.cfg.retry_delay
            wait = SweepWait(
                reason=SweepWaitReason.SMOKE_BUSY, detail=failure.describe()
            )
        if self._wait_slot(key, slot, wait):
            return
        slot["smoke_not_before"] = until
        self.record_result(
            key,
            "smoke",
            "waiting",
            reason=failure.describe(),
            disposition=Disposition.WAIT.value,
            failure_class=failure.klass,
            evidence=dict(failure.evidence),
            slot=slot,
        )

    def preempt(self, why: str) -> None:
        for key, slot in list(self.state.slots.items()):
            future = self.futures.pop(key, None)
            if future is not None:
                future.cancel()
            self.requeue(key, why)
            slot["phase"] = "finished"
        self.state.data["loads"].clear()

    # ------------------------------------------------------------- scheduling

    def enqueue_finished_cleanup(self) -> None:
        """Durably release finished lane occupancy without waiting on cleanup I/O."""
        pending = self.state.data.setdefault("cleanup", {})
        changed = False
        for key, slot in list(self.state.slots.items()):
            if slot["phase"] != "finished":
                continue
            pending.setdefault(slot["alias"], {"attempts": 0, "next_check": 0.0})
            self.futures.pop(key, None)
            stop = self.smoke_stops.pop(key, None)
            if stop is not None:
                stop.set()
            del self.state.slots[key]
            changed = True
        if changed:
            self.state.save()

    def cleanup_finished(self, *, profile_ready: bool = False) -> None:
        self.enqueue_finished_cleanup()
        pending = self.state.data.setdefault("cleanup", {})
        if pending and not profile_ready and not self._profile_edit_ready():
            return
        live_aliases = {slot["alias"] for slot in self.state.slots.values()}
        for alias, record in list(pending.items()):
            if alias in live_aliases or self.clock.now() < record["next_check"]:
                continue
            reply = self.vk.run(
                "profile", "remove", alias, "--yes", profile=self.cfg.sweep_profile
            )
            removed = reply.ok
            if not removed:
                # "absent or ambiguous" is not evidence of absence: read the exact names.
                exported = self.vk.run(
                    "profile", "export", profile=self.cfg.sweep_profile
                )
                document = exported.document
                assignments = (
                    document.get("assignments") if isinstance(document, dict) else None
                )
                removed = (
                    exported.ok
                    and isinstance(assignments, list)
                    and all(
                        isinstance(item, dict) and item.get("assignment_name") != alias
                        for item in assignments
                    )
                )
            if removed:
                del pending[alias]
            else:
                record["attempts"] += 1
                record["next_check"] = self.clock.now() + min(
                    self.cfg.retry_delay * 2 ** min(record["attempts"], 5), INFRA_CAP
                )
                self.state.event(
                    f"assignment cleanup deferred: {alias}; lane released, retry queued"
                )

    def ready(self, recipe: Recipe) -> bool:
        """The exact NAS assets are there (the download finished).

        Never fleet fit or the library's readiness: those cannot hold while someone else's
        workload fills the Sparks. ``profile load --review`` decides fit at placement.
        """
        if recipe.cache_ready:
            return True
        download = self.state.downloads.get(recipe.key, {})
        accepted = AcceptedRequest.read(download)
        if (
            download.get("state") == LifecycleState.SUCCEEDED
            and accepted is not None
            and content_identity(accepted.recipe) == content_identity(recipe)
            and (
                self.model_list.complete_started_at
                <= float(download.get("done_at") or 0.0)
                or recipe.model_set <= self.cached_models
            )
        ):
            return True
        return recipe.local == "cached" and all(
            d in self.cached_models for d in recipe.model_digests
        )

    def _review_transport_source(self, recipe: Recipe) -> str:
        # Back off the failed review topology, not a recipe verdict or the
        # whole scheduler. A dual preview outage must leave singles eligible.
        return f"{ReviewInfrastructureSource.TRANSPORT}:{self.cfg.sweep_profile}:{recipe.node_count}"

    def ready_candidates(self) -> list[Recipe]:
        now = self.clock.now()
        if self.controller_wait_issue() is not None:
            return []
        waits = self.controller_wait_holds()
        out = []
        for key in self.queue:
            recipe = self.recipes.get(key)
            entry = self.state.recipes.get(key, {})
            if recipe is None or not self.ready(recipe) or key in self.state.slots:
                continue
            if key in waits and waits[key].identity == current_identity(recipe):
                continue  # re-observe the original receipt, never replay unchanged intent
            if self.lanes_short(recipe):
                continue  # waits for its lanes to come online; the online ones carry on
            if now < float(entry.get("not_before", 0)) or now < float(
                entry.get("defer_until", 0)
            ):
                continue
            incident = self.state.data["infra"].get(
                self._review_transport_source(recipe)
            )
            if incident and now < float(incident["until"]):
                continue
            out.append(recipe)
        deferred = [r for r in out if self.state.entry(r.key).get("wait_deferral")]
        fresh = [r for r in out if not self.state.entry(r.key).get("wait_deferral")]
        return fresh + sorted(
            deferred, key=lambda r: (r.updated_at, r.key), reverse=True
        )

    def _bins(self) -> list[policy.SparkBin]:
        by_revision = {r.revision_id: r for r in self.recipes.values() if r.revision_id}
        bins: dict[str, policy.SparkBin] = {
            s.id: policy.SparkBin(
                s.id,
                s.name,
                s.memory_total,
                model_digests=frozenset(
                    digest
                    for revision in s.installed_revisions
                    if revision in by_revision and by_revision[revision].node_count == 1
                    for digest in by_revision[revision].model_digests
                ),
                installed_revisions=s.installed_revisions,
                disk_blocked_recipes=frozenset(
                    key
                    for key, entry in self.state.data["recipes"].items()
                    if entry.get("storage_wait") is not None
                    and (wait := SweepStorageWait.model_validate(entry["storage_wait"]))
                    and key in self.recipes
                    and self.recipes[key].content_sha256 == wait.content_sha256
                    and self.clock.now() < wait.next_check
                    and s.id in wait.node_ids
                ),
            )
            for s in self.sparks()
        }
        holds = self._lane_holds()
        return [bin_ for bin_ in bins.values() if bin_.id not in holds]

    def _lane_holds(self) -> dict[str, str]:
        """Why each online Spark takes no new placement now; a Spark not named is free.

        The one account the scheduler and the idle alert share.
        """
        holds: dict[str, str] = {}
        for slot in self.state.slots.values():
            if slot["phase"] != "finished":
                for node in slot["node_ids"]:
                    holds.setdefault(node, f"lane {slot['alias']} is {slot['phase']}")
        for node, why in self._cleanup_holds().items():
            holds.setdefault(node, why)
        for record in self.state.data.get("released", {}).values():
            for node in record["nodes"]:
                holds.setdefault(
                    node,
                    "given-up load "
                    + ", ".join(record.get("aliases", ()))
                    + " still shows its workload",
                )
        for node, why in self.adoption_holds.items():
            holds.setdefault(node, why)
        online = {s.id for s in self.sparks()}
        if wait := self.state.data.get("profile_edit_wait"):
            held = (
                f"profile edit held on {wait['waiting_on']} until "
                f"{datetime.fromtimestamp(wait['wait_deadline'], UTC).isoformat()}"
            )
        else:
            held = str(self.state.data.get("dual_reservation") or "")
        for node in online if held else ():
            holds.setdefault(node, held)
        return {node: why for node, why in holds.items() if node in online}

    def _hold_lanes(self, load: Mapping[str, Any]) -> None:
        """Remember the Sparks and workload names of a load the sweep gives up."""
        lanes = [
            s
            for s in self.state.slots.values()
            if s.get("request_key") == load["request_key"]
        ]
        if not (load.get("app_id") and lanes) or load.get("state") in SETTLED:
            return  # a settled application holds nothing
        now = self.clock.now()
        record = self.state.data.setdefault("released", {}).setdefault(
            load["request_key"],
            {
                "app_id": load["app_id"],
                "since": now,
                "cancel_at": now + self.cfg.retry_delay,
            },
        )
        record["nodes"] = sorted(
            {n for s in lanes for n in s["node_ids"]} | set(record.get("nodes", ()))
        )
        record["aliases"] = sorted(
            {s["alias"] for s in lanes} | set(record.get("aliases", ()))
        )

    def release_given_up(self, now: float) -> None:
        """A load the sweep gave up is cancelled until the Spark shows none of its workloads.

        Left alone, the platform's start child keeps retrying and holds the Spark, and every later
        load of the lane is refused. Only the lanes of that load wait; the other lane carries on,
        and the hold lasts exactly as long as the Controller shows the workload.
        """
        held = self.state.data.get("released")
        if not held:
            return
        try:
            fleet = fetch_fleet(self.vk, self.cfg.default_spark_memory)
        except VonkctlError as error:
            self.state.event(f"held lanes unreadable, still held: {str(error)[:120]}")
            return  # unreadable: keep holding, look again
        for key, record in list(held.items()):
            shown = any(
                p.alias in record["aliases"] and p.node_id in record["nodes"]
                for p in fleet.presences
            )
            # The Spark is occupied for as long as the Controller shows the workload.
            if not shown:
                del held[key]
                self.state.event(
                    f"released {record['aliases']} on {record['nodes']}: lane free again"
                )
            elif now >= record["cancel_at"] and not self._ridden_by_live_lane(
                record["app_id"], key
            ):
                self.vk.run(
                    "profile",
                    "cancel",
                    record["app_id"],
                    "--yes",
                    "--detach",
                    profile=self.cfg.sweep_profile,
                )
                record["cancel_at"] = now + self.cfg.retry_delay
        self.state.save()

    def release_orphan_load(self) -> None:
        """Forget a load whose lanes have all ended: nothing of ours waits on it any more.

        A lane fails on its own deadline (admission stalled, load timeout) and cancels the
        application detached; if the Controller keeps holding that application (a retried
        admission), the load record would otherwise stay forever and no lane would ever be
        scheduled again.
        """
        for load in list(self.state.data["loads"].values()):
            self._release_orphan_load(load)

    def _release_orphan_load(self, load: dict[str, Any]) -> None:
        if load.get("replanned_pending"):
            return  # observe until the fresh snapshot supersedes it, never cancel a race
        if self._adopted_slots_live(load):
            return  # selected aggregate still owns an active borrowed lane
        key = load.get("request_key")
        if any(
            s.get("request_key") == key and s["phase"] != "finished"
            for s in self.state.slots.values()
        ):
            return
        app = load.get("app_id")
        if (
            app
            and load.get("state") not in SETTLED
            # A stalled application released by its lane is the Controller's to heal.
            and not load.get("stall_released")
            and not self._ridden_by_live_lane(app, key)
        ):
            # Ours (the sweep's own request key): cancel it so it releases what it holds.
            self.vk.run(
                "profile",
                "cancel",
                app,
                "--yes",
                "--detach",
                profile=self.cfg.sweep_profile,
            )
        self.state.event(f"released load {app or key}: its lanes have ended")
        self.state.data["loads"].pop(key, None)

    def check_idle(self, now: float) -> None:
        """Alert when an idle lane has had ready work for too long (the sweep must never idle).

        Each idle lane is named with what holds it, from the same account the scheduler
        places by (``_lane_holds``), so the alert never reports a lane free that placement
        skips.
        """
        holds = self._lane_holds()
        lanes = {
            node
            for slot in self.state.slots.values()
            if slot["phase"] != "finished"
            for node in slot["node_ids"]
        }
        idle_lanes = [s for s in self.sparks() if s.id not in lanes]
        waiting = bool(idle_lanes) and not self.paused and bool(self.ready_candidates())
        if not waiting:
            self.idle_since = None
            pending_idle = self.state.data.get("pending_idle")
            self.state.data["alerts"] = [pending_idle] if pending_idle else []
            return
        if self.idle_since is None:
            self.idle_since = now
        idle = now - self.idle_since
        if idle < self.cfg.idle_alert_seconds:
            return
        unheld = (
            self.state.data.get("clearing_reason")
            or "placement review or lane admission"
        )
        causes = "; ".join(f"{s.name}: {holds.get(s.id) or unheld}" for s in idle_lanes)
        message = (
            f"IDLE WITH READY WORK: {len(idle_lanes)} idle lane(s) for {round(idle)}s "
            f"while recipes are ready (load record: "
            f"{len(self.state.data['loads'])}; {causes})"
        )
        if not self.state.data.get("alerts"):
            self.state.event(message)
        self.state.data["alerts"] = [message]

    def controller_wait_holds(self) -> dict[str, ControllerWaitHold]:
        return {
            key: ControllerWaitHold.model_validate(entry["controller_wait"])
            for key, entry in self.state.recipes.items()
            if entry.get("controller_wait") is not None
        }

    def controller_wait_issue(self) -> str | None:
        holds = list(self.controller_wait_holds().values())
        for hold in holds:
            controller = hold.wait.controller
            if controller is None or controller.identity is None:
                continue
            if any(
                other.identity != hold.identity
                and other.wait.controller is not None
                and other.wait.controller.identity == controller.identity
                for other in holds
            ):
                return (
                    f"shared Controller wait {controller.description} "
                    f"({controller.identity[1]}); placement paused pending changed evidence"
                )
        return None

    def _refresh_controller_waits(self, now: float) -> None:
        """Probe accepted receipts with bounded reads; time alone never resubmits a load."""
        documents: dict[str, Mapping[str, Any]] = {}
        recovered: set[tuple[str, str]] = set()
        platform = self._component_observations()
        for key, hold in self.controller_wait_holds().items():
            recipe = self.recipes.get(key)
            if recipe is not None and current_identity(recipe) != hold.identity:
                self.state.entry(key).pop("controller_wait", None)
                continue  # latest content gets a fresh canonical review
            if any(
                platform.get(owner) is not None and platform[owner] != fingerprint
                for owner, fingerprint in hold.platform.items()
            ):
                self.state.entry(key).pop("controller_wait", None)
                self.state.event(
                    f"deployed platform changed for {key}; re-planning placement"
                )
                continue
            if now < hold.next_check:
                continue
            hold.next_check = now + self.cfg.defer_delay
            self.state.entry(key)["controller_wait"] = hold.model_dump(mode="json")
            if hold.application_id not in documents:
                reply = self.vk.run(
                    "profile",
                    "progress",
                    "--application",
                    hold.application_id,
                    profile=self.cfg.sweep_profile,
                    timeout=30,
                )
                if not reply.ok or not isinstance(reply.document, Mapping):
                    continue  # unknown is no recovery proof; next probe remains visible
                if reply.document.get("id") != hold.application_id:
                    continue
                documents[hold.application_id] = reply.document
            document = documents[hold.application_id]
            if document.get("state") not in ACTIVE_APP | SETTLED:
                continue
            if (
                document.get("state") in SETTLED
                and document.get("state") != LifecycleState.SUCCEEDED
            ):
                # An ending is no recovery proof. A successor carries the wait;
                # without one nothing waits any more, so re-plan unless shared.
                replaced = policy.supersession(document)
                if replaced is not None and replaced.successor is not None:
                    hold.application_id = replaced.successor
                    self.state.entry(key)["controller_wait"] = hold.model_dump(
                        mode="json"
                    )
                elif self.controller_wait_issue() is None:
                    self.state.entry(key).pop("controller_wait", None)
                    self.state.event(
                        f"Controller wait receipt for {key} ended "
                        f"{document.get('state')} without a successor; re-planning"
                    )
                continue
            signal = verdict.application_signal(document)
            observed = signal.wait.controller if signal.wait is not None else None
            previous = hold.wait.controller
            resumed = (
                signal.wait is None or document.get("state") == LifecycleState.SUCCEEDED
            )
            if resumed or (
                observed is not None
                and previous is not None
                and (observed.reason, observed.identity)
                != (previous.reason, previous.identity)
            ):
                self.state.entry(key).pop("controller_wait", None)
                self.state.event(
                    f"Controller wait changed for {key}; re-planning placement"
                )
                if resumed and previous is not None and previous.identity is not None:
                    recovered.add(previous.identity)
        # One original receipt resuming is behaviour proof for the shared dependency.
        for key, hold in self.controller_wait_holds().items():
            if (
                hold.wait.controller is not None
                and hold.wait.controller.identity in recovered
            ):
                self.state.entry(key).pop("controller_wait", None)

    def schedule(self, now: float) -> None:
        self.state.data.pop("dual_reservation", None)
        self.adoption_holds.clear()
        self._refresh_controller_waits(now)
        self.release_given_up(now)
        self.release_orphan_load()
        candidates = self.ready_candidates()
        if not candidates:
            return
        if not self._take_over(now, place_ready=True):
            return
        attempted: set[str] = set()
        # Each plan consumes a candidate; retries happen on a later tick.
        for _ in range(len(candidates)):
            if not candidates:
                return
            if attempted:
                self._observe_owner(self.clock.now())
                if self.paused or self.owner_status.new_activity:
                    return
            bins = self._bins()
            if not bins:
                return
            # A gang can drain healthy lanes, but cannot reserve a free lane behind a
            # stalled application or unresolved cleanup on its other member.
            if (
                candidates[0].node_count == 2
                and len(bins) < 2
                and (
                    self._cleanup_nodes()
                    or self.state.data.get("released")
                    or self.unobserved_loads
                    or any(
                        load.get("stalled")
                        or load.get("app_id") in self.stalled_applications
                        for load in self.state.data["loads"].values()
                    )
                )
            ):
                candidates = [recipe for recipe in candidates if recipe.node_count == 1]
                if not candidates:
                    return
            # A dual at the head drains both lanes; do not refill one with later singles.
            if candidates[0].node_count == 2:
                if len(bins) < 2:
                    self._reserve_for_dual(candidates[0], {b.id for b in bins})
                    return
                self.state.data["mode"] = "dual"
                dual = policy.place_dual(candidates[:1], bins, self.cfg.reserve_bytes)
                placements = [dual] if dual else []
            else:
                self.state.data["mode"] = "single"
                singles = []
                for recipe in candidates:
                    if recipe.node_count == 2:
                        break
                    singles.append(recipe)
                placements = policy.place_singles(
                    singles,
                    bins,
                    self.cfg.reserve_bytes,
                    lambda r: alias_for(r, self.defs),
                )
            if not placements and not self.state.slots:
                # Declared demand beyond one Spark: let the platform's review decide.
                head = candidates[0]
                width = head.node_count
                spark_ids = tuple(b.id for b in bins[:width])
                if len(spark_ids) == width and all(
                    head.key not in b.disk_blocked_recipes for b in bins[:width]
                ):
                    placements = [policy.Placement(head, spark_ids)]
            if placements:
                attempted.update(p.recipe.key for p in placements)
                if self._apply(placements, self.clock.now()) is False:
                    return
            else:
                return
            candidates = [r for r in self.ready_candidates() if r.key not in attempted]

    def _reserve_for_dual(self, recipe: Recipe, free: set[str]) -> None:
        """Name why a free lane stays empty: the dual at the head drains both lanes."""
        busy = {why for node, why in self._lane_holds().items() if node not in free}
        lanes = [
            float(slot["deadline"])
            for slot in self.state.slots.values()
            if slot["phase"] != SweepSlotPhase.FINISHED
            and not set(slot["node_ids"]) <= free
        ]
        until = (
            f" until {datetime.fromtimestamp(max(lanes), UTC).isoformat()}"
            if lanes
            else ""
        )
        self.state.data["dual_reservation"] = (
            f"reserved for dual {recipe.key} behind {'; '.join(sorted(busy))}{until}"
        )

    def _cleanup_confirmed(self) -> dict[str, int]:
        confirmed: dict[str, int] = {}
        for load in self.state.data["cleanup_loads"].values():
            for row in load.get("effects", []):
                if cleanup.stopped(row):
                    signature = cleanup.stop_signature(row["stop_effect"])
                    confirmed[signature] = max(
                        confirmed.get(signature, 0), row["workload_intent_ordinal"]
                    )
        return confirmed

    def _cleanup_pending(self, load: Mapping[str, Any]) -> list[dict[str, Any]]:
        if load.get("end"):
            return []
        confirmed = self._cleanup_confirmed()
        return [
            row
            for row in load.get("effects", [])
            if confirmed.get(cleanup.stop_signature(row["stop_effect"]), 0)
            < row["workload_intent_ordinal"]
        ]

    def _cleanup_nodes(self) -> set[str]:
        """Unconfirmed exact stop scopes remain occupied, including whole gangs."""
        return set(self._cleanup_holds())

    def _cleanup_holds(self) -> dict[str, str]:
        """The online Sparks cleanup bookkeeping keeps from placement, each with why."""
        held: dict[str, str] = {}
        for key, load in self.state.data["cleanup_loads"].items():
            if load.get("end"):
                continue
            name = f"cleanup {str(load.get('app_id') or key)[:8]}"
            if load.get("terminal_receipt"):
                if load.get("projection_known") and not self._cleanup_pending(load):
                    continue
                # Terminal bookkeeping is not a physical occupancy certificate.
                # Current fleet facts and the fresh Controller review own admission.
                kept = self._own_aliases()
                for p in self.fleet.presences:
                    if p.alias not in kept:
                        held.setdefault(
                            p.node_id,
                            f"{name} ended with unconfirmed stops and runs "
                            f"{p.alias}, not the sweep's",
                        )
                continue
            if not load.get("projection_known"):
                # Before the current projection is known no lane may be inferred free.
                for s in self.sparks():
                    held.setdefault(s.id, f"{name} effects are not yet observed")
                continue
            for row in self._cleanup_pending(load):
                for node in row["node_ids"]:
                    held.setdefault(
                        node, f"{name} stop of run {row.get('target_id')} unconfirmed"
                    )
        if self.state.data.get("takeover_fallback"):
            # Bookkeeping is not occupancy. A fresh review still owns admission;
            # this observation never rewrites or declares an old receipt successful.
            own = self._own_aliases()
            held = {
                p.node_id: f"clearing fallback: runs {p.alias}, not the sweep's"
                for p in self.fleet.presences
                if p.alias not in own
            }
        online = {s.id for s in self.sparks()}
        return {node: why for node, why in held.items() if node in online}

    def _cleanup_target_absent(self, load: Mapping, *, terminal: bool = False) -> bool:
        """An exact absent receipt plus fresh online fleet facts ends bookkeeping.

        A ``terminal`` receipt is final: absence is judged against its own stop targets
        only, and one with none has nothing to wait for. An absent receipt without
        targets may have stopped anything, so only a fleet with no runs ends it.
        """
        reply = self.vk.run("fleet")
        if not reply.ok:
            return False
        try:
            observed = CleanupFleetObservation.model_validate(reply.document)
        except ValidationError:
            return False
        # Offline or incomplete facts cannot certify the absence of an old run.
        effects = load.get("effects") or []
        expected_nodes = {node for row in effects for node in row["node_ids"]}
        expected_runs = {row["target_id"] for row in effects}
        for signature in load.get("reviewed_stops") or []:
            stop = cleanup.reviewed_stop(signature)
            if stop is None:
                return False
            expected_nodes.update(stop.node_ids)
            expected_runs.add(stop.run_id)
        nodes = {node.id for node in observed.nodes}
        if not nodes or not expected_nodes <= nodes:
            return False
        relevant = [
            node
            for node in observed.nodes
            if not expected_nodes or node.id in expected_nodes
        ]
        if any(
            node.connection.online_state != FleetOnlineState.ONLINE for node in relevant
        ):
            return False
        present = {run.run_id for node in observed.nodes for run in node.loaded}
        if expected_runs or terminal:
            return not (expected_runs & present)
        return not present

    def _end_cleanup(self, load: dict[str, Any], now: float) -> None:
        """Fresh fleet facts show the cleanup's targets gone: its bookkeeping ends."""
        load["end"] = CleanupEnd(
            reason=CleanupEndReason.TARGET_ABSENT, observed_at=now
        ).model_dump(mode="json")
        load["observation_unknown"] = False
        load.pop("observation_budget", None)
        self.clear_infra("clearing")
        self.backoff_until = max(
            (
                item["until"]
                for source, item in self.state.data["infra"].items()
                if source != "download"
            ),
            default=0.0,
        )
        self.state.event(
            f"cleanup {str(load.get('app_id') or load['request_key'])[:8]} retired: "
            "its stop targets are gone from the fleet"
        )
        self.state.save()

    @staticmethod
    def _defer_cleanup(
        load: dict[str, Any], budget: CleanupObservationBudget, now: float
    ) -> None:
        budget.attempts += 1
        budget.next_check = now + min(
            INFRA_BASE * 2 ** min(budget.attempts - 1, 20), INFRA_CAP
        )
        load["observation_budget"] = budget.model_dump(mode="json")

    def observe_cleanup(self) -> None:
        """Reconcile original applications and stops; never replace intent on timeout."""
        for load in self.state.data["cleanup_loads"].values():
            if load.get("end"):
                continue
            if (
                load.get("terminal_receipt")
                and load.get("projection_known")
                and not self._cleanup_pending(load)
            ):
                continue  # only a retained terminal receipt is immutable history
            now = self.clock.now()
            budget = CleanupObservationBudget.model_validate(
                load.get("observation_budget", {})
            )
            if now < budget.next_check:
                continue
            if load.get("terminal_receipt"):
                # The Controller ended it with stops unconfirmed and its receipt is
                # final: only current fleet facts retire it, once its targets are gone.
                if self._cleanup_target_absent(load, terminal=True):
                    self._end_cleanup(load, now)
                else:
                    self._defer_cleanup(load, budget, now)
                continue
            args = ["progress"]
            if load.get("app_id"):
                args += ["--application", load["app_id"]]
            else:
                args += ["--request-key", load["request_key"]]
            reply = self.vk.run("profile", *args, profile=self.cfg.sweep_profile)
            document = reply.document
            if reply.is_not_found and self._cleanup_target_absent(load):
                self.clear_infra("clearing")
                self._end_cleanup(load, now)
                continue
            if not reply.ok:
                self._defer_cleanup(load, budget, now)
                # Status reports the next observation without pausing other lanes.
                if item := self.state.data["infra"].get("clearing"):
                    item["count"] += 1
                    item["until"] = budget.next_check
            else:
                load.pop("observation_budget", None)
                self.clear_infra("clearing")
            if (
                not load.get("app_id")
                and reply.is_not_found
                and load.get("reviewed_stops") is not None
            ):
                review = self.vk.run(
                    "profile", "load", "--review", profile=self.cfg.sweep_profile
                )
                current = review.document
                if not review.ok or not isinstance(current, Mapping):
                    continue
                stops = [
                    row
                    for row in dig(current, "effects", "runs", default=[])
                    if isinstance(row, Mapping) and row.get("action") == "stop"
                ]
                if (
                    current.get("allowed") is not True
                    or sorted(cleanup.stop_signature(row) for row in stops)
                    != load["reviewed_stops"]
                    or self._adopted_requests(current) is None
                ):
                    load["observation_unknown"] = True
                    continue
                try:
                    document = self._submit_load(
                        self.cfg.sweep_profile, load["request_key"], "takeover"
                    )
                except VonkctlError as error:
                    self.state.event(
                        f"cleanup re-entry unanswered, observing: {str(error)[:120]}"
                    )
                    continue
                # Exactly absent receipt permits same-key reentry, never a new UUID.
                load["app_id"] = document.get("id")
                load.pop("observation_budget", None)
                self.state.save()
                continue
            if not reply.ok or not isinstance(document, Mapping):
                load["observation_unknown"] = True
                self.state.event(
                    "exact cleanup receipt observation unavailable; retaining its original identity"
                )
                continue
            app = document.get("id")
            if (
                not isinstance(app, str)
                or document.get("request_key") != load["request_key"]
                or (load.get("app_id") and app != load["app_id"])
            ):
                load["observation_unknown"] = True
                continue
            if document.get("state") in SETTLED:
                load.update(
                    app_id=app,
                    state=document["state"],
                    terminal_receipt=dict(document),
                    observation_unknown=False,
                )
                # A terminal parent may lack its final child projection. That
                # cannot keep the old request at the head of fresh admission.
                self.state.save()
            rows = cleanup.stop_effects(document)
            expected = load.get("reviewed_stops")
            if rows is not None and (
                (not rows and expected != [])
                or (
                    isinstance(expected, list)
                    and not set(expected)
                    <= {cleanup.stop_signature(row["stop_effect"]) for row in rows}
                )
            ):
                # Acceptance can precede adapter initialization. An empty or partial
                # projection never erases the reviewed stops or invents free lanes.
                load["observation_unknown"] = True
                continue
            previous = {row["effect_id"]: row for row in load.get("effects", [])}
            if (
                rows is None
                or (
                    load.get("projection_known")
                    and set(previous) != {row["effect_id"] for row in rows}
                )
                or any(
                    row["effect_id"] not in {item["effect_id"] for item in rows}
                    for row in previous.values()
                )
                or any(
                    row["effect_id"] in previous
                    and (
                        not cleanup.same_effect(previous[row["effect_id"]], row)
                        or (
                            cleanup.stopped(previous[row["effect_id"]])
                            and not cleanup.stopped(row)
                        )
                    )
                    for row in rows or []
                )
            ):
                load["observation_unknown"] = True
                self.state.event(
                    "canonical cleanup effect projection unavailable or changed; retaining original scopes"
                )
                continue
            load.update(
                app_id=app,
                state=document.get("state"),
                effects=rows,
                projection_known=True,
                observation_unknown=False,
                observed_at=self.clock.now(),
            )
            if not self._cleanup_pending(load):
                # Child completion is not parent completion. Reconcile its exact
                # targets with fresh fleet facts before retiring the observer.
                if self._cleanup_target_absent(load, terminal=True):
                    self._end_cleanup(load, now)
                else:
                    self._defer_cleanup(load, budget, now)
            self.state.save()

    def _take_over(self, now: float, *, place_ready: bool = False) -> bool:
        """Bound clearing admission; fresh fleet facts keep independent lanes moving."""
        try:
            self.fleet = fetch_fleet(self.vk, self.cfg.default_spark_memory)
        except VonkctlError:
            self.state.data["clearing_reason"] = "clearing waits on fleet observation"
            return False  # no stale fleet may authorize placement
        budget = self.state.data.setdefault(
            "takeover_budget", {"attempts": 0, "next_check": 0.0}
        )
        offline = [s.name for s in self.fleet.sparks if not s.online]
        refusal = self.state.data.get("clearing_refusal")
        reason = (
            "clearing waits on offline " + ", ".join(offline)
            if offline
            else f"clearing refused: {refusal}"
            if refusal
            else "clearing waits on cleanup receipt or admission"
        )
        self.state.data["clearing_reason"] = reason
        if self.state.data.get("took_over") and all(
            load.get("projection_known") and load.get("state") not in SETTLED
            for load in self.state.data["cleanup_loads"].values()
        ):
            self.state.data.pop("clearing_reason", None)
            return True
        if now < budget["next_check"]:
            if place_ready:
                self.state.data["takeover_fallback"] = True
            return bool(self.state.data.get("takeover_fallback"))
        budget["attempts"] += 1
        budget["next_check"] = now + min(
            TAKEOVER_RETRY_SECONDS * 2 ** min(budget["attempts"] - 1, 5), INFRA_CAP
        )
        self.state.save()
        try:
            admitted = self._attempt_takeover(now, place_ready=place_ready)
        except VonkctlError as error:
            self.state.event(f"clearing attempt unanswered: {str(error)[:120]}")
            admitted = False
        if admitted:
            budget.update(attempts=0, next_check=0.0)
            self.state.data.pop("clearing_reason", None)
            self.state.data.pop("clearing_refusal", None)
            return True
        if self.state.data.get("profile_edit_wait"):
            return False
        self.state.data["takeover_fallback"] = True
        self.state.event(
            f"{reason}; reviewing free online lanes while clearing retries"
        )
        return True

    def _attempt_takeover(self, now: float, *, place_ready: bool = False) -> bool:
        """Begin one durable cleanup, then admit lanes from exact receipts."""
        loads = self.state.data["cleanup_loads"]
        if (
            self.state.data.get("cleared_for")
            and not self.state.data.get("cleared_by")
            and all(
                load.get("terminal_receipt") or load.get("end")
                for load in loads.values()
            )
        ):
            # A decided clearing precedes ready-lane shortcuts. In-flight intent
            # still follows its original receipt instead of being submitted again.
            return self._renew_cleanup(now)
        if loads:
            if all(
                load.get("terminal_receipt") or load.get("end")
                for load in loads.values()
            ):
                if not self.fleet.presences or (
                    place_ready
                    and any(s.id not in self._cleanup_nodes() for s in self.sparks())
                ):
                    # Prefer the ready lane's fresh reviewed load to an empty
                    # clearing that would introduce another adoption dependency.
                    return True
                return self._renew_cleanup(now)
            if not all(
                load.get("projection_known")
                or load.get("terminal_receipt")
                or load.get("end")
                for load in loads.values()
            ):
                return False
            unresolved = [
                load
                for load in loads.values()
                if not load.get("terminal_receipt")
                and not load.get("end")
                and self._cleanup_pending(load)
            ]
            if unresolved and any(
                load.get("state") not in SETTLED for load in unresolved
            ):
                return True
            if not unresolved and self.state.data.get("took_over"):
                return True
            # Only observed terminal intent or an explicit new clearing decision may
            # enter normal admission again. A read timeout never reaches this branch.
            return self._renew_cleanup(now)
        if self.state.data.get("took_over") or self.state.slots:
            return True
        # Recover the latest canonical owned cleanup before creating any new intent.
        original = next(
            (
                item
                for item in reversed(self.state.data["own_loads"])
                if item.get("kind") in ("takeover", "stop")
                and item.get("profile") == self.cfg.sweep_profile
            ),
            None,
        )
        if original is not None:
            key = original["request_key"]
            self.state.data["took_over"] = True
            loads[key] = {"request_key": key, "app_id": original.get("application_id")}
            self.state.save()
            self.observe_cleanup()
            return bool(loads[key].get("projection_known"))
        try:
            self.fleet = fetch_fleet(self.vk, self.cfg.default_spark_memory)
        except VonkctlError:
            return False
        if not self.fleet.presences:
            # This permits a fresh review, not a claim that an old stop succeeded.
            self.state.data["took_over"] = True
            return True
        return self._renew_cleanup(now)

    def _clearing_refused(self, why: str) -> bool:
        """A clearing that is not submitted says why: in the status, once in the events."""
        if self.state.data.get("clearing_refusal") != why:
            self.state.event(f"clearing refused: {why}"[:300])
        self.state.data["clearing_refusal"] = why
        self.state.data["clearing_reason"] = f"clearing refused: {why}"
        return False

    def _request_clearing(self, now: float) -> None:
        """A new clearing decision gets its own bounded budget, never an old one's exhaustion."""
        self.state.data["took_over"] = False
        self.state.data["takeover_budget"] = {"attempts": 0, "next_check": 0.0}
        self.state.data.pop("takeover_fallback", None)
        self.state.data.pop("cleared_by", None)

    def _profile_edit_ready(self) -> bool:
        """Even a rolled-back edit supersedes a lane whose admission is still unbound.

        A bound lane is carried forward: the review of the edited profile must adopt it.
        The last receipt decides; a load never observed (an unknown submit) is unbound.
        """
        loads = self.state.data["loads"]
        loading = [
            slot
            for slot in self.state.slots.values()
            if slot["phase"] == SweepSlotPhase.LOADING
            and slot["request_key"] in loads
            and loads[slot["request_key"]].get("admission_unbound", True)
        ]
        if not loading:
            self.state.data.pop("profile_edit_wait", None)
            return True
        deadline = min(
            float(slot["watch"]["wait_deadline"])
            if slot.get("watch", {}).get("wait")
            else float(slot.get("watch", {}).get("progressed_at", slot["started_at"]))
            + self._load_window(slot)
            for slot in loading
        )
        applications = [
            f"{self.state.data['loads'].get(slot['request_key'], {}).get('app_id') or slot['request_key']}"
            f" ({slot.get('watch', {}).get('waiting_on') or 'admission not yet bound'})"
            for slot in loading
        ]
        now = self.clock.now()
        since = float(
            (self.state.data.get("profile_edit_wait") or {}).get("wait_started_at", now)
        )
        # The lane owns this bound, re-derived each pass: resumed progress may replace
        # an old dependency wait, so the watch starts fresh from the same start.
        watch = self.state.data["profile_edit_wait"] = {}
        progress = verdict.bounded_wait(
            SweepWait(
                reason=SweepWaitReason.ADMISSION,
                detail=f"profile edit awaits bound admission: {', '.join(applications)}",
            ),
            watch,
            window=deadline - since,
            now=now,
            started_at=since,
        )
        if progress is Progress.WAIT_EXPIRED:
            self.state.data.pop("profile_edit_wait", None)
            return True
        return False

    def _prune_profile(self) -> ProfilePruneOutcome:
        """Before any load of the sweep profile, it holds only what live lanes need.

        A load applies the whole profile. An assignment no lane accounts for (a lost cleanup
        record, an edit the Controller applied after a timeout) is kept or started by every
        later load, so a clearing or stop load would run the very workload it should end.
        The exact export decides; only names the sweep itself uses are removed.
        """
        if not self._profile_edit_ready():
            return ProfilePruneOutcome(
                self.state.data["profile_edit_wait"]["waiting_on"]
            )
        reply = self.vk.run("profile", "export", profile=self.cfg.sweep_profile)
        document = reply.document
        assignments = (
            document.get("assignments") if isinstance(document, dict) else None
        )
        if not reply.ok or not isinstance(assignments, list):
            waiting_on = f"sweep profile unreadable: {reply.error_text[:200]}"
            self.note_infra("profile-edit", waiting_on)
            return ProfilePruneOutcome(waiting_on)
        live = {
            str(s["alias"])
            for s in self.state.slots.values()
            if s["phase"] != "finished"
        }
        own = self._own_aliases()
        stale = {
            str(item.get("assignment_name"))
            for item in assignments
            if isinstance(item, dict)
            and item.get("assignment_name") in own
            and item.get("assignment_name") not in live
        }
        if not stale:
            return ProfilePruneOutcome()
        pending = self.state.data.setdefault("cleanup", {})
        for name in stale:
            pending.setdefault(name, {"attempts": 0, "next_check": 0.0})
        self.cleanup_finished(profile_ready=True)
        remaining = stale & set(pending)
        return ProfilePruneOutcome(
            f"sweep profile still holds unused assignments: {', '.join(sorted(remaining))}"
            if remaining
            else None
        )

    def _renew_cleanup(self, now: float) -> bool:
        # Every new clear starts from current physical facts, including after
        # a terminal receipt with an absent or incomplete child projection.
        self.fleet = fetch_fleet(self.vk, self.cfg.default_spark_memory)
        loads = self.state.data["cleanup_loads"]
        pruned = self._prune_profile()
        if pruned.waiting_on is not None:
            return self._clearing_refused(pruned.waiting_on)
        review = self.vk.run(
            "profile", "load", "--review", profile=self.cfg.sweep_profile
        )
        document = review.document
        if not review.ok and not isinstance(document, Mapping):
            return self._clearing_refused(
                f"clearing review unanswered: {review.error_text[:200]}"
            )
        if not isinstance(document, Mapping) or document.get("allowed") is not True:
            return self._clearing_refused(
                "clearing review not allowed: " + _blocker_text(document)
            )
        runs = dig(document, "effects", "runs", default=[])
        stops = [
            row
            for row in runs
            if isinstance(row, Mapping) and row.get("action") == "stop"
        ]
        online = {s.id for s in self.sparks()}
        enrolled = {s.id for s in self.enrolled()}
        own = self._own_aliases()
        if any(
            not set(row.get("node_ids", [])) <= online
            and not (
                set(row.get("node_ids", [])) <= enrolled and row.get("alias") in own
            )
            for row in stops
        ):
            # An empty whole-fleet load would clear someone else's workload on an offline
            # (or unselected) Spark. Our own workload there is stopped by the Controller
            # once the Spark answers: an undeliverable stop of ours never holds the
            # clearing of the online lanes.
            return self._clearing_refused(
                "a reviewed stop reaches a workload that is not ours on an offline Spark"
            )
        kept = {
            slot["alias"]
            for slot in self.state.slots.values()
            if slot["phase"] in ("loading", "smoking")
        }
        if any(row.get("alias") in kept for row in stops):
            return self._clearing_refused("the clearing would stop a live lane")
        if self._adopted_requests(document) is None:
            return self._clearing_refused(
                "pending cleanup effects are not bound for adoption"
            )
        residual = [
            row
            for load in loads.values()
            if not load.get("terminal_receipt")
            for row in self._cleanup_pending(load)
        ]
        if any(
            not any(
                cleanup.stop_signature(stop)
                == cleanup.stop_signature(row["stop_effect"])
                for stop in stops
            )
            for row in residual
        ):
            return self._clearing_refused(
                "fresh cleanup admission does not bind every exact residual stop"
            )
        key = self._own_key("takeover", self.cfg.sweep_profile)
        loads[key] = {
            "request_key": key,
            "app_id": None,
            "reviewed_stops": sorted(cleanup.stop_signature(row) for row in stops),
        }
        self.state.data["took_over"] = True
        self.state.data["dirty"] = True
        self.state.save()
        try:
            document = self._submit_load(self.cfg.sweep_profile, key, "takeover")
            loads[key]["app_id"] = document.get("id")
        except VonkctlError as error:
            self.note_infra("clearing", str(error))
            # The same request is observed on reentry; unknown acceptance is not replayed.
            return False
        # Submitted: a review still blocked by the same holders now waits on their owner.
        self.state.data["cleared_by"] = key
        self.state.data.pop("clearing_refusal", None)
        self.state.save()
        self.observe_cleanup()
        self.fleet = fetch_fleet(self.vk, self.cfg.default_spark_memory)
        return bool(
            loads[key].get("projection_known") or loads[key].get("terminal_receipt")
        )

    def _alias(self, recipe: Recipe) -> str:
        """The recipe's assignment name in the sweep profile, unique within it.

        A lane keeps the name it was opened with. Another recipe that serves
        under the same reviewed alias (the Inkling variants) gets a short
        suffix, because a profile cannot hold two running assignments of one
        name.
        """
        slot = self.state.slots.get(recipe.key)
        if slot is not None:
            return str(slot["alias"])
        if recipe.key in self._assigned:
            return self._assigned[recipe.key]
        alias = alias_for(recipe, self.defs)
        taken = (
            {str(s["alias"]) for k, s in self.state.slots.items() if k != recipe.key}
            | {a for k, a in self._assigned.items() if k != recipe.key}
            | {p.alias for p in self.fleet.presences}  # a finished sibling still up
        )
        if alias in taken:
            alias = unique_profile_alias(alias, recipe.key)
        self._assigned[recipe.key] = alias
        return alias

    def _apply(self, placements: Sequence[policy.Placement], now: float) -> bool | None:
        """Return False when the shared profile cannot be edited this pass."""
        if len(placements) > 1:
            # Each new lane has its own receipt. The unchanged lane remains in
            # this same whole-fleet profile and requires canonical adoption.
            for index, placement in enumerate(placements):
                if index:
                    self._observe_owner(self.clock.now())
                    if self.paused or self.owner_status.new_activity:
                        return
                if self._apply([placement], now) is False:
                    return False
                # Resolve a first lane's admission wait before reviewing the
                # next snapshot; a pending application has no effect to adopt.
                self.advance_slots(now)
            return
        profile = self.cfg.sweep_profile
        if self._prune_profile().waiting_on is not None:
            return False
        self._assigned = {}
        added: list[policy.Placement] = []
        names = {s.id: s.name for s in self.sparks()}
        for placement in placements:
            recipe = placement.recipe
            args = ["add", recipe.key]
            for spark_id in placement.spark_ids:
                args += ["--spark", names[spark_id]]
            args += [
                "--as",
                self._alias(recipe),
                "--state",
                "running",
                "--yes",
            ]
            try:
                self.vk.call("profile", *args, profile=profile)
                # Ours from the edit on: a rollback that fails leaves it for the prune.
                self._remember_own(self._alias(recipe))
                added.append(placement)
                self.clear_infra("profile-edit")
            except VonkctlError as error:
                if verdict.reply_disposition(error.reply) is Disposition.WAIT:
                    # The client, protocol or transport failed, or the Controller refused
                    # something the sweep itself sent (the name, the Sparks, the state):
                    # nothing about this recipe. The next pass reviews afresh.
                    self._rollback(added, f"profile edit failed: {error}")
                    self.note_infra("profile-edit", str(error))
                    self.state.entry(recipe.key)["not_before"] = self.state.data[
                        "infra"
                    ]["profile-edit"]["until"]
                    remaining = [p for p in placements if p.recipe.key != recipe.key]
                    if remaining:
                        self._apply(remaining, now)
                    return
                # An explicit recipe-data code on the recipe's own data.
                refused = verdict.refused_field(str(error))
                self._record_failure(
                    recipe.key,
                    policy.classify(
                        "profile-edit" if refused else "review",
                        error.code,
                        str(error),
                        klass="recipe-data" if refused else "",
                    ),
                    None,
                )
                continue
        if not added:
            return
        overfull = self._overfull_sparks()
        crowded = [p for p in added if overfull & set(p.spark_ids)]
        if crowded:
            # A large recipe runs alone; the other waits for the next free slot.
            self._rollback(crowded, f"would overfill Spark(s) {sorted(overfull)}")
            added = [p for p in added if p not in crowded]
            if not added:
                return
        aliases = {self._alias(p.recipe): p for p in added}
        kept = {
            s["alias"]
            for s in self.state.slots.values()
            if s["phase"] in ("loading", "smoking")
        }
        review = self.vk.run("profile", "load", "--review", profile=profile)
        document = review.document if isinstance(review.document, dict) else {}
        if review.is_transport_error or review.is_error_document or not document:
            self._rollback(added, f"review failed: {review.error_text}")
            if verdict.reply_disposition(review) is Disposition.WAIT:
                # A failed observation says nothing about the recipe. Back off
                # this review topology, leaving independent reviews eligible.
                for placement in added:
                    self.state.entry(placement.recipe.key).pop("not_before", None)
                    self.note_infra(
                        self._review_transport_source(placement.recipe),
                        review.error_text,
                    )
                self.state.save()
                return
            # A recipe-data refusal of the review: a deterministic verdict, recorded once,
            # never the same plan reviewed again and again.
            failure = policy.classify(
                "review",
                str(document.get("code") or ""),
                review.error_text,
                evidence={"reply": policy.excerpt(document, 1000)},
            )
            for placement in added:
                self._record_failure(placement.recipe.key, failure, None)
            return
        for placement in added:
            self.clear_infra(self._review_transport_source(placement.recipe))
        stopped = [
            r["alias"]
            for r in dig(document, "effects", "runs", default=[])
            if isinstance(r, dict)
            and r.get("action") == "stop"
            and r.get("alias") in kept
        ]
        if stopped:
            self._rollback(added, f"review would stop running lanes {stopped}")
            return
        cleanup_requests, dependency = self._cleanup_adoption(document)
        if cleanup_requests is None:
            self._rollback(
                added, "pending original cleanup effects are not bound for adoption"
            )
            self.note_infra(
                "review",
                "waiting for exact original cleanup adoption",
                dependency=dependency,
            )
            return
        adopted_requests = self._adopted_requests(document)
        if adopted_requests is None:
            self._rollback(
                added, "unchanged loading effects are not bound for adoption"
            )
            self.note_infra(
                "review", "observing unchanged loading effects before overlap"
            )
            # The rest of this pass would pay the same add, review and remove.
            loading = sorted(
                s["alias"]
                for s in self.state.slots.values()
                if s["phase"] == SweepSlotPhase.LOADING
            )
            retry = datetime.fromtimestamp(self.clock.now() + self.cfg.retry_delay, UTC)
            why = (
                f"loading lane(s) {', '.join(loading)} not adopted by the review of "
                f"the edited profile; {', '.join(p.recipe.key for p in added)} "
                f"retry at {retry.isoformat()}"
            )
            self.adoption_holds.update(
                {node: why for p in added for node in p.spark_ids}
            )
            return
        # The Controller's disk decision includes reservations, reused objects
        # and evictable installations. Preparation being waitable cannot turn
        # a proven storage shortfall into a placement on that Spark.
        storage_reasons = SweepStorageReview.model_validate(document).insufficient
        storage_blocked = [
            p
            for p in added
            if any(set(p.spark_ids) & set(r.node_ids) for r in storage_reasons)
        ]
        if storage_blocked:
            self._rollback(
                storage_blocked, "storage does not fit after eviction: re-planning"
            )
            for placement in storage_blocked:
                entry = self.state.entry(placement.recipe.key)
                entry["storage_wait"] = SweepStorageWait(
                    reasons=[
                        r
                        for r in storage_reasons
                        if set(r.node_ids) & set(placement.spark_ids)
                    ],
                    content_sha256=placement.recipe.content_sha256,
                    next_check=now + self.cfg.defer_delay,
                ).model_dump(mode="json")
                entry.pop("not_before", None)
                self.record_result(
                    placement.recipe.key,
                    "review",
                    "waiting",
                    reason="; ".join(r.detail for r in storage_reasons),
                    disposition=Disposition.WAIT.value,
                    document=document,
                )
            self.state.save()
            remaining = [p for p in added if p not in storage_blocked]
            if remaining:
                self._apply(remaining, now)
            return
        if not (
            document.get("allowed") is True
            or document.get("waits_for_preparation") is True
        ):
            self._blocked(document, aliases, added, now)
            admitted = {
                decision.get("alias")
                for decision in document.get("admission_decisions", [])
                if isinstance(decision, dict) and decision.get("allowed") is True
            }
            remaining = [p for alias, p in aliases.items() if alias in admitted]
            if remaining and len(remaining) < len(added):
                # A rejected batch accepted nothing. Re-plan the smaller set,
                # including a fresh review and binding, so one held Spark cannot
                # starve an independent lane. Each re-plan removes an assignment.
                self._apply(remaining, now)
            return
        self.clear_infra("review")
        seq = int(self.state.data["load_seq"]) + 1
        key = request_key(
            "sweep-load",
            self.state.nonce,
            self.cfg.sweep_profile,
            seq,
        )
        self.state.data["load_seq"] = seq
        self.state.data["loads"][key] = {
            "request_key": key,
            "app_id": None,
            "seq": seq,
            "submitted_at": now,
            "adopted_requests": adopted_requests,
            "adopted_cleanup_requests": cleanup_requests,
        }
        for placement in added:
            self._open_slot(placement, key, now)
        self.state.data["dirty"] = (
            True  # the submitted effect may be accepted even if its reply is lost
        )
        self.state.save()
        try:
            submitted = self._submit_load(profile, key, "placement")
        except VonkctlError as error:
            if verdict.reply_disposition(error.reply) is Disposition.WAIT:
                self.note_infra("load-submit", str(error))
                # Unknown admission is resolved by request UUID before any edit or replay.
                self.state.data["loads"][key]["state"] = "observing"
                return
            # A recipe-data refusal of the reviewed load: record the verdict with the
            # refusal as evidence; the same placement is never resubmitted unchanged.
            self.state.data["loads"].pop(key, None)
            reply = error.reply.document if error.reply else None
            failure = policy.classify(
                "review",
                error.code,
                str(error),
                evidence={"reply": policy.excerpt(reply, 1000) if reply else None},
            )
            for placement in added:
                self._record_failure(placement.recipe.key, failure, None)
                self.state.slots.pop(placement.recipe.key, None)
            self._rollback(added, f"load refused: {error}")
            return
        self.state.data["loads"][key]["app_id"] = submitted.get("id")
        if isinstance(submitted, Mapping):
            self.documents[key] = submitted
            self.state.data["loads"][key]["admission_unbound"] = (
                verdict.admission_unbound(submitted)
            )
        self.state.data["dirty"] = True
        self.state.event("load submitted: " + ", ".join(p.recipe.key for p in added))
        self.state.save()

    def _adopted_requests(
        self, review: Mapping[str, Any], *, excluded: frozenset[str] = frozenset()
    ) -> list[str] | None:
        """Require bound whole-assignment adoption before overlapping accepted snapshots."""
        adopted = (review.get("effects") or {}).get("adopted") or []
        requests = []
        for key, slot in self.state.slots.items():
            if slot["phase"] != "loading" or key in excluded:
                continue
            request = slot["request_key"]
            original = self.state.data["loads"].get(request, {})
            match = next(
                (
                    item
                    for item in adopted
                    if isinstance(item, Mapping)
                    and isinstance(original.get("app_id"), str)
                    and item.get("application_id") == original["app_id"]
                    and isinstance(item.get("node_ids"), list)
                    and all(isinstance(node, str) for node in item["node_ids"])
                    and set(item["node_ids"]) == set(slot["node_ids"])
                    and isinstance(item.get("plan_digest"), str)
                    and len(item["plan_digest"]) == 64
                    and all(c in "0123456789abcdef" for c in item["plan_digest"])
                    and type(item.get("workload_intent_ordinal")) is int
                    and item["workload_intent_ordinal"] > 0
                    and isinstance(item.get("assignment_ids"), list)
                    and item["assignment_ids"]
                    and all(
                        isinstance(value, str) and value
                        for value in item["assignment_ids"]
                    )
                ),
                None,
            )
            if match is None:
                return None
            requests.append(request)
        return sorted(set(requests))

    def _adopted_cleanup_requests(self, review: Mapping[str, Any]) -> list[str] | None:
        return self._cleanup_adoption(review)[0]

    def _cleanup_adoption(
        self, review: Mapping[str, Any]
    ) -> tuple[list[str] | None, str | None]:
        requests = []
        for key, load in self.state.data["cleanup_loads"].items():
            if load.get("end"):
                continue
            if load.get("terminal_receipt") or load.get("state") in SETTLED:
                if self._cleanup_target_absent(load, terminal=True):
                    self._end_cleanup(load, self.clock.now())
                # An ended owner cannot adopt or execute another stop. The fresh
                # placement review admits our own stops for any remaining runs.
                continue
            owner = (
                f"cleanup request={key}, application={load.get('app_id') or 'unknown'}"
            )
            if not load.get("projection_known"):
                if self.state.data.get("takeover_fallback"):
                    continue  # fleet occupancy and the fresh review govern new work
                return None, f"{owner}: waiting for effects projection"
            online = {s.id for s in self.sparks()}
            pending = [
                row
                for row in self._cleanup_pending(load)
                if set(row["node_ids"]) & online
            ]
            if pending and self._cleanup_target_absent(load):
                self._end_cleanup(load, self.clock.now())
                continue
            for row in pending:
                if not cleanup.adopted(review, row):
                    return None, (
                        f"{owner}, effect={row['effect_id']}, "
                        f"operation={row.get('operation_id')}, run={row['target_id']}, "
                        f"nodes={','.join(row['node_ids'])}: "
                        "waiting for exact stop adoption or a fresh exact stop receipt"
                    )
            if pending:
                requests.append(key)
        return sorted(requests), None

    def _ridden_by_live_lane(self, app_id: Any, besides: Any) -> bool:
        """Does a live lane of another of the sweep's loads follow this application?

        Following a supersession or a retry can bind two load records to one application.
        The sweep never cancels an application that a lane of its own still rides: a load
        whose lanes ended only forgets it (the live lane's own ending decides).
        """
        if not app_id:
            return False
        return any(
            other.get("app_id") == app_id
            and other.get("request_key") != besides
            and (
                self._adopted_slots_live(other)
                or any(
                    slot.get("request_key") == other.get("request_key")
                    and slot["phase"] != "finished"
                    for slot in self.state.slots.values()
                )
            )
            for other in self.state.data["loads"].values()
        )

    def _adopted_slots_live(self, load: Mapping[str, Any]) -> bool:
        if any(
            bool(self._cleanup_pending(original))
            or not original.get("projection_known")
            or original.get("observation_unknown")
            for key in load.get("adopted_cleanup_requests", [])
            if (original := self.state.data["cleanup_loads"].get(key)) is not None
        ):
            return True
        requests = set(load.get("adopted_requests") or [])
        return any(
            slot.get("request_key") in requests and slot["phase"] != "finished"
            for slot in self.state.slots.values()
        )

    def _open_slot(self, placement: policy.Placement, request: str, now: float) -> None:
        recipe = placement.recipe
        model_bytes = (
            sum(self.sizes.get(d, 0) for d in recipe.model_digests)
            or recipe.artifact_bytes
        )
        timeout = policy.load_timeout(
            self.state.data["learned"], recipe.engine, model_bytes, self.cfg.timeouts
        )
        names = {s.id: s.name for s in self.sparks()}
        alias = self._alias(recipe)
        self._remember_own(alias)
        self.state.slots[recipe.key] = {
            "accepted_content": AcceptedRequest(
                request_key=request,
                recipe=recipe,
                library_commit=self.library_commit or None,
            ).model_dump(mode="json"),
            "alias": alias,
            "node_ids": list(placement.spark_ids),
            "spark_names": [names.get(i, i) for i in placement.spark_ids],
            "phase": "loading",
            "request_key": request,
            "started_at": now,
            "deadline": now + timeout,
            "model_bytes": model_bytes,
            "digests": list(recipe.model_digests),
            "prior_runs": sorted(  # runs of this alias that are not this load's
                p.run_id
                for p in self.fleet.presences
                if p.alias == alias and p.node_id in placement.spark_ids
            ),
        }

    def _remember_own(self, alias: str) -> None:
        own = self.state.data["own_aliases"]
        if alias not in own:
            own.append(alias)
            del own[:-500]

    def _overfull_sparks(self) -> set[str]:
        """Sparks the sweep profile would load beyond one Spark's memory.

        Two assignments share a Spark only when their declared peaks plus the reserve fit
        it; an unknown demand shares with nothing. The platform refuses such a load at
        review; the sweep does not propose it. An unreadable export decides nothing here.
        """
        reply = self.vk.run("profile", "export", profile=self.cfg.sweep_profile)
        document = reply.document
        assignments = (
            document.get("assignments") if isinstance(document, dict) else None
        )
        if not reply.ok or not isinstance(assignments, list):
            return set()
        peaks: dict[str, list[int]] = {}
        for item in assignments:
            if not isinstance(item, dict) or item.get("desired_state") != "running":
                continue
            recipe = self.recipes.get(str(item.get("recipe_selector")))
            for ref in item.get("spark_ids") or []:
                spark = self.fleet.by_ref(str(ref))
                if spark is not None:
                    peaks.setdefault(spark.id, []).append(
                        recipe.memory_bytes if recipe else 0
                    )
        capacity = {s.id: s.memory_total for s in self.fleet.sparks}
        return {
            spark_id
            for spark_id, declared in peaks.items()
            if len(declared) > 1
            and (
                0 in declared
                or sum(declared) + self.cfg.reserve_bytes > capacity[spark_id]
            )
        }

    def _rollback(self, placements: Sequence[policy.Placement], why: str) -> None:
        for placement in placements:
            try:
                self.vk.call(
                    "profile",
                    "remove",
                    self._alias(placement.recipe),
                    "--yes",
                    profile=self.cfg.sweep_profile,
                )
            except VonkctlError as error:
                self.state.event(
                    f"rollback of {self._alias(placement.recipe)} deferred: {str(error)[:120]}"
                )
                self.state.data.setdefault("cleanup", {}).setdefault(
                    self._alias(placement.recipe),
                    {
                        "attempts": 0,
                        "next_check": self.clock.now() + self.cfg.retry_delay,
                    },
                )
        for placement in placements:
            self.state.entry(placement.recipe.key)["not_before"] = (
                self.clock.now() + self.cfg.retry_delay
            )
        self.state.event(why[:200])

    def _blocked(
        self,
        document: Mapping[str, Any],
        aliases: Mapping[str, policy.Placement],
        added: Sequence[policy.Placement],
        now: float,
    ) -> None:
        """Attribute a blocked review to the new recipes; defer them while a lane is busy, else fail them."""
        reasons: dict[str, list[tuple[str, str]]] = {}
        for decision in document.get("admission_decisions", []):
            if isinstance(decision, dict) and decision.get("allowed") is False:
                for blocker in decision.get("blockers", []):
                    reasons.setdefault(str(decision.get("alias")), []).append(
                        (str(blocker.get("code", "")), str(blocker.get("detail", "")))
                    )
        busy = any(s["phase"] == "smoking" for s in self.state.slots.values())
        blocked = [p for a, p in aliases.items() if a in reasons]
        if not blocked:
            blocked = list(added)
            generic = [
                (str(r.get("code", "")), str(r.get("detail", "")))
                for r in document.get("reasons", [])
                if isinstance(r, dict) and r.get("severity") == "error"
            ]
            for placement in blocked:
                reasons.setdefault(
                    self._alias(placement.recipe),
                    generic or [("review.blocked", "the review was not allowed")],
                )
        self._rollback(
            blocked, "review blocked: " + ", ".join(p.recipe.key for p in blocked)
        )
        foreign, leftover = self._fleet_runs() if not busy else ([], [])
        if foreign or leftover:
            # Something is on the Sparks that no lane needs: that is the fit problem, not the
            # recipe's. Re-plan with that blocker as input: clear the fleet once for it. A
            # passed lane's workload keeps running after its slot is released: that one is
            # ours, not an owner's.
            holders = sorted({*foreign, *leftover})
            if self.state.data.get("cleared_for") != holders:
                self.state.data["cleared_for"] = holders
                self._request_clearing(now)
                self.state.event(
                    "review blocked by a workload that is not ours: clearing"
                    if foreign
                    else "review blocked by our own finished workload "
                    + ", ".join(leftover)
                    + ": clearing"
                )
                return
            if not self.state.data.get("cleared_by"):
                # Decided, never submitted: nothing has outlived anything. The refusal is
                # the evidence; decide the clearing again with a budget of its own.
                why = self.state.data.get("clearing_reason") or "not submitted"
                self._request_clearing(now)
                for placement in blocked:
                    self.record_result(
                        placement.recipe.key,
                        "review",
                        "waiting",
                        reason=f"review blocked by {', '.join(holders)}; their clearing "
                        f"was not submitted ({why}): clearing again",
                        disposition=Disposition.WAIT.value,
                        document=document,
                    )
                return
            # The same workloads outlived a submitted clearing: their owner keeps them. Wait
            # while they run (bounded by their owner, never by a count), no attempt spent.
            for placement in blocked:
                key = placement.recipe.key
                self.state.entry(key)["defer_until"] = now + self.cfg.defer_delay
                self.record_result(
                    key,
                    "review",
                    "waiting",
                    reason="review blocked by workloads that outlived a clearing: "
                    + ", ".join(holders),
                    disposition=Disposition.WAIT.value,
                    document=document,
                )
            return
        self.state.data.pop("cleared_for", None)
        for placement in blocked:
            key = placement.recipe.key
            entry = self.state.entry(key)
            code, detail = reasons.get(
                self._alias(placement.recipe), [("review.blocked", "")]
            )[0]
            # A preparation (build, image, download) the Controller has not finished is a
            # wait, never a verdict: the next review (alone, or with lanes that fit) takes the
            # wait-for-preparation path. A busy lane may hold the room: wait for it too.
            if code in verdict.PREPARATION_CODES or busy:
                entry["defer_until"] = now + self.cfg.defer_delay
                self.record_result(
                    key,
                    "review",
                    "waiting",
                    reason=f"{code}: {detail}".strip(": "),
                    disposition=Disposition.WAIT.value,
                    document=document,
                )
                continue
            failure = policy.classify(
                "review",
                code,
                detail,
                evidence={"reply": policy.excerpt(document, 1000)},
            )
            beyond_idle = failure.klass == "capacity" and self._beyond_idle_spark(
                document, self._alias(placement.recipe)
            )
            if beyond_idle:
                failure = replace(
                    failure,
                    evidence={**failure.evidence, "demand_exceeds_idle_spark": True},
                )
            freeing = int(entry.get("review_freeing", 0))
            if (
                verdict.disposition(failure) is Disposition.OBSERVE
                and failure.klass == "capacity"
                and freeing < MAX_REVIEW_FREEING
            ):
                # Not a verdict yet: stop what holds the room, evict, and review again.
                # Bounded, so a recipe that really does not fit is still recorded (with the
                # attempts in its evidence).
                entry["review_freeing"] = freeing + 1
                entry["defer_until"] = now + self.cfg.defer_delay
                self._request_clearing(now)
                self.record_result(
                    key,
                    "review",
                    "waiting",
                    reason=f"review refused ({code}): clearing the fleet and trying again "
                    f"({freeing + 1}/{MAX_REVIEW_FREEING})",
                    disposition=Disposition.OBSERVE.value,
                    document=document,
                )
                self.state.event(
                    f"review refused {key} ({code}): clearing the fleet and trying "
                    f"again ({freeing + 1}/{MAX_REVIEW_FREEING})"
                )
                continue
            if failure.klass == "capacity" and not beyond_idle:
                # Not beyond an idle Spark (or unknown): what holds the room (a Controller build, a
                # workload no clearing stops) is the platform's wait, never a verdict. Review
                # again later, no attempt spent.
                entry["defer_until"] = now + self.cfg.defer_delay
                self.record_result(
                    key,
                    "review",
                    "waiting",
                    reason=f"{code}: not beyond an idle Spark; waiting for the room",
                    disposition=Disposition.WAIT.value,
                    document=document,
                )
                continue
            if freeing:
                failure = replace(
                    failure,
                    evidence={**failure.evidence, "freeing_attempts": freeing},
                )
            # Only the observed demand, never a clearing count, can settle capacity.
            self._record_failure(key, failure, None, document=document)

    def _beyond_idle_spark(self, document: Mapping[str, Any], alias: str) -> bool:
        """The Controller's own demand for a node, with its reserve, exceeds that Spark's
        whole memory: no clearing or wait can make room. Unknown is never beyond."""
        totals = {s.id: s.memory_total for s in self.enrolled()}
        for decision in document.get("admission_decisions", []):
            if not isinstance(decision, dict) or decision.get("alias") != alias:
                continue
            for need in decision.get("requirements") or []:
                if not isinstance(need, dict):
                    continue
                total = totals.get(str(need.get("node_id")), 0)
                demand = as_int(need.get("memory_required_bytes"))
                if total and demand + as_int(need.get("memory_floor_bytes")) > total:
                    return True
        return False

    def note_infra(
        self, source: str, message: str, *, dependency: str | None = None
    ) -> None:
        """Pause that kind of work, preserving retry history until its predicate passes."""
        now = self.clock.now()
        item = InfrastructureIncident.model_validate(
            self.state.data["infra"].get(source) or {"since": now}
        )
        item.count += 1
        hint = (
            " (vonkctl may be older than the Controller: run `vonkctl update --apply`)"
            if "protocol_invalid" in message or "OpenAPI" in message
            else ""
        )
        item.message = message[:300] + hint
        if dependency is not None:
            item.dependency = dependency
            if item.escalate_at is None:
                item.escalate_at = item.since + self.cfg.idle_alert_seconds
        item.until = now + min(INFRA_BASE * 2 ** min(item.count - 1, 20), INFRA_CAP)
        self.state.data["infra"][source] = item.model_dump(exclude_none=True)
        if source == "download":
            self.prefetcher.pause_until = item.until
        if item.count in (1, 5) or item.count % 20 == 0:
            self.state.event(
                f"infrastructure problem ({source}, not a recipe failure), "
                f"retrying with backoff: {item.message[:160]}"
            )

    def clear_infra(self, source: str) -> None:
        if self.state.data["infra"].pop(source, None) is not None:
            self.state.event(f"infrastructure problem cleared ({source})")

    def _own_aliases(self) -> set[str]:
        """Every workload name the sweep has ever put in its profile (lanes come and go, the
        workload of a released lane runs until the next load replaces it)."""
        # Also every name ``_alias`` can derive for a catalog recipe, the plain one and its
        # recipe-keyed suffix: ownership follows from the name itself, not from a record
        # that a lost save or an older state file may lack.
        derived = {
            name
            for r in self.recipes.values()
            for name in (
                alias_for(r, self.defs),
                unique_profile_alias(alias_for(r, self.defs), r.key),
            )
        }
        return (
            {str(a) for a in self.state.data["own_aliases"]}
            | {str(s["alias"]) for s in self.state.slots.values()}
            | derived
        )

    def _fleet_runs(self) -> tuple[list[str], list[str]]:
        """What runs on the Sparks that no live lane accounts for: (not ours, ours but released)."""
        try:
            self.fleet = fetch_fleet(self.vk, self.cfg.default_spark_memory)
        except VonkctlError as error:
            self.state.event(f"fleet runs unreadable: {str(error)[:120]}")
            return [], []
        lanes = {str(s["alias"]) for s in self.state.slots.values()}
        own = self._own_aliases()
        online = {s.id for s in self.sparks()}
        names = sorted(
            {p.alias for p in self.fleet.presences if p.node_id in online} - lanes
        )
        return [a for a in names if a not in own], [a for a in names if a in own]

    def _foreign_runs(self) -> bool:
        return bool(self._fleet_runs()[0])

    # ---------------------------------------------------------- end of a run

    def interrupt(self) -> None:
        """Ctrl-C: cancel the lane's current load and give its recipes back to the queue."""
        for load in list(self.state.data["loads"].values()):
            self._cancel_load(load, force=True)
        for key in list(self.state.slots):
            self.futures.pop(key, None)
        self.state.event("interrupted: lane loads cancelled, recipes requeued")
        self.preempt("sweep interrupted")
        self.cleanup_finished()

    def _cancel_load(self, load: Mapping[str, Any], *, force: bool = False) -> None:
        if not force and self._adopted_slots_live(load):
            return  # root cancellation also owns adopted effects, so preserve their observer
        if not force and self._ridden_by_live_lane(
            load.get("app_id"), load.get("request_key")
        ):
            return
        if load.get("app_id"):
            self._hold_lanes(load)
            self.vk.run(
                "profile",
                "cancel",
                load["app_id"],
                "--yes",
                "--detach",
                profile=self.cfg.sweep_profile,
            )

    def finish(self, interrupted: bool = False) -> None:
        self.executor.shutdown(wait=False, cancel_futures=True)
        if not interrupted:
            self.cleanup_finished()
            if not any(self._open(k) for k in self.recipes):
                self.prefetcher.release_pins()
        self._leave_fleet()
        if not interrupted:
            self.remove_finished_models(final=True)
        write_status(self, final=True)
        self.state.save()

    def _own_key(self, kind: str, profile: int) -> str:
        """A request key never used before: nonce, load kind, profile and a persisted sequence number.

        The Controller answers a repeated request key with the application it already
        made. For a clearing, stop or restore load that would be an old, finished
        application and nothing would change, so every attempt gets a key of its own.
        """
        seq = int(self.state.data["own_seq"]) + 1
        self.state.data["own_seq"] = seq
        return request_key(kind, self.state.nonce, profile, seq)

    def _submit_reviewed_load(
        self, profile: int, kind: str, *, owner: bool = False, key: str | None = None
    ) -> Any:
        """Persist reviewed stop intent, including an empty scope, before effects."""
        if profile == self.cfg.sweep_profile:
            pruned = self._prune_profile()
            if pruned.waiting_on is not None:
                raise RuntimeError(pruned.waiting_on)
        review = self.vk.call(
            "profile",
            "load",
            "--review",
            profile=profile,
            allow_owner_write=owner,
        )
        if review.get("allowed") is not True:
            raise RuntimeError(f"the {kind} load has no allowed canonical review")
        key = key or self._own_key(kind, profile)
        if kind == FinalAction.STOP:
            runs = dig(review, "effects", "runs")
            adopted = self._adopted_cleanup_requests(review)
            if not isinstance(runs, list) or adopted is None:
                raise RuntimeError("final stop review cannot bind pending cleanup")
            self.state.data["cleanup_loads"][key] = {
                "request_key": key,
                "app_id": None,
                "reviewed_stops": sorted(
                    cleanup.stop_signature(row)
                    for row in runs
                    if isinstance(row, Mapping) and row.get("action") == "stop"
                ),
                "adopted_cleanup_requests": adopted,
            }
            self.state.save()
        return self._submit_load(profile, key, kind, owner=owner)

    def _submit_load(
        self, profile: int, key: str, kind: str, *, owner: bool = False
    ) -> Any:
        """Submit a load the sweep itself decided on.

        Its request key is written to the state *before* the call, and the application id
        after it, so the owner guard can tell it from an owner's load, in this process and
        in the next one (a restore of the owner's profile is otherwise an "owner load").
        """
        own: list[dict[str, Any]] = self.state.data["own_loads"]
        record = next((item for item in own if item["request_key"] == key), None)
        if record is None:
            record = {
                "request_key": key,
                "kind": kind,
                "profile": profile,
                "at": self.clock.now(),
            }
            own.append(record)
            protected = set(self.state.data["loads"]) | set(
                self.state.data["cleanup_loads"]
            )
            inactive = [item for item in own if item["request_key"] not in protected]
            keep = {item["request_key"] for item in inactive[-200:]} | protected
            own[:] = [item for item in own if item["request_key"] in keep]
        if kind in ("takeover", "stop"):
            self.state.data["cleanup_loads"].setdefault(
                key, {"request_key": key, "app_id": record.get("application_id")}
            )
        self.state.save()
        document = self.vk.call(
            "profile",
            "load",
            "--yes",
            "--detach",
            "--request-key",
            key,
            profile=profile,
            allow_owner_write=owner,
        )
        if document.get("request_key") != key:
            raise RuntimeError(
                "profile acceptance does not match the owned request; observing original request"
            )
        if isinstance(document.get("id"), str):
            record["application_id"] = document["id"]
            if kind in ("takeover", "stop"):
                self.state.data["cleanup_loads"][key]["app_id"] = document["id"]
            self.state.save()
        return document

    def _leave_fleet(self) -> None:
        """Keep owning stop/restore until its exact terminal receipt confirms it.

        Unknown acceptance always observes the persisted request first. Only a
        canonical absence permits replay; a proven terminal failure re-reviews
        with a new request. A newer owner load supersedes the sweep's end intent.
        """
        number = self.cfg.restore_owner
        if not self.state.data.get("dirty"):
            return
        if number is None and not self.cfg.stop_at_end:
            return
        action = FinalAction.RESTORE if number is not None else FinalAction.STOP
        profile = number if number is not None else self.cfg.sweep_profile
        saved = self.state.data.get("finalization")
        final = (
            Finalization.model_validate(saved)
            if saved
            else Finalization(
                action=action,
                profile=profile,
                request_key=self._own_key(action, profile),
                since=self.clock.now(),
            )
        )
        for _ in itertools.count():
            now = self.clock.now()
            if now < final.next_check:
                write_status(self)
                self.clock.sleep(min(self.cfg.poll_seconds, final.next_check - now))
                continue
            final.attempts += 1
            final.next_check = now + verdict.retry_delay(
                self.cfg.retry_delay, final.attempts
            )
            self.state.data["finalization"] = final.model_dump(mode="json")
            self.state.save()
            write_status(self)
            try:
                owner_status = self.guard.check(now)
                if owner_status.new_activity:
                    self.state.event(
                        "final fleet intent superseded by a newer owner load"
                    )
                    self.state.data["dirty"] = False
                    break
                own = next(
                    (
                        item
                        for item in self.state.data["own_loads"]
                        if item["request_key"] == final.request_key
                    ),
                    None,
                )
                if own is not None:
                    reply = self.vk.run(
                        "profile",
                        "progress",
                        "--request-key",
                        final.request_key,
                        profile=final.profile,
                    )
                    document = reply.document
                    if reply.ok and isinstance(document, Mapping):
                        if (
                            document.get("request_key") != final.request_key
                            or not isinstance(document.get("id"), str)
                            or (
                                own.get("application_id")
                                and document.get("id") != own["application_id"]
                            )
                        ):
                            raise RuntimeError(
                                "final receipt does not match the owned request/application"
                            )
                        if document.get("state") == LifecycleState.SUCCEEDED:
                            own["terminal_receipt"] = dict(document)
                            # A new invocation may request a different final state. Finish
                            # observing the old accepted effect before applying the new one.
                            if final.action != action or final.profile != profile:
                                final = Finalization(
                                    action=action,
                                    profile=profile,
                                    request_key=self._own_key(action, profile),
                                    since=now,
                                )
                                continue
                            self.state.data["dirty"] = False
                            self.state.event(
                                f"confirmed final {final.action} profile {final.profile}"
                            )
                            break
                        if document.get("state") in SETTLED:
                            own["terminal_receipt"] = dict(document)
                            final.request_key = self._own_key(
                                final.action, final.profile
                            )
                            raise RuntimeError(
                                "final load ended without confirming the requested state; re-reviewing"
                            )
                        continue
                    if not reply.is_not_found or own.get("application_id"):
                        raise RuntimeError(
                            "final receipt unavailable; observing the same request"
                        )
                    # Canonical absence proves this unaccepted cleanup intent has
                    # no remote effects to adopt into its own fresh review.
                    self.state.data["cleanup_loads"].pop(final.request_key, None)
                if owner_status.paused:
                    raise RuntimeError(owner_status.reason)
                self.cleanup_finished()
                self._submit_reviewed_load(
                    final.profile,
                    final.action,
                    owner=final.action is FinalAction.RESTORE,
                    key=final.request_key,
                )
            except (VonkctlError, RuntimeError, OSError) as error:
                self.state.event(
                    f"final fleet reconciliation pending: {str(error)[:150]}"
                )
                self.note_infra("finalization", str(error))
            self.state.data["finalization"] = final.model_dump(mode="json")
            write_status(self)
            self.state.save()
        self.state.data.pop("finalization", None)
        self.clear_infra("finalization")
        self.state.save()


OWNER_PROFILES_READ = (1, 2, 3)


def _blocker_text(document: Any) -> str:
    """The codes and details a refused review gives, as one line of evidence."""
    if not isinstance(document, Mapping):
        return "no review document"
    found = [
        f"{r.get('code', '')}: {r.get('detail', '')}"
        for r in document.get("reasons", [])
        if isinstance(r, Mapping) and r.get("severity") == "error"
    ]
    found += [
        f"{b.get('code', '')}: {b.get('detail', '')}"
        for d in document.get("admission_decisions", [])
        if isinstance(d, Mapping) and d.get("allowed") is False
        for b in d.get("blockers", [])
        if isinstance(b, Mapping)
    ]
    if not found and (error := document.get("error") or document.get("code")):
        found.append(str(error))
    return "; ".join(found)[:400] or "no blocker named"


def _bounded_document(document: Mapping[str, Any]) -> Any:
    """A Controller document as evidence: whole when small, else a bounded excerpt."""
    text = json.dumps(document, sort_keys=True, default=str)
    if len(text) <= DOCUMENT_EVIDENCE_LIMIT:
        return document
    return {"excerpt": text[:DOCUMENT_EVIDENCE_LIMIT], "truncated_bytes": len(text)}


def _done_future(result: SmokeResult) -> Future[SmokeResult]:
    future: Future[SmokeResult] = Future()
    future.set_result(result)
    return future
