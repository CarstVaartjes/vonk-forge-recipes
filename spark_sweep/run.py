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
from collections.abc import Callable, Iterable, Mapping, Sequence
from concurrent.futures import Future
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Protocol

from pydantic import ValidationError
from vonk_forge_contracts.sweep import (
    CleanupEnd,
    CleanupEndReason,
    CleanupFleetObservation,
    CleanupObservationBudget,
    FleetOnlineState,
)

from . import cleanup, memory, policy, recovery, verdict
from .catalog import (
    Fleet,
    Model,
    Recipe,
    dig,
    fetch_fleet,
    model_listing,
    recipe_listing,
    serving_run,
)
from .definitions import Definitions
from .lifecycle import ACTIVE as ACTIVE_APP
from .lifecycle import TERMINAL as SETTLED
from .owner import OwnerGuard, OwnerStatus
from .prefetch import SWEEP_LABEL, PrefetchConfig, Prefetcher
from .profile_alias import profile_alias, unique_profile_alias
from .report import write_status
from .smoke import HttpConfig, SmokeResult, smoke_readiness, smoke_service
from .state import ResultsLog, State, StateLock
from .verdict import Disposition, Progress
from .vonkctl import Reply, Vonkctl, VonkctlError, request_key

WAIT_EVENT_SECONDS = 60.0  # a wait says what it waits for at least this often
# Clearing the fleet and reviewing again, for a capacity refusal (an OBSERVE outcome),
# before it counts against the recipe.
MAX_REVIEW_FREEING = 2
# Clearing attempts before placement re-plans onto the online lanes that are free now.
# A re-plan, never a verdict: nothing is ended or recorded against a recipe.
MAX_TAKEOVER_ATTEMPTS = 3
TAKEOVER_RETRY_SECONDS = 30.0
SKEW_CHECK_INTERVAL = 300.0
INFRA_BASE = 30.0
INFRA_CAP = 600.0
RETEST_SECONDS = 24 * 3600.0
ENDPOINT_RECHECK_SECONDS = 20.0
# A results row carries the Controller document it ended on, bounded.
DOCUMENT_EVIDENCE_LIMIT = 65536


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


FLEET_SKIP = "fleet"


def _fleet_skip(entry: Mapping[str, Any]) -> bool:
    """A skip that depends on the fleet (also the older state files, which only carry the reason)."""
    return entry.get("status") == "skipped" and (
        entry.get("reason_kind") == FLEET_SKIP
        or str(entry.get("reason", "")).startswith("not testable on this fleet")
    )


@dataclass
class SweepConfig:
    state_dir: Path
    authority_id: str
    sweep_profile: int = 10
    pin_profile: int | None = 13
    sparks: tuple[str, ...] = ()
    reserve_bytes: int = 8_000_000_000
    default_spark_memory: int = 130_663_231_488
    poll_seconds: float = 10.0
    catalog_seconds: float = 60.0
    status_seconds: float = 15.0
    idle_alert_seconds: float = 600.0
    owner_hold_seconds: float = 1800.0
    owner_poll_seconds: float = 15.0
    retry_delay: float = 60.0
    defer_delay: float = 120.0
    max_attempts: int = 2
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
    # The stall window while bytes are copied to a Spark (the learned window covers the rest).
    copy_stall_seconds: float = 600.0
    # How long a load may wait on a *preparation* that names only some of its recipes before
    # it is re-planned without them: they leave the profile and wait for their preparation
    # (no attempt, no verdict) while the rest load. A re-plan delay, not a deadline.
    blocked_release_seconds: float = 300.0


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
        # The cached listings: scheduling reads these, a refresh updates them page by page.
        self.recipe_list = recipe_listing(vk)
        self.model_list = model_listing(vk)
        self.recipes: dict[str, Recipe] = self.recipe_list.rows
        self.models: dict[str, Model] = self.model_list.rows
        self.fleet = Fleet((), ())
        self.release_at = self.clock.now()  # check_client has just read it
        self.status_at = -1e18
        self.owner_at = -1e18
        self.owner_status = OwnerStatus()
        self.backoff_until = 0.0
        self.idle_since: float | None = None
        self.blind = False  # this tick could not observe the Controller
        self._skew_checked_at = float("-inf")
        self.library_commit = ""
        self.rate = policy.RateTracker()
        self.rate.rate = float(state.data["rate"].get("ema", 0.0))
        self.rate.samples = int(state.data["rate"].get("samples", 0))
        self.guard = OwnerGuard(vk, state, config.owner_hold_seconds)
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

    def sparks(self) -> list[Any]:
        chosen = [s for s in self.fleet.sparks if s.online]
        if self.cfg.sparks:
            picked = [self.fleet.by_ref(ref) for ref in self.cfg.sparks]
            chosen = [s for s in picked if s is not None and s.online]
        return chosen

    @property
    def catalog_loaded(self) -> bool:
        """Nothing is done, or stuck, before the recipe library has been read once."""
        return self.recipe_list.passes > 0

    def advance_catalog(self, now: float) -> None:
        """Read library pages and merge them into the cached listings.

        One page per tick once both listings have been read (several at start),
        so a slow or failed page never stalls scheduling, which reads the cache.
        """
        budget = 1 if self.recipe_list.passes and self.model_list.passes else 6
        for _ in range(budget):
            active = next(
                (item for item in (self.recipe_list, self.model_list) if item.in_pass),
                None,
            )
            if active is None:
                due = next(
                    (
                        item
                        for item in (self.recipe_list, self.model_list)
                        if now - item.pass_done_at >= self.cfg.catalog_seconds
                    ),
                    None,
                )
                if due is None:
                    break
                due.begin(now)
                active = due
            before = active.last_error
            try:
                finished = active.step(now)
            except VonkctlError as error:  # not expected: run() does not raise
                self.state.event(f"library page failed: {str(error)[:120]}")
                break
            if active.last_error and active.last_error != before:
                self.state.event(
                    f"library page failed, retrying: {active.last_error[:120]}"
                )
            if finished and active is self.recipe_list:
                self.library_commit = active.commit
        self._reconcile_all()

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
                        final=True,
                    )
                if key not in self.state.slots:
                    self.state.recipes.pop(key, None)
                record = self.state.downloads.pop(key, None)
                if record and record.get("state") not in SETTLED:
                    self.prefetcher._retire(key, record, "catalog.recipe_removed")
        self.sizes = policy.build_sizes(self.recipes.values(), self.models)
        usable = len(self.sparks())
        for recipe in list(self.recipes.values()):
            self._reconcile_entry(recipe, usable)

    def fleet_capacity(self) -> int:
        """Sparks that exist and are online, whatever they run: never lane occupancy."""
        return len(self.sparks())

    def heal_fleet_skips(self) -> None:
        """Fleet-dependent skips are derived, not verdicts: re-evaluate them against the fleet now."""
        if not any(_fleet_skip(e) for e in self.state.recipes.values()):
            return
        try:
            self.fleet = fetch_fleet(self.vk, self.cfg.default_spark_memory)
        except VonkctlError as error:
            self.state.event(f"fleet unreadable: {str(error)[:120]}")
            return
        usable = self.fleet_capacity()
        for key, entry in self.state.recipes.items():
            recipe = self.recipes.get(key)
            if recipe is None or not _fleet_skip(entry):
                continue
            if recipe.node_count <= usable:
                self._reconcile_entry(recipe, usable)

    def _selected(self, recipe: Recipe) -> bool:
        return not self.cfg.only or any(word in recipe.key for word in self.cfg.only)

    def _reconcile_entry(self, recipe: Recipe, usable_sparks: int) -> None:
        if not self._selected(recipe):
            return
        entry = self.state.entry(recipe.key)
        if recipe.node_count > usable_sparks and (
            entry.get("status") == "pending" or _fleet_skip(entry)
        ):
            reason = f"not testable on this fleet: needs {recipe.node_count} Sparks, {usable_sparks} usable"
            if entry.get("status") != "skipped" or entry.get("reason") != reason:
                self.record_result(recipe.key, "reconcile", "skipped", reason=reason)
            entry.update(
                status="skipped",
                reason=reason,
                reason_kind=FLEET_SKIP,
                content_sha256=recipe.content_sha256,
            )
            return
        if _fleet_skip(entry) and recipe.node_count <= usable_sparks:
            entry["status"] = "pending"
            entry.pop("reason", None)
            entry.pop("reason_kind", None)
            self.record_result(
                recipe.key,
                "reconcile",
                "pending",
                reason="testable on this fleet again",
            )
            self.state.event(f"{recipe.key} is testable on this fleet again: pending")
        if recipe.node_count > usable_sparks:
            return  # a Spark offline for a moment must not rewrite a recorded result
        why = policy.requeue_reason(entry, recipe)
        if why is None:
            return
        if why == "passed-on-older-revision" and not self.cfg.revalidate_passes:
            return
        entry["previous"] = {
            k: entry.get(k)
            for k in ("status", "content_sha256", "revision_id", "cluster")
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
                    "revision_id",
                )
            },
        )
        entry.update(
            status="pending",
            attempts=0,
            revalidate=why.startswith("passed"),
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
            if not self._selected(recipe):
                continue
            out.append(recipe)
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

    def done_waiting(self) -> None:
        self.state.data.pop("waiting", None)
        self._wait_event_at = -1e18

    def tick(self) -> None:
        """One pass. A Controller that stops answering costs a pass, never the sweep."""
        try:
            self._tick()
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

    def _startup(self, step: Callable[[], None]) -> bool:
        """One try of a startup step that only reads the fleet: a failing vonkctl waits, never exits.

        The caller tries again on its next pass; the wait has no deadline and backs off.
        """
        try:
            step()
        except (VonkctlError, OSError) as error:
            self.note_infra("startup", f"{type(error).__name__}: {error}")
            item = self.state.data["infra"]["startup"]
            self.waiting("the Controller to answer at startup", float(item["since"]))
            self.clock.sleep(
                max(
                    1.0,
                    min(
                        float(item["until"]) - self.clock.now(),
                        self.cfg.poll_seconds * 6,
                    ),
                )
            )
            return False
        self.clear_infra("startup")
        self.done_waiting()
        return True

    def _tick(self) -> None:
        now = self.clock.now()
        self.advance_catalog(now)
        self.check_release(now)
        if self.cfg.watch_seconds > 0:
            for key, entry in list(self.state.recipes.items()):
                if (
                    entry.get("status") in ("failed", "deferred")
                    and now >= float(entry.get("finished_at", now)) + RETEST_SECONDS
                    and self._selected_key(key)
                ):
                    self._schedule_retest(key, "scheduled-retest")
        if now - self.owner_at >= self.cfg.owner_poll_seconds:
            self.owner_at = now
            self.owner_status = self.guard.check(now)
            if self.owner_status.new_activity:
                self.state.data["took_over"] = (
                    False  # their load replaced what ours ran
                )
                self.state.data["owner"]["seen_at"] = now
                self.preempt("an owner profile was loaded")
        if self.owner_status.paused:
            # An owner's load (or ownership we cannot read: unknown) is waited for while it
            # lasts; it never ends the sweep.
            self.state.data["owner"].setdefault("paused_at", now)
        else:
            self.state.data["owner"].pop("paused_at", None)
        self.observe_platform()
        self.advance_slots(now)
        self.heal_fleet_skips()
        if not self.owner_status.paused:
            self.enqueue_finished_cleanup()
        pending = self.pending()
        for (
            key,
            record,
        ) in self.state.downloads.items():  # a state file from an older run
            if record.get("state") == "succeeded":
                self._download_done(key)
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
        )
        if not self.owner_status.paused:
            self.schedule(self.clock.now())
            self.cleanup_finished()
        # Previously confirmed immutable stop receipts may feed a fresh canonical
        # admission before optional progress reads of an offline borrowed child.
        self.observe_cleanup()
        # Ready cached placements and their durable request receipts come first.
        # Download observation, preparation and pin maintenance may wait on the API.
        self.last_prefetch = self.prefetcher.tick(
            self.recipes,
            self.models,
            self.pending(),
            self.sizes,
            present,
            self.boost,
            self.cached_models,
        )
        self.check_idle(self.clock.now())
        self.state.data["rate"] = {"ema": self.rate.rate, "samples": self.rate.samples}
        if now - self.status_at >= self.cfg.status_seconds:
            self.status_at = now
            write_status(self)
        self.state.save()

    def _cached_models(self) -> set[str]:
        """Models on the NAS: the library says so, a download of ours finished, or the library's own
        assessment of a recipe says its exact assets are ready. Never fleet fit or readiness."""
        cached = {d for d, m in self.models.items() if m.local == "cached"}
        cached |= set(self.state.data["models_done"])
        for recipe in self.recipes.values():
            if recipe.cache_ready:
                cached |= set(recipe.model_digests)
        return cached

    def _present(self) -> set[str]:
        """Models on the NAS or arriving."""
        return policy.present_models(self.models) | self.cached_models

    def _download_done(self, key: str) -> None:
        recipe = self.recipes.get(key)
        if recipe is not None:
            done = set(self.state.data["models_done"]) | set(recipe.model_digests)
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
            "authority": "deployed-controller",
        }
        if any(release.get(k) != value for k, value in observed.items()):
            self.state.data["release_history"].append(
                {**observed, "seen_at": self.clock.now()}
            )
            self.state.event(
                "deployed Controller observed: " + (observed["sha"] or "unknown")
            )
        if release and release.get("authority") != "deployed-controller":
            self.state.data["release_history"].append(
                {**release, "authority": "legacy-unverified-publication-proxy"}
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
                self._schedule_retest(key, "observed-fault-owner-changed", basis=basis)

    def _schedule_retest(
        self, key: str, reason: str, *, basis: Mapping[str, str] | None = None
    ) -> None:
        """Schedule an automatic retest without erasing its cause or retry history."""
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

    def requeue_for_rules(self) -> None:
        """Load timeouts recorded under the old rules (copy time counted) are tried once more."""
        for key, entry in self.state.recipes.items():
            if (
                entry.get("status") == "failed"
                and entry.get("failure_class") == "timeout"
                and int(entry.get("rules", 1)) < policy.LOAD_RULES
                and self._selected_key(key)
            ):
                self._requeue_failed(
                    key, "load-timeout rules changed: copying no longer counts"
                )
        for key, entry in self.state.recipes.items():
            if (
                entry.get("status") == "failed"
                and entry.get("phase") == "smoke"
                and entry.get("failure_class") == "smoke-assertion"
                and int(entry.get("smoke_rules", 1)) < policy.SMOKE_RULES
                and self._selected_key(key)
            ):
                self._requeue_failed(
                    key, "smoke rules changed: answer quality no longer fails a recipe"
                )

    def requeue_superseded_failures(self) -> None:
        """A load the Controller replaced is not the recipe's failure: records of it are pending again."""
        for key, entry in self.state.recipes.items():
            evidence = entry.get("evidence")
            reason = (
                evidence.get("reason") if isinstance(evidence, Mapping) else None
            ) or entry.get("error")
            if (
                entry.get("status") == "failed"
                and (
                    policy.superseded_text(reason)
                    or policy.superseded_text(entry.get("error"))
                )
                and self._selected_key(key)
            ):
                self._requeue_failed(key, "load-superseded: not a recipe failure")

    def requeue_unfreed_capacity_failures(self) -> None:
        """A capacity refusal recorded before the fleet was cleared and reviewed again is tried again."""
        for key, entry in self.state.recipes.items():
            evidence = entry.get("evidence")
            if (
                entry.get("status") == "failed"
                and entry.get("phase") == "review"
                and entry.get("failure_class") == "capacity"
                and not (
                    isinstance(evidence, Mapping) and evidence.get("freeing_attempts")
                )
                and self._selected_key(key)
            ):
                self._requeue_failed(
                    key, "review capacity refusal: the fleet was not cleared first"
                )

    def requeue_failed(self) -> int:
        """``--retry-failed``: every failed recipe (within ``--only``), whatever the cause."""
        keys = [
            key
            for key, entry in self.state.recipes.items()
            if entry.get("status") in ("failed", "deferred") and self._selected_key(key)
        ]
        for key in keys:
            self._requeue_failed(key, "operator: --retry-failed")
        return len(keys)

    def _selected_key(self, key: str) -> bool:
        return not self.cfg.only or any(word in key for word in self.cfg.only)

    def _requeue_failed(self, key: str, why: str) -> None:
        """Back to pending; the failure stays in results.jsonl, which is only ever appended to."""
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
                if why == "operator: --retry-failed":
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
        self.state.data.pop("end", None)
        stuck = 0
        try:
            self.check_client()
            self.requeue_for_rules()
            self.requeue_superseded_failures()
            self.requeue_unfreed_capacity_failures()
            if self.cfg.retry_failed:
                self.requeue_failed()
            started = False
            for _ in itertools.count():
                try:
                    if not started:
                        if not self._startup(self.preflight):
                            continue
                        self.adopt_in_flight()
                        if not self._startup(self.start_takeover):
                            continue
                        started = True
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
            or self.owner_status.paused
            or bool(self.state.data["infra"])
            or self.clock.now()
            < max(
                self.backoff_until,
                self.prefetcher.pressure_until,
                self.prefetcher.pause_until,
            )
        )
        return not busy and not waiting and not self.ready_candidates()

    # --------------------------------------------------------------- failures

    def _download_failed(
        self, key: str, failure: policy.Failure, op_id: str | None, model_level: bool
    ) -> None:
        recipe = self.recipes.get(key)
        # The prefetcher already waited out what is a wait: this one is final.
        self._record_failure(key, failure, op_id, final=True)
        if (
            model_level
            and failure.code in policy.RECIPE_FAILURE_CODES
            and recipe is not None
        ):
            for digest in recipe.model_digests:
                self.state.data["model_failures"][digest] = failure.cluster

    def _record_failure(
        self,
        key: str,
        failure: policy.Failure,
        op_id: str | None,
        *,
        inherited_from: str | None = None,
        final: bool = False,
    ) -> None:
        """Record what an attempt came to; only an OBSERVE outcome spends an attempt.

        WAIT (the platform or a dependency is not ready) puts the recipe back without an
        attempt; OBSERVE retries until ``max_attempts``; FAIL records at once. Every outcome
        writes a results row with its evidence.
        """
        now = self.clock.now()
        entry = self.state.entry(key)
        recipe = self.recipes.get(key)
        slot = self.state.slots.get(key)
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
        outcome = Disposition.FAIL if final else verdict.disposition(failure)
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
            )
            self.state.event(f"wait {key}: {failure.klass} ({failure.phase})")
            return
        entry["attempts"] = int(entry.get("attempts", 0)) + 1
        if outcome is Disposition.OBSERVE and entry["attempts"] < self.cfg.max_attempts:
            entry.update(
                status="pending",
                not_before=now + self.cfg.retry_delay,
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
            )
            self.state.event(f"retry {key}: {failure.klass} ({failure.phase})")
            return
        entry["evidence_seq"] = int(entry.get("evidence_seq", 0)) + 1
        bundle = (
            self._evidence(key, op_id, entry["evidence_seq"])
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
            release_authority="deployed-controller" if self.release_sha else None,
            recovery_basis=recovery.failure_basis(
                failure.code, self._component_observations()
            ),
            rules=policy.LOAD_RULES,
            smoke_rules=policy.SMOKE_RULES,
            content_sha256=recipe.content_sha256
            if recipe
            else entry.get("content_sha256"),
            revision_id=recipe.revision_id if recipe else entry.get("revision_id"),
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
        )
        self.state.event(f"{status.upper()} {key}: {failure.phase}/{failure.klass}")

    def record_result(
        self,
        key: str,
        step: str,
        status: str,
        *,
        slot: Mapping[str, Any] | None = None,
        document: Mapping[str, Any] | None = None,
        **fields: Any,
    ) -> None:
        """The one writer of ``results.jsonl``: every lane exit leaves a row.

        The row carries the recipe's facts and the Controller document its load was last
        seen in, so the evidence of a wait, a retry or a requeue is never only in the
        bounded event ring.
        """
        entry = self.state.entry(key)
        slot = slot if slot is not None else self.state.slots.get(key)
        if document is None and slot is not None:
            document = self.documents.get(str(slot.get("request_key", "")))
        row: dict[str, Any] = {
            "batch": "sweep",
            "recipe": key,
            "step": step,
            "status": status,
            **self._facts(key, entry, slot),
            **{k: v for k, v in fields.items() if v is not None},
        }
        if document:
            row["controller_document"] = _bounded_document(document)
        try:
            self.log.append(**row)
        except OSError as error:
            # The log is the evidence; a full disk is an infrastructure fact to show.
            self.note_infra("results-log", f"{type(error).__name__}: {error}")

    def _facts(
        self, key: str, entry: Mapping[str, Any], slot: Mapping[str, Any] | None
    ) -> dict[str, Any]:
        recipe = self.recipes.get(key)
        facts: dict[str, Any] = {
            "phase": entry.get("phase"),
            "failure_class": entry.get("failure_class"),
            "signature": entry.get("signature"),
            "cluster": entry.get("cluster"),
            "attempt": entry.get("attempts"),
            "evidence": entry.get("evidence"),
            "evidence_bundle": entry.get("evidence_bundle"),
            "content_sha256": recipe.content_sha256
            if recipe
            else entry.get("content_sha256"),
            "recipe_revision_id": recipe.revision_id
            if recipe
            else entry.get("revision_id"),
            "library_commit": self.library_commit or None,
            "controller_release": entry.get("release"),
            "controller_version": self.state.data["release"].get("version"),
            "engine": recipe.engine if recipe else None,
            "timings": entry.get("timings"),
        }
        if slot:
            facts["node_ids"] = list(slot.get("node_ids", []))
            facts["sparks"] = list(slot.get("spark_names", []))
            facts["run_id"] = slot.get("run_id")
        return {k: v for k, v in facts.items() if v is not None}

    def _evidence(self, key: str, op_id: str | None, attempt: int) -> str | None:
        if not op_id:
            return None
        directory = self.cfg.state_dir / "evidence"
        directory.mkdir(parents=True, exist_ok=True)
        target = directory / f"{key.partition('/')[2]}-{attempt}.json"
        reply = self.vk.run("fleet", "evidence", op_id, "--output", str(target))
        return str(target) if reply.ok else None

    def _record_pass(self, key: str, slot: dict[str, Any], result: SmokeResult) -> None:
        now = self.clock.now()
        entry = self.state.entry(key)
        recipe = self.recipes[key]
        download = self.state.downloads.get(key, {})
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
            release_authority="deployed-controller" if self.release_sha else None,
            content_sha256=recipe.content_sha256,
            revision_id=recipe.revision_id,
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

    # ------------------------------------------------------------------ slots

    def poll_application(self) -> bool:
        """Refresh the load's application; False when the Controller could not be observed."""
        observed = True
        for load in list(self.state.data["loads"].values()):
            observed = self._poll_application(load) and observed
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
        self.clear_infra("review")
        self.documents[load["request_key"]] = document
        load["observed_at"] = self.clock.now()
        load["app_id"] = document["id"]
        status = str(document.get("state", ""))
        load["state"] = status
        moving = policy.distribution(document)
        load["copying"] = moving.copying
        load["phase"] = moving.phase
        load["blockers"] = policy.admission_blockers(document)
        signal = verdict.application_signal(document)
        load["signal"] = {
            "signature": list(signal.signature),
            "waiting_on": signal.waiting_on,
        }
        failure = document.get("failure") or {}
        failure_code = str(failure.get("code") or document.get("code") or "")
        replaced = policy.supersession(document) if status in SETTLED else None
        if replaced is not None and replaced.successor not in (None, document["id"]):
            # The Controller continued this application as another one: follow it.
            self.state.event(
                f"application {document['id']} was superseded: following {replaced.successor}"
            )
            load["app_id"] = replaced.successor
            load["state"] = "queued"
            load["copying"] = False
            return True
        if status == "failed" and replaced is None:
            try:
                retry = self._retry_successor(document)
            except VonkctlError as error:
                # Unknown is not "ended": keep observing the failed application.
                self._blind(str(error), error.reply)
                return False
            if retry is not None:
                # The Controller is still retrying this load by itself: a newer retry
                # application continues it, and only its ending is the recipe's.
                self.state.event(
                    f"application {document['id']} failed but the Controller retries it: "
                    f"following {retry}"
                )
                load["app_id"] = retry
                load["state"] = "queued"
                load["copying"] = False
                return True
        if status in SETTLED:
            child = dig(document, "progress", "child_progress", "phase")
            self.state.data.setdefault("apps", {})[load["request_key"]] = {
                "state": status,
                "superseded": replaced is not None,
                "reason": document.get("status_reason"),
                "failure_code": failure_code,
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

    def _retry_successor(self, document: Mapping[str, Any]) -> str | None:
        """The application that continues a failed one the Controller is still retrying, if any.

        A failed application ends for good with the Controller's typed ``profile.failure_repeated``
        blocker or a terminal assignment failure; otherwise a later application of the profile
        that names this one as the application it retries is its continuation. An unreadable
        answer raises: unknown is never "it ended".
        """
        if verdict.retry_ended(document):
            return None
        reply = self.vk.run("profile", "progress", profile=self.cfg.sweep_profile)
        latest = reply.document
        if not reply.ok or not isinstance(latest, dict):
            raise VonkctlError(
                reply.argv, reply, "the profile's latest application is unreadable"
            )
        if (
            isinstance(latest.get("id"), str)
            and latest["id"] != document.get("id")
            and latest.get("retry_of_application_id") == document.get("id")
        ):
            return latest["id"]
        return None

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
        if not self.state.slots:
            self.poll_application()
            if not self.blind:
                self.clear_infra("observe")
            return
        try:
            self.fleet = fetch_fleet(self.vk, self.cfg.default_spark_memory)
        except VonkctlError as error:
            # An unreadable fleet is unknown: nothing is judged on it this pass.
            self._blind(str(error), error.reply)
        self.poll_application()
        if not self.blind:
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

        Copying a few hundred GiB to a Spark can take longer than any sensible load timeout while
        being perfectly healthy, so copy seconds are not load time and only a copy that stops
        moving for the copy window stalls. While the Controller holds the application (queue,
        admission, a blocker, its own retry, an operator) or cannot be read, the lane waits
        and the window restarts: a wait never ages into a verdict.
        """
        load = self.state.data["loads"].get(slot["request_key"])
        elapsed = now - float(slot.get("last_tick", slot["started_at"]))
        slot["last_tick"] = now
        watch = slot.setdefault("watch", {})
        if self.blind or load is None or not isinstance(load.get("signal"), Mapping):
            if self.blind:
                slot["blind_seconds"] = float(slot.get("blind_seconds", 0)) + elapsed
            return verdict.progress_verdict(
                None, watch, window=self._load_window(slot), now=now
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
            load["signal"].get("waiting_on"),
        )
        if signal.waiting_on is not None:
            slot["waited_seconds"] = float(slot.get("waited_seconds", 0)) + elapsed
        return verdict.progress_verdict(
            signal, watch, window=self._load_window(slot), now=now
        )

    def _observe_readiness(self, slot: dict[str, Any], now: float) -> Progress:
        """After the load succeeded: the run turning healthy and its route being published.

        A healthy run whose route is not published yet waits on the Controller's route
        publication; a run that is missing or unhealthy is the recipe's own work and stalls
        after the lane's window without any change.
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
            verdict.Signal(signature, "route publication" if publishing else None),
            slot.setdefault("ready_watch", {}),
            window=float(slot["deadline"]) - float(slot["started_at"]),
            now=now,
        )

    def _advance_loading(self, key: str, slot: dict[str, Any], now: float) -> None:
        recipe = self.recipes[key]
        if self.blind:
            self._observe_load(slot, now)
            return  # no recipe verdict on an unreadable Controller: unknown keeps waiting
        run_id = serving_run(
            self.fleet,
            slot["alias"],
            slot["node_ids"],
            recipe.node_count,
            slot.get("prior_runs", ()),
        )
        if run_id:
            slot.update(phase="smoking", run_id=run_id, ready_at=now)
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
            if (
                app.get("superseded")
                or policy.superseded_text(app.get("reason"))
                or (app["state"] == "cancelled" and app.get("cause") == "superseded")
            ):
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
                if self._observe_readiness(slot, now) is Progress.STALLED:
                    silent = verdict.silent_seconds(slot["ready_watch"], now)
                    self._fail_slot(
                        key,
                        slot,
                        policy.classify(
                            "readiness",
                            "run.not_published",
                            "the load succeeded but the run never became healthy and "
                            f"published (no change for {round(silent)}s)",
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
            detail = f"{app.get('reason') or app['state']} {app.get('failures') or ''}".strip()
            for member in group:
                failure = policy.classify(
                    phase,
                    next(
                        (
                            str(item["code"])
                            for item in app.get("failures") or []
                            if isinstance(item, Mapping)
                            and isinstance(item.get("code"), str)
                            and member in policy.blamed([item], self._lanes(group))
                        ),
                        app.get("failure_code") or f"application.{app['state']}",
                    ),
                    detail,
                )
                if unattributed:
                    # Nothing the Controller said names this lane: an uncertain outcome,
                    # observed again (one attempt each), never a verdict on one of them.
                    failure = replace(failure, klass=verdict.UNATTRIBUTED)
                self._fail_slot(member, self.state.slots[member], failure, app)
            return
        progress = self._observe_load(slot, now)
        held = slot.get("blocked_since")
        if (
            held is not None
            and now - float(held) > self.cfg.blocked_release_seconds
            and self._release_prepared(slot, now)
        ):
            return
        if progress is Progress.STALLED:
            self._stalled(key, slot, now)

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

    def _release_prepared(self, slot: Mapping[str, Any], now: float) -> bool:
        """A load waiting on the preparation of some recipes: re-plan it without them.

        The recipes a preparation blocker names leave the profile and wait for their
        preparation (requeued: no attempt, no verdict; the Controller keeps preparing), and
        the load is submitted again for the rest, so ready recipes do not wait for them.
        Returns False when nothing was done (no preparation names a subset of the load).
        """
        blockers = verdict.preparation_blockers(slot["blocked"])
        group = self._lane_group(slot)
        if len(group) < 2 or not blockers:
            return False
        culprits = [
            k for k in group if k in policy.blamed(blockers, self._lanes(group))
        ]
        survivors = [k for k in group if k not in culprits]
        if not culprits or not survivors:
            return False
        profile = self.cfg.sweep_profile
        try:
            for key in culprits:
                self.vk.call(
                    "profile",
                    "remove",
                    self.state.slots[key]["alias"],
                    "--yes",
                    profile=profile,
                )
        except VonkctlError as error:
            self.note_infra("review", f"re-plan without a preparing recipe: {error}")
            return False
        if not self._prune_profile():
            return False
        review = self.vk.run("profile", "load", "--review", profile=profile)
        document = review.document
        if not review.ok or not isinstance(document, Mapping):
            return False
        if not (
            document.get("allowed") is True
            or document.get("waits_for_preparation") is True
        ):
            return False
        cleanup_requests = self._adopted_cleanup_requests(document)
        adopted_requests = self._adopted_requests(
            document, excluded=frozenset(culprits)
        )
        if cleanup_requests is None or adopted_requests is None:
            return False
        lanes = self._lanes(culprits)
        # Fresh canonical admission supersedes only changed effects. Cancelling
        # this original root first would also abort the unchanged survivor.
        for key in culprits:
            own = [b for b in blockers if policy.blamed([b], {key: lanes[key]})]
            first = (own or blockers)[0]
            self.requeue(
                key,
                f"waiting on its preparation ({first['code']}): re-planned without it, "
                "the other recipes of the load go on",
                blockers=list(blockers),
            )
            self.state.entry(key)["not_before"] = now + self.cfg.defer_delay
        seq = int(self.state.data["load_seq"]) + 1
        request = request_key("sweep-load", self.state.nonce, profile, seq)
        self.state.data["load_seq"] = seq
        self.state.data["loads"][request] = {
            "request_key": request,
            "app_id": None,
            "seq": seq,
            "submitted_at": now,
            "adopted_requests": adopted_requests,
            "adopted_cleanup_requests": cleanup_requests,
        }
        for key in survivors:
            lane = self.state.slots[key]
            # Observation stays bound to the original accepted child/root.
            for stale in ("blocked_since", "blocked"):
                lane.pop(stale, None)
        self.state.save()
        try:
            submitted = self._submit_load(profile, request, "release")
        except VonkctlError as error:
            self.state.data["loads"][request]["state"] = "observing"
            self.note_infra("review", f"release outcome unknown: {str(error)[:100]}")
            return True
        self.state.data["loads"][request]["app_id"] = submitted.get("id")
        self.state.event(
            "released " + ", ".join(survivors) + " without " + ", ".join(culprits)
        )
        return True

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
            failure_class=failure.klass if failure else None,
            code=failure.code if failure else None,
            evidence={**(dict(failure.evidence) if failure else {}), **evidence}
            or None,
        )
        entry["status"] = "pending"
        if slot is not None:
            slot["phase"] = "finished"
        self.state.event(f"requeued {key}: {why}")

    def _submit_smoke(self, key: str, slot: dict[str, Any]) -> None:
        recipe = self.recipes[key]
        if not recipe.is_openai:
            self.futures[key] = _done_future(smoke_readiness(slot["run_id"]))
            return
        try:
            endpoints = self.vk.call(
                "profile", "endpoint", slot["alias"], profile=self.cfg.sweep_profile
            )
        except VonkctlError as error:
            if verdict.reply_disposition(error.reply) is Disposition.WAIT:
                self.note_infra("endpoint", str(error))
                slot["smoke_not_before"] = self.clock.now() + ENDPOINT_RECHECK_SECONDS
                return  # the run is up: look the endpoint up again on the next pass
            self._fail_slot(
                key, slot, policy.classify("readiness", "endpoint", str(error)), {}
            )
            return
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
            # The Controller publishes the route: a wait on it never expires. Look again.
            slot["smoke_not_before"] = self.clock.now() + ENDPOINT_RECHECK_SECONDS
            slot["waiting_on"] = (
                f"endpoint publication (assignment state {item.get('state')})"
            )
            return
        slot.pop("waiting_on", None)
        self.futures[key] = self.executor.submit(
            smoke_service, recipe, self.defs, base, slot["alias"], self.cfg.http
        )

    def _advance_smoking(self, key: str, slot: dict[str, Any]) -> None:
        future = self.futures.get(key)
        if (
            future is None
        ):  # resumed after a restart: the smoke is idempotent, run it again
            if self.clock.now() >= float(slot.get("smoke_not_before", 0)):
                self._submit_smoke(key, slot)
            return
        if not future.done():
            # Each request has its own inactivity bound (the socket timeout); the smoke as a
            # whole has no wall clock to expire.
            return
        del self.futures[key]
        try:
            result = future.result()
        except Exception as error:  # noqa: BLE001 - a bug in one smoke must not stop an unattended sweep
            result = SmokeResult(
                ok=False,
                kind="service",
                failure=policy.classify("smoke", "smoke.crash", repr(error)),
            )
        if result.ok:
            self.clear_infra("client-key")
            self._record_pass(key, slot, result)
            return
        assert result.failure is not None
        failure = result.failure
        if verdict.disposition(failure) is Disposition.WAIT:
            self._smoke_wait(key, slot, failure)
            return
        self._record_failure(key, failure, slot.get("operation_id"))

    def _smoke_wait(
        self, key: str, slot: dict[str, Any], failure: policy.Failure
    ) -> None:
        """A busy gateway or engine (429/503) or a refused client key: no attempt, smoke again.

        The lane is kept. A refused key waits on its renewal (the key file is read on every
        request); a busy answer that never changes for the lane's whole window is the
        engine's own stall and is observed (one attempt).
        """
        now = self.clock.now()
        if failure.klass == policy.CLIENT_CREDENTIAL:
            self.note_infra(
                "client-key",
                "the gateway refused the sweep's client key "
                f"({failure.describe()}); renew it in the --client-key-file",
            )
            until = float(self.state.data["infra"]["client-key"]["until"])
            waiting_on: str | None = "client key renewal"
        else:
            until = now + self.cfg.retry_delay
            waiting_on = None
        progress = verdict.progress_verdict(
            verdict.Signal((failure.signature,), waiting_on),
            slot.setdefault("smoke_watch", {}),
            window=float(slot["deadline"]) - float(slot["started_at"]),
            now=now,
        )
        if progress is Progress.STALLED:
            silent = round(verdict.silent_seconds(slot["smoke_watch"], now))
            self._record_failure(
                key,
                replace(
                    failure,
                    klass="smoke-timeout",
                    detail=f"{failure.detail}; busy without change for {silent}s",
                ),
                slot.get("operation_id"),
            )
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
            del self.state.slots[key]
            changed = True
        if changed:
            self.state.save()

    def cleanup_finished(self) -> None:
        self.enqueue_finished_cleanup()
        pending = self.state.data.setdefault("cleanup", {})
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
        if self.state.downloads.get(recipe.key, {}).get("state") == "succeeded":
            return True
        return recipe.local == "cached" and all(
            d in self.cached_models for d in recipe.model_digests
        )

    def ready_candidates(self) -> list[Recipe]:
        now = self.clock.now()
        out = []
        for key in self.queue:
            recipe = self.recipes.get(key)
            entry = self.state.recipes.get(key, {})
            if recipe is None or not self.ready(recipe) or key in self.state.slots:
                continue
            if now < float(entry.get("not_before", 0)) or now < float(
                entry.get("defer_until", 0)
            ):
                continue
            out.append(recipe)
        return out

    def _bins(self) -> list[policy.SparkBin]:
        bins: dict[str, policy.SparkBin] = {
            s.id: policy.SparkBin(s.id, s.name, s.memory_total) for s in self.sparks()
        }
        for key, slot in self.state.slots.items():
            if slot["phase"] == "finished":
                continue
            recipe = self.recipes[key]
            for node in slot["node_ids"]:
                if node in bins:
                    b = bins[node]
                    bins[node] = policy.SparkBin(
                        b.id,
                        b.name,
                        b.capacity,
                        b.used + recipe.memory_bytes,
                        b.ports | frozenset(recipe.ports),
                        b.aliases | {slot["alias"]},
                    )
        busy = {
            node
            for slot in self.state.slots.values()
            if slot["phase"] != "finished"
            for node in slot["node_ids"]
        }
        busy |= self._cleanup_nodes() | self._held_nodes()
        return [bin_ for bin_ in bins.values() if bin_.id not in busy]

    def _held_nodes(self) -> set[str]:
        """Sparks whose given-up load of ours still shows a workload: not reused until it is gone."""
        return {
            n for r in self.state.data.get("released", {}).values() for n in r["nodes"]
        }

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
        """Alert when a free lane has had ready work for too long (the sweep must never idle)."""
        busy = {n for s in self.state.slots.values() for n in s.get("node_ids", [])}
        busy |= self._held_nodes()
        free = [s for s in self.sparks() if s.id not in busy]
        waiting = (
            bool(free)
            and not self.owner_status.paused
            and bool(self.ready_candidates())
        )
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
        message = (
            f"IDLE WITH READY WORK: {len(free)} free lane(s) for {round(idle)}s "
            f"while recipes are ready (load record: "
            f"{len(self.state.data['loads'])}; "
            f"blocking reason: {self.state.data.get('clearing_reason', 'placement review or lane admission')})"
        )
        if not self.state.data.get("alerts"):
            self.state.event(message)
        self.state.data["alerts"] = [message]

    def schedule(self, now: float) -> None:
        self.release_given_up(now)
        self.release_orphan_load()
        if now < self.backoff_until:
            return
        candidates = self.ready_candidates()
        if not candidates:
            return
        if not self._take_over(now):
            return
        bins = self._bins()
        # A dual at the head drains both lanes; do not refill one with later singles.
        if candidates[0].node_count == 2:
            if len(bins) < 2:
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
            if len(spark_ids) == width:
                placements = [policy.Placement(head, spark_ids)]
        if placements:
            self._apply(placements, now)

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
        occupied: set[str] = set()
        for load in self.state.data["cleanup_loads"].values():
            if load.get("end"):
                continue
            if load.get("terminal_receipt"):
                if load.get("projection_known") and not self._cleanup_pending(load):
                    continue
                # Terminal bookkeeping is not a physical occupancy certificate.
                # Current fleet facts and the fresh Controller review own admission.
                kept = self._own_aliases()
                occupied.update(
                    p.node_id for p in self.fleet.presences if p.alias not in kept
                )
                continue
            if not load.get("projection_known"):
                # Before the current projection is known no lane may be inferred free.
                occupied.update(s.id for s in self.sparks())
                continue
            for row in self._cleanup_pending(load):
                occupied.update(row["node_ids"])
        online = {s.id for s in self.sparks()}
        if self.state.data.get("takeover_fallback"):
            # Bookkeeping is not occupancy. A fresh review still owns admission;
            # this observation never rewrites or declares an old receipt successful.
            own = self._own_aliases()
            return {
                p.node_id for p in self.fleet.presences if p.alias not in own
            } & online
        return occupied & online

    def _cleanup_target_absent(self, load: Mapping) -> bool:
        """An exact absent receipt plus fresh online fleet facts ends bookkeeping."""
        reply = self.vk.run("fleet")
        if not reply.ok:
            return False
        try:
            observed = CleanupFleetObservation.model_validate(reply.document)
        except ValidationError:
            return False
        # Offline or incomplete facts cannot certify the absence of an old run.
        expected_nodes = {
            node for row in load.get("effects", []) for node in row["node_ids"]
        }
        expected_runs = {row["target_id"] for row in load.get("effects", [])}
        for signature in load.get("reviewed_stops", []):
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
        return not (expected_runs & present) if expected_runs else not present

    def observe_cleanup(self) -> None:
        """Observe original stop receipts once; never replace intent after a read timeout."""
        for load in self.state.data["cleanup_loads"].values():
            if load.get("end"):
                continue
            if load.get("terminal_receipt"):
                continue
            if load.get("projection_known") and not self._cleanup_pending(load):
                continue  # immutable successful receipt/history needs no repeated I/O
            now = self.clock.now()
            budget = CleanupObservationBudget.model_validate(
                load.get("observation_budget", {})
            )
            if now < budget.next_check:
                continue
            args = ["progress"]
            if load.get("app_id"):
                args += ["--application", load["app_id"]]
            else:
                args += ["--request-key", load["request_key"]]
            reply = self.vk.run("profile", *args, profile=self.cfg.sweep_profile)
            document = reply.document
            if reply.is_not_found and self._cleanup_target_absent(load):
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
                self.state.save()
                continue
            if not reply.ok:
                budget.attempts += 1
                budget.next_check = now + min(
                    INFRA_BASE * 2 ** min(budget.attempts - 1, 20), INFRA_CAP
                )
                load["observation_budget"] = budget.model_dump(mode="json")
                # Legacy status entries describe the next actual observation, not
                # an expired submission retry. This does not pause other lanes.
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
            self.state.save()

    def _take_over(self, now: float) -> bool:
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
        if budget["attempts"] >= MAX_TAKEOVER_ATTEMPTS:
            # Spent on this decision: review free online lanes, and say so (not a wait).
            self.state.data["takeover_fallback"] = True
            self.state.data["clearing_reason"] = (
                f"{reason}; clearing budget spent, reviewing free online lanes"
            )
            return True
        if self.state.data.get("took_over") and all(
            load.get("projection_known") and load.get("state") not in SETTLED
            for load in self.state.data["cleanup_loads"].values()
        ):
            self.state.data.pop("clearing_reason", None)
            return True
        if now < budget["next_check"]:
            return False
        budget["attempts"] += 1
        budget["next_check"] = now + TAKEOVER_RETRY_SECONDS
        self.state.save()
        try:
            admitted = self._attempt_takeover(now)
        except VonkctlError as error:
            self.state.event(f"clearing attempt unanswered: {str(error)[:120]}")
            admitted = False
        if admitted:
            budget.update(attempts=0, next_check=0.0)
            self.state.data.pop("clearing_reason", None)
            self.state.data.pop("clearing_refusal", None)
            return True
        if budget["attempts"] >= MAX_TAKEOVER_ATTEMPTS:
            self.state.data["takeover_fallback"] = True
            self.state.event(
                f"{reason}: clearing budget exhausted; reviewing free online lanes"
            )
            return True
        return False

    def _attempt_takeover(self, now: float) -> bool:
        """Begin one durable cleanup, then admit lanes from exact receipts."""
        loads = self.state.data["cleanup_loads"]
        if loads:
            if all(
                load.get("terminal_receipt") or load.get("end")
                for load in loads.values()
            ):
                if not self.fleet.presences:
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
        self.backoff_until = now

    def _prune_profile(self) -> bool:
        """Before any load of the sweep profile, it holds only what live lanes need.

        A load applies the whole profile. An assignment no lane accounts for (a lost cleanup
        record, an edit the Controller applied after a timeout) is kept or started by every
        later load, so a clearing or stop load would run the very workload it should end.
        The exact export decides; only names the sweep itself uses are removed.
        """
        reply = self.vk.run("profile", "export", profile=self.cfg.sweep_profile)
        document = reply.document
        assignments = (
            document.get("assignments") if isinstance(document, dict) else None
        )
        if not reply.ok or not isinstance(assignments, list):
            self.note_infra(
                "profile-edit", f"sweep profile unreadable: {reply.error_text[:200]}"
            )
            return False
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
            return True
        pending = self.state.data.setdefault("cleanup", {})
        for name in stale:
            pending.setdefault(name, {"attempts": 0, "next_check": 0.0})
        self.cleanup_finished()
        return not stale & set(pending)

    def _renew_cleanup(self, now: float) -> bool:
        # Every new clear starts from current physical facts, including after
        # a terminal receipt with an absent or incomplete child projection.
        self.fleet = fetch_fleet(self.vk, self.cfg.default_spark_memory)
        loads = self.state.data["cleanup_loads"]
        if not self._prune_profile():
            return self._clearing_refused(
                "the sweep profile still holds assignments no lane needs"
            )
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
        if any(not set(row.get("node_ids", [])) <= online for row in stops):
            # an empty whole-fleet load would clear an offline Spark
            return self._clearing_refused("a reviewed stop reaches an offline Spark")
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
            "predecessor_effects": [row["effect_id"] for row in residual],
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

    def _apply(self, placements: Sequence[policy.Placement], now: float) -> None:
        profile = self.cfg.sweep_profile
        if not self._prune_profile():
            return  # a load now would also start what no lane needs; the next pass retries
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
                added.append(placement)
                self.clear_infra("profile-edit")
            except VonkctlError as error:
                if verdict.reply_disposition(error.reply) is Disposition.WAIT:
                    # The client, protocol or transport failed, or the Controller refused
                    # something the sweep itself sent (the name, the Sparks, the state):
                    # nothing about this recipe. The next pass reviews afresh.
                    self._rollback(added, f"profile edit failed: {error}")
                    self.note_infra("profile-edit", str(error))
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
        aliases = {self._alias(p.recipe): p for p in added}
        kept = {
            s["alias"]
            for s in self.state.slots.values()
            if s["phase"] in ("loading", "smoking")
        }
        review = self.vk.run("profile", "load", "--review", profile=profile)
        document = review.document if isinstance(review.document, dict) else {}
        if review.is_error_document or not document:
            self._rollback(added, f"review failed: {review.error_text}")
            if verdict.reply_disposition(review) is Disposition.WAIT:
                self.note_infra("review", review.error_text)
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
        self.clear_infra("review")
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
        cleanup_requests = self._adopted_cleanup_requests(document)
        if cleanup_requests is None:
            self._rollback(
                added, "pending original cleanup effects are not bound for adoption"
            )
            self.note_infra("review", "waiting for exact original cleanup adoption")
            return
        adopted_requests = self._adopted_requests(document)
        if adopted_requests is None:
            self._rollback(
                added, "unchanged loading effects are not bound for adoption"
            )
            self.note_infra(
                "review", "observing unchanged loading effects before overlap"
            )
            return
        if not (
            document.get("allowed") is True
            or document.get("waits_for_preparation") is True
        ):
            self._blocked(document, aliases, added, now)
            return
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
                self.note_infra("review", str(error))
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

    def _cleanup_reaccepted(self, original: Mapping[str, Any]) -> bool:
        return any(
            original["effect_id"] in load.get("predecessor_effects", [])
            and load.get("projection_known")
            and load.get("state") not in SETTLED
            and any(
                cleanup.stop_signature(row["stop_effect"])
                == cleanup.stop_signature(original["stop_effect"])
                for row in load.get("effects", [])
            )
            for load in self.state.data["cleanup_loads"].values()
        )

    def _adopted_cleanup_requests(self, review: Mapping[str, Any]) -> list[str] | None:
        requests = []
        for key, load in self.state.data["cleanup_loads"].items():
            if load.get("terminal_receipt") or load.get("end"):
                continue
            if not load.get("projection_known"):
                if self.state.data.get("takeover_fallback"):
                    continue  # fleet occupancy and the fresh review govern new work
                return None
            online = {s.id for s in self.sparks()}
            pending = [
                row
                for row in self._cleanup_pending(load)
                if set(row["node_ids"]) & online
            ]
            if load.get("state") in SETTLED:
                # Historical cancelled claims stay occupied. A newer accepted exact
                # stop may own that residual effect; never try to adopt terminal work.
                if any(not self._cleanup_reaccepted(row) for row in pending):
                    return None
                continue
            if any(not cleanup.adopted(review, row) for row in pending):
                return None
            if pending:
                requests.append(key)
        return sorted(requests)

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
        own = self.state.data["own_aliases"]
        if alias not in own:
            own.append(alias)
            del own[:-500]
        self.state.slots[recipe.key] = {
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
        self.backoff_until = self.clock.now() + self.cfg.retry_delay
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
                )
                continue
            failure = policy.classify("review", code, detail)
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
                )
                self.state.event(
                    f"review refused {key} ({code}): clearing the fleet and trying "
                    f"again ({freeing + 1}/{MAX_REVIEW_FREEING})"
                )
                continue
            if freeing:
                failure = replace(
                    failure,
                    evidence={**failure.evidence, "freeing_attempts": freeing},
                )
            # The clearings were this outcome's observations: once spent, it is recorded.
            self._record_failure(key, failure, None, final=freeing > 0)

    def note_infra(self, source: str, message: str) -> None:
        """A client, protocol or transport problem: pause that kind of work, back off, show it."""
        now = self.clock.now()
        item = self.state.data["infra"].get(source) or {"since": now, "count": 0}
        item["count"] += 1
        hint = (
            " (vonkctl may be older than the Controller: run `vonkctl update --apply`)"
            if "protocol_invalid" in message or "OpenAPI" in message
            else ""
        )
        item["message"] = message[:300] + hint
        item["until"] = now + min(
            INFRA_BASE * 2 ** min(item["count"] - 1, 20), INFRA_CAP
        )
        self.state.data["infra"][source] = item
        if source == "download":
            self.prefetcher.pause_until = item["until"]
        else:
            self.backoff_until = max(self.backoff_until, item["until"])
        if item["count"] in (1, 5) or item["count"] % 20 == 0:
            self.state.event(
                f"infrastructure problem ({source}, not a recipe failure), "
                f"retrying with backoff: {item['message'][:160]}"
            )

    def clear_infra(self, source: str) -> None:
        if self.state.data["infra"].pop(source, None) is not None:
            self.state.event(f"infrastructure problem cleared ({source})")

    def _own_aliases(self) -> set[str]:
        """Every workload name the sweep has ever put in its profile (lanes come and go, the
        workload of a released lane runs until the next load replaces it)."""
        # (also the names the catalog would give: a state file from before own_aliases existed)
        return (
            {str(a) for a in self.state.data["own_aliases"]}
            | {str(s["alias"]) for s in self.state.slots.values()}
            | {alias_for(r, self.defs) for r in self.recipes.values()}
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
        self, profile: int, kind: str, *, owner: bool = False
    ) -> Any:
        """Persist reviewed stop intent, including an empty scope, before effects."""
        if profile == self.cfg.sweep_profile and not self._prune_profile():
            raise RuntimeError("the sweep profile holds assignments no lane needs")
        review = self.vk.call(
            "profile",
            "load",
            "--review",
            profile=profile,
            allow_owner_write=owner,
        )
        if review.get("allowed") is not True:
            raise RuntimeError(f"the {kind} load has no allowed canonical review")
        key = self._own_key(kind, profile)
        if kind == "stop":
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
        number = self.cfg.restore_owner
        if not self.state.data.get("dirty"):
            return  # the sweep never changed what the Sparks run
        try:
            if number is not None:
                self._submit_reviewed_load(number, "restore-owner", owner=True)
                self.state.event(f"restored owner profile {number}")
                self.state.data["dirty"] = False
            elif self.cfg.stop_at_end:
                self._submit_reviewed_load(self.cfg.sweep_profile, "stop")
                self.state.data["dirty"] = False
        except (VonkctlError, RuntimeError) as error:
            self.state.event(f"leaving the fleet failed: {str(error)[:150]}")


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
