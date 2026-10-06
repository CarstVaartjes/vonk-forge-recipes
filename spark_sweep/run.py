"""The sweep itself: lanes over one sweep profile, fed by the prefetcher.

Profiles cover the whole fleet (an unassigned Spark is idle and its workload is
stopped by a load), so the lanes are *assignments of one sweep profile*, not
one profile per lane. A lane swap edits that profile and loads it; the planner
keeps every unchanged assignment (``--review`` is checked to prove it).
Applications are serialised: one load in flight at a time, while smoke tests of
the other lane run beside it.

``Sweep.tick`` is a small state machine, so the whole sweep is deterministic
under a fake clock and a fake ``vonkctl``:

    refresh catalog -> owner guard -> advance slots -> prefetch -> schedule -> status
"""

from __future__ import annotations

import itertools
import signal
import threading
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from concurrent.futures import Future
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Protocol

from . import policy
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
from .lifecycle import WAITING
from .owner import OwnerGuard, OwnerStatus
from .prefetch import PrefetchConfig, Prefetcher
from .profile_alias import profile_alias, unique_profile_alias
from .report import write_status
from .smoke import HttpConfig, SmokeResult, smoke_readiness, smoke_service
from .state import ResultsLog, State, StateLock
from .vonkctl import Vonkctl, VonkctlError, is_infrastructure, request_key

CLEAR_TIMEOUT_SECONDS = 900.0
IDLE_WAIT_SECONDS = 120.0
CLEARING_ATTEMPTS = 3
WAIT_EVENT_SECONDS = 60.0  # a wait says what it waits for at least this often
OWNER_WAIT_SECONDS = (
    3600.0  # an owner load is waited for this long, then it is an infra event
)
STARTUP_WAIT_SECONDS = (
    1800.0  # a startup step that keeps failing is an infra event after this
)
TICK_WATCHDOG_SECONDS = 1800.0  # a pass that has not finished in this long is abandoned
MAX_REVIEW_DEFERS = 3
# A shared load that ended with no blocker naming a recipe requeues its lanes unblamed this often.
MAX_UNATTRIBUTED_REQUEUES = 2
# Clearing the fleet and reviewing again, for a capacity refusal, before it counts against the recipe.
MAX_REVIEW_FREEING = 2
SKEW_CHECK_INTERVAL = 300.0
INFRA_BASE = 30.0
INFRA_CAP = 600.0
RETEST_SECONDS = 24 * 3600.0
LOAD_WALL_SECONDS = 24 * 3600.0
SMOKE_WALL_SECONDS = 1800.0


class SweepOwnerIntentSuperseded(RuntimeError):
    """An owner intent ends the sweep's bounded observation period."""


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
    readiness_grace: float = 180.0
    retry_delay: float = 60.0
    defer_delay: float = 120.0
    max_attempts: int = 2
    dual_batch_min: int = 2
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
    copy_stall_seconds: float = 600.0
    # How long an application may sit behind its own admission blockers (capacity, stale
    # inventory, a retried phase) before the lane fails as admission-stalled.
    blocked_seconds: float = 900.0
    # How long a load may sit behind a *preparation* blocker that names only some of its
    # recipes before the others are released: the blocked one leaves the profile and the
    # rest are loaded without it.
    blocked_release_seconds: float = 300.0


def alias_for(recipe: Recipe, definitions: Definitions) -> str:
    """The name the lane serves under: the reviewed service alias, else the slug.

    Always a valid profile alias, so a spelling the contract refuses (capitals,
    a ``/``) never reaches the Controller.
    """
    return profile_alias(definitions.alias(recipe.key) or recipe.slug)


class TickStuck(RuntimeError):
    """The watchdog abandons a pass that has not finished in time."""


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
        self._tick_started: float | None = None
        self._wait_event_at = -1e18

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
        except VonkctlError:
            if not self.fleet.sparks:
                return
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
        except VonkctlError:
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
            entry.update(
                status="skipped",
                reason=f"not testable on this fleet: needs {recipe.node_count} Sparks, {usable_sparks} usable",
                reason_kind=FLEET_SKIP,
                content_sha256=recipe.content_sha256,
            )
            return
        if _fleet_skip(entry) and recipe.node_count <= usable_sparks:
            entry["status"] = "pending"
            entry.pop("reason", None)
            entry.pop("reason_kind", None)
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

    def waiting(self, what: str, since: float, *, watch: bool = True) -> None:
        """Called on every poll of a wait: save state and status, say what is awaited and since when.

        A wait is never silent: the state file's ``updated_at`` and the status page move on every
        poll. When the pass the wait is part of exceeds the watchdog limit, the step is abandoned.
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
        if (
            watch
            and self._tick_started is not None
            and now - self._tick_started > TICK_WATCHDOG_SECONDS
        ):
            raise TickStuck(f"the pass is stuck waiting for {what}")

    def done_waiting(self) -> None:
        self.state.data.pop("waiting", None)
        self._wait_event_at = -1e18

    def tick(self) -> None:
        """One pass. A Controller that stops answering costs a pass, never the sweep."""
        self._tick_started = self.clock.now()
        try:
            self._tick()
        except SweepOwnerIntentSuperseded:
            raise
        except TickStuck as error:
            self.note_infra("watchdog", str(error))
            self.state.event(
                f"watchdog: abandoned a stuck pass, continuing: {str(error)[:150]}"
            )
            self._save_quietly()
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
            self._tick_started = None
            self.done_waiting()

    def _save_quietly(self) -> None:
        try:
            self.state.save()
        except Exception as error:  # noqa: BLE001 - the next pass saves again
            self.note_infra("state-save", f"{type(error).__name__}: {error}")

    def _startup(self, step: Callable[[], None]) -> bool:
        """A startup step that only reads the fleet: a missing or failing vonkctl waits, never exits."""
        started = self.clock.now()
        deadline = started + STARTUP_WAIT_SECONDS
        while self.clock.now() < deadline:
            try:
                step()
                self.clear_infra("startup")
                self.done_waiting()
                return True
            except (VonkctlError, OSError) as error:
                self.note_infra("startup", f"{type(error).__name__}: {error}")
                until = self.state.data["infra"]["startup"]["until"]
                self.waiting(
                    "the Controller to answer at startup", started, watch=False
                )
                self.clock.sleep(
                    max(1.0, min(until - self.clock.now(), self.cfg.poll_seconds * 6))
                )

        self.state.event(
            "startup observation budget exhausted; retrying in the next cycle"
        )
        return False

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
                    self._requeue_failed(key, "scheduled-retest")
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
            paused_at = self.state.data["owner"].setdefault("paused_at", now)
            if now - float(paused_at) >= OWNER_WAIT_SECONDS:
                self.state.data["end"] = {
                    "code": "sweep.owner_intent_superseded",
                    "reason": self.owner_status.reason,
                }
                raise SweepOwnerIntentSuperseded(self.owner_status.reason)
        else:
            self.state.data["owner"].pop("paused_at", None)
        self.advance_slots(now)
        self.heal_fleet_skips()
        pending = self.pending()
        for (
            key,
            record,
        ) in self.state.downloads.items():  # a state file from an older run
            if record.get("state") == "succeeded":
                self._download_done(key)
        self.cached_models = self._cached_models()
        self.prefetcher.library_fresh_since = self.recipe_list.complete_started_at
        self.last_prefetch = self.prefetcher.tick(
            self.recipes,
            self.models,
            pending,
            self.sizes,
            self._present(),
            self.boost,
            self.cached_models,
        )
        self.queue = policy.order_queue(
            self.prefetcher.last_plans,
            self.recipes,
            revalidate={
                k for k, e in self.state.recipes.items() if e.get("revalidate")
            },
            deprioritised=self._deprioritised(pending),
            variants_last=self.cfg.variants_last,
        )
        if not self.owner_status.paused:
            self.cleanup_finished()
            self.schedule(now)
        self.check_idle(now)
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
                slot["deadline"] = float(slot["deadline"]) + gap
                if "copy_progress_at" in slot:
                    slot["copy_progress_at"] = now
            slot["last_tick"] = now
            adopted.append(key)
        if adopted:
            self.state.event(
                "adopted an in-flight load of "
                + ", ".join(adopted)
                + " (time away not counted)"
            )

    def start_takeover(self) -> None:
        """``run --yes`` means the sweep owns the fleet: clear it once, now, so reviews see idle Sparks."""
        started = self.clock.now()
        deadline = started + OWNER_WAIT_SECONDS
        while self.owner_status.paused and self.clock.now() < deadline:
            self.waiting(
                f"an owner load ({self.owner_status.reason})", started, watch=False
            )
            self.clock.sleep(self.cfg.poll_seconds)
            self.owner_status = self.guard.check(self.clock.now())
        self.done_waiting()
        if self.owner_status.paused:
            self.note_infra("owner-wait", f"still paused: {self.owner_status.reason}")
            return
        self._take_over(self.clock.now())

    def check_client(self) -> None:
        """Refuse to run with a vonkctl that is not the accepted release.

        The Controller exposes no version of its own; the accepted signed release is what it
        is deployed from, so ``vonkctl update`` (a read-only check) tells whether this client
        has drifted. An older client fails at runtime with protocol errors that look like
        every recipe failing.
        """
        version = self.vk.run("--version")
        if isinstance(version.document, dict):
            self.state.data["client"] = {
                k: version.document.get(k) for k in ("version", "source_sha")
            }
        check = self.vk.run("update", timeout=60)
        document = check.document if isinstance(check.document, dict) else {}
        if not check.ok or "update_available" not in document:
            self.state.event(
                "could not compare vonkctl with the accepted release: "
                f"{check.error_text[:120]}"
            )
            return
        self.observe_release(document)
        if document["update_available"] is not True:
            return
        current = dig(document, "current", "version", default="unknown")
        accepted = document.get("accepted_version", "unknown")
        message = f"vonkctl {current} differs from the accepted release {accepted}"
        if self.cfg.allow_version_skew:
            self.state.event(f"WARNING: {message}; continuing as asked")
            return
        upgraded = self.vk.run("update", "--apply", timeout=120)
        if upgraded.ok:
            self.state.event(f"{message}; accepted client update applied")
            version = self.vk.run("--version")
            if isinstance(version.document, dict):
                self.state.data["client"] = {
                    k: version.document.get(k) for k in ("version", "source_sha")
                }
            self.clear_infra("client-update")
        else:
            self.note_infra("client-update", f"{message}; update will be retried")

    @property
    def release_sha(self) -> str | None:
        value = self.state.data["release"].get("sha")
        return value if isinstance(value, str) else None

    def observe_release(self, document: Mapping[str, Any]) -> None:
        """Note the accepted Controller release; a change requeues failures a fix may have cured.

        The Controller has no version endpoint, so the accepted signed release (the source it
        is deployed from) stands in for it, as in the client check.
        """
        version = document.get("accepted_version")
        identity = document.get("accepted_source_sha") or version
        if not isinstance(identity, str) or not identity:
            return
        release = self.state.data["release"]
        if release.get("sha") != identity:
            if release:
                self.state.event(
                    f"Controller release changed: {release.get('sha')} -> {identity}"
                )
            release.update(sha=identity, version=version, seen_at=self.clock.now())
            self.state.data["release_history"].append(
                {"sha": identity, "version": version, "seen_at": self.clock.now()}
            )
        self.requeue_for_release()

    def check_release(self, now: float) -> None:
        if now - self.release_at < self.cfg.release_seconds:
            return
        self.release_at = now
        self.check_client()

    def requeue_for_release(self) -> None:
        """Failures from another Controller release, of a platform-side class, are tried again."""
        current = self.release_sha
        if current is None:
            return
        for key, entry in self.state.recipes.items():
            if (
                entry.get("status") in ("failed", "deferred")
                and entry.get("release") != current
                and policy.platform_side(entry.get("failure_class"))
                and self._selected_key(key)
            ):
                self._requeue_failed(key, f"controller-release-changed ({current})")

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
        self.log.append(
            batch="sweep",
            recipe=key,
            step="requeue",
            status="requeued",
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
            # Ask again with the next attempt number (a new request key: the old key would
            # replay the failed operation). A finished download stands.
            record["state"] = "retired"
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
                except SweepOwnerIntentSuperseded:
                    self.interrupt()
                    self.finish(interrupted=True)
                    return 0
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
        # The prefetcher already retried what is worth retrying: this one is final.
        self._record_failure(key, failure, op_id, retryable=False)
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
        retryable: bool = True,
    ) -> None:
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
                key, f"{failure.phase} failure overlapped an owner profile load"
            )
            return
        entry["attempts"] = int(entry.get("attempts", 0)) + 1
        if slot is not None:
            slot["phase"] = "finished"
        if (
            retryable
            and failure.transient
            and entry["attempts"] < self.cfg.max_attempts
        ):
            entry.update(
                status="pending",
                not_before=now + self.cfg.retry_delay,
                last_failure=failure.signature,
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
        attributable = (
            failure.code in policy.RECIPE_FAILURE_CODES
            or failure.klass == "smoke-assertion"
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
        self.log.append(
            batch="sweep",
            recipe=key,
            step="smoke",
            status=status,
            error=f"{failure.phase}/{failure.klass}: {failure.describe()}"[:1536],
            **self._facts(key, entry, slot),
        )
        self.state.event(f"{status.upper()} {key}: {failure.phase}/{failure.klass}")

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
            content_sha256=recipe.content_sha256,
            revision_id=recipe.revision_id,
            timings=timings,
            smoke=result.kind,
            perf=result.perf,
            smoke_rules=policy.SMOKE_RULES,
            revalidate=False,
        )
        if result.quality_notes:
            entry["quality"] = result.quality_notes
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
        self.log.append(
            batch="sweep",
            recipe=key,
            step="smoke",
            status="passed",
            result={"endpoint_alias": slot["alias"], **result.record()},
            **self._facts(key, entry, slot),
        )
        notes = len(result.quality_notes)
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
        if (
            not reply.ok
            and not load.get("app_id")
            and isinstance(document, dict)
            and document.get("code") in {"not_found", "profile.application_not_found"}
        ):
            # An exact fresh lookup proved there is no receipt. Replay the same request.
            try:
                document = self._submit_load(
                    self.cfg.sweep_profile, load["request_key"], "placement"
                )
            except VonkctlError as error:
                self._blind(str(error), error.reply)
                return False
        elif not reply.ok and is_infrastructure(reply):
            self._blind(reply.error_text, reply)
            return False
        if not isinstance(document, dict) or not isinstance(document.get("id"), str):
            self._blind("application observation is unreadable")
            return False
        self.clear_infra("review")
        load["observed_at"] = self.clock.now()
        load["app_id"] = document["id"]
        status = str(document.get("state", ""))
        load["state"] = status
        moving = policy.distribution(document)
        load["copying"] = moving.copying
        load["phase"] = moving.phase
        load["progress"] = list(moving.signature) if moving.signature else None
        load["blockers"] = policy.admission_blockers(document)
        failure = document.get("failure") or {}
        failure_code = str(failure.get("code") or document.get("code") or "")
        gate = next(
            (
                b["code"]
                for b in load["blockers"]
                if b["code"]
                .lower()
                .endswith(("deletion_in_progress", "gate_held", "needs_operator"))
            ),
            None,
        )
        if gate or (
            status == "failed"
            and failure_code.startswith(("controller.", "http.", "profile."))
        ):
            self._end_load(load, gate or failure_code)
            return True
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
            retry = self._retry_successor(document)
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
        if status in WAITING:
            self._end_load(load, f"application.{status}")
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

    def _end_load(self, load: dict[str, Any], reason: str) -> None:
        self._cancel_load(load)
        if not self._adopted_slots_live(load):
            self.state.data["loads"].pop(load["request_key"], None)
        for key, slot in list(self.state.slots.items()):
            if (
                slot.get("request_key") == load["request_key"]
                and slot["phase"] == "loading"
            ):
                self.requeue(key, reason)
                self.state.entry(key)["not_before"] = (
                    self.clock.now() + self.cfg.retry_delay
                )
                slot["phase"] = "finished"
        self.state.event(
            f"released application {load.get('app_id')}: {reason}; retry queued"
        )

    def _retry_successor(self, document: Mapping[str, Any]) -> str | None:
        """The application that continues a failed one the Controller is still retrying, if any.

        A failed application ends for good with the Controller's typed ``profile.failure_repeated``
        blocker or a terminal assignment failure; otherwise a later application of the profile
        that names this one as the application it retries is its continuation.
        """
        if policy.retry_ended(document):
            return None
        reply = self.vk.run("profile", "progress", profile=self.cfg.sweep_profile)
        latest = reply.document if reply.ok else None
        if (
            isinstance(latest, dict)
            and isinstance(latest.get("id"), str)
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
            if error.infrastructure():
                self._blind(str(error), error.reply)
        self.poll_application()
        if not self.blind:
            self.clear_infra("observe")
        for key, slot in list(self.state.slots.items()):
            if slot["phase"] == "loading":
                self._advance_loading(key, slot, now)
            elif slot["phase"] == "smoking":
                self._advance_smoking(key, slot)

    def _track_distribution(self, slot: dict[str, Any], now: float) -> bool:
        """Is the slot's load still putting bytes in place? Those seconds are not load time.

        Copying a few hundred GiB to a Spark can take longer than any sensible load timeout while
        being perfectly healthy. The wall-clock limit therefore only runs while the application is
        in the phases after the bytes are in place; while copying, only a copy that has stopped
        moving counts against the load.
        """
        load = self.state.data["loads"].get(slot["request_key"])
        elapsed = now - float(slot.get("last_tick", slot["started_at"]))
        slot["last_tick"] = now
        if self.blind or slot.pop("blind_tick", False):
            # The sweep could not see the Controller (this tick, or the one before, so the gap
            # between them was blind too): those seconds are never load time, and nothing that
            # was waiting (a stalled copy, a held application) ages while we are blind.
            slot["blind_seconds"] = float(slot.get("blind_seconds", 0)) + elapsed
            for waiting in ("copy_progress_at", "blocked_since"):
                if waiting in slot:
                    slot[waiting] = float(slot[waiting]) + elapsed
            if self.blind:
                slot["blind_tick"] = True
            return bool(slot.get("copying"))
        copying = bool(
            load
            and load.get("request_key") == slot["request_key"]
            and load.get("copying")
        )
        slot["copying"] = copying
        blockers = (
            load.get("blockers")
            if load and load.get("request_key") == slot["request_key"] and not copying
            else None
        )
        if blockers:
            # The Controller is holding the application back, not the engine loading: those
            # seconds are not load time, and a hold that never ends is its own failure.
            slot["blocked_seconds"] = float(slot.get("blocked_seconds", 0)) + elapsed
            slot.setdefault("blocked_since", now)
            slot["blocked"] = blockers
        else:
            slot.pop("blocked_since", None)
            slot.pop("blocked", None)
        if load and load.get("phase"):
            slot["load_phase"] = load["phase"]
        if not copying:
            slot.pop("copy_progress_at", None)
            return False
        slot["copy_seconds"] = float(slot.get("copy_seconds", 0)) + elapsed
        signature = load.get("progress") if load else None
        if signature != slot.get("copy_signature") or "copy_progress_at" not in slot:
            slot["copy_signature"] = signature
            slot["copy_progress_at"] = now
        return True

    def _advance_loading(self, key: str, slot: dict[str, Any], now: float) -> None:
        recipe = self.recipes[key]
        copying = self._track_distribution(slot, now)
        if self.blind:
            if now >= float(
                slot.get("wall_deadline", slot["started_at"] + LOAD_WALL_SECONDS)
            ):
                load = self.state.data["loads"].get(slot["request_key"])
                if load:
                    self._end_load(load, "application.observation_budget_exhausted")
            return  # no recipe verdict on an unreadable Controller
        run_id = serving_run(
            self.fleet, slot["alias"], slot["node_ids"], recipe.node_count
        )
        if run_id:
            slot.update(phase="smoking", run_id=run_id, ready_at=now)
            policy.learn(
                self.state.data["learned"],
                recipe.engine,
                slot["model_bytes"],
                now - slot["started_at"] - float(slot.get("copy_seconds", 0)),
            )
            self._submit_smoke(key, slot)
            return
        if now >= float(
            slot.get("wall_deadline", slot["started_at"] + LOAD_WALL_SECONDS)
        ):
            load = self.state.data["loads"].get(slot["request_key"])
            if load:
                self._end_load(load, "application.observation_budget_exhausted")
            else:
                self.requeue(key, "application.observation_budget_exhausted")
                self.state.entry(key)["not_before"] = now + self.cfg.retry_delay
                slot["phase"] = "finished"
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
            if app["state"] == "succeeded":
                slot.setdefault("settled_at", now)
                if now - slot["settled_at"] > self.cfg.readiness_grace:
                    self._fail_slot(
                        key,
                        slot,
                        policy.classify(
                            "readiness",
                            "run.not_published",
                            "the load succeeded but the run never became healthy and published",
                        ),
                        app,
                    )
                return
            group = self._lane_group(slot)
            if len(group) > 1:
                failures = [
                    f for f in app.get("failures") or [] if isinstance(f, Mapping)
                ]
                culprits, collateral = self._split(
                    group, policy.blamed(failures, self._lanes(group)), now
                )
                self._requeue_collateral(collateral, culprits, "its application failed")
                group = culprits
            else:
                group = [key]
            phase = policy.child_phase(app.get("child_phase")) or "start"
            detail = f"{app.get('reason') or app['state']} {app.get('failures') or ''}".strip()
            for member in group:
                self._fail_slot(
                    member,
                    self.state.slots[member],
                    policy.classify(
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
                    ),
                    app,
                )
            return
        held = slot.get("blocked_since")
        if held is not None and now - float(held) > self.cfg.blocked_seconds:
            self._stalled(slot, now)
        elif (
            held is not None
            and now - float(held) > self.cfg.blocked_release_seconds
            and self._release_prepared(slot, now)
        ):
            return
        elif copying:
            stalled = now - float(slot["copy_progress_at"])
            if stalled > self.cfg.copy_stall_seconds:
                self._timeout(
                    key,
                    slot,
                    policy.classify(
                        "install",
                        "copy.stalled",
                        f"the copy to the Spark made no progress for {round(stalled)}s "
                        f"(phase {slot.get('load_phase')})",
                        "copy-stalled",
                    ),
                )
        elif now > slot["deadline"] + float(slot.get("copy_seconds", 0)) + float(
            slot.get("blocked_seconds", 0)
        ) + float(slot.get("blind_seconds", 0)):
            self._timeout(key, slot)

    def _lane_group(self, slot: Mapping[str, Any]) -> list[str]:
        """The lanes still loading in the same application as this one."""
        return [
            k
            for k, s in self.state.slots.items()
            if s["phase"] == "loading"
            and s["request_key"] == slot["request_key"]
            and not serving_run(
                self.fleet, s["alias"], s["node_ids"], self.recipes[k].node_count
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
        self, group: Sequence[str], blamed: set[str], now: float
    ) -> tuple[list[str], list[str]]:
        """Who of a load that ended or stalled carries the failure, and who is collateral.

        Only the lanes the Controller's blockers or failures name are the recipes' own. When
        nothing names a lane of a shared load, none is blamed (it is retried in a later load)
        unless it was already left unblamed too often, so a recipe that really fails is
        never retried for ever.
        """
        culprits = [k for k in group if k in blamed]
        if culprits:
            return culprits, [k for k in group if k not in blamed]
        if len(group) < 2:
            return list(group), []
        exhausted = [
            k
            for k in group
            if int(self.state.entry(k).get("unattributed_requeues", 0))
            >= MAX_UNATTRIBUTED_REQUEUES
        ]
        for key in group:
            if key not in exhausted:
                entry = self.state.entry(key)
                entry["unattributed_requeues"] = (
                    int(entry.get("unattributed_requeues", 0)) + 1
                )
        return exhausted, [k for k in group if k not in exhausted]

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

    def _stalled(self, slot: Mapping[str, Any], now: float) -> None:
        """The Controller held a load back for good: only the recipes it names fail."""
        blockers = list(slot["blocked"])
        group = self._lane_group(slot)
        culprits, collateral = self._split(
            group,
            policy.blamed(blockers, self._lanes(group))
            if len(group) > 1
            else set(group),
            now,
        )
        self._requeue_collateral(collateral, culprits, "the Controller held it back")
        if not culprits:
            load = self.state.data["loads"].get(slot["request_key"])
            if load:
                self._cancel_load(load)
        lanes = self._lanes(culprits)
        for key in culprits:
            own = [b for b in blockers if policy.blamed([b], {key: lanes[key]})]
            first = (own or blockers)[0]
            held = round(now - float(self.state.slots[key]["blocked_since"]))
            self._timeout(
                key,
                self.state.slots[key],
                policy.classify(
                    "start",
                    first["code"],
                    f"the Controller held this application back for {held}s: "
                    f"{first['code']}: {first['detail']}",
                    "admission-stalled",
                    {"blockers": blockers},
                ),
            )

    def _release_prepared(self, slot: Mapping[str, Any], now: float) -> bool:
        """A load stalled on the preparation of some recipes: load the others without them.

        The blocked recipes fail (they are the ones named), leave the profile, and the load is
        submitted again for the rest, so ready recipes do not wait out the whole deadline.
        Returns False when nothing was done (nothing names a subset of the load).
        """
        blockers = [
            b
            for b in slot["blocked"]
            if any(
                w in b["code"] for w in ("preparation", "build-unavailable", "cache")
            )
        ]
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
        except VonkctlError:
            return False  # leave it to the stall handling
        lanes = self._lanes(culprits)
        previous = self.state.data["loads"].get(slot["request_key"], {})
        self._cancel_load(previous)
        for key in culprits:
            own = [b for b in blockers if policy.blamed([b], {key: lanes[key]})]
            first = (own or blockers)[0]
            held = round(now - float(self.state.slots[key]["blocked_since"]))
            self._fail_slot(
                key,
                self.state.slots[key],
                policy.classify(
                    "start",
                    first["code"],
                    f"the Controller held this application back for {held}s: "
                    f"{first['code']}: {first['detail']}; the other recipes of the load "
                    "were released without it",
                    "admission-stalled",
                    {"blockers": blockers},
                ),
                previous,
            )
        if not self._adopted_slots_live(previous):
            self.state.data["loads"].pop(slot["request_key"], None)
        seq = int(self.state.data["load_seq"]) + 1
        request = request_key("sweep-load", self.state.nonce, profile, seq)
        self.state.data["load_seq"] = seq
        self.state.data["loads"][request] = {
            "request_key": request,
            "app_id": None,
            "seq": seq,
            "submitted_at": now,
            "adopted_requests": list(previous.get("adopted_requests") or []),
        }
        for key in survivors:
            lane = self.state.slots[key]
            lane["request_key"] = request
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
        seconds = round(
            self.clock.now()
            - slot["started_at"]
            - float(slot.get("copy_seconds", 0))
            - float(slot.get("blocked_seconds", 0))
            - float(slot.get("blind_seconds", 0))
        )
        self._fail_slot(
            key,
            slot,
            failure
            or policy.classify(
                "timeout",
                "load.timeout",
                f"not serving {seconds}s after the bytes were in place "
                f"(limit {round(slot['deadline'] - slot['started_at'])}s, "
                f"{round(float(slot.get('copy_seconds', 0)))}s of copying not counted; "
                f"last phase {slot.get('load_phase')})",
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

    def requeue(self, key: str, why: str) -> None:
        """Back to pending without blame: the recipe did not get a fair run."""
        entry = self.state.entry(key)
        entry["status"] = "pending"
        slot = self.state.slots.get(key)
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
            if error.infrastructure():
                self.note_infra("endpoint", str(error))
                return  # the run is up: look the endpoint up again on the next pass
            self._fail_slot(
                key, slot, policy.classify("readiness", "endpoint", str(error)), {}
            )
            return
        base = next(
            (
                dig(item, "endpoint", "api_base")
                for item in dig(endpoints, "assignments", default=[])
                if isinstance(item, dict) and item.get("alias") == slot["alias"]
            ),
            None,
        )
        if not isinstance(base, str):
            self._fail_slot(
                key,
                slot,
                policy.classify(
                    "readiness", "endpoint.unpublished", "no published API endpoint"
                ),
                {},
            )
            return
        slot["smoke_deadline"] = self.clock.now() + SMOKE_WALL_SECONDS
        self.futures[key] = self.executor.submit(
            smoke_service, recipe, self.defs, base, slot["alias"], self.cfg.http
        )

    def _advance_smoking(self, key: str, slot: dict[str, Any]) -> None:
        future = self.futures.get(key)
        if (
            future is None
        ):  # resumed after a restart: the smoke is idempotent, run it again
            self._submit_smoke(key, slot)
            return
        if not future.done():
            if self.clock.now() >= float(
                slot.get("smoke_deadline", slot["ready_at"] + SMOKE_WALL_SECONDS)
            ):
                future.cancel()
                del self.futures[key]
                self.requeue(key, "smoke.observation_budget_exhausted")
                self.state.entry(key)["not_before"] = (
                    self.clock.now() + self.cfg.retry_delay
                )
                slot["phase"] = "finished"
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
            self._record_pass(key, slot, result)
        else:
            assert result.failure is not None
            self._record_failure(key, result.failure, slot.get("operation_id"))

    def preempt(self, why: str) -> None:
        for key, slot in list(self.state.slots.items()):
            future = self.futures.pop(key, None)
            if future is not None:
                future.cancel()
            self.requeue(key, why)
            slot["phase"] = "finished"
        self.state.data["loads"].clear()

    # ------------------------------------------------------------- scheduling

    def cleanup_finished(self) -> None:
        pending = self.state.data.setdefault("cleanup", {})
        for key, slot in list(self.state.slots.items()):
            if slot["phase"] != "finished":
                continue
            pending.setdefault(slot["alias"], {"attempts": 0, "next_check": 0.0})
            self.futures.pop(key, None)
            del self.state.slots[key]
        for alias, record in list(pending.items()):
            if self.clock.now() < record["next_check"]:
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
        return [bin_ for bin_ in bins.values() if bin_.id not in busy]

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
        key = load.get("request_key")
        if any(
            s.get("request_key") == key and s["phase"] != "finished"
            for s in self.state.slots.values()
        ):
            return
        app = load.get("app_id")
        if app and load.get("state") not in SETTLED:
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
            f"{len(self.state.data['loads'])})"
        )
        if not self.state.data.get("alerts"):
            self.state.event(message)
        self.state.data["alerts"] = [message]

    def schedule(self, now: float) -> None:
        self.release_orphan_load()
        if now < self.backoff_until:
            return
        candidates = self.ready_candidates()
        if not candidates:
            return
        bins = self._bins()
        mode = policy.choose_mode(
            self.state.data["mode"],
            sum(1 for r in candidates if r.node_count == 1),
            sum(1 for r in candidates if r.node_count == 2),
            self.cfg.dual_batch_min,
        )
        self.state.data["mode"] = mode
        alias_of = lambda r: alias_for(r, self.defs)
        if mode == "dual":
            dual = policy.place_dual(candidates, bins, self.cfg.reserve_bytes)
            placements = [dual] if dual else []
            if not placements:
                # No dual fits now: free lanes run ready singles instead of idling.
                placements = policy.place_singles(
                    candidates, bins, self.cfg.reserve_bytes, alias_of
                )
        else:
            placements = policy.place_singles(
                candidates, bins, self.cfg.reserve_bytes, alias_of
            )
        if not placements and not self.state.slots:
            # Declared demand beyond one Spark: let the platform's review decide.
            head = candidates[0]
            width = head.node_count
            spark_ids = tuple(b.id for b in bins[:width])
            if len(spark_ids) == width:
                placements = [policy.Placement(head, spark_ids)]
        if placements and self._take_over(now):
            self._apply(placements, now)

    def _take_over(self, now: float) -> bool:
        """Stop whatever else the Sparks run, once (again after an owner load).

        A profile load replaces the whole fleet's workloads anyway; doing it as
        its own step (an empty sweep profile) means every review sees idle
        Sparks instead of arguing about someone else's workload. ``run --yes``
        is the operator's consent. Returns False to try again on the next tick.
        """
        if self.state.data.get("took_over") or self.state.slots:
            return True
        try:
            self.fleet = fetch_fleet(self.vk, self.cfg.default_spark_memory)
        except VonkctlError:
            return False
        if not self.fleet.presences:
            self.state.data["took_over"] = True
            return True
        self.state.event(
            "stopping what the Sparks run: loading the empty sweep profile"
        )
        self.state.data["dirty"] = True
        self.state.save()
        for attempt in range(1, CLEARING_ATTEMPTS + 1):
            try:
                submitted = self._submit_fresh_load(self.cfg.sweep_profile, "takeover")
            except VonkctlError as error:
                self.state.event(f"could not clear the Sparks: {str(error)[:150]}")
                self.backoff_until = now + self.cfg.retry_delay
                if error.infrastructure(reviewing=True):
                    self.note_infra("review", str(error))
                return False
            application = str(submitted.get("id"))
            status = self._await_application(application, self.clock.now())
            if status not in SETTLED:
                # Bounded: the clearing load never settled. It is ours: cancel it and start over.
                self.state.event(
                    f"the clearing load {application} did not settle in time: cancelling it"
                )
                self.note_infra(
                    "clearing",
                    f"clearing load {application} stuck ({status or 'unknown'})",
                )
                try:
                    self.vk.run(
                        "profile",
                        "cancel",
                        application,
                        "--yes",
                        "--detach",
                        profile=self.cfg.sweep_profile,
                    )
                except VonkctlError as error:
                    self.state.event(
                        f"could not cancel the clearing load: {str(error)[:150]}"
                    )
                self.backoff_until = self.clock.now() + self.cfg.retry_delay
                return False
            if status != "succeeded":
                self.state.event(
                    f"clearing the Sparks did not succeed (last state {status})"
                )
                self.note_infra("clearing", f"clearing load ended {status}")
                self.backoff_until = self.clock.now() + self.cfg.retry_delay
                return False
            if self._wait_idle():
                self.clear_infra("clearing")
                self.state.data["took_over"] = True
                return True
            self.state.event(
                f"the Sparks still run something after clearing load {attempt}: trying again"
            )
        # Bounded: read the fleet again. Whatever still runs is reported; the next pass retries.
        names = ", ".join(sorted({p.alias for p in self.fleet.presences})) or "nothing"
        self.note_infra(
            "clearing",
            f"the Sparks still run workloads after {CLEARING_ATTEMPTS} clearing loads: {names}",
        )
        self.backoff_until = self.clock.now() + self.cfg.retry_delay
        return False

    def _await_application(self, application_id: str, started: float) -> str:
        """Wait for an application to settle; bounded, and never silent. Returns its last state."""
        status = ""
        deadline = started + CLEAR_TIMEOUT_SECONDS
        while self.clock.now() < deadline:
            reply = self.vk.run(
                "profile",
                "progress",
                "--application",
                application_id,
                profile=self.cfg.sweep_profile,
            )
            status = str(dig(reply.document, "state", default=""))
            replaced = (
                policy.supersession(reply.document)
                if status in SETTLED and isinstance(reply.document, dict)
                else None
            )
            if replaced is not None and replaced.successor:
                application_id = replaced.successor
                self.waiting(f"application {application_id}", started)
                self.clock.sleep(self.cfg.poll_seconds)
                continue
            if status in SETTLED:
                self.done_waiting()
                return status
            self.waiting(
                f"application {application_id} ({status or 'unknown'})", started
            )
            self.clock.sleep(self.cfg.poll_seconds)
        return status if status in SETTLED else ""

    def _wait_idle(self) -> bool:
        """Did the load really empty the Sparks? Look at the fleet, not at the application."""
        started = self.clock.now()
        deadline = started + IDLE_WAIT_SECONDS
        while self.clock.now() < deadline:
            try:
                self.fleet = fetch_fleet(self.vk, self.cfg.default_spark_memory)
                if not self.fleet.presences:
                    self.done_waiting()
                    return True
            except VonkctlError:
                pass
            if self.clock.now() >= deadline:
                self.done_waiting()
                return False
            self.waiting("the Sparks to become idle", started)
            self.clock.sleep(self.cfg.poll_seconds)

        self.done_waiting()
        return False

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
        taken = {
            str(s["alias"]) for k, s in self.state.slots.items() if k != recipe.key
        } | {a for k, a in self._assigned.items() if k != recipe.key}
        if alias in taken:
            alias = unique_profile_alias(alias, recipe.key)
        self._assigned[recipe.key] = alias
        return alias

    def _apply(self, placements: Sequence[policy.Placement], now: float) -> None:
        profile = self.cfg.sweep_profile
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
                if error.infrastructure(reviewing=True):
                    self._rollback(added, f"profile edit failed: {error}")
                    self.note_infra("profile-edit", str(error))
                    return
                refused = policy.refused_profile_field(str(error))
                if refused is not None and refused not in policy.RECIPE_PROFILE_FIELDS:
                    # The Controller refused something the sweep itself sent (the
                    # name, the Sparks, the state): nothing about this recipe.
                    self._rollback(added, f"profile edit refused {refused}: {error}")
                    self.note_infra("profile-edit", str(error))
                    return
                if refused is not None:
                    self._record_failure(
                        recipe.key,
                        policy.classify(
                            "profile-edit", error.code, str(error), klass="recipe-data"
                        ),
                        None,
                    )
                    continue
                if error.infrastructure(reviewing=True):
                    # The client, protocol or transport failed: nothing about this recipe.
                    self._rollback(added, f"profile edit failed: {error}")
                    self.note_infra("review", str(error))
                    return
                self._record_failure(
                    recipe.key, policy.classify("review", error.code, str(error)), None
                )
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
            if is_infrastructure(review, reviewing=True):
                self.note_infra("review", review.error_text)
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
            if error.infrastructure(reviewing=True):
                self.note_infra("review", str(error))
                # Unknown admission is resolved by request UUID before any edit or replay.
                self.state.data["loads"][key]["state"] = "observing"
                return
            self.state.data["loads"].pop(key, None)
            for placement in added:
                self.state.slots.pop(placement.recipe.key, None)
            self._rollback(added, f"load refused: {error}")
            return
        self.state.data["loads"][key]["app_id"] = submitted.get("id")
        self.state.data["dirty"] = True
        self.state.event("load submitted: " + ", ".join(p.recipe.key for p in added))

    def _adopted_requests(self, review: Mapping[str, Any]) -> list[str] | None:
        """Require bound whole-assignment adoption before overlapping accepted snapshots."""
        adopted = (review.get("effects") or {}).get("adopted") or []
        requests = []
        for slot in self.state.slots.values():
            if slot["phase"] != "loading":
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

    def _adopted_slots_live(self, load: Mapping[str, Any]) -> bool:
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
            "wall_deadline": now + LOAD_WALL_SECONDS,
            "model_bytes": model_bytes,
            "digests": list(recipe.model_digests),
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
            except VonkctlError:
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
            # recipe's. Clear the fleet again, then retry. A passed lane's workload keeps
            # running after its slot is released: that one is ours, not an owner's.
            self.state.data["took_over"] = False
            self.backoff_until = now
            self.state.event(
                "review blocked by a workload that is not ours: clearing"
                if foreign
                else "review blocked by our own finished workload "
                + ", ".join(leftover)
                + ": clearing"
            )
            return
        for placement in blocked:
            key = placement.recipe.key
            entry = self.state.entry(key)
            code, detail = reasons.get(
                self._alias(placement.recipe), [("review.blocked", "")]
            )[0]
            cache = "cache" in code or "not-cached" in detail or "not cached" in detail
            if cache and int(entry.get("review_defers", 0)) < MAX_REVIEW_DEFERS:
                entry["review_defers"] = int(entry.get("review_defers", 0)) + 1
                entry["defer_until"] = now + self.cfg.defer_delay
            elif busy:
                entry["defer_until"] = now + self.cfg.defer_delay
            else:
                failure = policy.classify("review", code, detail)
                freeing = int(entry.get("review_freeing", 0))
                if policy.freeable_refusal(failure) and freeing < MAX_REVIEW_FREEING:
                    # Not a terminal verdict yet: stop what holds the room, evict, and
                    # review again. Bounded, so a recipe that really does not fit is
                    # still recorded (with the attempts in its evidence).
                    entry["review_freeing"] = freeing + 1
                    entry["defer_until"] = now + self.cfg.defer_delay
                    self.state.data["took_over"] = False
                    self.backoff_until = now
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
                self._record_failure(key, failure, None)

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
        except VonkctlError:
            return [], []
        lanes = {str(s["alias"]) for s in self.state.slots.values()}
        own = self._own_aliases()
        names = sorted({p.alias for p in self.fleet.presences} - lanes)
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
        if load.get("app_id"):
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

    def _known_applications(self) -> set[str]:
        known = {
            str(item["application_id"])
            for item in self.state.data["own_loads"]
            if item.get("application_id")
        }
        known |= {
            str(load["app_id"])
            for load in self.state.data["loads"].values()
            if load.get("app_id")
        }
        known |= {
            item["id"]
            for item in self.state.data["owner"]["baseline"].values()
            if item.get("id")
        }
        return known

    def _submit_fresh_load(
        self, profile: int, kind: str, *, owner: bool = False
    ) -> Any:
        """Submit a load of our own and make sure the Controller started a *new* application.

        An application that is not new (an id seen before, or a request key that is not ours)
        is stale: nothing happened. Try again with another key, a few times.
        """
        for _ in range(CLEARING_ATTEMPTS):
            known = self._known_applications()
            key = self._own_key(kind, profile)
            document = self._submit_load(profile, key, kind, owner=owner)
            returned = dig(document, "request_key", default=key)
            if document.get("id") not in known and returned == key:
                return document
            self.state.event(
                f"the Controller returned an old application for the {kind} load "
                f"({dig(document, 'id')}): trying again with a new request key"
            )
        raise RuntimeError(
            f"the Controller keeps answering the {kind} load of profile {profile} with an "
            "application that is not new; nothing was started"
        )

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
            del own[:-200]
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
        if isinstance(document.get("id"), str):
            record["application_id"] = document["id"]
            self.state.save()
        return document

    def _leave_fleet(self) -> None:
        number = self.cfg.restore_owner
        if not self.state.data.get("dirty"):
            return  # the sweep never changed what the Sparks run
        try:
            if number is not None:
                self._submit_fresh_load(number, "restore-owner", owner=True)
                self.state.event(f"restored owner profile {number}")
                self.state.data["dirty"] = False
            elif self.cfg.stop_at_end:
                self._submit_fresh_load(self.cfg.sweep_profile, "stop")
                self.state.data["dirty"] = False
        except (VonkctlError, RuntimeError) as error:
            self.state.event(f"leaving the fleet failed: {str(error)[:150]}")


OWNER_PROFILES_READ = (1, 2, 3)
SWEEP_LABEL = ("purpose", "hardware-sweep")


def _done_future(result: SmokeResult) -> Future[SmokeResult]:
    future: Future[SmokeResult] = Future()
    future.set_result(result)
    return future
