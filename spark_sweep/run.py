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
MAX_REVIEW_DEFERS = 3
INFRA_BASE = 30.0
INFRA_CAP = 600.0
ACTIVE_APP = frozenset({"queued", "running"})
SETTLED = frozenset({"succeeded", "failed", "cancelled", "waiting-for-operator"})


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

    # ------------------------------------------------------------------ setup

    def _build_boost(self) -> policy.Boost:
        return policy.make_boost(
            self.cfg.boost, self.cfg.seed, self.clock.now(), self.cfg.recent_days
        )

    def preflight(self) -> None:
        """Read-only facts before anything is changed: fleet, owner baseline, owner profile export."""
        self.fleet = fetch_fleet(self.vk, self.cfg.default_spark_memory)
        if not any(s.online for s in self.fleet.sparks):
            raise RuntimeError("no Spark is online")
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
        definition = (
            reply.document if reply.ok and isinstance(reply.document, dict) else {}
        )
        labels = definition.get("labels") or {}
        if labels.get(SWEEP_LABEL[0]) == SWEEP_LABEL[1]:
            return
        if definition.get("assignments"):
            raise RuntimeError(
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
        if not self.fleet.sparks:
            try:
                self.fleet = fetch_fleet(self.vk, self.cfg.default_spark_memory)
            except VonkctlError:
                return
        self.sizes = policy.build_sizes(self.recipes.values(), self.models)
        usable = len(self.sparks())
        for recipe in list(self.recipes.values()):
            self._reconcile_entry(recipe, usable)

    def _selected(self, recipe: Recipe) -> bool:
        return not self.cfg.only or any(word in recipe.key for word in self.cfg.only)

    def _reconcile_entry(self, recipe: Recipe, usable_sparks: int) -> None:
        if not self._selected(recipe):
            return
        entry = self.state.entry(recipe.key)
        if recipe.node_count > usable_sparks and entry.get("status") in (
            "pending",
            "skipped",
        ):
            entry.update(
                status="skipped",
                reason=f"not testable on this fleet: needs {recipe.node_count} Sparks, {usable_sparks} usable",
                content_sha256=recipe.content_sha256,
            )
            return
        if entry.get("status") == "skipped" and recipe.node_count <= usable_sparks:
            entry["status"] = "pending"
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
        for stale in ("phase", "failure_class", "signature", "cluster", "error"):
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

    def tick(self) -> None:
        """One pass. A Controller that stops answering costs a pass, never the sweep."""
        try:
            self._tick()
        except VonkctlError as error:
            self.state.event(f"vonkctl failed, will retry: {str(error)[:150]}")
            self.state.save()

    def _tick(self) -> None:
        now = self.clock.now()
        self.advance_catalog(now)
        self.check_release(now)
        if now - self.owner_at >= self.cfg.owner_poll_seconds:
            self.owner_at = now
            self.owner_status = self.guard.check(now)
            if self.owner_status.new_activity:
                self.state.data["took_over"] = (
                    False  # their load replaced what ours ran
                )
                self.state.data["owner"]["seen_at"] = now
                self.preempt("an owner profile was loaded")
        self.advance_slots(now)
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
        if any(
            r.get("state") in ("queued", "running", "partial")
            for r in self.state.downloads.values()
        ):
            return False
        return not any(self._open(k) for k in self.recipes)

    def _open(self, key: str) -> bool:
        entry = self.state.recipes.get(key)
        return bool(
            entry
            and entry.get("status") == "pending"
            and self._selected(self.recipes[key])
        )

    def start_takeover(self) -> None:
        """``run --yes`` means the sweep owns the fleet: clear it once, now, so reviews see idle Sparks."""
        while self.owner_status.paused:  # an owner load is in progress: wait for it
            self.clock.sleep(self.cfg.poll_seconds)
            self.owner_status = self.guard.check(self.clock.now())
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
        message = (
            f"vonkctl {current} differs from the accepted release {accepted}: "
            "run `vonkctl update --apply`"
        )
        if not self.cfg.allow_version_skew:
            raise RuntimeError(message + " (or pass --allow-version-skew)")
        self.state.event(f"WARNING: {message}; continuing as asked")

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
        reply = self.vk.run("update", timeout=60)
        if reply.ok and isinstance(reply.document, dict):
            self.observe_release(reply.document)

    def requeue_for_release(self) -> None:
        """Failures from another Controller release, of a platform-side class, are tried again."""
        current = self.release_sha
        if current is None:
            return
        for key, entry in self.state.recipes.items():
            if (
                entry.get("status") == "failed"
                and entry.get("release") != current
                and policy.platform_side(entry.get("failure_class"))
                and self._selected_key(key)
            ):
                self._requeue_failed(key, f"controller-release-changed ({current})")

    def requeue_failed(self) -> int:
        """``--retry-failed``: every failed recipe (within ``--only``), whatever the cause."""
        keys = [
            key
            for key, entry in self.state.recipes.items()
            if entry.get("status") == "failed" and self._selected_key(key)
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
        entry["previous"] = {"status": "failed", **previous}
        entry.update(status="pending", attempts=0, requeued_because=why)
        for stale in (
            "phase",
            "failure_class",
            "code",
            "error",
            "signature",
            "cluster",
            "evidence",
            "inherited_from",
            "not_before",
            "defer_until",
            "release",
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
            if self.cfg.retry_failed:
                self.requeue_failed()
            self.preflight()
            self.start_takeover()
            while True:
                self.tick()
                if self.done():
                    if self.cfg.watch_seconds <= 0:
                        break
                    self.clock.sleep(self.cfg.watch_seconds)
                    self.recipe_list.pass_done_at = self.model_list.pass_done_at = -1e18
                    continue
                stuck = stuck + 1 if self._nothing_moves() else 0
                if stuck >= 30:
                    self.state.event(
                        "nothing is runnable and nothing is downloading: stopping"
                    )
                    break
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
            r.get("state") in ("queued", "running", "partial", "accepted")
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
        if model_level and recipe is not None:
            for digest in recipe.model_digests:
                self.state.data["model_failures"][digest] = failure.cluster
            for other in self.recipes.values():
                if (
                    other.key != key
                    and other.model_set == recipe.model_set
                    and self.state.status(other.key) == "pending"
                ):
                    self._record_failure(
                        other.key, failure, op_id, inherited_from=key, retryable=False
                    )

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
        evidence = (
            self._evidence(key, op_id, entry["evidence_seq"])
            if inherited_from is None
            else None
        )
        entry.update(
            status="failed",
            phase=failure.phase,
            failure_class=failure.klass,
            code=failure.code,
            error=failure.detail,
            signature=failure.signature,
            cluster=failure.cluster,
            finished_at=now,
            release=self.release_sha,
            content_sha256=recipe.content_sha256
            if recipe
            else entry.get("content_sha256"),
            revision_id=recipe.revision_id if recipe else entry.get("revision_id"),
        )
        if inherited_from:
            entry["inherited_from"] = inherited_from
        if evidence:
            entry["evidence"] = evidence
        if failure.model_level and recipe is not None:
            for digest in recipe.model_digests:
                self.state.data["model_failures"][digest] = failure.cluster
        self.log.append(
            batch="sweep",
            recipe=key,
            step="smoke",
            status="failed",
            error=f"{failure.phase}/{failure.klass}: {failure.detail}"[:1024],
            **self._facts(key, entry, slot),
        )
        self.state.event(f"FAILED {key}: {failure.phase}/{failure.klass}")

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
            revalidate=False,
        )
        for stale in (
            "phase",
            "failure_class",
            "signature",
            "cluster",
            "error",
            "evidence",
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
        self.state.event(f"passed {key}")

    # ------------------------------------------------------------------ slots

    def poll_application(self) -> None:
        load = self.state.data["load"]
        if not load:
            return
        args = ["progress"]
        if load.get("app_id"):
            args += ["--application", load["app_id"]]
        else:
            args += ["--request-key", load["request_key"]]
        reply = self.vk.run("profile", *args, profile=self.cfg.sweep_profile)
        document = reply.document
        if not isinstance(document, dict) or not isinstance(document.get("id"), str):
            return
        load["app_id"] = document["id"]
        status = str(document.get("state", ""))
        load["state"] = status
        if status in SETTLED:
            child = dig(document, "progress", "child_progress", "phase")
            self.state.data.setdefault("apps", {})[load["request_key"]] = {
                "state": status,
                "reason": document.get("status_reason"),
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
            self.state.data["load"] = None

    def advance_slots(self, now: float) -> None:
        if not self.state.slots:
            self.poll_application()
            return
        try:
            self.fleet = fetch_fleet(self.vk, self.cfg.default_spark_memory)
        except VonkctlError:
            pass
        self.poll_application()
        for key, slot in list(self.state.slots.items()):
            if slot["phase"] == "loading":
                self._advance_loading(key, slot, now)
            elif slot["phase"] == "smoking":
                self._advance_smoking(key, slot)

    def _advance_loading(self, key: str, slot: dict[str, Any], now: float) -> None:
        recipe = self.recipes[key]
        run_id = serving_run(
            self.fleet, slot["alias"], slot["node_ids"], recipe.node_count
        )
        if run_id:
            slot.update(phase="smoking", run_id=run_id, ready_at=now)
            policy.learn(
                self.state.data["learned"],
                recipe.engine,
                slot["model_bytes"],
                now - slot["started_at"],
            )
            self._submit_smoke(key, slot)
            return
        app = self.state.data.get("apps", {}).get(slot["request_key"])
        if app is not None:
            if app["state"] == "cancelled" and app.get("cause") == "superseded":
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
            phase = policy.child_phase(app.get("child_phase")) or "start"
            detail = f"{app.get('reason') or app['state']} {app.get('failures') or ''}".strip()
            self._fail_slot(
                key,
                slot,
                policy.classify(phase, f"application.{app['state']}", detail),
                app,
            )
            return
        if now > slot["deadline"]:
            self._timeout(key, slot)

    def _timeout(self, key: str, slot: dict[str, Any]) -> None:
        load = self.state.data["load"]
        others = [
            k
            for k, s in self.state.slots.items()
            if k != key and s["phase"] == "loading"
        ]
        if load and load.get("app_id") and not others:
            self.vk.run(
                "profile",
                "cancel",
                load["app_id"],
                "--yes",
                "--detach",
                profile=self.cfg.sweep_profile,
            )
        seconds = round(self.clock.now() - slot["started_at"])
        self._fail_slot(
            key,
            slot,
            policy.classify(
                "timeout",
                "load.timeout",
                f"not serving after {seconds}s (limit {round(slot['deadline'] - slot['started_at'])}s)",
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
        self.state.data["load"] = None

    # ------------------------------------------------------------- scheduling

    def cleanup_finished(self) -> None:
        for key, slot in list(self.state.slots.items()):
            if slot["phase"] != "finished":
                continue
            try:
                self.vk.call(
                    "profile",
                    "remove",
                    slot["alias"],
                    "--yes",
                    profile=self.cfg.sweep_profile,
                )
            except VonkctlError as error:
                if "absent" not in str(error):
                    self.state.event(
                        f"could not remove {slot['alias']} from the sweep profile: {str(error)[:100]}"
                    )
                    continue
            del self.state.slots[key]

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
        return list(bins.values())

    def schedule(self, now: float) -> None:
        if self.state.data["load"] is not None or now < self.backoff_until:
            return
        if any(
            s["phase"] in ("loading", "finished") for s in self.state.slots.values()
        ):
            return  # settling: a lane is between its load and its smoke
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
            status = self._await_application(str(dig(submitted, "id")), now)
            if status != "succeeded":
                raise RuntimeError(
                    f"clearing the Sparks did not succeed (last state {status or 'unknown'})"
                )
            if self._wait_idle():
                self.state.data["took_over"] = True
                return True
            self.state.event(
                f"the Sparks still run something after clearing load {attempt}: trying again"
            )
        raise RuntimeError(
            f"the Sparks still run workloads after {CLEARING_ATTEMPTS} clearing loads: "
            + ", ".join(sorted({p.alias for p in self.fleet.presences}))
        )

    def _await_application(self, application_id: str, started: float) -> str:
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
            if status in SETTLED:
                break
            self.clock.sleep(self.cfg.poll_seconds)
        return status

    def _wait_idle(self) -> bool:
        """Did the load really empty the Sparks? Look at the fleet, not at the application."""
        deadline = self.clock.now() + IDLE_WAIT_SECONDS
        while True:
            try:
                self.fleet = fetch_fleet(self.vk, self.cfg.default_spark_memory)
                if not self.fleet.presences:
                    return True
            except VonkctlError:
                pass
            if self.clock.now() >= deadline:
                return False
            self.clock.sleep(self.cfg.poll_seconds)

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
            s["alias"] for s in self.state.slots.values() if s["phase"] == "smoking"
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
        self.state.data["load"] = {
            "request_key": key,
            "app_id": None,
            "seq": seq,
            "submitted_at": now,
        }
        self.state.save()
        try:
            submitted = self._submit_load(profile, key, "placement")
        except VonkctlError as error:
            self.state.data["load"] = None
            self._rollback(added, f"load refused: {error}")
            if error.infrastructure(reviewing=True):
                self.note_infra("review", str(error))
            return
        self.state.data["load"]["app_id"] = dig(submitted, "id")
        self.state.data["dirty"] = True
        for placement in added:
            self._open_slot(placement, key, now)
        self.state.event("load submitted: " + ", ".join(p.recipe.key for p in added))

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
        self.state.slots[recipe.key] = {
            "alias": self._alias(recipe),
            "node_ids": list(placement.spark_ids),
            "spark_names": [names.get(i, i) for i in placement.spark_ids],
            "phase": "loading",
            "request_key": request,
            "started_at": now,
            "deadline": now + timeout,
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
                pass
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
        for decision in dig(document, "admission_decisions", default=[]):
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
                for r in dig(document, "reasons", default=[])
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
        if not busy and self._foreign_runs():
            # Someone else's workload is back on the Sparks (the owner loaded a profile):
            # that is the fit problem, not the recipe's. Clear the fleet again, then retry.
            self.state.data["took_over"] = False
            self.backoff_until = now
            self.state.event("review blocked by a workload that is not ours: clearing")
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
                self._record_failure(key, policy.classify("review", code, detail), None)

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
        item["until"] = now + min(INFRA_BASE * 2 ** (item["count"] - 1), INFRA_CAP)
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

    def _foreign_runs(self) -> bool:
        try:
            self.fleet = fetch_fleet(self.vk, self.cfg.default_spark_memory)
        except VonkctlError:
            return False
        ours = {s["alias"] for s in self.state.slots.values()}
        return any(p.alias not in ours for p in self.fleet.presences)

    # ---------------------------------------------------------- end of a run

    def interrupt(self) -> None:
        """Ctrl-C: cancel the lane's current load and give its recipes back to the queue."""
        load = self.state.data["load"]
        if load and load.get("app_id"):
            self.vk.run(
                "profile",
                "cancel",
                load["app_id"],
                "--yes",
                "--detach",
                profile=self.cfg.sweep_profile,
            )
        for key in list(self.state.slots):
            self.futures.pop(key, None)
        self.state.event("interrupted: lane loads cancelled, recipes requeued")
        self.preempt("sweep interrupted")
        self.state.data["load"] = None
        self.cleanup_finished()

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
        load = self.state.data["load"]
        if load and load.get("app_id"):
            known.add(str(load["app_id"]))
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
            if dig(document, "id") not in known and returned == key:
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
        if isinstance(dig(document, "id"), str):
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
