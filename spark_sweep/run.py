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

import time
from collections.abc import Iterable, Mapping, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
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
    fetch_models,
    fetch_recipes,
    serving_run,
)
from .definitions import Definitions
from .owner import OwnerGuard, OwnerStatus
from .prefetch import PrefetchConfig, Prefetcher
from .report import write_status
from .smoke import HttpConfig, SmokeResult, smoke_readiness, smoke_service
from .state import ResultsLog, State
from .vonkctl import Vonkctl, VonkctlError, request_key

ACTIVE_APP = frozenset({"queued", "running"})
SETTLED = frozenset({"succeeded", "failed", "cancelled", "waiting-for-operator"})


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
    seed: frozenset[str] = frozenset()
    only: tuple[str, ...] = ()
    limit: int = 0
    variants_last: bool = False
    revalidate_passes: bool = True
    restore_owner: int | None = None
    stop_at_end: bool = True
    watch_seconds: float = 0.0


def alias_for(recipe: Recipe, definitions: Definitions) -> str:
    """The name the lane serves under: the reviewed service alias, else the slug."""
    return definitions.alias(recipe.key) or recipe.slug[:60]


class Sweep:
    def __init__(
        self,
        config: SweepConfig,
        vk: Vonkctl,
        state: State,
        log: ResultsLog,
        definitions: Definitions,
        clock: Clock | None = None,
        executor: ThreadPoolExecutor | None = None,
    ) -> None:
        self.cfg = config
        self.vk = vk
        self.state = state
        self.log = log
        self.defs = definitions
        self.clock = clock or RealClock()
        self.executor = executor or ThreadPoolExecutor(
            max_workers=4, thread_name_prefix="smoke"
        )
        self.futures: dict[str, Future[SmokeResult]] = {}
        self.recipes: dict[str, Recipe] = {}
        self.models: dict[str, Model] = {}
        self.fleet = Fleet((), ())
        self.catalog_at = -1e18
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
            vk, state, config.prefetch, self.rate, self.clock.now, self._download_failed
        )
        self.boost = self._build_boost()
        self.queue: list[str] = []
        self.sizes: dict[str, int] = {}
        self.last_prefetch: dict[str, Any] = {}
        self.idle_ticks = 0

    # ------------------------------------------------------------------ setup

    def _build_boost(self) -> policy.Boost:
        keyword = policy.keyword_boost(self.cfg.boost)
        seed = self.cfg.seed

        def boost(recipe: Recipe) -> float:
            return keyword(recipe) * (
                1000.0 if recipe.slug in seed or recipe.key in seed else 1.0
            )

        return boost

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

    def refresh_catalog(self) -> None:
        now = self.clock.now()
        try:
            if not self.fleet.sparks:
                self.fleet = fetch_fleet(self.vk, self.cfg.default_spark_memory)
            recipes, self.library_commit = fetch_recipes(self.vk)
            self.models = fetch_models(self.vk)
        except VonkctlError as error:
            self.state.event(f"library refresh failed: {str(error)[:120]}")
            self.catalog_at = now
            return
        self.recipes = {r.key: r for r in recipes}
        self.catalog_at = now
        self.sizes = policy.build_sizes(recipes, self.models)
        usable = len(self.sparks())
        for recipe in recipes:
            self._reconcile_entry(recipe, usable)

    def _selected(self, recipe: Recipe) -> bool:
        return not self.cfg.only or any(word in recipe.key for word in self.cfg.only)

    def _reconcile_entry(self, recipe: Recipe, usable_sparks: int) -> None:
        if not self._selected(recipe):
            return
        entry = self.state.entry(recipe.key)
        if recipe.node_count > usable_sparks:
            entry.update(
                status="skipped",
                reason=f"not testable on this fleet: needs {recipe.node_count} Sparks, {usable_sparks} usable",
                content_sha256=recipe.content_sha256,
            )
            return
        if entry.get("status") == "skipped":
            entry["status"] = "pending"
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
        now = self.clock.now()
        if now - self.catalog_at >= self.cfg.catalog_seconds:
            self.refresh_catalog()
        if now - self.owner_at >= self.cfg.owner_poll_seconds:
            self.owner_at = now
            self.owner_status = self.guard.check(now)
            if self.owner_status.new_activity:
                self.state.data["owner"]["seen_at"] = now
                self.preempt("an owner profile was loaded")
        self.advance_slots(now)
        pending = self.pending()
        self.last_prefetch = self.prefetcher.tick(
            self.recipes,
            self.models,
            pending,
            self.sizes,
            policy.present_models(self.models),
            self.boost,
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

    def _deprioritised(self, pending: Iterable[Recipe]) -> set[str]:
        failures = self.state.data["model_failures"]
        return {r.key for r in pending if any(d in failures for d in r.model_digests)}

    def done(self) -> bool:
        if self.state.slots:
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

    def run(self) -> int:
        self.preflight()
        stuck = 0
        try:
            while True:
                self.tick()
                if self.done():
                    if self.cfg.watch_seconds <= 0:
                        break
                    self.clock.sleep(self.cfg.watch_seconds)
                    self.catalog_at = -1e18
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

    def _nothing_moves(self) -> bool:
        busy = bool(self.state.slots) or any(
            r.get("state") in ("queued", "running", "partial", "accepted")
            or (r.get("retry_at") and r.get("state") == "failed")
            for r in self.state.downloads.values()
        )
        waiting = self.owner_status.paused or self.clock.now() < max(
            self.backoff_until, self.prefetcher.pressure_until
        )
        return not busy and not waiting and not self.ready_candidates()

    # --------------------------------------------------------------- failures

    def _download_failed(
        self, key: str, failure: policy.Failure, op_id: str | None, model_level: bool
    ) -> None:
        recipe = self.recipes.get(key)
        self._record_failure(key, failure, op_id)
        if model_level and recipe is not None:
            for digest in recipe.model_digests:
                self.state.data["model_failures"][digest] = failure.cluster
            for other in self.recipes.values():
                if (
                    other.key != key
                    and other.model_set == recipe.model_set
                    and self.state.status(other.key) == "pending"
                ):
                    self._record_failure(other.key, failure, op_id, inherited_from=key)

    def _record_failure(
        self,
        key: str,
        failure: policy.Failure,
        op_id: str | None,
        *,
        inherited_from: str | None = None,
        penalise: bool = True,
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
        if penalise:
            entry["attempts"] = int(entry.get("attempts", 0)) + 1
        if slot is not None:
            slot["phase"] = "finished"
        if penalise and failure.transient and entry["attempts"] < self.cfg.max_attempts:
            entry.update(
                status="pending",
                not_before=now + self.cfg.retry_delay,
                last_failure=failure.signature,
            )
            self.state.event(f"retry {key}: {failure.klass} ({failure.phase})")
            return
        evidence = (
            self._evidence(key, op_id, entry["attempts"])
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
        if recipe.local != "cached":
            return False
        return all(
            (model := self.models.get(d)) is not None and model.local == "cached"
            for d in recipe.model_digests
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
        if placements:
            self._apply(placements, now)

    def _apply(self, placements: Sequence[policy.Placement], now: float) -> None:
        profile = self.cfg.sweep_profile
        added: list[policy.Placement] = []
        names = {s.id: s.name for s in self.sparks()}
        for placement in placements:
            recipe = placement.recipe
            args = ["add", recipe.key]
            for spark_id in placement.spark_ids:
                args += ["--spark", names[spark_id]]
            args += [
                "--as",
                alias_for(recipe, self.defs),
                "--state",
                "running",
                "--yes",
            ]
            try:
                self.vk.call("profile", *args, profile=profile)
                added.append(placement)
            except VonkctlError as error:
                self._record_failure(
                    recipe.key, policy.classify("review", error.code, str(error)), None
                )
        if not added:
            return
        aliases = {alias_for(p.recipe, self.defs): p for p in added}
        kept = {
            s["alias"] for s in self.state.slots.values() if s["phase"] == "smoking"
        }
        review = self.vk.run("profile", "load", "--review", profile=profile)
        document = review.document if isinstance(review.document, dict) else {}
        if review.is_error_document or not document:
            self._rollback(added, f"review failed: {review.error_text}")
            return
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
            self.cfg.sweep_profile,
            self.state.data.get("started_at", 0),
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
            submitted = self.vk.call(
                "profile",
                "load",
                "--yes",
                "--detach",
                "--request-key",
                key,
                profile=profile,
            )
        except VonkctlError as error:
            self.state.data["load"] = None
            self._rollback(added, f"load refused: {error}")
            return
        self.state.data["load"]["app_id"] = dig(submitted, "id")
        self.state.data["loaded_any"] = True
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
            "alias": alias_for(recipe, self.defs),
            "node_ids": list(placement.spark_ids),
            "spark_names": [names.get(i, i) for i in placement.spark_ids],
            "phase": "loading",
            "request_key": request,
            "started_at": now,
            "deadline": now + timeout,
            "model_bytes": model_bytes,
        }

    def _rollback(self, placements: Sequence[policy.Placement], why: str) -> None:
        for placement in placements:
            try:
                self.vk.call(
                    "profile",
                    "remove",
                    alias_for(placement.recipe, self.defs),
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
                    alias_for(placement.recipe, self.defs),
                    generic or [("review.blocked", "the review was not allowed")],
                )
        self._rollback(
            blocked, "review blocked: " + ", ".join(p.recipe.key for p in blocked)
        )
        for placement in blocked:
            key = placement.recipe.key
            code, detail = reasons.get(
                alias_for(placement.recipe, self.defs), [("review.blocked", "")]
            )[0]
            if busy:
                self.state.entry(key)["defer_until"] = now + self.cfg.defer_delay
            else:
                self._record_failure(key, policy.classify("review", code, detail), None)

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

    def _leave_fleet(self) -> None:
        number = self.cfg.restore_owner
        try:
            started = self.state.data.get("started_at", 0)
            if number is not None:
                key = request_key(
                    "restore-owner", number, started, self.state.data["load_seq"]
                )
                self.vk.call(
                    "profile",
                    "load",
                    "--yes",
                    "--detach",
                    "--request-key",
                    key,
                    profile=number,
                    allow_owner_write=True,
                )
                self.state.event(f"restored owner profile {number}")
            elif self.cfg.stop_at_end and self.state.data.get("loaded_any"):
                key = request_key(
                    "sweep-stop",
                    self.cfg.sweep_profile,
                    started,
                    self.state.data["load_seq"],
                )
                self.vk.call(
                    "profile",
                    "load",
                    "--yes",
                    "--detach",
                    "--request-key",
                    key,
                    profile=self.cfg.sweep_profile,
                )
        except VonkctlError as error:
            self.state.event(f"leaving the fleet failed: {str(error)[:150]}")


OWNER_PROFILES_READ = (1, 2, 3)
SWEEP_LABEL = ("purpose", "hardware-sweep")


def _done_future(result: SmokeResult) -> Future[SmokeResult]:
    future: Future[SmokeResult] = Future()
    future.set_result(result)
    return future
