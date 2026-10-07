"""Keep the NAS ahead of the Sparks: bounded downloads, and pins against eviction.

``recipe download`` fetches a recipe's model (Hugging Face) and its runtime
image (GHCR) as children of one operation, so the two sources already run
concurrently inside an operation; the prefetcher adds concurrency across
operations. Two budgets bound it (model downloads in flight, image-only pulls
in flight) plus a byte budget for everything downloaded but not yet tested.

Nothing in vonkctl or the Controller reports the NAS's free space, so the byte
budget is the operator's (``--nas-budget``); a ``free_space`` failure also
pauses new model downloads for a while.

The Controller never evicts what a saved profile names, loaded or not. The
prefetcher therefore keeps one never-loaded *pin profile* that names every
recipe that has been downloaded (or is downloading) and not yet tested, and
removes each recipe from it once tested.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from .catalog import Model, Recipe
from .lifecycle import TERMINAL, WAITING, active
from .policy import (
    Boost,
    Failure,
    GroupPlan,
    RateTracker,
    classify,
    excerpt,
    plan_groups,
)
from .state import State
from .vonkctl import Vonkctl, VonkctlError, request_key

TIB = 1024**4
DOWNLOAD_WALL_SECONDS = 24 * 3600.0
DOWNLOAD_STALL_SECONDS = 3600.0
FailureHandler = Callable[[str, Failure, str | None, bool], None]
DoneHandler = Callable[[str], None]
MAX_REREQUESTS = (
    2  # after a success, ask again at most this often when the cache looks empty
)


@dataclass(frozen=True)
class PrefetchConfig:
    max_model_downloads: int = 3
    max_image_pulls: int = 2
    nas_budget_bytes: int = 2 * TIB
    pin_profile: int | None = None
    pins_per_tick: int = 8
    retry_cooldown: float = 120.0
    pressure_pause: float = 600.0
    group_window: int = 16
    pin_spark: str = ""  # a Spark to name in the (never loaded) pin profile


# The Controller's FleetProfileInput contract: $.assignments maxItems. The contract schema is
# not vendored in contracts/, so the limit is named here.
PIN_PROFILE_MAX_ASSIGNMENTS = 64
PIN_RETRY_MAX_SECONDS = 3600.0


PIN_ALIAS_PREFIX = "pin-"


def pin_alias(key: str) -> str:
    return PIN_ALIAS_PREFIX + hashlib.sha1(key.encode()).hexdigest()[:12]


class Prefetcher:
    def __init__(
        self,
        vk: Vonkctl,
        state: State,
        config: PrefetchConfig,
        rate: RateTracker,
        clock: Callable[[], float],
        on_failure: FailureHandler,
        on_done: DoneHandler = lambda _key: None,
    ) -> None:
        self.vk = vk
        self.state = state
        self.config = config
        self.rate = rate
        self.clock = clock
        self.on_failure = on_failure
        self.on_done = on_done
        # Set by the sweep: a client, protocol or transport problem is reported, never a recipe's failure.
        self.on_infra: Callable[[str, str], None] = lambda _source, _message: None
        self.on_ok: Callable[[str], None] = lambda _source: None
        self.pause_until = 0.0
        self.library_fresh_since = 0.0  # start of the newest complete library pass
        self.pressure_until = 0.0
        self.pin_error: str | None = None
        self.pin_failures = 0
        self.pin_retry_at = 0.0
        self.last_plans: list[GroupPlan] = []

    # -- operations ---------------------------------------------------------

    def _poll(self) -> float:
        """Refresh our operations; returns the summed live throughput."""
        total_rate = 0.0
        now = self.clock()
        for key, record in list(self.state.downloads.items()):
            if (
                record.get("state") == "external"
                or (record.get("state") == "retired" and not record.get("request_key"))
                or record.get("state") in TERMINAL
            ):
                continue
            record.setdefault("started_at", now)
            record.setdefault("progress_at", now)
            deadline = min(
                float(record["started_at"]) + DOWNLOAD_WALL_SECONDS,
                float(record["progress_at"]) + DOWNLOAD_STALL_SECONDS,
            )
            if now >= deadline and not record.get("observation_budget_exhausted_at"):
                record["observation_budget_exhausted_at"] = now
                self._retire(key, record, "download.observation_budget_exhausted")
                continue
            if now < float(record.get("next_check", 0)):
                continue
            reply = self.vk.run("recipe", "progress", *self._progress_args(record))
            if reply.is_not_found and not record.get("operation_id"):
                # Only a canonical absence before any accepted parent permits
                # replay. Keep the persisted intent and request UUID unchanged.
                intent = record.get("requested_intent")
                action: list[str] | None = None
                if isinstance(intent, Mapping):
                    if (
                        intent.get("kind") == "selector"
                        and isinstance(intent.get("selector"), str)
                        and intent.get("force") is False
                    ):
                        action = ["download", intent["selector"]]
                    elif intent.get("kind") == "retry" and isinstance(
                        intent.get("operation_id"), str
                    ):
                        action = ["retry", intent["operation_id"]]
                if action is not None:
                    reply = self.vk.run(
                        "recipe",
                        *action,
                        "--yes",
                        "--detach",
                        "--request-key",
                        str(record["request_key"]),
                    )
            document = reply.document
            record["next_check"] = now + min(self.config.retry_cooldown, 60.0)
            if not reply.ok or not isinstance(document, dict):
                continue  # keep the last truthful state; try again next tick
            if not record.get("operation_id"):
                intent = record.get("requested_intent")
                accepted = document.get("request")
                if (
                    not isinstance(document.get("id"), str)
                    or document.get("request_id") != record.get("request_key")
                    or not isinstance(intent, Mapping)
                    or not isinstance(accepted, Mapping)
                    or accepted.get("kind") != intent.get("kind")
                    or (
                        intent.get("kind") == "selector"
                        and (
                            accepted.get("selector") != intent.get("selector")
                            or accepted.get("force", False) is not False
                        )
                    )
                    or (
                        intent.get("kind") == "retry"
                        and accepted.get("operation_id") != intent.get("operation_id")
                    )
                    or intent.get("kind") not in ("selector", "retry")
                ):
                    continue
                record["accepted_intent"] = dict(accepted)
                self.on_ok("download")
            record["observed_at"] = now
            record["next_check"] = (
                now  # normal polling cadence; backoff is for unreadable replies
            )
            old_bytes = record.get("bytes_done")
            record["operation_id"] = document.get("id", record.get("operation_id"))
            observed_state = document.get("state")
            record["state"] = (
                observed_state if isinstance(observed_state, str) else "observing"
            )
            progress = document.get("progress") or {}
            record["bytes_done"] = progress.get("completed_bytes", 0)
            record["bytes_total"] = progress.get("total_bytes")
            if record["bytes_done"] != old_bytes:
                record["progress_at"] = now
            speed = (
                progress.get("smoothed_bytes_per_second")
                or progress.get("bytes_per_second")
                or 0
            )
            record["bps"] = speed
            if record["state"] in WAITING:
                self._retire(key, record, f"download.{record['state']}")
            elif active(record["state"]):
                total_rate += float(speed)
            elif record["state"] == "failed":
                self._failed(key, record, document)
            elif record["state"] in ("cancelled", "superseded"):
                self._retire(key, record, f"download.{record['state']}")
            if record["state"] == "succeeded" and not record.get("done_at"):
                record["done_at"] = self.clock()
                self.on_done(key)
        self.rate.add(total_rate)
        return total_rate

    def _retire(self, key: str, record: dict[str, Any], reason: str) -> None:
        if record.get("request_key"):
            # A read budget or waiting condition is not terminal proof. Preserve
            # the exact request and re-observe it without cancelling its effects.
            record.update(
                state="observing",
                reason=reason,
                next_check=self.clock() + self.config.retry_cooldown,
            )
            self.on_infra("download", f"{key}: {reason}; observing original request")
            self.state.event(f"download observation deferred: {key}: {reason}")
            return
        record.update(
            state="retired",
            reason=reason,
            retry_at=self.clock() + self.config.retry_cooldown,
        )
        self.on_infra("download", f"{key}: {reason}; observing and retrying")
        self.state.event(f"download released: {key}: {reason}")

    def _track_external(self, recipes: Mapping[str, Recipe], now: float) -> None:
        """Follow downloads someone else started until they land.

        The library lists models and recipes separately and a page at a time, so right
        after an external download finishes a stale model row can still say "missing".
        Remembering that we saw it in flight lets its landing count as done.
        """
        for key, recipe in recipes.items():
            record = self.state.downloads.get(key)
            if recipe.local == "preparing" and record is None:
                self.state.downloads[key] = {"state": "external", "started_at": now}
            elif record is not None and record.get("state") == "external":
                if recipe.local == "cached":
                    record.update(state="succeeded", done_at=now)
                    self.on_done(key)
                elif now >= float(record["started_at"]) + DOWNLOAD_WALL_SECONDS:
                    self._retire(
                        key, record, "download.external_observation_budget_exhausted"
                    )
                elif recipe.local in ("failed", "not_cached"):
                    del self.state.downloads[key]  # it did not land: ours to ask for

    @staticmethod
    def _progress_args(record: Mapping[str, Any]) -> list[str]:
        if record.get("operation_id"):
            return [str(record["operation_id"])]
        return ["--request-key", str(record["request_key"])]

    def _failed(
        self, key: str, record: dict[str, Any], document: Mapping[str, Any]
    ) -> None:
        failure_doc = document.get("failure") or {}
        code = str(failure_doc.get("code", ""))
        detail = str(failure_doc.get("detail", "") or document.get("detail", ""))
        actions = [str(a) for a in failure_doc.get("recovery_actions", [])]
        failure = classify(
            "download",
            code,
            detail,
            evidence={
                "operation_id": document.get("id"),
                "state": document.get("state"),
                "reason": excerpt(failure_doc.get("detail") or detail, 500),
                "recovery_actions": actions,
                "children": [
                    {
                        k: child.get(k)
                        for k in ("kind", "state", "id")
                        if child.get(k) is not None
                    }
                    for child in document.get("children", [])
                    if isinstance(child, dict) and child.get("state") == "failed"
                ][:8],
                "failure": excerpt(failure_doc, 1000),
            },
        )
        if "free_space" in actions or failure.klass == "capacity":
            self.pressure_until = self.clock() + self.config.pressure_pause
            self.state.event(f"NAS pressure on {key}: new model downloads paused")
        model_child = any(
            child.get("kind") == "model-cache" and child.get("state") == "failed"
            for child in document.get("children", [])
            if isinstance(child, dict)
        )
        record["failure"] = failure.signature
        retry = failure.transient and record.get("attempt", 1) < 2
        if retry:
            record["retry_at"] = self.clock() + self.config.retry_cooldown
            return
        self.on_failure(
            key,
            failure,
            str(document.get("id") or ""),
            model_child or failure.model_level,
        )

    def _request(self, recipe: Recipe, kind: str) -> None:
        if self.clock() < self.pause_until:
            return  # backing off after an infrastructure error
        record = self.state.downloads.get(recipe.key, {})
        attempt = int(record.get("attempt", 0)) + 1
        rerequests = int(record.get("rerequests", 0)) + (
            1 if record.get("state") == "succeeded" else 0
        )
        key = request_key(
            "download", self.state.nonce, recipe.key, recipe.content_sha256, attempt
        )
        retry_of = (
            record.get("operation_id") if record.get("state") == "failed" else None
        )
        requested_intent = (
            {"kind": "retry", "operation_id": retry_of}
            if retry_of
            else {"kind": "selector", "selector": recipe.key, "force": False}
        )
        if record:
            history = self.state.data["download_history"].setdefault(recipe.key, [])
            if not history or history[-1] != record:
                history.append(dict(record))
        # Persist the next request before effects. An interrupted reply reconnects
        # this same UUID rather than losing it or creating a different attempt.
        self.state.downloads[recipe.key] = {
            "request_key": key,
            "state": "observing",
            "kind": kind,
            "attempt": attempt,
            "requested_intent": requested_intent,
            "retry_of_operation_id": retry_of,
            "started_at": self.clock(),
            "request_intent": {
                "recipe_key": recipe.key,
                "content_sha256": recipe.content_sha256,
                "revision_id": recipe.revision_id,
            },
        }
        entry = self.state.entry(recipe.key)
        retest = entry.get("retest")
        if isinstance(retest, Mapping):
            entry["download_retest_consumed_at"] = retest.get("at")
        self.state.save()
        action = ["retry", str(retry_of)] if retry_of else ["download", recipe.key]
        try:
            document = self.vk.call(
                "recipe",
                *action,
                "--yes",
                "--detach",
                "--request-key",
                key,
                tolerate=(1, 2),
            )
        except VonkctlError as error:
            if error.infrastructure():
                self.on_infra("download", str(error))
                return
            document = (
                error.reply.document
                if error.reply and isinstance(error.reply.document, dict)
                else {}
            )
            failure = classify(
                "download",
                error.code,
                str(document.get("detail", error)),
                evidence={
                    "request_key": key,
                    "reason": excerpt(document.get("detail") or error, 500),
                    "reply": excerpt(document, 1000),
                },
            )
            self.state.downloads[recipe.key] = {
                "request_key": key,
                "state": "failed",
                "kind": kind,
                "attempt": attempt,
                "failure": failure.signature,
                "request_intent": {
                    "recipe_key": recipe.key,
                    "content_sha256": recipe.content_sha256,
                    "revision_id": recipe.revision_id,
                },
            }
            if failure.transient and attempt < 2:
                self.state.downloads[recipe.key]["retry_at"] = (
                    self.clock() + self.config.retry_cooldown
                )
            else:
                self.on_failure(recipe.key, failure, None, failure.model_level)
            return
        self.on_ok("download")
        op_state = str(document.get("state", "queued"))
        self.state.downloads[recipe.key] = {
            "operation_id": document.get("id"),
            "request_key": key,
            "state": op_state,
            "kind": kind,
            "attempt": attempt,
            "rerequests": rerequests,
            "accepted_intent": document.get("request"),
            "retry_of_operation_id": retry_of,
            "request_intent": {
                "recipe_key": recipe.key,
                "content_sha256": recipe.content_sha256,
                "revision_id": recipe.revision_id,
            },
            "started_at": self.clock(),
        }
        self.state.event(f"download {kind}: {recipe.key}")

    # -- planning -----------------------------------------------------------

    def _in_flight(
        self, recipes: Mapping[str, Recipe], models: Mapping[str, Model]
    ) -> tuple[int, int]:
        model_ops = image_ops = 0
        counted: set[str] = set()
        for key, record in self.state.downloads.items():
            if record.get("state") not in ("retired", "external") and active(
                record.get("state")
            ):
                counted.add(key)
                if record.get("kind") == "model":
                    model_ops += 1
                else:
                    image_ops += 1
        for (
            key,
            recipe,
        ) in recipes.items():  # started by someone else: do not re-request, do count
            if (
                recipe.local == "preparing"
                and key not in counted
                and self.state.downloads.get(key, {}).get("state") != "retired"
            ):
                missing = any(
                    models.get(d, Model("", d, 0, "unknown")).local != "cached"
                    for d in recipe.model_digests
                )
                if missing:
                    model_ops += 1
                else:
                    image_ops += 1
        return model_ops, image_ops

    def plan_queue(
        self,
        pending: Sequence[Recipe],
        sizes: Mapping[str, int],
        present: set[str],
        boost: Boost,
    ) -> list[GroupPlan]:
        """Plan cached ready work without observing or changing preparation operations."""
        self.last_plans = plan_groups(pending, sizes, present, boost)
        return self.last_plans

    def tick(
        self,
        recipes: Mapping[str, Recipe],
        models: Mapping[str, Model],
        pending: Sequence[Recipe],
        sizes: Mapping[str, int],
        present: set[str],
        boost: Boost,
        cached: set[str] | None = None,
    ) -> dict[str, Any]:
        now = self.clock()
        live_rate = self._poll()
        self._track_external(recipes, now)
        plans = self.plan_queue(pending, sizes, present, boost)
        model_ops, image_ops = self._in_flight(recipes, models)
        by_key = {r.key: r for r in pending}

        tracked = {
            digest
            for plan in plans
            for digest in plan.digests
            if digest in present or any(self._requested(k) for k in plan.recipes)
        }
        pinned_bytes = sum(sizes.get(d, 0) for d in tracked)

        if cached is None:
            cached = {d for d, model in models.items() if model.local == "cached"}
        for plan in plans[: self.config.group_window]:
            leads = [by_key[k] for k in plan.recipes if k in by_key]
            if plan.digests - cached:
                # The model is not on the NAS yet: one lead recipe fetches it; siblings wait for it.
                if any(
                    self._active(r.key)
                    or (
                        r.local == "preparing"
                        and self.state.downloads.get(r.key, {}).get("state")
                        != "retired"
                    )
                    for r in leads
                ):
                    continue
                lead = next((r for r in leads if self._can_request(r, now)), None)
                if (
                    lead is not None
                    and model_ops < self.config.max_model_downloads
                    and now >= self.pressure_until
                    and (
                        pinned_bytes == 0
                        or pinned_bytes + plan.new_bytes <= self.config.nas_budget_bytes
                    )
                ):
                    self._request(lead, "model")
                    model_ops += 1
                    pinned_bytes += plan.new_bytes
                continue
            for recipe in leads:  # model cached: only the runtime image is missing
                if (
                    (
                        recipe.local not in ("cached", "preparing")
                        or self.state.downloads.get(recipe.key, {}).get("state")
                        == "retired"
                    )
                    and not recipe.cache_ready
                    and image_ops < self.config.max_image_pulls
                    and self._can_request(recipe, now)
                ):
                    self._request(recipe, "image")
                    image_ops += 1
        self._reconcile_pins(plans, by_key, models, sizes, present)
        return {
            "model_ops": model_ops,
            "image_ops": image_ops,
            "rate": self.rate.rate,
            "live_rate": live_rate,
            "pinned_bytes": pinned_bytes,
            "pressure": now < self.pressure_until,
        }

    def _requested(self, key: str) -> bool:
        return key in self.state.downloads

    def _active(self, key: str) -> bool:
        record = self.state.downloads.get(key)
        return (
            record is not None
            and record.get("state") != "retired"
            and active(record.get("state"))
        )

    def _can_request(self, recipe: Recipe, now: float) -> bool:
        if self.state.status(recipe.key) in ("passed", "failed", "skipped"):
            return False
        record = self.state.downloads.get(recipe.key)
        if record is None:
            return True
        if active(record.get("state")) and record.get("state") != "retired":
            return False
        if recipe.cache_ready:
            return False
        if record.get("state") == "retired":
            if record.get("request_key"):
                return self._terminal_retest(recipe, record, now)
            return now >= float(record.get("retry_at", 0))
        if record.get("state") == "succeeded":
            # A finished download stands. Ask again only if a library read made *after*
            # it still says the cache is empty (evicted), and only a couple of times: a
            # lagging listing must never turn into a download loop.
            done_at = float(record.get("done_at", now))
            return (
                recipe.local == "not_cached"
                and int(record.get("rerequests", 0)) < MAX_REREQUESTS
                and now - done_at > 600
                and self.library_fresh_since > done_at
            )
        if record.get("state") == "failed" and record.get("operation_id"):
            # Accepted parents always need a fresh exact terminal receipt. An old
            # retry timer must never override an active or unreadable observation.
            return self._terminal_retest(recipe, record, now)
        return bool(record.get("retry_at")) and now >= float(record["retry_at"])

    def _terminal_retest(
        self, recipe: Recipe, record: dict[str, Any], now: float
    ) -> bool:
        """Every accepted-parent retry needs an observed terminal exact receipt."""
        entry = self.state.entry(recipe.key)
        retest = entry.get("retest")
        scheduled = (
            entry.get("status") == "pending"
            and isinstance(retest, Mapping)
            and retest.get("at") != entry.get("download_retest_consumed_at")
            and retest.get("reason")
            in ("scheduled-retest", "observed-fault-owner-changed")
            and entry.get("content_sha256") == recipe.content_sha256
            and entry.get("revision_id") == recipe.revision_id
        )
        bounded_retry = (
            bool(record.get("retry_at"))
            and int(record.get("attempt", 1)) < 2
            and isinstance(record.get("request_intent"), Mapping)
        )
        if (
            not (scheduled or bounded_retry)
            or record.get("state") not in ("failed", "retired")
            or not record.get("operation_id")
            or not record.get("request_key")
            or now < float(record.get("retry_at", 0))
            or now < float(entry.get("finished_at", 0)) + self.config.retry_cooldown
            or now < float(entry.get("not_before", 0))
            or now < float(entry.get("defer_until", 0))
        ):
            return False
        intent = record.get("request_intent")
        if isinstance(intent, Mapping) and (
            intent.get("recipe_key") != recipe.key
            or intent.get("content_sha256") != recipe.content_sha256
            or intent.get("revision_id") != recipe.revision_id
        ):
            return False
        # A request lookup binds the saved UUID to its actual terminal receipt.
        # Absence, permission errors, client timeouts and active states cannot
        # authorize a replacement request or erase its cache/ownership history.
        reply = self.vk.run(
            "recipe", "progress", "--request-key", str(record["request_key"])
        )
        document = reply.document
        if (
            not reply.ok
            or not isinstance(document, Mapping)
            or document.get("id") != record["operation_id"]
            or document.get("request_id") != record["request_key"]
            or document.get("recipe_revision_id") != recipe.revision_id
            or document.get("recipe_content_sha256") != recipe.content_sha256
            or not isinstance(document.get("failure"), Mapping)
            or not isinstance(document.get("request"), Mapping)
            or document.get("state") != "failed"
        ):
            return False
        original_intent = record.get("accepted_intent")
        observed_intent = document["request"]
        if isinstance(original_intent, Mapping):
            # Deterministic identity comparison after installed-CLI wire validation;
            # this is not another parser or a reconstruction of execution parameters.
            if observed_intent != original_intent:
                return False
        elif (
            observed_intent.get("kind") != "selector"
            or observed_intent.get("selector") != recipe.key
            or observed_intent.get("force", False) is not False
        ):
            # An older owned selector receipt can be bound by its original request
            # UUID plus frozen revision/content; unknown other intent is not guessed.
            return False
        record["terminal_receipt"] = dict(document)
        record["terminal_verified_at"] = now
        record["state"] = "failed"
        return True

    # -- pins ---------------------------------------------------------------

    def _live_assignments(self, number: int) -> list[str] | None:
        """The assignment names the Controller really holds in the pin profile, or None if unreadable.

        The sweep's own bookkeeping can drift from the profile (a timed-out add the Controller
        still applied, a lost state file, an assignment someone else added), and the contract
        limit applies to the real document, so every pin decision counts on this.
        """
        reply = self.vk.run("profile", "export", profile=number)
        document = reply.document if reply.ok else None
        if not isinstance(document, dict) or not isinstance(
            document.get("assignments", []), list
        ):
            return None
        return [
            str(item.get("assignment_name") or "")
            for item in document.get("assignments", [])
            if isinstance(item, dict)
        ]

    def release_pins(self) -> None:
        """Nothing is left to protect: empty the pin profile."""
        number = self.config.pin_profile
        if number is None or self.clock() < self.pin_retry_at:
            return
        live = self._live_assignments(number)
        if live is None:
            self._pin_failed()
            return
        for alias in [n for n in live if n.startswith(PIN_ALIAS_PREFIX)]:
            try:
                self.vk.call("profile", "remove", alias, "--yes", profile=number)
            except VonkctlError:
                self._pin_failed()
                return
        self.state.data["pins"] = []

    def _pin_failed(self) -> None:
        """Back off (never give up for good): the next attempt is later, the sweep carries on."""
        self.pin_failures += 1
        delay = min(
            self.config.retry_cooldown * 2 ** (self.pin_failures - 1),
            PIN_RETRY_MAX_SECONDS,
        )
        self.pin_retry_at = self.clock() + delay

    def desired_pins(
        self,
        plans: Sequence[Any],
        by_key: Mapping[str, Recipe],
        models: Mapping[str, Model],
        sizes: Mapping[str, int],
        present: set[str],
    ) -> list[str]:
        """One pin per model set that is on (or heading to) the NAS and not yet tested, within the byte budget."""
        wanted = list(
            self.state.slots
        )  # a recipe under test already protects its models
        covered = {
            frozenset(slot.get("digests", ())) for slot in self.state.slots.values()
        }
        spent = 0
        for plan in plans:
            if plan.digests - present and not any(
                k in self.state.downloads for k in plan.recipes
            ):
                continue  # not downloaded, not requested: nothing to protect
            cost = sum(sizes.get(d, 0) for d in plan.digests)
            if spent + cost > self.config.nas_budget_bytes and spent:
                continue
            spent += cost
            # The model is what needs protecting: one recipe per model set, plus any recipe
            # that needs a different combination of models.
            members = [k for k in plan.recipes if k in by_key]
            wanted += [
                k
                for i, k in enumerate(members)
                if (i == 0 and plan.digests not in covered)
                or by_key[k].model_set != plan.digests
            ]
        return wanted[:PIN_PROFILE_MAX_ASSIGNMENTS]  # nearest in the queue first

    def _reconcile_pins(
        self,
        plans: Sequence[Any],
        by_key: Mapping[str, Recipe],
        models: Mapping[str, Model],
        sizes: Mapping[str, int],
        present: set[str],
    ) -> None:
        number = self.config.pin_profile
        if (
            number is None
            or self.clock() < self.pin_retry_at
            or not self.config.pin_spark
        ):
            return
        live = self._live_assignments(number)
        if live is None:
            self._pin_failed()
            self.pin_error = "pin profile could not be read"
            return
        # Everything that is not one of our pins counts against the limit and is never removed.
        foreign = [n for n in live if not n.startswith(PIN_ALIAS_PREFIX)]
        cap = max(PIN_PROFILE_MAX_ASSIGNMENTS - len(foreign), 0)
        wanted = self.desired_pins(plans, by_key, models, sizes, present)[:cap]
        wanted_aliases = {pin_alias(k): k for k in wanted}
        held = [n for n in live if n.startswith(PIN_ALIAS_PREFIX)]
        budget = self.config.pins_per_tick
        try:
            stale = [n for n in held if n not in wanted_aliases]
            # Over the limit (or at it): removals come first and are not rationed by the budget.
            overflow = len(live) - PIN_PROFILE_MAX_ASSIGNMENTS
            for alias in stale[: max(budget, overflow, 0)]:
                self.vk.call("profile", "remove", alias, "--yes", profile=number)
                held.remove(alias)
                live.remove(alias)
                budget -= 1
            for key in [k for k in wanted if pin_alias(k) not in held][
                : max(budget, 0)
            ]:
                if len(live) + 1 > PIN_PROFILE_MAX_ASSIGNMENTS:
                    break  # the document would break the contract: never submit it
                self.vk.call(
                    "profile",
                    "add",
                    key,
                    "--spark",
                    self.config.pin_spark,
                    "--as",
                    pin_alias(key),
                    "--state",
                    "installed",
                    "--yes",
                    profile=number,
                )
                held.append(pin_alias(key))
                live.append(pin_alias(key))
            self.pin_failures = 0
            self.pin_error = None
        except VonkctlError as error:
            self._pin_failed()
            self.pin_error = str(error)[:200]
            self.state.event(
                f"pin profile {number} could not be updated: {self.pin_error}"
            )
        current = [k for k in wanted if pin_alias(k) in held]
        self.state.data["pins"] = current
