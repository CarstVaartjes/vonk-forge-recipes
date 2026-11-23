"""Adopted requests, blocked slots, review waits and infrastructure notes (methods of ``Sweep``)."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import replace
from functools import cache, partial
from typing import TYPE_CHECKING, Any

from vonk_forge_contracts import (
    QualificationBlame,
)

from . import cleanup, policy, verdict
from .catalog import (
    Recipe,
    fetch_fleet,
)
from .identity import (
    AcceptedRequest,
)
from .lifecycle import TERMINAL as SETTLED
from .profile_alias import unique_profile_alias
from .sweep_contract import (
    InfrastructureIncident,
)
from .sweep_support import (
    INFRA_BASE,
    INFRA_CAP,
    MAX_REVIEW_FREEING,
    alias_for,
)
from .verdict import Disposition, Progress
from .vonkctl import Reply, VonkctlError

if TYPE_CHECKING:
    from .run import Sweep


class SweepAdopt:
    @staticmethod
    def _reviewed_revisions(review: Mapping[str, Any]) -> dict[str, str]:
        """The recipe revision the Controller's review resolves for each alias, before any effect."""
        return {
            str(r["alias"]): str(r["recipe_revision_id"])
            for r in review.get("resolved_assignments") or []
            if isinstance(r, Mapping) and r.get("alias") and r.get("recipe_revision_id")
        }

    def _revision_mismatch(
        self: Sweep,  # pyright: ignore[reportGeneralTypeIssues]
        recipe: Recipe,
        reviewed: str,
        document: Mapping[str, Any],
        now: float,
    ) -> None:
        """The platform would load another revision than the requested one: nothing loads.

        A wait paced by how often it happened; the platform's fault, never the recipe's.
        """
        entry = self.state.entry(recipe.key)
        entry["revision_mismatches"] = int(entry.get("revision_mismatches", 0)) + 1
        entry["not_before"] = now + verdict.retry_delay(
            self.cfg.retry_delay, entry["revision_mismatches"]
        )
        self.record_result(
            recipe.key,
            "identity",
            "waiting",
            blame=QualificationBlame.PLATFORM.value,
            code=policy.REVISION_MISMATCH,
            disposition=Disposition.WAIT.value,
            reason="the review binds another recipe revision than the one requested",
            evidence={
                "requested_revision": recipe.revision_id,
                "reviewed_revision": reviewed,
                "occurrences": entry["revision_mismatches"],
            },
            document=document,
        )
        self.state.event(f"waiting {recipe.key}: {policy.REVISION_MISMATCH}")

    def _adopted_requests(
        self: Sweep,  # pyright: ignore[reportGeneralTypeIssues]
        review: Mapping[str, Any],
        *,
        excluded: frozenset[str] = frozenset(),
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

    def _adopted_cleanup_requests(
        self: Sweep,  # pyright: ignore[reportGeneralTypeIssues]
        review: Mapping[str, Any],
    ) -> list[str] | None:
        return self._cleanup_adoption(review)[0]

    def _cleanup_adoption(
        self: Sweep,  # pyright: ignore[reportGeneralTypeIssues]
        review: Mapping[str, Any],
    ) -> tuple[list[str] | None, str | None]:
        requests = []
        fleet = cache(partial(self.vk.run, "fleet"))
        for key, load in self.state.data["cleanup_loads"].items():
            if load.get("end"):
                continue
            if load.get("terminal_receipt") or load.get("state") in SETTLED:
                if self._cleanup_target_absent(load, terminal=True, fleet=fleet):
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
            if pending and self._cleanup_target_absent(load, fleet=fleet):
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

    def _ridden_by_live_lane(self: Sweep, app_id: Any, besides: Any) -> bool:  # pyright: ignore[reportGeneralTypeIssues]
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

    def _adopted_slots_live(self: Sweep, load: Mapping[str, Any]) -> bool:  # pyright: ignore[reportGeneralTypeIssues]
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

    def _open_slot(
        self: Sweep,  # pyright: ignore[reportGeneralTypeIssues]
        placement: policy.Placement,
        request: str,
        now: float,
    ) -> None:
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
            "release": self.release_sha,
            "deadline": now + timeout,
            "model_bytes": model_bytes,
            "digests": list(recipe.model_digests),
            "prior_runs": sorted(  # runs of this alias that are not this load's
                p.run_id
                for p in self.fleet.presences
                if p.alias == alias and p.node_id in placement.spark_ids
            ),
        }

    def _remember_own(self: Sweep, alias: str) -> None:  # pyright: ignore[reportGeneralTypeIssues]
        own = self.state.data["own_aliases"]
        if alias not in own:
            own.append(alias)
            del own[:-500]

    def _overfull_sparks(self: Sweep) -> set[str]:  # pyright: ignore[reportGeneralTypeIssues]
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

    def _rollback(
        self: Sweep,  # pyright: ignore[reportGeneralTypeIssues]
        placements: Sequence[policy.Placement],
        why: str,
    ) -> None:
        for placement in placements:
            try:  # sequential: every removal edits the one shared sweep profile
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

    def _clear_holders(
        self: Sweep,  # pyright: ignore[reportGeneralTypeIssues]
        holders: list[str],
        blocked: Sequence[policy.Placement],
        document: Mapping[str, Any],
        now: float,
        event: str,
        labels: Sequence[str] | None = None,
    ) -> None:
        """Workloads in the way of ``blocked`` are cleared once; if they outlive it, wait for their owner.

        ``holders`` identify them (a run id, or a name); ``labels`` is what the text shows.
        """
        labels = labels if labels is not None else holders
        if self.state.data.get("cleared_for") != holders:
            self.state.data["cleared_for"] = holders
            self._request_clearing(now)
            self.state.event(event)
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
                    reason=f"review blocked by {', '.join(labels)}; their clearing "
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
                + ", ".join(labels),
                disposition=Disposition.WAIT.value,
                document=document,
            )

    def _blocked(
        self: Sweep,  # pyright: ignore[reportGeneralTypeIssues]
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
            self._clear_holders(
                sorted({*foreign, *leftover}),
                blocked,
                document,
                now,
                "review blocked by a workload that is not ours: clearing"
                if foreign
                else "review blocked by our own finished workload "
                + ", ".join(leftover)
                + ": clearing",
            )
            return
        self.state.data.pop("cleared_for", None)
        for placement in blocked:
            key = placement.recipe.key
            entry = self.state.entry(key)
            blockers = reasons.get(self._alias(placement.recipe), [])
            code, detail = policy.review_blocker(blockers, verdict.PREPARATION_CODES)
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
            if failure.klass == "capacity":
                # What holds the room (a Controller build, a workload no clearing stops) is the
                # platform's wait, never a verdict: the Controller decides the fit. Review again
                # later, no attempt spent.
                entry["defer_until"] = now + self.cfg.defer_delay
                self.record_result(
                    key,
                    "review",
                    "waiting",
                    reason=f"{code}: waiting for the room",
                    disposition=Disposition.WAIT.value,
                    document=document,
                )
                continue
            if freeing:
                failure = replace(
                    failure,
                    evidence={**failure.evidence, "freeing_attempts": freeing},
                )
            if verdict.disposition(failure) is Disposition.WAIT:
                codes = tuple(sorted({c for c, _ in blockers}))
                self._watch_review_wait(key, entry, placement, failure, codes, now)
            elif verdict.disposition(failure) is Disposition.FAIL:
                self._clear_review_wait(entry)
            # Only the observed demand, never a clearing count, can settle capacity.
            self._record_failure(key, failure, None, document=document)

    def _watch_review_wait(
        self: Sweep,  # pyright: ignore[reportGeneralTypeIssues]
        key: str,
        entry: dict[str, Any],
        placement: policy.Placement,
        failure: policy.Failure,
        codes: tuple[str, ...],
        now: float,
    ) -> None:
        """A review wait is never silent: the same blocker words past ``idle_alert_seconds``
        are an escalation (informational; it pauses nothing and ends nothing). The detail
        carries live readings, so only the codes are the signature; it is the alarm text."""
        watch = entry.setdefault("review_watch", {})
        identity = placement.recipe.content_sha256
        if watch.get("content_sha256") != identity:
            self._clear_review_wait(entry)
            watch = entry.setdefault("review_watch", {"content_sha256": identity})
        signal = verdict.Signal(signature=codes)
        progress = verdict.progress_verdict(
            signal, watch, window=self.cfg.idle_alert_seconds, now=now
        )
        if progress is Progress.STALLED:
            source = f"{verdict.REVIEW_WAIT_SOURCE}:{failure.code}"
            if watch.get("source") not in (None, source):
                self.clear_infra(watch["source"])
            watch["source"] = source
            self.note_infra(
                source, failure.detail, dependency=failure.detail, escalated=True
            )

    def _clear_review_wait(self: Sweep, entry: dict[str, Any]) -> None:  # pyright: ignore[reportGeneralTypeIssues]
        """Admission, a content change or a final verdict ends a review wait's alarm and watch."""
        watch = entry.pop("review_watch", None)
        if isinstance(watch, dict) and watch.get("source"):
            self.clear_infra(watch["source"])

    def note_infra(
        self: Sweep,  # pyright: ignore[reportGeneralTypeIssues]
        source: str,
        message: str,
        *,
        dependency: str | None = None,
        escalated: bool = False,
        typed: Reply | None = None,
    ) -> None:
        """Pause that kind of work, preserving retry history until its predicate passes."""
        now = self.clock.now()
        item = InfrastructureIncident.model_validate(
            self.state.data["infra"].get(source) or {"since": now}
        )
        item.count += 1
        hint = (
            " (vonkctl may be older than the Controller: run `vonkctl update --apply`)"
            if not (typed is not None and typed.is_contract_mismatch)
            and ("protocol_invalid" in message or "OpenAPI" in message)
            else ""
        )
        item.message = message[:300] + hint
        if dependency is not None:
            item.dependency = dependency
            if item.escalate_at is None:
                item.escalate_at = item.since + (
                    0 if escalated else self.cfg.idle_alert_seconds
                )
        item.until = now + min(INFRA_BASE * 2 ** min(item.count - 1, 20), INFRA_CAP)
        self.state.data["infra"][source] = item.model_dump(exclude_none=True)
        if source == "download":
            self.prefetcher.pause_until = item.until
        if item.count in (1, 5) or item.count % 20 == 0:
            self.state.event(
                f"infrastructure problem ({source}, not a recipe failure), "
                f"retrying with backoff: {item.message[:160]}"
            )

    def clear_infra(self: Sweep, source: str) -> None:  # pyright: ignore[reportGeneralTypeIssues]
        if self.state.data["infra"].pop(source, None) is not None:
            self.state.event(f"infrastructure problem cleared ({source})")

    def _own_aliases(self: Sweep) -> set[str]:  # pyright: ignore[reportGeneralTypeIssues]
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

    def _fleet_runs(self: Sweep) -> tuple[list[str], list[str]]:  # pyright: ignore[reportGeneralTypeIssues]
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

    def _foreign_runs(self: Sweep) -> bool:  # pyright: ignore[reportGeneralTypeIssues]
        return bool(self._fleet_runs()[0])
