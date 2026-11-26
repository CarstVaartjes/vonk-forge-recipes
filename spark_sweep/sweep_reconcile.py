"""Catalog advance, source observation and reconciliation (methods of ``Sweep``)."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

from vonk_forge_contracts import (
    QualificationIdentity,
    QualificationStatus,
)

from . import policy, verdict
from .catalog import (
    Recipe,
    fetch_fleet,
)
from .identity import (
    VerdictIdentity,
)
from .lifecycle import (
    STICKY_REQUEUES,
    RecipeStatus,
    RetestReason,
)
from .lifecycle import TERMINAL as SETTLED
from .qualification import (
    current_identity,
    pass_marker,
    recheck_due,
    recheck_named,
    unpublishable_requeue_due,
)
from .source_serving import (
    SourceEvent,
    SourceRowStatus,
    SourceServing,
    Unserved,
    advance,
    describe,
    describe_missing,
    digests,
    missing_from_source,
    unserved,
)
from .sweep_contract import (
    SweepSlotPhase,
    SweepWait,
    SweepWaitCode,
    SweepWaitReason,
)
from .sweep_support import (
    CATALOG_SKIP,
    FLEET_SKIP,
    PLATFORM_SKIP,
    _fleet_skip,
    retest_reason,
)
from .verdict import Progress
from .vonkctl import VonkctlError

if TYPE_CHECKING:
    from .run import Sweep


class SweepReconcile:
    def advance_catalog(self: Sweep, now: float) -> None:  # pyright: ignore[reportGeneralTypeIssues]
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

    def refresh_catalog(self: Sweep, max_steps: int = 200) -> None:  # pyright: ignore[reportGeneralTypeIssues]
        """Read both libraries completely now (tests, and anyone who needs a fresh view)."""
        now = self.clock.now()
        for item in (self.recipe_list, self.model_list):
            item.begin(now)
            for _ in range(max_steps):
                if item.step(now):
                    break
        self.library_commit = self.recipe_list.commit
        self._reconcile_all()

    def _reconcile_all(self: Sweep) -> None:  # pyright: ignore[reportGeneralTypeIssues]
        # Fleet capacity is read fresh every time: a snapshot taken during a restart, a takeover or
        # a dual load in flight must never decide a recipe's fate for longer than one pass.
        try:
            self.fleet = fetch_fleet(self.vk, self.cfg.default_spark_memory)
        except VonkctlError as error:
            self.state.event(f"fleet unreadable: {str(error)[:120]}")
            if not self.fleet.sparks:
                return
        self.observe_source()
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

    def fleet_capacity(self: Sweep) -> int:  # pyright: ignore[reportGeneralTypeIssues]
        """Sparks the fleet enrolls, online or not: never how many are online or free now."""
        return len(self.enrolled())

    def _fleet_evidence(self: Sweep) -> dict[str, Any]:  # pyright: ignore[reportGeneralTypeIssues]
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

    def observe_lanes(self: Sweep) -> None:  # pyright: ignore[reportGeneralTypeIssues]
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

    def heal_fleet_skips(self: Sweep) -> None:  # pyright: ignore[reportGeneralTypeIssues]
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

    def _selected(self: Sweep, recipe: Recipe) -> bool:  # pyright: ignore[reportGeneralTypeIssues]
        return self._selected_key(recipe.key)

    def observe_source(self: Sweep) -> None:  # pyright: ignore[reportGeneralTypeIssues]
        """Until the catalog serves this sweep's source, its differing recipes wait.

        A served recipe whose content differs from the source is not tested yet: its
        verdict would be the old content's again, and a run with only such recipes
        left would end having tested nothing. The wait is bounded
        (``source_wait_seconds``, checked every poll by ``observe_source_wait``);
        at the bound the sweep tests what the catalog serves.
        """
        complete = bool(self.recipe_list.passes) and not self.recipe_list.in_pass
        if not complete:
            return
        self._observe_source_behind()
        if self.source_serving is SourceServing.SERVED:
            return
        differing = unserved(
            {key: recipe.content_sha256 for key, recipe in self.recipes.items()},
            self.authored,
            self._selected_key,
        )
        self._source_transition(
            SourceEvent.UNSERVED_PASS if differing else SourceEvent.SERVED_PASS,
            differing,
        )

    def _observe_source_behind(self: Sweep) -> None:  # pyright: ignore[reportGeneralTypeIssues]
        """A recipe the library serves but this sweep's source lacks is not tested here.

        Not a verdict on the recipe: the sweep's source is older than the library.
        """
        missing = missing_from_source(
            self.recipes.keys(), self.authored, self._selected_key
        )
        if frozenset(missing) == self.source_missing_keys:
            return
        self.source_missing_keys = frozenset(missing)
        if not missing:
            return
        reason = describe_missing(missing, self.library_commit)
        self.record_result(
            None,
            "catalog",
            SourceRowStatus.SOURCE_BEHIND,
            reason=reason,
            library_commit=self.library_commit or None,
        )
        self.state.event(reason)

    def observe_source_wait(self: Sweep) -> None:  # pyright: ignore[reportGeneralTypeIssues]
        """Every poll of the wait: the bounded wait's budget, said through ``waiting``."""
        if self.source_serving is not SourceServing.UNSERVED:
            return
        now = self.clock.now()
        wait = SweepWait(
            reason=SweepWaitReason.CATALOG_SOURCE, detail=self.source_alert
        )
        progress = verdict.bounded_wait(
            wait, self.source_watch, window=self.cfg.source_wait_seconds, now=now
        )
        if progress is Progress.WAIT_EXPIRED:
            self._source_transition(
                SourceEvent.WAIT_EXPIRED,
                self.unserved_recipes,
                expiry=verdict.wait_expiry(self.source_watch, now).model_dump(
                    mode="json"
                ),
            )
            return
        self.waiting(wait.description, float(self.source_watch["wait_started_at"]))

    def _source_transition(
        self: Sweep,  # pyright: ignore[reportGeneralTypeIssues]
        event: SourceEvent,
        differing: tuple[Unserved, ...],
        expiry: Mapping[str, Any] | None = None,
    ) -> None:
        before_state, before = self.source_serving, self.unserved_recipes
        state = advance(before_state, event)
        self.source_serving = state
        if state is SourceServing.SERVED:
            self.source_watch.clear()
        elif self.source_watch.get("authored") != self.authored_digest:
            # bounded_wait starts the budget on its first poll
            self.source_watch.clear()
            self.source_watch["authored"] = self.authored_digest
        self.unserved_recipes = differing if state is SourceServing.UNSERVED else ()
        self.unserved_keys = frozenset(item.recipe for item in self.unserved_recipes)
        if event is SourceEvent.UNSERVED_PASS and differing == before:
            return  # the same wait, already recorded
        if event is SourceEvent.SERVED_PASS and before_state is SourceServing.UNREAD:
            return  # the catalog served the source from the first pass
        commit = self.library_commit[:12] or "unknown"
        if event is SourceEvent.UNSERVED_PASS:
            status = SourceRowStatus.WAITING
            reason = "WAITING FOR THE CATALOG: " + describe(
                differing, self.library_commit
            )
            self.source_alert = reason
        elif event is SourceEvent.SERVED_PASS:
            status = SourceRowStatus.SERVED
            reason = f"library {commit} serves this sweep's source"
            self.source_alert = ""
        else:
            status = SourceRowStatus.WAIT_EXPIRED
            reason = (
                "source not served within bound "
                f"({self.cfg.source_wait_seconds:.0f}s): testing what library "
                f"{commit} serves; check the publish and `vonkctl recipe "
                "sync-status`, or update the sweep's source checkout"
            )
            self.source_alert = (
                f"CATALOG DOES NOT SERVE THE SOURCE: {reason} ("
                + ", ".join(item.recipe for item in differing[:3])
                + (f" and {len(differing) - 3} more" if len(differing) > 3 else "")
                + ")"
            )
        self.record_result(
            None,
            "catalog",
            status,
            reason=reason,
            code=SweepWaitCode.BUDGET_EXHAUSTED if expiry is not None else None,
            wait_expiry=expiry,
            library_commit=self.library_commit or None,
            unserved=digests(differing) or None,
        )
        self.state.event(reason)

    def _reconcile_entry(self: Sweep, recipe: Recipe, enrolled_sparks: int) -> None:  # pyright: ignore[reportGeneralTypeIssues]
        """Testable is the enrolled topology; fewer lanes online now is a wait, never a skip."""
        if not self._selected(recipe):
            return
        entry = self.state.entry(recipe.key)
        slot = self.state.slots.get(recipe.key)
        if slot is not None and slot.get("phase") != "finished":
            return
        evidence = self._recipe_evidence(recipe)
        try:
            identity: QualificationIdentity | None = current_identity(recipe)
        except (ValueError, TypeError):
            identity = None  # malformed observed identity cannot certify public proof
        view = (
            self.qualification_view(recipe, evidence, identity)
            if identity is not None
            else None
        )
        # A local pass the catalog would not publish is asked for again, once per content
        # and catalog image, until the re-run records a verdict (then it stands).
        marker = pass_marker(recipe, identity) if identity is not None else ""
        unpublishable = unpublishable_requeue_due(
            view, evidence, identity, recipe, entry, self.release_sha
        )
        # An operator-named recipe runs once on this release, whatever its status.
        recheck = recheck_due(
            self.cfg.recheck,
            recipe.key,
            recipe.content_sha256,
            evidence,
            self.release_sha,
        )
        requeued = (
            entry.get("status") == RecipeStatus.PENDING
            and entry.get("requeued_because") in STICKY_REQUEUES
        )
        # While still named, a recheck that ended deferred stays visible, not passed.
        held = (
            entry.get("status") == RecipeStatus.DEFERRED
            and entry.get("requeued_because") == RetestReason.OPERATOR_RECHECK
            and recheck_named(self.cfg.recheck, recipe.key)
        )
        if (
            view is not None
            and view.status == QualificationStatus.WORKING
            and view.last_pass is not None
            and not unpublishable
            and not recheck
            and not held
            and not requeued  # a pending requeue is not undone by the next catalog pass
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
        if entry.get("reason_kind") == PLATFORM_SKIP:
            entry["status"] = "pending"
            entry.pop("reason", None)
            entry.pop("reason_kind", None)
            self.record_result(
                recipe.key, "reconcile", "pending", reason="loadable on this platform"
            )
            self.state.event(f"{recipe.key} is loadable on this platform: pending")
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
        why = retest_reason(
            view,
            lambda: identity or current_identity(recipe),
            unpublishable=unpublishable,
            revalidate_passes=self.cfg.revalidate_passes,
            recheck=recheck,
        )
        if entry.get("status") == RecipeStatus.PENDING:
            return
        if why is None:
            return
        if why == RetestReason.UNPUBLISHABLE_PASS:
            entry["unpublishable_requeue"] = {"at": self.clock.now(), "marker": marker}
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
