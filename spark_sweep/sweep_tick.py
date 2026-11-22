"""The tick: what is pending, waiting, paused and stalled (methods of ``Sweep``)."""

# pyright: reportAttributeAccessIssue=false
# The methods reach the state set in ``Sweep.__init__`` and the sibling mixins through ``self``.
from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from functools import partial
from typing import Any

from vonk_forge_contracts import (
    QualificationStatus,
    QualificationStatusView,
)

from . import policy
from .catalog import (
    Recipe,
    fetch_fleet,
)
from .fanout import gather
from .identity import (
    AcceptedRequest,
)
from .lifecycle import ACTIVE as ACTIVE_APP
from .lifecycle import (
    LifecycleState,
    RecipeStatus,
    RetestReason,
)
from .qualification import (
    scheduling_priority,
)
from .receipt import CANCELLED_FAILURE
from .removal_contract import RemovalWait
from .report import write_status
from .sweep_contract import (
    SweepWaitReason,
)
from .sweep_support import (
    RETEST_SECONDS,
    WAIT_EVENT_SECONDS,
)
from .vonkctl import VonkctlError


class SweepTick:
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
            if recipe.key in self.unserved_keys:
                continue  # the catalog still serves other content than the source
            if recipe.key not in self.authored:
                continue  # this sweep's source is older than the library
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
                and float(e.get("finished_at", 0)) >= self.run_started_at
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
            write_status(self)  # pyright: ignore[reportArgumentType]
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
            self._reads.clear()
            self.guard.reads = {}
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
            self._request_clearing(now)  # their load replaced what ours ran
            self.state.data["owner"]["seen_at"] = now
            self.preempt("an owner profile was loaded")
        if self.owner_status.paused:
            # Readable owner activity holds; unreadable observations have a per-profile budget.
            self.state.data["owner"].setdefault("paused_at", now)
        else:
            self.state.data["owner"].pop("paused_at", None)

    def _observe_concurrently(self) -> None:
        """The independent read-only observations of a pass, all at once, right before the
        owner check (after catalog and release work).

        Run after catalog/release work (which may update vonkctl or re-read the fleet), so no
        prefetched read predates it. Each consumer takes its outcome at its own place in
        the pass and an error is raised there; writes stay sequential and unchanged.
        """
        calls: dict[str, Callable[[], Any]] = {
            f"owner:{n}": partial(self.guard.read_latest, n)
            for n in self.guard.profiles
        }
        calls["platform"] = lambda: self.vk.run("platform", timeout=30)
        if self.state.slots:
            calls["fleet"] = lambda: fetch_fleet(self.vk, self.cfg.default_spark_memory)
        self._reads = gather(calls, self.read_executor)
        self.guard.reads = {
            n: self._reads.pop(f"owner:{n}") for n in self.guard.profiles
        }

    def _tick_read(self, name: str, fallback: Callable[[], Any]) -> Any:
        outcome = self._reads.pop(name, None)
        return outcome.result() if outcome is not None else fallback()

    def _tick(self) -> None:
        # Placement/cleanup re-derive these observations only when asked this pass.
        self.state.data.pop("profile_edit_wait", None)
        self.state.data.pop("dual_reservation", None)
        self.adoption_holds.clear()
        self.fit_holds.clear()
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
        self._observe_concurrently()
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
            ready=self.ready_keys(key for plan in plans for key in plan.recipes),
            sizes=self.sizes,
            revalidate={
                k for k, e in self.state.recipes.items() if e.get("revalidate")
            },
            deprioritised=self._deprioritised(pending),
            variants_last=self.cfg.variants_last,
            boost=self._placement_priority(),
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
            write_status(self)  # pyright: ignore[reportArgumentType]
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
        """Models on the NAS: the library says so or a download of ours finished. Per model, never
        recipe readiness (``ready``) and never fleet fit."""
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
        return cached

    def _present(self) -> set[str]:
        """Models on the NAS or arriving."""
        return policy.present_models(self.models) | self.cached_models

    def _download_done(self, key: str) -> None:
        entry = self.state.recipes.get(key, {})
        repeat = entry.get("exit_repeat") or {}
        if (repeat.get("key") or {}).get("klass") == CANCELLED_FAILURE:
            entry.pop("exit_repeat")  # a download that landed ends the run of cancels
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
        if not self.catalog_loaded or self.state.slots or self.unserved_recipes:
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
            and key in self.authored  # an older source cannot test what it lacks
            and self._selected(self.recipes[key])
        )
