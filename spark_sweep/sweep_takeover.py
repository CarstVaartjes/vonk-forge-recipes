"""Taking over the fleet and editing the profile (methods of ``Sweep``)."""

# pyright: reportAttributeAccessIssue=false
# The methods reach the state set in ``Sweep.__init__`` and the sibling mixins through ``self``.
from __future__ import annotations

from collections.abc import Mapping

from . import cleanup, verdict
from .catalog import (
    Recipe,
    dig,
    fetch_fleet,
)
from .lifecycle import TERMINAL as SETTLED
from .profile_alias import unique_profile_alias
from .sweep_contract import (
    SweepSlotPhase,
    SweepWait,
    SweepWaitReason,
)
from .sweep_support import (
    INFRA_CAP,
    TAKEOVER_RETRY_SECONDS,
    ProfilePruneOutcome,
    _blocker_text,
    alias_for,
)
from .verdict import Progress
from .vonkctl import VonkctlError


class SweepTakeover:
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
            self.state.data.pop("takeover_fallback", None)
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

    def _begin_run(self) -> None:
        """A new run takes the fleet from fresh facts, never from the last run's takeover."""
        self._request_clearing(self.clock.now())
        self.state.data.pop("cleared_for", None)

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
