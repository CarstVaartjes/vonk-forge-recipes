"""Applying a plan to the fleet (methods of ``Sweep``)."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from . import policy, verdict
from .catalog import (
    dig,
)
from .sweep_contract import (
    SweepSlotPhase,
    SweepStorageReview,
    SweepStorageWait,
)
from .verdict import Disposition
from .vonkctl import VonkctlError, request_key

if TYPE_CHECKING:
    from .run import Sweep


class SweepApply:
    def _apply(
        self: Sweep,  # pyright: ignore[reportGeneralTypeIssues]
        placements: Sequence[policy.Placement],
        now: float,
    ) -> bool | None:
        """Return False when the shared profile cannot be edited this pass."""
        if len(placements) > 1:
            # Each new lane has its own receipt. The unchanged lane remains in
            # this same whole-fleet profile and requires canonical adoption.
            for index, placement in enumerate(placements):
                if index:
                    self._observe_owner(self.clock.now())
                    if self.paused or self.owner_status.new_activity:
                        return None
                if self._apply([placement], now) is False:
                    return False
                # Resolve a first lane's admission wait before reviewing the
                # next snapshot; a pending application has no effect to adopt.
                self.advance_slots(now)
            return None
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
                    return None
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
            return None
        overfull = self._overfull_sparks()
        crowded = [p for p in added if overfull & set(p.spark_ids)]
        if crowded:
            # A large recipe runs alone; the other waits for the next free slot.
            self._rollback(crowded, f"would overfill Spark(s) {sorted(overfull)}")
            added = [p for p in added if p not in crowded]
            if not added:
                return None
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
                return None
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
            return None
        for placement in added:
            self.clear_infra(self._review_transport_source(placement.recipe))
        # The revision comes first: every reply that names assignments (blocked ones too)
        # is checked before any verdict can be recorded for the requested revision.
        reviewed = self._reviewed_revisions(document)
        unnamed = [p for alias, p in aliases.items() if alias not in reviewed]
        wrong = [
            p
            for alias, p in aliases.items()
            if reviewed.get(alias, p.recipe.revision_id) != p.recipe.revision_id
        ]
        if unnamed or wrong:
            self._rollback(
                unnamed + wrong, "review names another or no recipe revision"
            )
            for placement in unnamed:
                # An unreadable reply says nothing about the recipe: back off this topology.
                self.state.entry(placement.recipe.key).pop("not_before", None)
                self.note_infra(
                    self._review_transport_source(placement.recipe),
                    "the review names no recipe revision for " + placement.recipe.key,
                )
            for placement in wrong:
                self._revision_mismatch(
                    placement.recipe,
                    reviewed[self._alias(placement.recipe)],
                    document,
                    now,
                )
            self.state.save()
            remaining = [p for p in added if p not in unnamed + wrong]
            if remaining:
                self._apply(remaining, now)
            return None
        runs = dig(document, "effects", "runs", default=[])
        stopped = [
            r["alias"]
            for r in runs
            if isinstance(r, dict)
            and r.get("action") == "stop"
            and r.get("alias") in kept
        ]
        if stopped:
            self._rollback(added, f"review would stop running lanes {stopped}")
            return None
        # A test needs a cold start (startup peak, real load). The Controller keeps a run
        # whose execution identity matches and only renames it: clear it first, then load.
        # A holder is the kept run itself (its run_id): a later run is a new holder.
        reused = {
            str(r["alias"]): r
            for r in runs
            if isinstance(r, dict)
            and r.get("action") == "keep"
            and str(r.get("alias")) in aliases
        }
        if reused:
            held = [aliases[alias] for alias in reused]
            names = sorted(str(r.get("previous_alias") or a) for a, r in reused.items())
            self._rollback(
                held, f"already running, a test needs a fresh start: {names}"
            )
            self._clear_holders(
                sorted(str(r.get("run_id")) for r in reused.values()),
                held,
                document,
                now,
                "review would keep a running " + ", ".join(names) + ": clearing",
                names,
            )
            remaining = [p for alias, p in aliases.items() if alias not in reused]
            if remaining:
                self._apply(remaining, now)
            return None
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
            return None
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
            return None
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
            return None
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
            for placement in remaining:
                self._clear_review_wait(self.state.entry(placement.recipe.key))
            if remaining and len(remaining) < len(added):
                # A rejected batch accepted nothing. Re-plan the smaller set,
                # including a fresh review and binding, so one held Spark cannot
                # starve an independent lane. Each re-plan removes an assignment.
                self._apply(remaining, now)
            return None
        self.clear_infra("review")
        for placement in added:
            self._clear_review_wait(self.state.entry(placement.recipe.key))
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
            self.state.entry(placement.recipe.key).pop("revision_mismatches", None)
            self.state.slots[placement.recipe.key]["ran_revision"] = reviewed[
                self._alias(placement.recipe)
            ]
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
                return None
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
            return None
        self.state.data["loads"][key]["app_id"] = submitted.get("id")
        if isinstance(submitted, Mapping):
            self.documents[key] = submitted
            self.state.data["loads"][key]["admission_unbound"] = (
                verdict.admission_unbound(submitted)
            )
        self.state.data["dirty"] = True
        self.state.event("load submitted: " + ", ".join(p.recipe.key for p in added))
        self.state.save()
        return None
