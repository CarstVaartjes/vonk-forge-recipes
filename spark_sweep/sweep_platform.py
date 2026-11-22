"""Adopting in-flight work, the client and release checks, requeues (methods of ``Sweep``)."""

# pyright: reportAttributeAccessIssue=false
# The methods reach the state set in ``Sweep.__init__`` and the sibling mixins through ``self``.
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from vonk_forge_contracts import (
    QualificationStatus,
)

from . import policy, recovery, verdict
from .catalog import (
    dig,
)
from .fanout import gather
from .lifecycle import (
    LifecycleState,
    RecipeStatus,
    ReleaseAuthority,
    RetestReason,
)
from .prefetch import DOWNLOAD_RETIRED
from .sweep_contract import (
    SweepSlotPhase,
)
from .sweep_support import (
    INFRA_BASE,
    selects,
)


class SweepPlatform:
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

    def check_client(self) -> bool:
        """Observe deployment and let the signed CLI updater select a compatible client.

        True when an update was applied.
        """
        reads = gather(
            {
                "version": lambda: self.vk.run("--version"),
                "platform": lambda: self.vk.run("platform", timeout=30),
                "update": lambda: self.vk.run("update", timeout=60),
            },
            self.read_executor,
        )
        version = reads["version"].result()
        if version.ok and isinstance(version.document, dict):
            self.state.data["client"] = {
                k: version.document.get(k)
                for k in ("version", "source_sha", "control_contract_sha256")
            }
        self._reads["platform"] = reads["platform"]
        self.observe_platform()
        check = reads["update"].result()
        document = check.document if isinstance(check.document, dict) else {}
        if not check.ok or "update_available" not in document:
            self.state.event(
                "could not compare vonkctl with the signed publication: "
                f"{check.error_text[:120]}"
            )
            return False
        self.observe_publication(document)
        if document["update_available"] is not True:
            self.requeue_for_changed_owner()
            return False
        current = dig(document, "current", "version", default="unknown")
        accepted = document.get("accepted_version", "unknown")
        message = f"vonkctl {current} differs from the signed publication {accepted}"
        if self.cfg.allow_version_skew:
            self.state.event(f"WARNING: {message}; continuing as asked")
            return False
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
            return False
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
            return True
        self.note_infra("client-update", f"{message}; update will be retried")
        return False

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
            self.sweep_source,
        )

    def observe_platform(self) -> None:
        reply = self._tick_read("platform", lambda: self.vk.run("platform", timeout=30))
        document = (
            reply.document if reply.ok and isinstance(reply.document, dict) else {}
        )
        self.state.data["platform"] = document
        identity = self._component_observations().get("api")
        release = self.state.data["release"]
        observed = {
            "sha": identity.source_sha if identity else None,
            "control_contract_sha256": identity.contract_sha256 if identity else None,
            "authority": ReleaseAuthority.DEPLOYED_CONTROLLER,
        }
        if any(release.get(k) != value for k, value in observed.items()):
            self.state.data["release_history"].append(
                {**observed, "seen_at": self.clock.now()}
            )
            self.state.event(
                "deployed Controller observed: " + (observed["sha"] or "unknown")
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
        with self._contract_lock:
            healed = self._mismatch_healed and not self._mismatched
            self._mismatch_healed = False
            pending = self._mismatch_pending
        if healed:
            self.clear_infra(policy.CLIENT_CONTRACT)
        # A typed contract mismatch asks for the client check now (at most one per
        # ``INFRA_BASE``, however many calls met it); otherwise the normal cadence.
        if pending and now - self.release_at >= INFRA_BASE:
            with self._contract_lock:
                self._mismatch_pending = False
            self.release_at = now
            if not self.check_client():
                self._contract_alert()
            return
        if now - self.release_at < self.cfg.release_seconds:
            return
        self.release_at = now
        self.check_client()
        # The client's contract fingerprint equals the Controller's: direct evidence the
        # mismatch is gone, whichever command met it (a rare one may not run for hours).
        if self._component_observations().get("client") is not None:
            self.clear_infra(policy.CLIENT_CONTRACT)

    def _contract_alert(self) -> None:
        text = (
            "client.contract_mismatch: the Controller runs another control contract and no "
            "compatible vonkctl update was applied (--allow-version-skew, nothing published, or "
            "the update failed). Not a recipe failure: work waits. Run `vonkctl update --apply`."
        )
        self.note_infra(policy.CLIENT_CONTRACT, text, dependency=text, escalated=True)

    def requeue_for_changed_owner(self) -> None:
        observations = self._component_observations()
        api = observations.get("api")
        release = api.source_sha if api is not None else None
        for key, entry in self.state.recipes.items():
            if entry.get("status") not in (
                RecipeStatus.FAILED,
                RecipeStatus.DEFERRED,
            ) or not self._selected_key(key):
                continue
            basis = recovery.changed_owner(entry, observations)
            reason = RetestReason.OWNER_CHANGED
            if basis is None:
                basis = recovery.changed_release(entry, release)
                reason = RetestReason.RELEASE_CHANGED
            if basis is not None:
                self._schedule_retest(key, reason, basis=basis)

    def _schedule_retest(
        self, key: str, reason: str, *, basis: Mapping[str, str] | None = None
    ) -> None:
        """Schedule an automatic retest without erasing its cause or retry history."""
        if key in self.recipes:
            view = self.qualification_view(self.recipes[key])
            if view is not None and view.status == QualificationStatus.WORKING:
                return
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

    def requeue_uncertain(self) -> None:
        """Reopen outcomes whose evidence no longer justifies a terminal result."""
        for key, entry in self.state.recipes.items():
            if (
                entry.get("status") == RecipeStatus.DEFERRED
                and (
                    entry.get("failure_class") in verdict.OBSERVE_CLASSES
                    # Skew is per process: a restart re-reads the authored source.
                    or entry.get("failure_class") == verdict.SKEWED
                    # A review refusal the sweep once ended as a fit verdict: only the
                    # Controller's final words still do (the rest are its waits). A
                    # review capacity refusal is never a verdict (stored rows are stale).
                    or (
                        entry.get("phase") == "review"
                        and entry.get("failure_class") == "capacity"
                    )
                    or (
                        entry.get("phase") == "review"
                        and entry.get("failure_class") == policy.FIT
                        and entry.get("code") not in verdict.REVIEW_VERDICT_CODES
                    )
                )
                and self._selected_key(key)
            ):
                # A saved count-exhausted uncertainty still has an owner on restart.
                # Keep its diagnostic count and original download identity.
                attempts = int(entry.get("attempts", 0))
                self._requeue_failed(key, verdict.RecoveryReason.UNCERTAIN)
                due = self.clock.now() + verdict.retry_delay(
                    self.cfg.retry_delay, attempts
                )
                entry.update(attempts=attempts, not_before=due)
                record = self.state.downloads.get(key)
                if record is not None:
                    record["retry_at"] = due

    def requeue_failed(self) -> int:
        """``--retry-failed``: every failed recipe (within ``--only``), whatever the cause."""
        keys = [
            recipe.key
            for recipe in self.recipes.values()
            if self._selected(recipe)
            and (view := self.qualification_view(recipe)) is not None
            and view.status
            in (
                QualificationStatus.FAILED_RECIPE,
                QualificationStatus.FIX_TO_VERIFY,
                QualificationStatus.BLOCKED_PLATFORM,
            )
            and self.state.status(recipe.key) != RecipeStatus.PENDING
            and (
                recipe.key not in self.state.slots
                or self.state.slots[recipe.key].get("phase") == SweepSlotPhase.FINISHED
            )
        ]
        for key in keys:
            self._requeue_failed(key, RetestReason.OPERATOR)
        return len(keys)

    def _selected_key(self, key: str) -> bool:
        return selects(self.cfg.only, key)

    def _requeue_failed(self, key: str, why: str) -> None:
        """Back to pending; the failure stays in results.jsonl, which is only ever appended to."""
        if key in self.recipes:
            view = self.qualification_view(self.recipes[key])
            if view is not None and view.status == QualificationStatus.WORKING:
                return
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
        if record is not None and record.get("state") in (
            LifecycleState.FAILED,
            LifecycleState.CANCELLED,
        ):
            if record.get("operation_id") or record.get("request_key"):
                # Explicit operator intent qualifies a retest, not a terminal
                # receipt. Keep the accepted parent unchanged until fresh exact
                # failure proof permits the canonical parent-bound retry.
                if why == RetestReason.OPERATOR:
                    entry["retest"] = {
                        "reason": why,
                        "at": self.clock.now(),
                        "previous": previous,
                    }
            else:
                record["state"] = DOWNLOAD_RETIRED  # genuinely unaccepted failure
        self.state.event(f"requeued {key}: {why}")
