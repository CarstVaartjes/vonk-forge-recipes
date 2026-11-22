"""Ready candidates, releases, controller waits and scheduling (methods of ``Sweep``)."""

# pyright: reportAttributeAccessIssue=false
# The methods reach the state set in ``Sweep.__init__`` and the sibling mixins through ``self``.
from __future__ import annotations

import random
from collections.abc import Iterable, Mapping
from datetime import UTC, datetime
from functools import partial
from typing import Any

from . import policy, verdict
from .catalog import (
    Recipe,
    fetch_fleet,
)
from .fanout import gather
from .identity import (
    AcceptedRequest,
    ControllerWaitHold,
    content_identity,
)
from .lifecycle import ACTIVE as ACTIVE_APP
from .lifecycle import TERMINAL as SETTLED
from .lifecycle import (
    LifecycleState,
    RecipeStatus,
)
from .qualification import (
    current_identity,
)
from .sweep_contract import (
    ReviewInfrastructureSource,
    SweepSlotPhase,
    SweepStorageWait,
)
from .sweep_support import (
    ENDED_RECEIPT_REPLAN_CAP,
    INFRA_CAP,
    alias_for,
)
from .vonkctl import VonkctlError


class SweepSchedule:
    # ------------------------------------------------------------- scheduling

    def enqueue_finished_cleanup(self) -> None:
        """Durably release finished lane occupancy without waiting on cleanup I/O."""
        pending = self.state.data.setdefault("cleanup", {})
        changed = False
        for key, slot in list(self.state.slots.items()):
            if slot["phase"] != "finished":
                continue
            pending.setdefault(slot["alias"], {"attempts": 0, "next_check": 0.0})
            self.futures.pop(key, None)
            stop = self.smoke_stops.pop(key, None)
            if stop is not None:
                stop.set()
            del self.state.slots[key]
            changed = True
        if changed:
            self.state.save()

    def cleanup_finished(self, *, profile_ready: bool = False) -> None:
        self.enqueue_finished_cleanup()
        pending = self.state.data.setdefault("cleanup", {})
        if pending and not profile_ready and not self._profile_edit_ready():
            return
        live_aliases = {slot["alias"] for slot in self.state.slots.values()}
        for alias, record in list(pending.items()):
            if alias in live_aliases or self.clock.now() < record["next_check"]:
                continue
            # sequential: every removal edits the one shared sweep profile
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
        """The exact NAS assets are there: the recipe row says cached, or our download finished.

        Never fleet fit: it cannot hold while someone else's
        workload fills the Sparks. ``profile load --review`` decides fit at placement.
        """
        download = self.state.downloads.get(recipe.key, {})
        accepted = AcceptedRequest.read(download)
        if (
            download.get("state") == LifecycleState.SUCCEEDED
            and accepted is not None
            and content_identity(accepted.recipe) == content_identity(recipe)
            and (
                self.model_list.complete_started_at
                <= float(download.get("done_at") or 0.0)
                or recipe.model_set <= self.cached_models
            )
        ):
            return True
        return policy.assets_cached(recipe)

    def _review_transport_source(self, recipe: Recipe) -> str:
        # Back off the failed review topology, not a recipe verdict or the
        # whole scheduler. A dual preview outage must leave singles eligible.
        return f"{ReviewInfrastructureSource.TRANSPORT}:{self.cfg.sweep_profile}:{recipe.node_count}"

    def ready_keys(self, keys: Iterable[str]) -> set[str]:
        """Readiness of the planned recipes only, never the whole catalog."""
        return {k for k in keys if k in self.recipes and self.ready(self.recipes[k])}

    def ready_candidates(self) -> list[Recipe]:
        now = self.clock.now()
        # A shared Controller wait holds only the recipes behind it (their own
        # holds below); unrelated content keeps placing.
        waits = self.controller_wait_holds()
        ready = self.ready_keys(self.queue)
        out = []
        for key in self.queue:
            recipe = self.recipes.get(key)
            entry = self.state.recipes.get(key, {})
            if recipe is None or key not in ready or key in self.state.slots:
                continue
            if (
                key in waits
                and waits[key].identity == current_identity(recipe)
                and (waits[key].live or now < float(waits[key].not_before or 0))
            ):
                continue  # re-observe the original receipt, never replay unchanged intent
            if self.lanes_short(recipe):
                continue  # waits for its lanes to come online; the online ones carry on
            if now < float(entry.get("not_before", 0)) or now < float(
                entry.get("defer_until", 0)
            ):
                continue
            incident = self.state.data["infra"].get(
                self._review_transport_source(recipe)
            )
            if incident and now < float(incident["until"]):
                continue
            out.append(recipe)
        deferred = [r for r in out if self.state.entry(r.key).get("wait_deferral")]
        fresh = [r for r in out if not self.state.entry(r.key).get("wait_deferral")]
        return fresh + sorted(
            deferred, key=lambda r: (r.updated_at, r.key), reverse=True
        )

    def _bins(self) -> list[policy.SparkBin]:
        by_revision = {r.revision_id: r for r in self.recipes.values() if r.revision_id}
        bins: dict[str, policy.SparkBin] = {
            s.id: policy.SparkBin(
                s.id,
                s.name,
                s.memory_total,
                model_digests=frozenset(
                    digest
                    for revision in s.installed_revisions
                    if revision in by_revision and by_revision[revision].node_count == 1
                    for digest in by_revision[revision].model_digests
                ),
                installed_revisions=s.installed_revisions,
                disk_blocked_recipes=frozenset(
                    key
                    for key, entry in self.state.data["recipes"].items()
                    if entry.get("storage_wait") is not None
                    and (wait := SweepStorageWait.model_validate(entry["storage_wait"]))
                    and key in self.recipes
                    and self.recipes[key].content_sha256 == wait.content_sha256
                    and self.clock.now() < wait.next_check
                    and s.id in wait.node_ids
                ),
            )
            for s in self.sparks()
        }
        holds = self._lane_holds()
        return [bin_ for bin_ in bins.values() if bin_.id not in holds]

    def _lane_holds(self) -> dict[str, str]:
        """Why each online Spark takes no new placement now; a Spark not named is free.

        The one account the scheduler and the idle alert share.
        """
        holds: dict[str, str] = {}
        for slot in self.state.slots.values():
            if slot["phase"] != "finished":
                for node in slot["node_ids"]:
                    holds.setdefault(node, f"lane {slot['alias']} is {slot['phase']}")
        for node, why in self._cleanup_holds().items():
            holds.setdefault(node, why)
        for record in self.state.data.get("released", {}).values():
            for node in record["nodes"]:
                holds.setdefault(
                    node,
                    "given-up load "
                    + ", ".join(record.get("aliases", ()))
                    + " still shows its workload",
                )
        for node, why in self.adoption_holds.items():
            holds.setdefault(node, why)
        for node, why in self.fit_holds.items():
            holds.setdefault(node, why)
        online = {s.id for s in self.sparks()}
        if wait := self.state.data.get("profile_edit_wait"):
            held = (
                f"profile edit held on {wait['waiting_on']} until "
                f"{datetime.fromtimestamp(wait['wait_deadline'], UTC).isoformat()}"
            )
        else:
            held = str(self.state.data.get("dual_reservation") or "")
        for node in online if held else ():
            holds.setdefault(node, held)
        return {node: why for node, why in holds.items() if node in online}

    def _hold_lanes(self, load: Mapping[str, Any]) -> None:
        """Remember the Sparks and workload names of a load the sweep gives up."""
        lanes = [
            s
            for s in self.state.slots.values()
            if s.get("request_key") == load["request_key"]
        ]
        if not (load.get("app_id") and lanes) or load.get("state") in SETTLED:
            return  # a settled application holds nothing
        now = self.clock.now()
        record = self.state.data.setdefault("released", {}).setdefault(
            load["request_key"],
            {
                "app_id": load["app_id"],
                "since": now,
                "cancel_at": now + self.cfg.retry_delay,
            },
        )
        record["nodes"] = sorted(
            {n for s in lanes for n in s["node_ids"]} | set(record.get("nodes", ()))
        )
        record["aliases"] = sorted(
            {s["alias"] for s in lanes} | set(record.get("aliases", ()))
        )

    def release_given_up(self, now: float) -> None:
        """A load the sweep gave up is cancelled until the Spark shows none of its workloads.

        Left alone, the platform's start child keeps retrying and holds the Spark, and every later
        load of the lane is refused. Only the lanes of that load wait; the other lane carries on,
        and the hold lasts exactly as long as the Controller shows the workload.
        """
        held = self.state.data.get("released")
        if not held:
            return
        try:
            fleet = fetch_fleet(self.vk, self.cfg.default_spark_memory)
        except VonkctlError as error:
            self.state.event(f"held lanes unreadable, still held: {str(error)[:120]}")
            return  # unreadable: keep holding, look again
        for key, record in list(held.items()):
            shown = any(
                p.alias in record["aliases"] and p.node_id in record["nodes"]
                for p in fleet.presences
            )
            # The Spark is occupied for as long as the Controller shows the workload.
            if not shown:
                del held[key]
                self.state.event(
                    f"released {record['aliases']} on {record['nodes']}: lane free again"
                )
            elif now >= record["cancel_at"] and not self._ridden_by_live_lane(
                record["app_id"], key
            ):
                self.vk.run(
                    "profile",
                    "cancel",
                    record["app_id"],
                    "--yes",
                    "--detach",
                    profile=self.cfg.sweep_profile,
                )
                record["cancel_at"] = now + self.cfg.retry_delay
        self.state.save()

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
        if load.get("replanned_pending"):
            return  # observe until the fresh snapshot supersedes it, never cancel a race
        if self._adopted_slots_live(load):
            return  # selected aggregate still owns an active borrowed lane
        key = load.get("request_key")
        if any(
            s.get("request_key") == key and s["phase"] != "finished"
            for s in self.state.slots.values()
        ):
            return
        app = load.get("app_id")
        if (
            app
            and load.get("state") not in SETTLED
            # A stalled application released by its lane is the Controller's to heal.
            and not load.get("stall_released")
            and not self._ridden_by_live_lane(app, key)
            # A newer profile load adopted this request: its intent owns the workload,
            # and the Controller rejects a cancel of it as an invalid scope.
            and not any(
                key in (other.get("adopted_requests") or [])
                for other in self.state.data["loads"].values()
                # Only a live, admitted load owns what it adopted.
                if other is not load
                and other.get("app_id")
                and other.get("state") not in SETTLED
            )
        ):
            # Ours (the sweep's own request key): cancel it so it releases what it holds.
            result = self.vk.run(
                "profile",
                "cancel",
                app,
                "--yes",
                "--detach",
                profile=self.cfg.sweep_profile,
            )
            if result.exit_code:
                cause = (result.stderr or result.stdout or "").strip()[:300]
                self.state.event(
                    f"cancel of {app} failed ({result.exit_code}): {cause}"
                )
        self.state.event(f"released load {app or key}: its lanes have ended")
        self.state.data["loads"].pop(key, None)

    def check_idle(self, now: float) -> None:
        """Alert when an idle lane has had ready work for too long (the sweep must never idle).

        Each idle lane is named with what holds it, from the same account the scheduler
        places by (``_lane_holds``), so the alert never reports a lane free that placement
        skips.
        """
        holds = self._lane_holds()
        lanes = {
            node
            for slot in self.state.slots.values()
            if slot["phase"] != "finished"
            for node in slot["node_ids"]
        }
        idle_lanes = [s for s in self.sparks() if s.id not in lanes]
        waiting = bool(idle_lanes) and not self.paused and bool(self.ready_candidates())
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
        unheld = (
            self.state.data.get("clearing_reason")
            or "placement review or lane admission"
        )
        causes = "; ".join(f"{s.name}: {holds.get(s.id) or unheld}" for s in idle_lanes)
        message = (
            f"IDLE WITH READY WORK: {len(idle_lanes)} idle lane(s) for {round(idle)}s "
            f"while recipes are ready (load record: "
            f"{len(self.state.data['loads'])}; {causes})"
        )
        if not self.state.data.get("alerts"):
            self.state.event(message)
        self.state.data["alerts"] = [message]

    def controller_wait_holds(self) -> dict[str, ControllerWaitHold]:
        return {
            key: ControllerWaitHold.model_validate(entry["controller_wait"])
            for key, entry in self.state.recipes.items()
            if entry.get("controller_wait") is not None
        }

    def controller_wait_issue(self) -> str | None:
        # An ended receipt never resumes: it can neither prove nor release an incident.
        holds = [hold for hold in self.controller_wait_holds().values() if hold.live]
        for hold in holds:
            controller = hold.wait.controller
            if controller is None or controller.identity is None:
                continue
            if any(
                other.identity != hold.identity
                and other.wait.controller is not None
                and other.wait.controller.identity == controller.identity
                for other in holds
            ):
                operations = ", ".join(
                    f"{kind} {operation}"
                    for kind, operation in sorted(controller.identity[1])
                )
                return (
                    f"shared Controller wait {controller.description} ({operations}); "
                    "its recipes wait for changed evidence"
                )
        return None

    def _read_wait_receipt(self, application_id: str) -> Any:
        return self.vk.run(
            "profile",
            "progress",
            "--load",
            application_id,
            profile=self.cfg.sweep_profile,
            timeout=30,
        )

    def _refresh_controller_waits(self, now: float) -> None:
        """Probe live receipts with bounded reads.

        Time alone never resubmits a load on a live receipt; an ended receipt's recipe
        is planned again at its ``not_before``.
        """
        documents: dict[str, Mapping[str, Any]] = {}
        recovered: set[tuple[str, frozenset[tuple[str, str]]]] = set()
        platform = self._component_observations()
        probing: list[tuple[str, ControllerWaitHold]] = []
        for key, hold in self.controller_wait_holds().items():
            recipe = self.recipes.get(key)
            if recipe is not None and current_identity(recipe) != hold.identity:
                self.state.entry(key).pop("controller_wait", None)
                continue  # latest content gets a fresh canonical review
            if any(
                platform.get(owner) is not None and platform[owner] != fingerprint
                for owner, fingerprint in hold.platform.items()
            ):
                self.state.entry(key).pop("controller_wait", None)
                self.state.event(
                    f"deployed platform changed for {key}; re-planning placement"
                )
                continue
            if not hold.live:
                if self.state.entry(key).get("status") != RecipeStatus.PENDING:
                    self.state.entry(key).pop("controller_wait", None)  # a verdict
                continue  # planned again at its not_before; nothing left to observe
            if now < hold.next_check:
                continue
            hold.next_check = now + self.cfg.defer_delay
            self.state.entry(key)["controller_wait"] = hold.model_dump(mode="json")
            probing.append((key, hold))
        # The held applications' receipts are independent reads; their effects apply in order.
        replies = gather(
            {
                hold.application_id: partial(
                    self._read_wait_receipt, hold.application_id
                )
                for _, hold in probing
            },
            self.read_executor,
        )
        for key, hold in probing:
            if hold.application_id not in documents:
                reply = replies[hold.application_id].result()
                if reply.is_not_found:
                    # The Controller's definite answer: no such receipt waits any more.
                    self._replan_ended(key, hold, "is absent at the Controller", now)
                    self.state.entry(key)["controller_wait"] = hold.model_dump(
                        mode="json"
                    )
                    continue
                if not reply.ok or not isinstance(reply.document, Mapping):
                    continue  # unknown is no recovery proof; next probe remains visible
                if reply.document.get("id") != hold.application_id:
                    continue
                documents[hold.application_id] = reply.document
            document = documents[hold.application_id]
            if document.get("state") not in ACTIVE_APP | SETTLED:
                continue
            if (
                document.get("state") in SETTLED
                and document.get("state") != LifecycleState.SUCCEEDED
            ):
                # An ending is no recovery proof. The Controller's continuation of
                # this request carries the wait (as the poll path follows it); another
                # recipe's load does not, and then nothing waits on this receipt.
                replaced = policy.supersession(document)
                successor = replaced.successor if replaced is not None else None
                if successor not in (None, document["id"]) and self._own_load_request(
                    str(successor)
                ) in (None, hold.request_key):
                    hold.application_id = str(successor)
                else:
                    self._replan_ended(
                        key,
                        hold,
                        f"ended {document.get('state')}"
                        + (f", taken over by {successor}" if successor else ""),
                        now,
                    )
                self.state.entry(key)["controller_wait"] = hold.model_dump(mode="json")
                continue
            signal = verdict.application_signal(document)
            observed = signal.wait.controller if signal.wait is not None else None
            previous = hold.wait.controller
            resumed = (
                signal.wait is None or document.get("state") == LifecycleState.SUCCEEDED
            )
            # A new child id under the same reason is the Controller continuing
            # the same work (a re-dispatched child, a retry), not changed evidence.
            if resumed or (
                observed is not None
                and previous is not None
                and observed.reason != previous.reason
            ):
                self.state.entry(key).pop("controller_wait", None)
                self.state.event(
                    f"Controller wait changed for {key}; re-planning placement"
                )
                if resumed and previous is not None and previous.identity is not None:
                    recovered.add(previous.identity)
            elif observed is not None and signal.wait is not None:
                # Matching across recipes uses the operations waited on now.
                hold.wait = signal.wait
                self.state.entry(key)["controller_wait"] = hold.model_dump(mode="json")
        # One original receipt resuming is behaviour proof for the shared dependency.
        for key, hold in self.controller_wait_holds().items():
            if (
                hold.live
                and hold.wait.controller is not None
                and hold.wait.controller.identity in recovered
            ):
                self.state.entry(key).pop("controller_wait", None)

    def _replan_ended(
        self, key: str, hold: ControllerWaitHold, why: str, now: float
    ) -> None:
        """Plan a recipe whose wait receipt ended again later, never straight back.

        The unchanged recipe would meet the same admission wait at once. The delay is
        as long as it has kept ending (doubling by time, no attempt count), jittered,
        from the sweep's retry pace up to ENDED_RECEIPT_REPLAN_CAP.
        """
        since = hold.ended_since if hold.ended_since is not None else now
        delay = min(ENDED_RECEIPT_REPLAN_CAP, max(self.cfg.defer_delay, now - since))
        hold.ended_since = since
        hold.not_before = now + delay * random.uniform(1.0, 1.25)
        self.state.event(
            f"Controller wait receipt {hold.application_id} for {key} {why}; "
            f"planning it again at {datetime.fromtimestamp(hold.not_before, UTC).isoformat()}"
        )

    def schedule(self, now: float) -> None:
        self.state.data.pop("dual_reservation", None)
        self.adoption_holds.clear()
        self.fit_holds.clear()
        self._refresh_controller_waits(now)
        self.release_given_up(now)
        self.release_orphan_load()
        candidates = self.ready_candidates()
        if not candidates:
            return
        if not self._take_over(now, place_ready=True):
            return
        attempted: set[str] = set()
        # Each plan consumes a candidate; retries happen on a later tick.
        for _ in range(len(candidates)):
            if not candidates:
                return
            if attempted:
                self._observe_owner(self.clock.now())
                if self.paused or self.owner_status.new_activity:
                    return
            bins = self._bins()
            if not bins:
                return
            # A gang can drain healthy lanes, but cannot reserve a free lane behind a
            # stalled application or unresolved cleanup on its other member.
            if (
                candidates[0].node_count == 2
                and len(bins) < 2
                and (
                    self._cleanup_nodes()
                    or self.state.data.get("released")
                    or self.unobserved_loads
                    or any(
                        load.get("stalled")
                        or load.get("app_id") in self.stalled_applications
                        for load in self.state.data["loads"].values()
                    )
                )
            ):
                candidates = [recipe for recipe in candidates if recipe.node_count == 1]
                if not candidates:
                    return
            # A dual at the head drains both lanes; do not refill one with later singles.
            offered = candidates[:1]
            if candidates[0].node_count == 2:
                if len(bins) < 2:
                    self._reserve_for_dual(candidates[0], {b.id for b in bins})
                    return
                self.state.data["mode"] = "dual"
                dual = policy.place_dual(candidates[:1], bins)
                placements = [dual] if dual else []
            else:
                self.state.data["mode"] = "single"
                singles = []
                for recipe in candidates:
                    if recipe.node_count == 2:
                        break
                    singles.append(recipe)
                offered = singles
                placements = policy.place_singles(
                    singles,
                    bins,
                    lambda r: alias_for(r, self.defs),
                )
            if not placements:
                # Declared demand beyond one Spark: let the platform's review decide.
                # Only a live disk wait keeps a free Spark from being offered.
                for head in offered:
                    open_bins = [
                        b for b in bins if head.key not in b.disk_blocked_recipes
                    ]
                    if len(open_bins) >= head.node_count:
                        spark_ids = tuple(b.id for b in open_bins[: head.node_count])
                        placements = [policy.Placement(head, spark_ids)]
                        break
            if placements:
                attempted.update(p.recipe.key for p in placements)
                if self._apply(placements, self.clock.now()) is False:
                    return
            else:
                head = candidates[0]
                wait = self.state.entry(head.key).get("storage_wait")
                if wait is not None:
                    wait = SweepStorageWait.model_validate(wait)
                    retry = datetime.fromtimestamp(wait.next_check, UTC).isoformat()
                    for b in bins:
                        if head.key in b.disk_blocked_recipes:
                            detail = "; ".join(
                                r.detail for r in wait.reasons if b.id in r.node_ids
                            )
                            self.fit_holds[b.id] = (
                                f"ready {head.key} waits for disk on {b.name}: "
                                f"{detail}; reviewed again at {retry}"
                            )
                return
            candidates = [r for r in self.ready_candidates() if r.key not in attempted]

    def _reserve_for_dual(self, recipe: Recipe, free: set[str]) -> None:
        """Name why a free lane stays empty: the dual at the head drains both lanes."""
        busy = {why for node, why in self._lane_holds().items() if node not in free}
        lanes = [
            float(slot["deadline"])
            for slot in self.state.slots.values()
            if slot["phase"] != SweepSlotPhase.FINISHED
            and not set(slot["node_ids"]) <= free
        ]
        until = (
            f" until {datetime.fromtimestamp(max(lanes), UTC).isoformat()}"
            if lanes
            else ""
        )
        self.state.data["dual_reservation"] = (
            f"reserved for dual {recipe.key} behind {'; '.join(sorted(busy))}{until}"
        )
