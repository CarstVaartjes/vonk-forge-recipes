"""Polling an application and advancing loading slots (methods of ``Sweep``)."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import replace
from functools import partial
from typing import TYPE_CHECKING, Any

from . import policy, verdict
from .catalog import (
    dig,
    fetch_fleet,
    serving_run,
)
from .fanout import gather
from .identity import (
    AcceptedRequest,
    IdentityReason,
)
from .lifecycle import ACTIVE as ACTIVE_APP
from .lifecycle import TERMINAL as SETTLED
from .lifecycle import (
    LifecycleState,
)
from .qualification import (
    current_identity,
)
from .sweep_contract import (
    SweepSlotPhase,
    SweepWait,
    SweepWaitReason,
    WorkloadFailureCode,
)
from .sweep_support import (
    SKEW_CHECK_INTERVAL,
)
from .verdict import Disposition, Progress
from .vonkctl import Reply, VonkctlError

if TYPE_CHECKING:
    from .run import Sweep


class SweepLoading:
    # ------------------------------------------------------------------ slots

    def poll_application(self: Sweep) -> bool:  # pyright: ignore[reportGeneralTypeIssues]
        """Refresh the load's application; False when the Controller could not be observed."""
        observed = True
        fleet_blind = self.blind
        loads = list(self.state.data["loads"].values())
        # The lanes' progress reads are independent, bounded by the live lanes (one thread
        # each); the state they change is applied in order.
        replies = gather(
            {load["request_key"]: partial(self._read_progress, load) for load in loads},
            self.read_executor,
        )
        for load in loads:
            self.blind = fleet_blind
            fresh = self._poll_application(load, replies[load["request_key"]].result())
            if not fresh:
                self.unobserved_loads.add(load["request_key"])
            observed = fresh and observed
        self.blind = fleet_blind
        return observed

    def _read_progress(self: Sweep, load: dict[str, Any]) -> Any:  # pyright: ignore[reportGeneralTypeIssues]
        args = ["progress"]
        if load.get("app_id"):
            args += ["--load", load["app_id"]]
        else:
            args += ["--request-key", load["request_key"]]
        return self.vk.run("profile", *args, profile=self.cfg.sweep_profile)

    def _poll_application(self: Sweep, load: dict[str, Any], reply: Any) -> bool:  # pyright: ignore[reportGeneralTypeIssues]
        document = reply.document
        if reply.is_not_found and not load.get("app_id"):
            # An exact fresh lookup proved there is no receipt. Replay the same request.
            try:
                document = self._submit_load(
                    self.cfg.sweep_profile, load["request_key"], "placement"
                )
            except VonkctlError as error:
                self._blind(str(error), error.reply)
                return False
        elif reply.is_not_found:
            # A known application the Controller does not find: ask by our own request key,
            # which names the same intent, before concluding anything.
            by_key = self.vk.run(
                "profile",
                "progress",
                "--request-key",
                load["request_key"],
                profile=self.cfg.sweep_profile,
            )
            if by_key.is_not_found:
                self._load_absent(load, by_key)
                return True
            if not by_key.ok:
                self._blind(by_key.error_text, by_key)
                return False
            document = by_key.document
        elif not reply.ok:
            self._blind(reply.error_text, reply)
            return False
        if not isinstance(document, dict) or not isinstance(document.get("id"), str):
            self._blind("application observation is unreadable")
            return False
        # This receipt resolves submission uncertainty, never another review's wait.
        self.clear_infra("load-submit")
        owner = self._own_load_request(document["id"])
        if owner not in (None, load["request_key"]):
            self._carry_loading_slots(owner)
            # Only lanes absent from the successor's canonical adoption requeue.
            self.state.data.setdefault("apps", {})[load["request_key"]] = {
                "state": LifecycleState.SUPERSEDED,
                "reason": f"superseded by this sweep's own load {owner}",
            }
            self.state.data["loads"].pop(load["request_key"], None)
            return True
        self.documents[load["request_key"]] = document
        load["admission_unbound"] = verdict.admission_unbound(document)
        load["observed_at"] = self.clock.now()
        load["app_id"] = document["id"]
        status = str(document.get("state", ""))
        load["state"] = status
        moving = policy.distribution(
            document,
            previous_bytes=int(load["copy_bytes"]) if "copy_bytes" in load else None,
        )
        load["copy_bytes"] = moving.completed_bytes
        load["copying"] = moving.copying
        load["phase"] = moving.phase
        load["blockers"] = policy.admission_blockers(document)
        signal = verdict.application_signal(document)
        # Capture known deployed content before a long wait ages its provenance.
        if (
            signal.wait is not None
            and signal.wait.controller is not None
            and not load.get("wait_platform")
        ):
            load["wait_platform"] = {
                owner: fingerprint.record()
                for owner, fingerprint in self._component_observations().items()
                if owner in ("api", "worker") and fingerprint is not None
            }
        load["signal"] = {
            "signature": list(signal.signature),
            "wait": signal.wait.model_dump(mode="json") if signal.wait else None,
            "marks": list(signal.marks),
        }
        load["stalled"] = verdict.stalled_detail(document)
        replaced = policy.supersession(document) if status in SETTLED else None
        if replaced is not None and replaced.successor is not None:
            successor = self._own_load_request(replaced.successor)
            if successor is not None:
                self._carry_loading_slots(successor)
        if (
            replaced is not None
            and replaced.successor not in (None, document["id"])
            and self._own_load_request(replaced.successor)
            in (None, load["request_key"])
        ):
            # The Controller continued this application as another one: follow it.
            # Our own successors carry adopted lanes on their existing receipt.
            self.state.event(
                f"application {document['id']} was superseded: following {replaced.successor}"
            )
            load["app_id"] = replaced.successor
            load["state"] = "queued"
            load["copying"] = False
            # Unobserved, the successor may still be pending admission.
            load.pop("admission_unbound", None)
            return True
        if status in SETTLED:
            child = dig(document, "progress", "child_progress", "phase")
            self.state.data.setdefault("apps", {})[load["request_key"]] = {
                "state": status,
                "reason": document.get("status_reason"),
                "adopted_requests": load.get("adopted_requests", []),
                "operation_id": document.get("current_operation_id") or document["id"],
                "child_phase": child,
                "cause": dig(document, "cancellation", "cause"),
                "failures": [
                    f.as_state() for f in policy.assignment_failures(document)
                ],
            }
            self.state.data["loads"].pop(load["request_key"], None)
        return True

    def _carry_loading_slots(self: Sweep, request: str) -> None:  # pyright: ignore[reportGeneralTypeIssues]
        """Observe unchanged lanes through the latest accepted whole-fleet load.

        Admission already bound each adopted request's exact application, plan,
        assignment and Spark scope in `_adopted_requests`. Keep its accepted
        recipe content and deadline; only the aggregate observer changes.
        """
        successor = self.state.data["loads"].get(request) or self.state.data.get(
            "apps", {}
        ).get(request)
        if successor is None:
            return
        adopted = set(successor.get("adopted_requests") or [])
        for slot in self.state.slots.values():
            if (
                slot["phase"] == SweepSlotPhase.LOADING
                and slot["request_key"] in adopted
            ):
                slot["request_key"] = request

    def _own_load_request(self: Sweep, app_id: str) -> str | None:  # pyright: ignore[reportGeneralTypeIssues]
        """The request key of this sweep's own load whose application ``app_id`` is."""
        for item in self.state.data.get("own_loads") or []:
            if isinstance(item, dict) and item.get("application_id") == app_id:
                request = item.get("request_key")
                return request if isinstance(request, str) else None
        return None

    def _load_absent(self: Sweep, load: dict[str, Any], reply: Reply) -> None:  # pyright: ignore[reportGeneralTypeIssues]
        """The Controller has no application for this load, by id or by request key.

        That is its answer, not a wait: the lanes are requeued without blame or attempt,
        each with a results row carrying the not-found document.
        """
        evidence = reply.document if isinstance(reply.document, Mapping) else None
        self.state.data["loads"].pop(load["request_key"], None)
        for key, slot in list(self.state.slots.items()):
            if (
                slot.get("request_key") == load["request_key"]
                and slot["phase"] == "loading"
            ):
                self.requeue(
                    key,
                    f"application.absent: the Controller has no application {load.get('app_id')}",
                    document=evidence,
                )
                self.state.entry(key)["not_before"] = (
                    self.clock.now() + self.cfg.retry_delay
                )

    def _blind(self: Sweep, message: str, reply: Any = None) -> None:  # pyright: ignore[reportGeneralTypeIssues]
        """The sweep cannot observe the Controller (outage, 502/503, protocol or version skew)."""
        self.blind = True
        self.note_infra("observe", message, typed=reply)
        # The typed ``client.contract_mismatch`` is answered in ``Vonkctl.run`` (its message also
        # says "OpenAPI"). Untyped ``protocol_invalid`` remains: older clients, and new ones for a
        # damaged stamp or an unanswered compatibility probe.
        typed = reply is not None and getattr(reply, "is_contract_mismatch", False)
        if not typed and ("protocol_invalid" in message or "OpenAPI" in message):
            self._diagnose_skew()

    def contract_mismatch(self: Sweep, command: tuple[str, ...]) -> None:  # pyright: ignore[reportGeneralTypeIssues]
        """Any thread: a Controller command met the typed ``client.contract_mismatch``.

        Only a flag is set; ``check_release`` on the main loop owns the answer.
        """
        with self._contract_lock:
            self._mismatched.add(command)
            self._mismatch_pending = True

    def controller_answered(self: Sweep, command: tuple[str, ...]) -> None:  # pyright: ignore[reportGeneralTypeIssues]
        """Any thread: a command succeeded. Only one that met the mismatch heals it."""
        with self._contract_lock:
            if command in self._mismatched:
                self._mismatched.discard(command)
                self._mismatch_healed = not self._mismatched

    def _diagnose_skew(self: Sweep) -> None:  # pyright: ignore[reportGeneralTypeIssues]
        """protocol_invalid means vonkctl is older than the Controller: say so, loudly, once in a while."""
        now = self.clock.now()
        if now < self._skew_checked_at + SKEW_CHECK_INTERVAL:
            return
        self._skew_checked_at = now
        check = self.vk.run("update", timeout=60)  # read-only: never --apply
        document = check.document if isinstance(check.document, dict) else {}
        current = dig(document, "current", "version", default="unknown")
        accepted = document.get("accepted_version", "unknown")
        self.state.event(
            "WARNING: the Controller answers in a protocol this vonkctl does not speak "
            f"(vonkctl {current}, accepted release {accepted}): the CLIENT MUST BE UPDATED, "
            "run `vonkctl update --apply`. Not a recipe failure: loads are paused, not timed."
        )

    def advance_slots(self: Sweep, now: float) -> None:  # pyright: ignore[reportGeneralTypeIssues]
        self.blind = False
        self.unobserved_loads.clear()
        if not self.state.slots:
            self.poll_application()
            if not self.blind and not self.unobserved_loads:
                self.clear_infra("observe")
            return
        try:
            self.fleet = self._tick_read(
                "fleet", lambda: fetch_fleet(self.vk, self.cfg.default_spark_memory)
            )
        except VonkctlError as error:
            # An unreadable fleet is unknown: nothing is judged on it this pass.
            self._blind(str(error), error.reply)
        self.poll_application()
        if not self.blind and not self.unobserved_loads:
            self.clear_infra("observe")
        for key, slot in list(self.state.slots.items()):
            if slot["phase"] == "loading":
                self._advance_loading(key, slot, now)
            elif slot["phase"] == "smoking":
                self._advance_smoking(key, slot)

    @staticmethod
    def _engine_load_seconds(slot: Mapping[str, Any], now: float) -> float:
        """The engine's own load time: held Controller waits and copies are not it."""
        return max(
            0.0,
            now
            - slot["started_at"]
            - float(slot.get("copy_seconds", 0))
            - float(slot.get("waited_seconds", 0)),
        )

    def _load_window(self: Sweep, slot: Mapping[str, Any]) -> float:  # pyright: ignore[reportGeneralTypeIssues]
        """The lane's stall window: learned per engine, or the copy window while bytes move."""
        if slot.get("copying"):
            return float(self.cfg.copy_stall_seconds)
        return float(slot["deadline"]) - float(slot["started_at"])

    def _observe_load(self: Sweep, slot: dict[str, Any], now: float) -> Progress:  # pyright: ignore[reportGeneralTypeIssues]
        """What the slot's load is doing: PROGRESSING, WAITING or STALLED (``progress_verdict``).

        Moving copies retain their inactivity window. Dependency waits use the lane
        budget from admission, restarted only when a forward-only counter rises; changing
        reasons, retries and unreadable observations cannot extend it.
        """
        load = self.state.data["loads"].get(slot["request_key"])
        elapsed = now - float(slot.get("last_tick", slot["started_at"]))
        slot["last_tick"] = now
        watch = slot.setdefault("watch", {})
        unreadable = self.blind or slot["request_key"] in self.unobserved_loads
        if unreadable or load is None or not isinstance(load.get("signal"), Mapping):
            if unreadable:
                slot["blind_seconds"] = float(slot.get("blind_seconds", 0)) + elapsed
            return verdict.progress_verdict(
                None,
                watch,
                window=float(slot["deadline"]) - float(slot["started_at"]),
                now=now,
                started_at=float(slot["started_at"]),
            )
        copying = bool(load.get("copying"))
        slot["copying"] = copying
        if copying:
            slot["copy_seconds"] = float(slot.get("copy_seconds", 0)) + elapsed
        blockers = None if copying else load.get("blockers")
        if blockers:
            slot["blocked_seconds"] = float(slot.get("blocked_seconds", 0)) + elapsed
            slot.setdefault("blocked_since", now)
            slot["blocked"] = blockers
        else:
            slot.pop("blocked_since", None)
            slot.pop("blocked", None)
        if load.get("phase"):
            slot["load_phase"] = load["phase"]
        signal = verdict.Signal(
            tuple(load["signal"].get("signature") or ()),
            SweepWait.model_validate(load["signal"]["wait"])
            if load["signal"].get("wait")
            else None,
            tuple(load["signal"].get("marks") or ()),
        )
        if (
            copying
            and load.get("state") in ACTIVE_APP
            and (signal.wait is None or signal.wait.controller is None)
        ):
            # Distribution retries still own the same copy. Only changed bytes
            # or phases restart its window; a retry timestamp cannot keep a lane
            # forever. The existing stalled-copy path releases and observes it.
            signal = replace(signal, wait=None)
        if signal.waiting_on is not None and not copying:
            # A copying tick is already counted in copy_seconds; counting it again
            # made learn() subtract it twice and learn a load time of zero.
            slot["waited_seconds"] = float(slot.get("waited_seconds", 0)) + elapsed
        window = (
            float(slot["deadline"]) - float(slot["started_at"])
            if signal.wait
            else self._load_window(slot)
        )
        if signal.wait and verdict.preparation_blockers(load.get("blockers") or []):
            window = min(window, self.cfg.blocked_release_seconds)
        return verdict.progress_verdict(
            signal, watch, window=window, now=now, started_at=float(slot["started_at"])
        )

    def _observe_readiness(self: Sweep, slot: dict[str, Any], now: float) -> Progress:  # pyright: ignore[reportGeneralTypeIssues]
        """After the load succeeded: the run turning healthy and its route being published.

        A healthy run whose route is not published yet waits on the Controller's route
        publication; a run that is missing or unhealthy is the recipe's own work and stalls
        after the lane's window without any change. The publication wait has no forward
        counter, so a new run or route state never renews it.
        """
        prior = set(slot.get("prior_runs", ()))
        found = [
            p
            for p in self.fleet.presences
            if p.alias == slot["alias"]
            and p.node_id in slot["node_ids"]
            and p.run_id not in prior
        ]
        signature = tuple(
            sorted(
                (p.node_id, p.run_id, p.run_state, p.healthy, p.route_state)
                for p in found
            )
        )
        publishing = bool(found) and all(
            p.healthy and p.run_state == "running" for p in found
        )
        return verdict.progress_verdict(
            verdict.Signal(
                signature,
                SweepWait(reason=SweepWaitReason.ROUTE_PUBLICATION)
                if publishing
                else None,
            ),
            slot.setdefault("ready_watch", {}),
            window=float(slot["deadline"]) - float(slot["started_at"]),
            now=now,
            started_at=float(slot["started_at"]),
        )

    def _advance_loading(
        self: Sweep,  # pyright: ignore[reportGeneralTypeIssues]
        key: str,
        slot: dict[str, Any],
        now: float,
    ) -> None:
        accepted = AcceptedRequest.read(slot)
        if accepted is None:
            self.requeue(key, IdentityReason.UNBOUND)
            return
        recipe = accepted.recipe
        latest = self.recipes.get(key)
        # The Controller owns a held receipt's end; newer content steers past it.
        held = (
            verdict.held_receipt(self.state.data["loads"].get(slot["request_key"]))
            and latest is not None
            and current_identity(latest) == current_identity(recipe)
        )
        if self.blind or slot["request_key"] in self.unobserved_loads:
            # Unknown is no ending: a receipt last seen held stays held.
            if self._observe_load(slot, now) is Progress.WAIT_EXPIRED and not held:
                self._expire_wait(key, slot, slot["watch"])
            return
        run_id = serving_run(
            self.fleet,
            slot["alias"],
            slot["node_ids"],
            recipe.node_count,
            slot.get("prior_runs", ()),
            job=recipe.is_job,
        )
        if run_id:
            slot.pop("waiting_on", None)
            slot.update(
                phase="smoking", run_id=run_id, ready_at=now, smoke_started_at=now
            )
            self.clear_infra("load-stall")
            policy.learn(
                self.state.data["learned"],
                recipe.engine,
                slot["model_bytes"],
                self._engine_load_seconds(slot, now),
            )
            self._submit_smoke(key, slot)
            return
        app = self.state.data.get("apps", {}).get(slot["request_key"])
        if app is not None:
            if app["state"] == LifecycleState.SUPERSEDED:
                self.requeue(key, "load superseded by another application")
                return
            if app["state"] == LifecycleState.CANCELLED:
                # A cancel is an ending somebody requested (the operator, the sweep itself,
                # a superseding load), never the recipe's outcome. A child the cancel could
                # not confirm keeps its own lifecycle: the next load's admission waits on it.
                self.requeue(
                    key,
                    f"application cancelled (cause {app.get('cause') or 'unknown'}): "
                    "a cancel is no verdict on the recipe",
                    status_reason=app.get("reason"),
                    operation_id=app.get("operation_id"),
                )
                self.state.entry(key)["not_before"] = now + self.cfg.retry_delay
                return
            if app["state"] == LifecycleState.SUCCEEDED:
                progress = self._observe_readiness(slot, now)
                if progress is Progress.WAIT_EXPIRED:
                    self._expire_wait(key, slot, slot["ready_watch"])
                elif progress is Progress.STALLED:
                    silent = verdict.silent_seconds(slot["ready_watch"], now)
                    self._fail_slot(
                        key,
                        policy.classify(
                            "readiness",
                            "run.not_published",
                            "the load succeeded but the run never became healthy and "
                            f"{'held' if recipe.is_job else 'published'} "
                            f"(no change for {round(silent)}s)",
                        ),
                        app,
                    )
                return
            group = self._lane_group(slot)
            stored = [
                policy.AssignmentFailure.from_state(f)
                for f in app.get("failures") or []
                if isinstance(f, Mapping)
            ]
            unattributed = False
            if len(group) > 1:
                blamed = policy.blamed(
                    [f.as_state() for f in stored], self._lanes(group)
                )
                culprits, collateral = self._split(group, blamed)
                self._requeue_collateral(collateral, culprits, "its application failed")
                unattributed = not blamed
                group = culprits
            else:
                group = [key]
            phase = policy.child_phase(app.get("child_phase")) or "start"
            detail = str(app.get("reason") or app["state"])
            for member in group:
                own_failure = next(
                    (
                        f
                        for f in stored
                        if member in policy.blamed([f.as_state()], self._lanes(group))
                    ),
                    None,
                )
                failure = policy.classify_application_failure(
                    phase,
                    WorkloadFailureCode.APPLICATION_FAILED,
                    own_failure.reason or detail if own_failure else detail,
                    evidence={"assignment_failures": app.get("failures") or []},
                )
                if unattributed:
                    # Nothing the Controller said names this lane: an uncertain outcome,
                    # observed again (one attempt each), never a verdict on one of them.
                    failure = replace(failure, klass=verdict.UNATTRIBUTED)
                self._fail_slot(member, failure, app)
            return
        load = self.state.data["loads"].get(slot["request_key"]) or {}
        if load.get("stalled"):
            self._controller_stalled(key, slot, load)
            return
        load.pop("stall_released", None)
        slot.pop("waiting_on", None)
        progress = self._observe_load(slot, now)
        if held and progress in (Progress.WAITING, Progress.WAIT_EXPIRED):
            # Held time is the Controller's, never the recipe's own silence.
            slot["watch"]["progressed_at"] = now
        if progress is Progress.WAIT_EXPIRED:
            # The local window alone never ends a receipt the Controller still works.
            if not held:
                self._expire_wait(key, slot, slot["watch"])
            return
        if progress is Progress.STALLED:
            self._stalled(key, slot, now)

    def _controller_stalled(
        self: Sweep,  # pyright: ignore[reportGeneralTypeIssues]
        key: str,
        slot: dict[str, Any],
        load: dict[str, Any],
    ) -> None:
        """The Controller reports the lane's application stalled: a platform problem.

        Observe the original receipt within the lane budget. Expiry re-plans locally
        without asserting a remote ending or spending a recipe attempt.
        """
        declared = (load.get("signal") or {}).get("wait")
        wait = SweepWait.model_validate(declared) if declared is not None else None
        if self._wait_slot(
            key,
            slot,
            wait
            if wait is not None and wait.controller is not None
            else SweepWait(
                reason=SweepWaitReason.CONTROLLER_STALLED,
                detail=f"application {load.get('app_id')}: {load['stalled']}",
            ),
            watch_name="watch",
            started_at=float(slot["started_at"]),
        ):
            return
        if load.get("stall_released"):
            return
        load["stall_released"] = True
        detail = str(load["stalled"])
        self.record_result(
            key,
            "load",
            LifecycleState.OBSERVING,
            reason=f"the Controller reports the load stalled: {detail}",
            slot=slot,
            disposition=Disposition.OBSERVE.value,
            failure_class=verdict.PLATFORM_STALLED,
            code=verdict.STALLED_CODE,
            evidence={"application_id": load.get("app_id")},
        )

    def _stalled(self: Sweep, key: str, slot: dict[str, Any], now: float) -> None:  # pyright: ignore[reportGeneralTypeIssues]
        """The load's own progress was silent for a whole window: the one verdict that ends it."""
        silent = round(verdict.silent_seconds(slot.get("watch", {}), now))
        if slot.get("copying"):
            self._timeout(
                key,
                slot,
                policy.classify(
                    "install",
                    "copy.stalled",
                    f"the copy to the Spark made no progress for {silent}s "
                    f"(phase {slot.get('load_phase')})",
                    "copy-stalled",
                ),
            )
            return
        self._timeout(key, slot)

    def _lane_group(self: Sweep, slot: Mapping[str, Any]) -> list[str]:  # pyright: ignore[reportGeneralTypeIssues]
        """The lanes still loading in the same application as this one."""
        return [
            k
            for k, s in self.state.slots.items()
            if s["phase"] == "loading"
            and s["request_key"] == slot["request_key"]
            and not serving_run(
                self.fleet,
                s["alias"],
                s["node_ids"],
                self.recipes[k].node_count,
                s.get("prior_runs", ()),
            )  # a lane that already serves goes on to its smoke
        ]

    def _lanes(self: Sweep, group: Sequence[str]) -> dict[str, dict[str, list[str]]]:  # pyright: ignore[reportGeneralTypeIssues]
        lanes: dict[str, dict[str, list[str]]] = {}
        for key in group:
            slot = self.state.slots[key]
            lanes[key] = {
                "names": [slot["alias"], key, key.rsplit("/", 1)[-1]],
                "places": [*slot["node_ids"], *slot.get("spark_names", [])],
            }
        return lanes

    def _split(
        self: Sweep,  # pyright: ignore[reportGeneralTypeIssues]
        group: Sequence[str],
        blamed: set[str],
    ) -> tuple[list[str], list[str]]:
        """Who of a load that ended carries the failure, and who is collateral.

        Only the lanes the Controller's failures name are the recipes' own. When nothing names
        a lane of a shared load, every lane is observed again (an uncertain outcome that
        spends one attempt each), so a recipe that really fails is still recorded and none
        is blamed for another's failure on one load.
        """
        culprits = [k for k in group if k in blamed]
        if culprits:
            return culprits, [k for k in group if k not in blamed]
        return list(group), []

    def _requeue_collateral(
        self: Sweep,  # pyright: ignore[reportGeneralTypeIssues]
        collateral: Sequence[str],
        culprits: Sequence[str],
        cause: str,
    ) -> None:
        for key in collateral:
            why = (
                "collateral of " + ", ".join(culprits)
                if culprits
                else f"collateral: {cause}, and no blocker names a recipe"
            )
            self.requeue(key, why)

    def _timeout(
        self: Sweep,  # pyright: ignore[reportGeneralTypeIssues]
        key: str,
        slot: dict[str, Any],
        failure: policy.Failure | None = None,
    ) -> None:
        load = self.state.data["loads"].get(slot["request_key"])
        others = [
            k
            for k, s in self.state.slots.items()
            if k != key
            and s["phase"] == "loading"
            and s.get("request_key") == slot["request_key"]
        ]
        if load and load.get("app_id") and not others:
            self._cancel_load(load)
        seconds = round(verdict.silent_seconds(slot.get("watch", {}), self.clock.now()))
        self._fail_slot(
            key,
            failure
            or policy.classify(
                "timeout",
                "load.timeout",
                f"not serving and no progress for {seconds}s "
                f"(stall window {round(slot['deadline'] - slot['started_at'])}s, "
                f"{round(float(slot.get('copy_seconds', 0)))}s of copying not counted; "
                f"{round(float(slot.get('waited_seconds', 0)))}s of waiting on the "
                f"Controller not counted; last phase {slot.get('load_phase')})",
            ),
            load or {},
        )

    def _fail_slot(
        self: Sweep,  # pyright: ignore[reportGeneralTypeIssues]
        key: str,
        failure: policy.Failure,
        app: Mapping[str, Any],
    ) -> None:
        context = {
            "application_state": app.get("state"),
            "reason": app.get("reason"),
            "child_phase": app.get("child_phase"),
            "assignment_failures": policy.excerpt(app.get("failures") or "", 500)
            or None,
            "application_id": app.get("app_id"),
        }
        failure = replace(
            failure,
            evidence={
                **{k: v for k, v in context.items() if v},
                **failure.evidence,
            },
        )
        self._record_failure(key, failure, app.get("operation_id") or app.get("app_id"))

    def owner_overlapped(self: Sweep, slot: Mapping[str, Any]) -> bool:  # pyright: ignore[reportGeneralTypeIssues]
        """An owner load happened during this slot's life: the Sparks were taken from under it."""
        return float(self.state.data["owner"].get("seen_at", 0)) >= float(
            slot["started_at"]
        )

    def requeue(
        self: Sweep,  # pyright: ignore[reportGeneralTypeIssues]
        key: str,
        why: str,
        *,
        failure: policy.Failure | None = None,
        document: Mapping[str, Any] | None = None,
        disposition: str | None = None,
        **evidence: Any,
    ) -> None:
        """Back to pending without blame or attempt: the recipe did not get a fair run.

        Every requeue leaves a results row with the Controller document it was last seen in.
        """
        entry = self.state.entry(key)
        slot = self.state.slots.get(key)
        self.record_result(
            key,
            "requeue",
            "requeued",
            reason=why,
            slot=slot,
            document=document,
            disposition=disposition,
            failure_class=failure.klass if failure else None,
            code=failure.code if failure else None,
            evidence={**(dict(failure.evidence) if failure else {}), **evidence}
            or None,
        )
        entry["status"] = "pending"
        if slot is not None:
            slot["phase"] = "finished"
        self.state.event(f"requeued {key}: {why}")
