"""Cleanup observation (methods of ``Sweep``)."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from functools import cache, partial
from typing import TYPE_CHECKING, Any

from pydantic import ValidationError

from . import cleanup
from .catalog import (
    dig,
)
from .fanout import Outcome, gather
from .lifecycle import TERMINAL as SETTLED
from .sweep_contract import (
    CleanupEnd,
    CleanupEndReason,
    CleanupFleetObservation,
    CleanupObservationBudget,
    FleetOnlineState,
)
from .sweep_support import (
    INFRA_BASE,
    INFRA_CAP,
)
from .vonkctl import Reply, VonkctlError

if TYPE_CHECKING:
    from .run import Sweep


class SweepCleanup:
    def _cleanup_confirmed(self: Sweep) -> dict[str, int]:  # pyright: ignore[reportGeneralTypeIssues]
        confirmed: dict[str, int] = {}
        for load in self.state.data["cleanup_loads"].values():
            for row in load.get("effects", []):
                if cleanup.stopped(row):
                    signature = cleanup.stop_signature(row["stop_effect"])
                    confirmed[signature] = max(
                        confirmed.get(signature, 0), row["workload_intent_ordinal"]
                    )
        return confirmed

    def _cleanup_pending(self: Sweep, load: Mapping[str, Any]) -> list[dict[str, Any]]:  # pyright: ignore[reportGeneralTypeIssues]
        if load.get("end"):
            return []
        confirmed = self._cleanup_confirmed()
        return [
            row
            for row in load.get("effects", [])
            if confirmed.get(cleanup.stop_signature(row["stop_effect"]), 0)
            < row["workload_intent_ordinal"]
        ]

    def _cleanup_nodes(self: Sweep) -> set[str]:  # pyright: ignore[reportGeneralTypeIssues]
        """Unconfirmed exact stop scopes remain occupied, including whole gangs."""
        return set(self._cleanup_holds())

    def _cleanup_holds(self: Sweep) -> dict[str, str]:  # pyright: ignore[reportGeneralTypeIssues]
        """The online Sparks cleanup bookkeeping keeps from placement, each with why."""
        held: dict[str, str] = {}
        for key, load in self.state.data["cleanup_loads"].items():
            if self._cleanup_settled(load):
                continue
            name = f"cleanup {str(load.get('app_id') or key)[:8]}"
            if load.get("terminal_receipt"):
                # Terminal bookkeeping is not a physical occupancy certificate.
                # Current fleet facts and the fresh Controller review own admission.
                kept = self._own_aliases()
                for p in self.fleet.presences:
                    if p.alias not in kept:
                        held.setdefault(
                            p.node_id,
                            f"{name} ended with unconfirmed stops and runs "
                            f"{p.alias}, not the sweep's",
                        )
                continue
            if not load.get("projection_known"):
                # Before the current projection is known no lane may be inferred free.
                for s in self.sparks():
                    held.setdefault(s.id, f"{name} effects are not yet observed")
                continue
            for row in self._cleanup_pending(load):
                for node in row["node_ids"]:
                    held.setdefault(
                        node, f"{name} stop of run {row.get('target_id')} unconfirmed"
                    )
        if self.state.data.get("takeover_fallback"):
            # Bookkeeping is not occupancy. A fresh review still owns admission;
            # this observation never rewrites or declares an old receipt successful.
            own = self._own_aliases()
            held = {
                p.node_id: f"clearing fallback: runs {p.alias}, not the sweep's"
                for p in self.fleet.presences
                if p.alias not in own
            }
        online = {s.id for s in self.sparks()}
        return {node: why for node, why in held.items() if node in online}

    def _cleanup_target_absent(
        self: Sweep,  # pyright: ignore[reportGeneralTypeIssues]
        load: Mapping,
        *,
        terminal: bool = False,
        fleet: Callable[[], Reply],
    ) -> bool:
        """An exact absent receipt plus fresh online fleet facts ends bookkeeping.

        A ``terminal`` receipt is final: absence is judged against its own stop targets
        only, and one with none has nothing to wait for. An absent receipt without
        targets may have stopped anything, so only a fleet with no runs ends it.
        A pass hands in one ``fleet`` read (``functools.cache``) for all its loads, so the
        Controller is asked once per pass, not once per load.
        """
        reply = fleet()
        if not reply.ok:
            return False
        try:
            observed = CleanupFleetObservation.model_validate(reply.document)
        except ValidationError:
            return False
        # Offline or incomplete facts cannot certify the absence of an old run.
        effects = load.get("effects") or []
        expected_nodes = {node for row in effects for node in row["node_ids"]}
        expected_runs = {row["target_id"] for row in effects}
        for signature in load.get("reviewed_stops") or []:
            stop = cleanup.reviewed_stop(signature)
            if stop is None:
                return False
            expected_nodes.update(stop.node_ids)
            expected_runs.add(stop.run_id)
        nodes = {node.id for node in observed.nodes}
        if not nodes or not expected_nodes <= nodes:
            return False
        relevant = [
            node
            for node in observed.nodes
            if not expected_nodes or node.id in expected_nodes
        ]
        if any(
            node.connection.online_state != FleetOnlineState.ONLINE for node in relevant
        ):
            return False
        present = {run.run_id for node in observed.nodes for run in node.loaded}
        if expected_runs or terminal:
            return not (expected_runs & present)
        return not present

    def _end_cleanup(self: Sweep, load: dict[str, Any], now: float) -> None:  # pyright: ignore[reportGeneralTypeIssues]
        """Fresh fleet facts show the cleanup's targets gone: its bookkeeping ends."""
        load["end"] = CleanupEnd(
            reason=CleanupEndReason.TARGET_ABSENT, observed_at=now
        ).model_dump(mode="json")
        load["observation_unknown"] = False
        load.pop("observation_budget", None)
        self.clear_infra("clearing")
        self.backoff_until = max(
            (
                item["until"]
                for source, item in self.state.data["infra"].items()
                if source != "download"
            ),
            default=0.0,
        )
        self.state.event(
            f"cleanup {str(load.get('app_id') or load['request_key'])[:8]} retired: "
            "its stop targets are gone from the fleet"
        )
        self.state.save()

    @staticmethod
    def _defer_cleanup(
        load: dict[str, Any], budget: CleanupObservationBudget, now: float
    ) -> None:
        budget.attempts += 1
        budget.next_check = now + min(
            INFRA_BASE * 2 ** min(budget.attempts - 1, 20), INFRA_CAP
        )
        load["observation_budget"] = budget.model_dump(mode="json")

    @staticmethod
    def _cleanup_due(load: Mapping[str, Any], now: float) -> bool:
        budget = CleanupObservationBudget.model_validate(
            load.get("observation_budget", {})
        )
        return now >= budget.next_check

    def _read_cleanup_progress(self: Sweep, now: float) -> dict[str, Outcome]:  # pyright: ignore[reportGeneralTypeIssues]
        """The due observers' progress reads, at once (``observe_cleanup`` applies them)."""
        due = [
            load
            for load in self.state.data["cleanup_loads"].values()
            if not self._cleanup_settled(load)
            and not load.get("terminal_receipt")
            and self._cleanup_due(load, now)
        ]
        return gather(
            {load["request_key"]: partial(self._read_progress, load) for load in due},
            self.read_executor,
        )

    def _cleanup_settled(self: Sweep, load: Mapping[str, Any]) -> bool:  # pyright: ignore[reportGeneralTypeIssues]
        """Nothing is left to observe: retired, or a terminal receipt whose projection is known
        and has no pending stop. It reads the other observers' confirmations, so it can change
        when they do."""
        return bool(load.get("end")) or bool(
            load.get("terminal_receipt")
            and load.get("projection_known")
            and not self._cleanup_pending(load)
        )

    def _prune_cleanup_loads(self: Sweep) -> None:  # pyright: ignore[reportGeneralTypeIssues]
        """Keep the observers still needed and the newest settled one; drop older settled ones.

        The newest settled observer stays as the marker that a clearing ran, and any a live
        load adopted stays too. A dropped observer's stopped rows no longer confirm a survivor's
        row; that survivor heals by retiring once fleet facts show its target absent.
        """
        loads = self.state.data["cleanup_loads"]
        settled = [key for key, load in loads.items() if self._cleanup_settled(load)]
        if len(settled) < 2:
            return
        order = {
            item["request_key"]: i
            for i, item in enumerate(self.state.data["own_loads"])
        }
        newest = max(settled, key=lambda key: order.get(key, -1))
        adopted = {
            key
            for load in self.state.data["loads"].values()
            for key in load.get("adopted_cleanup_requests", [])
        }
        for key in settled:
            if key != newest and key not in adopted:
                del loads[key]

    def observe_cleanup(self: Sweep) -> None:  # pyright: ignore[reportGeneralTypeIssues]
        """Reconcile original applications and stops; never replace intent on timeout."""
        self._prune_cleanup_loads()
        now = self.clock.now()
        reads = self._read_cleanup_progress(now)
        fleet = cache(partial(self.vk.run, "fleet"))
        for load in self.state.data["cleanup_loads"].values():
            if self._cleanup_settled(load):
                continue
            if not self._cleanup_due(load, now):
                continue
            budget = CleanupObservationBudget.model_validate(
                load.get("observation_budget", {})
            )
            if load.get("terminal_receipt"):
                # The Controller ended it with stops unconfirmed and its receipt is
                # final: only current fleet facts retire it, once its targets are gone.
                if self._cleanup_target_absent(load, terminal=True, fleet=fleet):
                    self._end_cleanup(load, now)
                else:
                    self._defer_cleanup(load, budget, now)
                continue
            reply = reads[load["request_key"]].result()
            document = reply.document
            if reply.is_not_found and self._cleanup_target_absent(load, fleet=fleet):
                self.clear_infra("clearing")
                self._end_cleanup(load, now)
                continue
            if not reply.ok:
                self._defer_cleanup(load, budget, now)
                # Status reports the next observation without pausing other lanes.
                if item := self.state.data["infra"].get("clearing"):
                    item["count"] += 1
                    item["until"] = budget.next_check
            else:
                load.pop("observation_budget", None)
                self.clear_infra("clearing")
            if (
                not load.get("app_id")
                and reply.is_not_found
                and load.get("reviewed_stops") is not None
            ):
                review = self.vk.run(
                    "profile", "load", "--review", profile=self.cfg.sweep_profile
                )
                current = review.document
                if not review.ok or not isinstance(current, Mapping):
                    continue
                stops = [
                    row
                    for row in dig(current, "effects", "runs", default=[])
                    if isinstance(row, Mapping) and row.get("action") == "stop"
                ]
                if (
                    current.get("allowed") is not True
                    or sorted(cleanup.stop_signature(row) for row in stops)
                    != load["reviewed_stops"]
                    or self._adopted_requests(current) is None
                ):
                    load["observation_unknown"] = True
                    continue
                try:
                    document = self._submit_load(
                        self.cfg.sweep_profile, load["request_key"], "takeover"
                    )
                except VonkctlError as error:
                    self.state.event(
                        f"cleanup re-entry unanswered, observing: {str(error)[:120]}"
                    )
                    continue
                # Exactly absent receipt permits same-key reentry, never a new UUID.
                load["app_id"] = document.get("id")
                load.pop("observation_budget", None)
                self.state.save()
                continue
            if not reply.ok or not isinstance(document, Mapping):
                load["observation_unknown"] = True
                self.state.event(
                    "exact cleanup receipt observation unavailable; retaining its original identity"
                )
                continue
            app = document.get("id")
            if (
                not isinstance(app, str)
                or document.get("request_key") != load["request_key"]
                or (load.get("app_id") and app != load["app_id"])
            ):
                load["observation_unknown"] = True
                continue
            if document.get("state") in SETTLED:
                load.update(
                    app_id=app,
                    state=document["state"],
                    terminal_receipt=dict(document),
                    observation_unknown=False,
                )
                # A terminal parent may lack its final child projection. That
                # cannot keep the old request at the head of fresh admission.
                self.state.save()
            rows = cleanup.stop_effects(document)
            expected = load.get("reviewed_stops")
            if rows is not None and (
                (not rows and expected != [])
                or (
                    isinstance(expected, list)
                    and not set(expected)
                    <= {cleanup.stop_signature(row["stop_effect"]) for row in rows}
                )
            ):
                # Acceptance can precede adapter initialization. An empty or partial
                # projection never erases the reviewed stops or invents free lanes.
                load["observation_unknown"] = True
                continue
            previous = {row["effect_id"]: row for row in load.get("effects", [])}
            if (
                rows is None
                or (
                    load.get("projection_known")
                    and set(previous) != {row["effect_id"] for row in rows}
                )
                or any(
                    row["effect_id"] not in {item["effect_id"] for item in rows}
                    for row in previous.values()
                )
                or any(
                    row["effect_id"] in previous
                    and (
                        not cleanup.same_effect(previous[row["effect_id"]], row)
                        or (
                            cleanup.stopped(previous[row["effect_id"]])
                            and not cleanup.stopped(row)
                        )
                    )
                    for row in rows or []
                )
            ):
                load["observation_unknown"] = True
                self.state.event(
                    "canonical cleanup effect projection unavailable or changed; retaining original scopes"
                )
                continue
            load.update(
                app_id=app,
                state=document.get("state"),
                effects=rows,
                projection_known=True,
                observation_unknown=False,
                observed_at=self.clock.now(),
            )
            if not self._cleanup_pending(load):
                # Child completion is not parent completion. Reconcile its exact
                # targets with fresh fleet facts before retiring the observer.
                if self._cleanup_target_absent(load, terminal=True, fleet=fleet):
                    self._end_cleanup(load, now)
                else:
                    self._defer_cleanup(load, budget, now)
            self.state.save()
