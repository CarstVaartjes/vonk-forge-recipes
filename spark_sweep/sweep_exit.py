"""Interrupting, finishing and leaving the fleet (methods of ``Sweep``)."""

from __future__ import annotations

import itertools
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

from . import cleanup, verdict
from .catalog import (
    dig,
)
from .finalization import FinalAction, Finalization
from .lifecycle import TERMINAL as SETTLED
from .lifecycle import (
    LifecycleState,
)
from .report import write_status
from .vonkctl import VonkctlError, request_key

if TYPE_CHECKING:
    from .run import Sweep


class SweepExit:
    # ---------------------------------------------------------- end of a run

    def interrupt(self: Sweep) -> None:  # pyright: ignore[reportGeneralTypeIssues]
        """Ctrl-C: cancel the lane's current load and give its recipes back to the queue."""
        for load in list(self.state.data["loads"].values()):
            self._cancel_load(load, force=True)
        for key in list(self.state.slots):
            self.futures.pop(key, None)
        self.state.event("interrupted: lane loads cancelled, recipes requeued")
        self.preempt("sweep interrupted")
        self.cleanup_finished()

    def _cancel_load(
        self: Sweep,  # pyright: ignore[reportGeneralTypeIssues]
        load: Mapping[str, Any],
        *,
        force: bool = False,
    ) -> None:
        if not force and self._adopted_slots_live(load):
            return  # root cancellation also owns adopted effects, so preserve their observer
        if not force and self._ridden_by_live_lane(
            load.get("app_id"), load.get("request_key")
        ):
            return
        if load.get("app_id"):
            self._hold_lanes(load)
            self.vk.run(
                "profile",
                "cancel",
                load["app_id"],
                "--yes",
                "--detach",
                profile=self.cfg.sweep_profile,
            )

    def finish(self: Sweep, interrupted: bool = False) -> None:  # pyright: ignore[reportGeneralTypeIssues]
        self.executor.shutdown(wait=False, cancel_futures=True)
        if not interrupted:
            self.cleanup_finished()
            if not any(self._open(k) for k in self.recipes):
                self.prefetcher.release_pins()
        self._leave_fleet()
        if not interrupted:
            self.remove_finished_models(final=True)
        write_status(self, final=True)
        self.state.save()

    def _own_key(self: Sweep, kind: str, profile: int) -> str:  # pyright: ignore[reportGeneralTypeIssues]
        """A request key never used before: nonce, load kind, profile and a persisted sequence number.

        The Controller answers a repeated request key with the application it already
        made. For a clearing, stop or restore load that would be an old, finished
        application and nothing would change, so every attempt gets a key of its own.
        """
        seq = int(self.state.data["own_seq"]) + 1
        self.state.data["own_seq"] = seq
        return request_key(kind, self.state.nonce, profile, seq)

    def _submit_reviewed_load(
        self: Sweep,  # pyright: ignore[reportGeneralTypeIssues]
        profile: int,
        kind: str,
        *,
        owner: bool = False,
        key: str | None = None,
    ) -> Any:
        """Persist reviewed stop intent, including an empty scope, before effects."""
        if profile == self.cfg.sweep_profile:
            pruned = self._prune_profile()
            if pruned.waiting_on is not None:
                raise RuntimeError(pruned.waiting_on)
        review = self.vk.call(
            "profile",
            "load",
            "--review",
            profile=profile,
            allow_owner_write=owner,
        )
        if review.get("allowed") is not True:
            raise RuntimeError(f"the {kind} load has no allowed canonical review")
        key = key or self._own_key(kind, profile)
        if kind == FinalAction.STOP:
            runs = dig(review, "effects", "runs")
            adopted = self._adopted_cleanup_requests(review)
            if not isinstance(runs, list) or adopted is None:
                raise RuntimeError("final stop review cannot bind pending cleanup")
            self.state.data["cleanup_loads"][key] = {
                "request_key": key,
                "app_id": None,
                "reviewed_stops": sorted(
                    cleanup.stop_signature(row)
                    for row in runs
                    if isinstance(row, Mapping) and row.get("action") == "stop"
                ),
                "adopted_cleanup_requests": adopted,
            }
            self.state.save()
        return self._submit_load(profile, key, kind, owner=owner)

    def _submit_load(
        self: Sweep,  # pyright: ignore[reportGeneralTypeIssues]
        profile: int,
        key: str,
        kind: str,
        *,
        owner: bool = False,
    ) -> Any:
        """Submit a load the sweep itself decided on.

        Its request key is written to the state *before* the call, and the application id
        after it, so the owner guard can tell it from an owner's load, in this process and
        in the next one (a restore of the owner's profile is otherwise an "owner load").
        """
        own: list[dict[str, Any]] = self.state.data["own_loads"]
        record = next((item for item in own if item["request_key"] == key), None)
        if record is None:
            record = {
                "request_key": key,
                "kind": kind,
                "profile": profile,
                "at": self.clock.now(),
            }
            own.append(record)
            protected = set(self.state.data["loads"]) | set(
                self.state.data["cleanup_loads"]
            )
            inactive = [item for item in own if item["request_key"] not in protected]
            keep = {item["request_key"] for item in inactive[-200:]} | protected
            own[:] = [item for item in own if item["request_key"] in keep]
        if kind in ("takeover", "stop"):
            self.state.data["cleanup_loads"].setdefault(
                key, {"request_key": key, "app_id": record.get("application_id")}
            )
        self.state.save()
        document = self.vk.call(
            "profile",
            "load",
            "--yes",
            "--detach",
            "--request-key",
            key,
            profile=profile,
            allow_owner_write=owner,
        )
        if document.get("request_key") != key:
            raise RuntimeError(
                "profile acceptance does not match the owned request; observing original request"
            )
        if isinstance(document.get("id"), str):
            record["application_id"] = document["id"]
            if kind in ("takeover", "stop"):
                self.state.data["cleanup_loads"][key]["app_id"] = document["id"]
            self.state.save()
        return document

    def _leave_fleet(self: Sweep) -> None:  # pyright: ignore[reportGeneralTypeIssues]
        """Keep owning stop/restore until its exact terminal receipt confirms it.

        Unknown acceptance always observes the persisted request first. Only a
        canonical absence permits replay; a proven terminal failure re-reviews
        with a new request. A newer owner load supersedes the sweep's end intent.
        """
        number = self.cfg.restore_owner
        if not self.state.data.get("dirty"):
            return
        if number is None and not self.cfg.stop_at_end:
            return
        action = FinalAction.RESTORE if number is not None else FinalAction.STOP
        profile = number if number is not None else self.cfg.sweep_profile
        saved = self.state.data.get("finalization")
        final = (
            Finalization.model_validate(saved)
            if saved
            else Finalization(
                action=action,
                profile=profile,
                request_key=self._own_key(action, profile),
                since=self.clock.now(),
            )
        )
        for _ in itertools.count():
            now = self.clock.now()
            if now < final.next_check:
                write_status(self)
                self.clock.sleep(min(self.cfg.poll_seconds, final.next_check - now))
                continue
            final.attempts += 1
            final.next_check = now + verdict.retry_delay(
                self.cfg.retry_delay, final.attempts
            )
            self.state.data["finalization"] = final.model_dump(mode="json")
            self.state.save()
            write_status(self)
            try:
                owner_status = self.guard.check(now)
                if owner_status.new_activity:
                    self.state.event(
                        "final fleet intent superseded by a newer owner load"
                    )
                    self.state.data["dirty"] = False
                    break
                own = next(
                    (
                        item
                        for item in self.state.data["own_loads"]
                        if item["request_key"] == final.request_key
                    ),
                    None,
                )
                if own is not None:
                    reply = self.vk.run(
                        "profile",
                        "progress",
                        "--request-key",
                        final.request_key,
                        profile=final.profile,
                    )
                    document = reply.document
                    if reply.ok and isinstance(document, Mapping):
                        if (
                            document.get("request_key") != final.request_key
                            or not isinstance(document.get("id"), str)
                            or (
                                own.get("application_id")
                                and document.get("id") != own["application_id"]
                            )
                        ):
                            raise RuntimeError(
                                "final receipt does not match the owned request/application"
                            )
                        if document.get("state") == LifecycleState.SUCCEEDED:
                            # A new invocation may request a different final state. Finish
                            # observing the old accepted effect before applying the new one.
                            if final.action != action or final.profile != profile:
                                final = Finalization(
                                    action=action,
                                    profile=profile,
                                    request_key=self._own_key(action, profile),
                                    since=now,
                                )
                                continue
                            self.state.data["dirty"] = False
                            self.state.event(
                                f"confirmed final {final.action} profile {final.profile}"
                            )
                            break
                        if document.get("state") in SETTLED:
                            final.request_key = self._own_key(
                                final.action, final.profile
                            )
                            raise RuntimeError(
                                "final load ended without confirming the requested state; re-reviewing"
                            )
                        continue
                    if not reply.is_not_found or own.get("application_id"):
                        raise RuntimeError(
                            "final receipt unavailable; observing the same request"
                        )
                    # Canonical absence proves this unaccepted cleanup intent has
                    # no remote effects to adopt into its own fresh review.
                    self.state.data["cleanup_loads"].pop(final.request_key, None)
                if owner_status.paused:
                    raise RuntimeError(owner_status.reason)
                self.cleanup_finished()
                self._submit_reviewed_load(
                    final.profile,
                    final.action,
                    owner=final.action is FinalAction.RESTORE,
                    key=final.request_key,
                )
            except (VonkctlError, RuntimeError, OSError) as error:
                self.state.event(
                    f"final fleet reconciliation pending: {str(error)[:150]}"
                )
                self.note_infra("finalization", str(error))
            self.state.data["finalization"] = final.model_dump(mode="json")
            write_status(self)
            self.state.save()
        self.state.data.pop("finalization", None)
        self.clear_infra("finalization")
        self.state.save()
