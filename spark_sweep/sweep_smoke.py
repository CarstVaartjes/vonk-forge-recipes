"""Waiting slots and the smoke phase (methods of ``Sweep``)."""

from __future__ import annotations

import threading
from collections.abc import Mapping
from dataclasses import replace
from typing import TYPE_CHECKING, Any

from . import memory, policy, verdict
from .auth_contract import SmokeAuthClass
from .catalog import (
    Recipe,
    dig,
)
from .chat_probe import ChatFacts, chat_facts
from .definitions import CaseRenderError
from .identity import (
    AcceptedRequest,
    ControllerWaitHold,
    IdentityReason,
)
from .qualification import (
    current_identity,
)
from .smoke import (
    HttpConfig,
    SmokeResult,
    case_render_failure,
    smoke_readiness,
    smoke_service,
)
from .smoke_job import job_budget_seconds, smoke_job
from .sweep_contract import (
    SweepWait,
    SweepWaitReason,
)
from .sweep_support import (
    ENDPOINT_RECHECK_SECONDS,
    _done_future,
    crash_evidence,
)
from .verdict import Disposition, Progress
from .vonkctl import VonkctlError

if TYPE_CHECKING:
    from .run import Sweep


class SweepSmoke:
    def _expire_wait(
        self: Sweep,  # pyright: ignore[reportGeneralTypeIssues]
        key: str,
        slot: dict[str, Any],
        watch: Mapping[str, Any],
    ) -> None:
        expiry = verdict.wait_expiry(watch, self.clock.now())
        stop = self.smoke_stops.pop(key, None)
        if stop is not None:
            stop.set()
        future = self.futures.pop(key, None)
        if future is not None:
            future.cancel()
        # Keep the original receipt for observation; the next canonical review
        # supersedes changed intent. Local expiry is never a remote terminal claim.
        load = self.state.data["loads"].get(slot["request_key"])
        if load is not None:
            load["replanned_pending"] = True
        accepted = AcceptedRequest.read(slot)
        if (
            expiry.wait.controller is not None
            and expiry.wait.controller.identity is not None
            and accepted is not None
            and load is not None
            and load.get("app_id")
        ):
            identity = current_identity(accepted.recipe)
            previous = self.controller_wait_holds().get(key)
            hold = ControllerWaitHold(
                wait=expiry.wait,
                identity=identity,
                application_id=load["app_id"],
                next_check=self.clock.now() + self.cfg.defer_delay,
                request_key=slot["request_key"],
                # Content that kept ending keeps its re-plan pace.
                ended_since=previous.ended_since
                if previous is not None and previous.identity == identity
                else None,
                platform=load.get("wait_platform")
                or {
                    owner: fingerprint
                    for owner, fingerprint in self._component_observations().items()
                    if owner in ("api", "worker") and fingerprint is not None
                },
            )
            self.state.entry(key)["controller_wait"] = hold.model_dump(mode="json")
            self.state.event(
                f"placement waits for Controller dependency: {hold.wait.description}; "
                f"observing {hold.application_id} again at {hold.next_check}"
            )
        failure = policy.classify(
            slot["phase"],
            expiry.code,
            f"wait budget exhausted: {expiry.wait.description}",
            klass=expiry.code,
            evidence=expiry.model_dump(mode="json"),
        )
        group = [
            k
            for k, lane in self.state.slots.items()
            if lane.get("request_key") == slot["request_key"]
            and lane["phase"] == "loading"
        ]
        self.requeue(
            key, failure.detail, failure=failure, disposition=Disposition.WAIT.value
        )
        self.state.entry(key)["wait_deferral"] = expiry.model_dump(mode="json")
        # requeue marked this slot finished; include it when attributing a shared hold.
        lanes = self._lanes(sorted(set(group) | {key}))
        held = policy.blamed(load.get("blockers") or [], lanes) if load else set()
        delay = 0.0 if held and key not in held else self.cfg.defer_delay
        self.state.entry(key)["not_before"] = self.clock.now() + delay

    def _wait_slot(
        self: Sweep,  # pyright: ignore[reportGeneralTypeIssues]
        key: str,
        slot: dict[str, Any],
        wait: SweepWait,
        *,
        watch_name: str = "smoke_watch",
        started_at: float | None = None,
    ) -> bool:
        watch = slot.setdefault(watch_name, {})
        since = (
            float(
                slot.get("smoke_started_at", slot.get("ready_at", slot["started_at"]))
            )
            if started_at is None
            else started_at
        )
        window = float(
            slot.get(
                "smoke_budget_s", float(slot["deadline"]) - float(slot["started_at"])
            )
        )
        progress = verdict.bounded_wait(
            wait, watch, window=window, now=self.clock.now(), started_at=since
        )
        if progress is Progress.WAIT_EXPIRED:
            self._expire_wait(key, slot, watch)
            return True
        return False

    def _submit_smoke(self: Sweep, key: str, slot: dict[str, Any]) -> None:  # pyright: ignore[reportGeneralTypeIssues]
        accepted = AcceptedRequest.read(slot)
        if accepted is None:
            self.requeue(key, IdentityReason.UNBOUND)
            return
        recipe = accepted.recipe
        slot.setdefault("smoke_started_at", slot.get("ready_at", self.clock.now()))
        if recipe.is_job:
            # A job run has no endpoint: its smoke is one artifact job against the run.
            spec = self.defs.artifact(key)
            if spec is not None:
                # Each smoke enforces its own budget from its start: the lane shows that one.
                slot.update(
                    smoke_budget_s=job_budget_seconds(spec),
                )
            self.futures[key] = self.executor.submit(
                smoke_job,
                recipe,
                self.defs,
                self.vk,
                slot["run_id"],
                slot["request_key"],
            )
            return
        if not recipe.is_openai:
            self.futures[key] = _done_future(smoke_readiness(slot["run_id"]))
            return
        try:
            endpoints = self.vk.call(
                "profile", "endpoint", slot["alias"], profile=self.cfg.sweep_profile
            )
        except VonkctlError as error:
            if verdict.reply_disposition(error.reply) is Disposition.WAIT:
                if self._wait_slot(
                    key,
                    slot,
                    SweepWait(
                        reason=SweepWaitReason.ENDPOINT_PUBLICATION, detail=str(error)
                    ),
                ):
                    return
                self.note_infra("endpoint", str(error))
                slot["smoke_not_before"] = self.clock.now() + ENDPOINT_RECHECK_SECONDS
                return  # the run is up: look the endpoint up again on the next pass
            self._fail_slot(
                key, policy.classify("readiness", "endpoint", str(error)), {}
            )
            return
        self.clear_infra("endpoint")
        item = next(
            (
                item
                for item in dig(endpoints, "assignments", default=[])
                if isinstance(item, dict) and item.get("alias") == slot["alias"]
            ),
            {},
        )
        base = dig(item, "endpoint", "api_base")
        if not isinstance(base, str):
            if self._wait_slot(
                key,
                slot,
                SweepWait(
                    reason=SweepWaitReason.ENDPOINT_PUBLICATION,
                    detail=f"assignment state {item.get('state')}",
                ),
            ):
                return
            slot["smoke_not_before"] = self.clock.now() + ENDPOINT_RECHECK_SECONDS
            return
        slot.pop("waiting_on", None)
        stop = threading.Event()
        self.smoke_stops[key] = stop
        self.futures[key] = self.executor.submit(
            self._smoke_service,
            recipe,
            base,
            str(slot["alias"]),
            replace(self.cfg.http, cancelled=stop),
            self.documents.get(str(slot.get("request_key", ""))),
        )

    def _smoke_service(
        self: Sweep,  # pyright: ignore[reportGeneralTypeIssues]
        recipe: Recipe,
        base: str,
        alias: str,
        config: HttpConfig,
        application: Any,
    ) -> SmokeResult:
        """The service smoke, on an executor thread: resolving its facts may read the Controller."""
        return smoke_service(
            recipe,
            self.defs,
            base,
            alias,
            config,
            self._chat_facts(recipe, alias, application),
        )

    def _chat_facts(
        self: Sweep,  # pyright: ignore[reportGeneralTypeIssues]
        recipe: Recipe,
        alias: str,
        application: Any,
    ) -> ChatFacts | None:
        """The recipe-api facts of the launch that ran, when its smoke is the declared check.

        Reviewed smoke cases carry their own requests; only the fallback needs the facts.
        None when the recipe publishes none or the launch cannot be read: the declared check
        then runs as written.
        """
        api = self.defs.recipe_api(recipe.slug)
        if self.defs.has_service_cases(recipe.key) or api is None:
            return None
        definition = self._ran_recipe(recipe)
        if definition is None:
            return None
        try:
            choices = memory.intended_assignment(application, alias).option_choices
            return chat_facts(definition, api, choices)
        except StopIteration, ValueError, TypeError:
            return None

    def _advance_smoking(self: Sweep, key: str, slot: dict[str, Any]) -> None:  # pyright: ignore[reportGeneralTypeIssues]
        future = self.futures.get(key)
        watch = slot.get("smoke_watch") or {}
        wait = (
            SweepWait.model_validate(watch["wait"])
            if watch.get("wait")
            else SweepWait(reason=SweepWaitReason.SMOKE_RESPONSE)
        )
        if (future is None or not future.done()) and self._wait_slot(key, slot, wait):
            return
        if (
            future is None
        ):  # resumed after a restart: the smoke is idempotent, run it again
            if self.clock.now() >= float(slot.get("smoke_not_before", 0)):
                self._submit_smoke(key, slot)
            return
        if not future.done():
            self._wait_slot(key, slot, SweepWait(reason=SweepWaitReason.SMOKE_RESPONSE))
            return
        del self.futures[key]
        self.smoke_stops.pop(key, None)
        try:
            result = future.result()
        except CaseRenderError as error:
            # The recipe's own gap: a typed failure, never a tool crash to retry.
            self._record_failure(
                key, case_render_failure(error), slot.get("operation_id")
            )
            return
        except Exception as error:  # noqa: BLE001 - a bug in one smoke must not stop an unattended sweep
            self._smoke_crashed(key, slot, error)
            return
        if result.ok:
            self.clear_infra("client-key")
            self.clear_infra("smoke-tool")
            self.state.entry(key).pop("exit_repeat", None)
            self._record_pass(key, slot, result)
            return
        assert result.failure is not None
        failure = verdict.smoke_workload_exit(
            result.failure,
            [p.run_state for p in self.fleet.presences if p.run_id == slot["run_id"]],
        )
        if failure.klass == SmokeAuthClass.REFUSAL:
            self.clear_infra("client-key")
        if verdict.disposition(failure) is Disposition.WAIT:
            self._smoke_wait(key, slot, failure)
            return
        self._record_failure(key, failure, slot.get("operation_id"))

    def _smoke_crashed(
        self: Sweep,  # pyright: ignore[reportGeneralTypeIssues]
        key: str,
        slot: dict[str, Any],
        error: BaseException,
    ) -> None:
        """The sweep's own smoke code raised: a tool fault, never a verdict on the recipe.

        The exception's type, message (an ``OSError`` keeps its filename) and last frames
        go into the row and the event log. The recipe is requeued without spending an
        attempt, and this recipe backs off (``note_infra``) until a smoke passes again, so a
        crash that repeats costs backed-off loads, never the recipe. The same crash on the same
        content identity three times in a row ends as the platform's FAIL with its evidence
        (``_bound_identical_exit``); another exception or identity retries again.
        """
        evidence = crash_evidence(error)
        failure = policy.classify(
            "smoke",
            "smoke.crash",
            f"{evidence['type']}: {evidence['message']}",
            verdict.SWEEP_CRASH,
            evidence,
        )
        accepted = AcceptedRequest.read(slot)
        if accepted is not None:
            bounded = self._bound_identical_exit(
                self.state.entry(key), failure, accepted.recipe
            )
            if bounded.evidence.get("identical_exits"):
                self._record_failure(key, bounded, slot.get("operation_id"))
                return
        self.state.event(
            f"smoke of {key} crashed in the sweep (a tool fault, not the recipe's): "
            f"{failure.detail[:200]} at {evidence['frames'][-1][:200] if evidence['frames'] else '?'}"
        )
        self.note_infra("smoke-tool", f"the sweep's smoke crashed: {failure.detail}")
        self.requeue(
            key,
            f"the sweep's smoke crashed: {failure.detail}",
            failure=failure,
            disposition=Disposition.OBSERVE.value,
        )
        self.state.entry(key)["not_before"] = self.state.data["infra"]["smoke-tool"][
            "until"
        ]

    def _smoke_wait(
        self: Sweep,  # pyright: ignore[reportGeneralTypeIssues]
        key: str,
        slot: dict[str, Any],
        failure: policy.Failure,
    ) -> None:
        """Retry busy responses and typed key expiry without spending an attempt.

        Both reasons use the smoke phase budget, even during retry backoff.
        Expiry releases the lane without spending a recipe attempt.
        """
        now = self.clock.now()
        if failure.klass == policy.CLIENT_CREDENTIAL:
            self.note_infra(
                "client-key",
                "the gateway refused the sweep's client key "
                f"({failure.describe()}); renew it in the --client-key-file",
            )
            until = float(self.state.data["infra"]["client-key"]["until"])
            wait = SweepWait(
                reason=SweepWaitReason.CLIENT_KEY, detail=failure.describe()
            )
        else:
            until = now + self.cfg.retry_delay
            wait = SweepWait(
                reason=SweepWaitReason.SMOKE_BUSY, detail=failure.describe()
            )
        if self._wait_slot(key, slot, wait):
            return
        slot["smoke_not_before"] = until
        self.record_result(
            key,
            "smoke",
            "waiting",
            reason=failure.describe(),
            disposition=Disposition.WAIT.value,
            failure_class=failure.klass,
            evidence=dict(failure.evidence),
            slot=slot,
        )

    def preempt(self: Sweep, why: str) -> None:  # pyright: ignore[reportGeneralTypeIssues]
        for key, slot in list(self.state.slots.items()):
            future = self.futures.pop(key, None)
            if future is not None:
                future.cancel()
            self.requeue(key, why)
            slot["phase"] = "finished"
        self.state.data["loads"].clear()
