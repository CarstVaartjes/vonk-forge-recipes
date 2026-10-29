"""The sweep follows the Controller's verdicts: one place decides wait, observe or fail.

A Controller wait is not a recipe failure. Every wait has a finite observation
budget; expiry releases the local lane and re-plans without spending an attempt.
The original Controller receipt remains authoritative for remote effects.

The vocabulary is what actually reaches ``vonkctl --json``: the application's
``LifecycleState``, its typed blocker codes, ``next_attempt_at`` and admission flags,
and the CLI error document (``code``, ``retryable``, ``decision``, ``http_status``).
The document's typed ``wait`` is authoritative, including informational blockers.
The closed code sets used for other observations are vendored from published enums and
checked against ``tests/fixtures/sweep-vocabulary.json``.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, MutableMapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from vonk_forge_contracts.sweep import (
    ControllerWait,
    SweepFailureClass,
    SweepWait,
    SweepWaitCode,
    SweepWaitExpiry,
    SweepWaitReason,
)
from vonk_forge_contracts.sweep import SweepProgress as Progress

from .auth_contract import SmokeAuthClass
from .lifecycle import ACTIVE, LifecycleState
from .policy import (
    CLIENT_CREDENTIAL,
    CONTROLLER_WAIT,
    RECIPE_FAILURE_CODES,
    REVIEW_FINAL_CODES,
    SWAP_SHORT,
    Failure,
    activity,
    forward_marks,
)
from .vonkctl import Reply


class Disposition(StrEnum):
    WAIT = (
        "wait"  # the platform or a dependency is not ready: no attempt, never a verdict
    )
    OBSERVE = "observe"  # an uncertain outcome: look again, spending one attempt
    FAIL = "fail"  # a definite outcome: record it now, with its evidence
    SKIP = "skip"  # not the sweep's to test any more: record the reason now, no verdict


class RecoveryReason(StrEnum):
    UNCERTAIN = "sweep.uncertain_outcome"
    REQUEST_REFUSED = "sweep.request_refused"


# -- vendored published vocabulary (tests/fixtures/sweep-vocabulary.json) -----------

#: RunSwitchCode.PHASE_RETRY: the load's own phase keeps being retried. Own work, not a wait.
PHASE_RETRY = "run-switch.phase-retry"
#: ProgressAlarmCode.STALLED: the Controller saw the same wait repeat with nothing moving.
#: A platform problem, never the recipe's: only its lane waits on the original receipt.
STALLED_CODE = "progress.stalled"
#: ModelCacheCode.DOWNLOAD_BLOCKED: the Controller holds a model download for NAS space
#: while retention evicts. The Controller owns that wait; it is no infrastructure problem.
NAS_SPACE_CODE = "model_cache.download_blocked"
#: Source name of a long Controller review wait's escalation (``review-wait:<code>``).
REVIEW_WAIT_SOURCE = "review-wait"
#: The codes a refused review may end the recipe on.
REVIEW_VERDICT_CODES = frozenset(REVIEW_FINAL_CODES | RECIPE_FAILURE_CODES.keys())
#: Blockers that name a preparation (build, image, download, install) the Controller runs.
PREPARATION_CODES = frozenset(
    {
        "profile.preparation_not_started",
        "profile.preparation_unavailable",
        "profile.recovery_cache_pending",
        "profile.runtime_image_rebuild_pending",
        "run-switch.container-build-required",
        "run-switch.container-build-unavailable",
        "run-switch.install-preparation-unavailable",
        "run-switch.installation-preparation-unavailable",
        "run-switch.nas-download-required",
        "run-switch.recipe-build-unavailable",
    }
)
#: The CLI's ``ErrorDecision`` words that say "ask again later".
CLIENT_WAIT_DECISIONS = frozenset({"retry", "defer"})
#: The CLI's ``ErrorDecision`` word for a final answer: asking again gets the same one.
CLIENT_EXIT_DECISION = "exit"
#: The Controller refused the request's own contents (``controller.conflict`` and
#: ``controller.invalid_request``), as opposed to who asked (401/403) or what it names (404).
REQUEST_REFUSAL_HTTP = frozenset({409, 422})
#: Standard temporary HTTP answers (the Controller's TemporaryHTTPError is 429/503).
TEMPORARY_HTTP = frozenset({408, 429, 502, 503, 504})

# -- the sweep's own outcome classes ----------------------------------------------

#: The smoke saw a temporary HTTP answer: the gateway or engine is busy, not wrong.
SMOKE_BUSY = "smoke-busy"
#: A shared load ended and nothing the Controller said names which recipe caused it.
UNATTRIBUTED = "unattributed"
#: A fleet swap shortfall (``policy.SWAP_SHORT``): the Controller's remedy is swap or time,
#: never the recipe's verdict; its never-fits word is ``demand_exceeds_capacity``.
WAIT_CLASSES = frozenset(
    {
        SMOKE_BUSY,
        CLIENT_CREDENTIAL,
        SweepWaitCode.BUDGET_EXHAUSTED,
        SWAP_SHORT,
        CONTROLLER_WAIT,
    }
)
#: Uncertain outcomes. ``network`` is a Controller operation that ended failed on a transport
#: error after its own retries: the cause is uncertain, so it is observed once more.
SMOKE_TIMEOUT = "smoke-timeout"
OBSERVE_CLASSES = frozenset(
    {
        "copy-stalled",
        "network",
        SMOKE_TIMEOUT,
        UNATTRIBUTED,
        SweepFailureClass.PLATFORM_EXIT,
    }
)
#: The Controller reports the lane's application as stalled (``STALLED_CODE``): the lane is
#: kept with an ``observe`` row that spends no attempt (never through ``disposition``).
PLATFORM_STALLED = "platform-stalled"
#: The sweep's own smoke code raised (a bug, a source tree changed under it): a tool fault,
#: never the recipe's. The lane is requeued with an ``observe`` row that spends no attempt.
SWEEP_CRASH = "sweep-crash"
#: A job recipe's smoke job ended failed and the Controller names no platform cause.
JOB_FAILED = "job-failed"
FAIL_CLASSES = frozenset(
    {"smoke-assertion", "recipe-data", JOB_FAILED, SmokeAuthClass.REFUSAL}
)
#: The recipe left the complete catalog (``catalog.recipe_removed``): deterministic, so it is
#: skipped at once; a recipe that comes back is picked up like any new one.
SKIP_CLASSES = frozenset({"catalog"})
#: The profile fields that carry the recipe's own data; the rest the sweep chose itself.
RECIPE_PROFILE_FIELDS = frozenset(
    {"recipe_selector", "option_choices", "model_variant"}
)
_HTTP_FIELDS = ("status", "status_code", "http_status")
# The published refusal field path: the Controller's ``request is invalid: body.<path>``
# validation detail and the CLI contract check's ``$.<path>``. vonkctl does not pass
# ``issues[].loc`` through, so this is the one reader of that structured path.
_ASSIGNMENT_FIELD = re.compile(r"assignments(?:\[\d+\]|\.\d+)\.([a-z_]+)")
_PROFILE_FIELD = re.compile(r"(?:\$|\bbody)\.([a-z_]+)")


# -- progress ---------------------------------------------------------------------


@dataclass(frozen=True)
class Signal:
    """One observation of a piece of work.

    ``signature`` changes only when the work's own progress moves (bytes, items, phase,
    the Controller's last-progress time, a run turning healthy). ``waiting_on`` names what
    the Controller reports owning while the work waits (a queue, admission, a blocker, a
    retry, a route publication); ``None`` means the work itself should be moving.
    ``marks`` are the work's forward-only counters (``None``: not reported); only one
    rising past its high-water mark renews a wait (``progress_verdict``).
    """

    signature: tuple[Any, ...]
    wait: SweepWait | None = None
    marks: tuple[int | None, ...] = ()

    @property
    def waiting_on(self) -> str | None:
        return self.wait.description if self.wait else None


def _blocker_codes(document: Mapping[str, Any], *, holding: bool = False) -> list[str]:
    """The blocker codes; ``holding`` keeps only error severity (a warning is a brief hold)."""
    progress = document.get("progress")
    held = document.get("blockers") or (
        progress.get("blockers") if isinstance(progress, Mapping) else None
    )
    return [
        str(item.get("code") or "")
        for item in held or []
        if isinstance(item, Mapping)
        and item.get("code")
        and (not holding or item.get("severity") == "error")
    ]


def application_signal(document: Mapping[str, Any]) -> Signal:
    """A profile application's own movement and what the Controller holds it on."""
    progress = document.get("progress")
    progress = progress if isinstance(progress, Mapping) else {}
    state = str(document.get("state") or "")
    held = [
        code
        for code in _blocker_codes(document, holding=True)
        if code not in (PHASE_RETRY, STALLED_CODE)
    ]
    wait = None
    if document.get("wait") is not None:
        wait = SweepWait(
            reason=SweepWaitReason.CONTROLLER_WAIT,
            controller=ControllerWait.model_validate(document["wait"]),
        )
    elif state == LifecycleState.NEEDS_OPERATOR:
        wait = SweepWait(
            reason=SweepWaitReason.OPERATOR,
            detail=str(document.get("status_reason") or state),
        )
    elif progress.get("admission_pending"):
        wait = SweepWait(
            reason=SweepWaitReason.ADMISSION, detail=held[0] if held else ""
        )
    elif state == LifecycleState.QUEUED:
        wait = SweepWait(reason=SweepWaitReason.APPLICATION_QUEUED)
    elif state == LifecycleState.BACKOFF:
        wait = SweepWait(reason=SweepWaitReason.APPLICATION_BACKOFF)
    elif held:
        wait = SweepWait(reason=SweepWaitReason.BLOCKER, detail=held[0])
    elif document.get("next_attempt_at"):
        wait = SweepWait(reason=SweepWaitReason.CONTROLLER_RETRY)
    return Signal(tuple(activity(document)), wait, forward_marks(document))


def admission_unbound(document: Mapping[str, Any]) -> bool:
    """Whether a newer saved profile would supersede this application.

    The Controller's pending-admission observer supersedes only an application
    still pending admission once the profile was saved again. A bound application
    continues: a newer decision adopts its unchanged assignments.
    """
    progress = document.get("progress")
    return isinstance(progress, Mapping) and progress.get("admission_pending") is True


def held_receipt(load: Mapping[str, Any] | None) -> bool:
    """Whether the Controller works this load's bound, live receipt behind a wait it reports.

    Only the Controller ends such a wait: the receipt's end, its stall alarm
    (``stalled_detail``) or a newer request. Releasing the lane on a local window
    would drop its assignment from the sweep profile, and the next whole-profile
    load would supersede the very receipt the Controller is still working on
    (2026-10-24). An unbound admission is not held: any saved profile supersedes
    it anyway, so its bounded profile-edit wait stays.
    """
    if (
        load is None
        or load.get("state") not in ACTIVE
        or load.get("admission_unbound", True) is not False
        or load.get("stalled")
    ):
        return False
    signal = load.get("signal")
    return isinstance(signal, Mapping) and signal.get("wait") is not None


def stalled_detail(document: Mapping[str, Any]) -> str | None:
    """The Controller's stalled-work alarm on an application, or ``None`` (absent: no stall)."""
    progress = document.get("progress")
    held = document.get("blockers") or (
        progress.get("blockers") if isinstance(progress, Mapping) else None
    )
    for item in held or []:
        if isinstance(item, Mapping) and item.get("code") == STALLED_CODE:
            return str(item.get("detail") or STALLED_CODE)
    return None


def download_signal(document: Mapping[str, Any]) -> Signal:
    """A recipe download operation: bytes, items, phase and the last-progress time."""
    progress = document.get("progress")
    progress = progress if isinstance(progress, Mapping) else {}
    state = str(document.get("state") or "")
    children = tuple(
        (child.get("kind"), child.get("state"))
        for child in document.get("children") or []
        if isinstance(child, Mapping)
    )
    signature = (
        progress.get("completed_bytes"),
        progress.get("completed_items"),
        progress.get("phase"),
        progress.get("last_progress_at"),
        state,
        children,
    )
    space = next(
        (
            child["failure"]
            for child in document.get("children") or []
            if isinstance(child, Mapping)
            and child.get("kind") == "model-cache"
            and child.get("state") in (LifecycleState.QUEUED, LifecycleState.BACKOFF)
            and isinstance(child.get("failure"), Mapping)
            and child["failure"].get("code") == NAS_SPACE_CODE
        ),
        None,
    )
    wait = None
    if space is not None:
        wait = SweepWait(
            reason=SweepWaitReason.NAS_SPACE, detail=str(space.get("detail") or "")
        )
    elif state == LifecycleState.QUEUED:
        wait = SweepWait(reason=SweepWaitReason.OPERATION_QUEUED)
    elif state == LifecycleState.BACKOFF:
        wait = SweepWait(reason=SweepWaitReason.OPERATION_BACKOFF)
    marks = (progress.get("completed_bytes"), progress.get("completed_items"))
    return Signal(signature, wait, marks)


def _comparable(value: Any) -> Any:
    """Signatures are stored as JSON: compare tuples and lists alike."""
    if isinstance(value, (list, tuple)):
        return [_comparable(item) for item in value]
    return value


def bounded_wait(
    wait: SweepWait,
    watch: MutableMapping[str, Any],
    *,
    window: float,
    now: float,
    started_at: float | None = None,
) -> Progress:
    """The only waiting_on writer. Changing reasons or heartbeats cannot renew a budget.

    Only ``progress_verdict`` renews it, when the waiting work itself moved.
    """
    since = float(
        watch.setdefault("wait_started_at", now if started_at is None else started_at)
    )
    deadline = min(float(watch.get("wait_deadline", since + window)), since + window)
    if wait.controller is not None and wait.controller.dependency_deadline is not None:
        deadline = min(deadline, wait.controller.dependency_deadline.timestamp())
    watch.update(
        observed_at=now,
        wait=wait.model_dump(mode="json"),
        waiting_on=wait.description,
        wait_deadline=deadline,
    )
    return Progress.WAIT_EXPIRED if now > deadline else Progress.WAITING


def wait_expiry(watch: Mapping[str, Any], now: float) -> SweepWaitExpiry:
    return SweepWaitExpiry(
        wait=SweepWait.model_validate(watch["wait"]),
        started_at=float(watch["wait_started_at"]),
        deadline=float(watch["wait_deadline"]),
        observed_at=now,
    )


def progress_verdict(
    signal: Signal | None,
    watch: MutableMapping[str, Any],
    *,
    window: float,
    now: float,
    started_at: float | None = None,
) -> Progress:
    """Own-work silence stalls; dependency waits expire through bounded_wait.

    A wait renews only when the waiting work moves strictly forward: one of its
    ``marks`` rises above the highest value seen (``wait_marks``). The window then
    restarts from that movement, so moving work is bounded by how long it goes
    without moving. A retry that re-copies the same bytes, a flapping phase, a new
    run, a changed reason or an unreadable observation raises no mark and renews
    nothing; the Controller's ``dependency_deadline`` still caps every renewal.
    """
    signature = _comparable(signal.signature) if signal is not None else None
    if signal is not None:
        seen = watch.get("wait_marks")
        if not isinstance(seen, list) or len(seen) != len(signal.marks):
            # A first look or an older watch records the marks and renews nothing.
            seen = list(signal.marks)
        rose = any(
            value is not None and mark is not None and value > mark
            for value, mark in zip(signal.marks, seen, strict=True)
        )
        watch["wait_marks"] = [
            mark if value is None else value if mark is None else max(value, mark)
            for value, mark in zip(signal.marks, seen, strict=True)
        ]
        if rose and signal.wait is not None:
            watch["wait_started_at"] = now
            watch.pop("wait_deadline", None)
    if signal is None or signal.wait is not None:
        wait = signal.wait if signal is not None else None
        if (
            wait is not None
            and wait.reason is SweepWaitReason.NAS_SPACE
            and now > float(watch.get("wait_deadline", now))
        ):
            # The Controller holds a NAS-space wait under its own bounded window while
            # retention evicts: each expiry of ours renews it (the caller records the
            # expired window), so it is looked at once per window and pauses no work.
            watch["wait_started_at"] = now
            watch.pop("wait_deadline", None)
        return bounded_wait(
            wait or SweepWait(reason=SweepWaitReason.OBSERVATION),
            watch,
            window=window,
            now=now,
            started_at=started_at,
        )
    watch["observed_at"] = now
    watch.pop("waiting_on", None)
    watch.pop("wait", None)
    if "progressed_at" not in watch or watch.get("signature") != signature:
        watch["signature"] = signature
        watch["progressed_at"] = now
        return Progress.PROGRESSING
    if now - float(watch["progressed_at"]) > window:
        return Progress.STALLED
    return Progress.PROGRESSING


def silent_seconds(watch: Mapping[str, Any], now: float) -> float:
    return max(0.0, now - float(watch.get("progressed_at", now)))


# -- disposition ------------------------------------------------------------------


def disposition(failure: Failure) -> Disposition:
    """What an outcome the sweep observed means for the recipe's attempt budget."""
    if failure.klass in SKIP_CLASSES:
        return Disposition.SKIP
    if failure.klass == UNATTRIBUTED:
        # Even a recipe code is nobody's verdict while nothing names whose it is.
        return Disposition.OBSERVE
    if (
        failure.klass in FAIL_CLASSES
        or failure.code in RECIPE_FAILURE_CODES
        or failure.code in REVIEW_FINAL_CODES
    ):
        return Disposition.FAIL
    if failure.klass in WAIT_CLASSES:
        return Disposition.WAIT
    if failure.klass == "capacity":
        if failure.evidence.get("demand_exceeds_idle_spark") is True:
            return (
                Disposition.FAIL
            )  # deterministic physical demand proof, not retry exhaustion
        # NAS space is a resource prerequisite the collector frees: a wait. A placement
        # refusal may be the recipe not fitting: the fleet is cleared and it is observed.
        return Disposition.WAIT if failure.phase == "download" else Disposition.OBSERVE
    if failure.klass in OBSERVE_CLASSES:
        return Disposition.OBSERVE
    return Disposition.FAIL


def _document(reply: Reply | None) -> Mapping[str, Any]:
    document = reply.document if reply is not None else None
    return document if isinstance(document, Mapping) else {}


def refused_field(text: str) -> str | None:
    """The profile field a contract refusal names (``$.assignments[0].assignment_name``)."""
    match = _ASSIGNMENT_FIELD.search(text) or _PROFILE_FIELD.search(text)
    return match.group(1) if match else None


def reply_disposition(reply: Reply | None) -> Disposition:
    """A vonkctl error: only an explicit recipe-data code on the recipe's own data fails.

    Everything else (HTTP and transport failures, Controller codes, an unknown outcome, a
    refusal of a field the sweep chose) waits: the next pass reviews afresh and the
    Controller's review, not the CLI error, decides admission.
    """
    document = _document(reply)
    code = str(document.get("code") or "")
    if (
        str(document.get("decision") or "") in CLIENT_WAIT_DECISIONS
        or document.get("retryable") is True
        or (
            any(document.get(name) is not None for name in _HTTP_FIELDS)
            and not request_refused(reply)
        )
        or code not in RECIPE_FAILURE_CODES
    ):
        return Disposition.WAIT
    field = refused_field(
        f"{document.get('detail') or ''} {document.get('error') or ''}"
    )
    if field is not None and field not in RECIPE_PROFILE_FIELDS:
        return Disposition.WAIT
    return Disposition.FAIL


def request_refused(reply: Reply | None) -> bool:
    """Did the Controller finally refuse the request's own contents (a 409/422, ``exit``)?"""
    document = _document(reply)
    return (
        document.get("decision") == CLIENT_EXIT_DECISION
        and document.get("retryable") is False
        and document.get("http_status") in REQUEST_REFUSAL_HTTP
    )


def http_wait(status: object) -> bool:
    return isinstance(status, int) and status in TEMPORARY_HTTP


def preparation_blockers(
    blockers: Sequence[Mapping[str, Any]],
) -> list[Mapping[str, Any]]:
    return [b for b in blockers if str(b.get("code") or "") in PREPARATION_CODES]


def retry_delay(base: float, attempts: int) -> float:
    """Counters pace observation, never end uncertain work."""
    return min(base * 2 ** min(max(0, attempts - 1), 10), 600.0)
