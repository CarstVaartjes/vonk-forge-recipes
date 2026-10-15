"""The sweep follows the Controller's verdicts: one place decides wait, observe or fail.

Rule: the sweep records what the platform does; it never turns a wait into a verdict.

* A lane ends a load only on the Controller's own terminal state or on a ``STALLED``
  verdict of the work's own progress (``progress_verdict``).
* A wait on something the Controller reports as owned, queued or in progress
  (``WAITING``) never expires: it restarts the stall window instead of ageing it.
* Attempts are spent only on ``OBSERVE`` outcomes (``disposition``); a ``WAIT`` spends
  nothing, a ``FAIL`` ends at once with its evidence and a ``SKIP`` (the recipe is no
  longer the sweep's to test) is recorded at once with its reason, never a verdict.

The vocabulary is what actually reaches ``vonkctl --json``: the application's
``LifecycleState``, its typed blocker codes, ``next_attempt_at`` and admission flags,
and the CLI error document (``code``, ``retryable``, ``decision``, ``http_status``).
The Controller's ``WaitReason``/``ErrorCategory`` enums are not on the profile wire, so
the closed code sets used here are vendored from the platform's published enums and
checked against ``tests/fixtures/sweep-vocabulary.json``.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, MutableMapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from .policy import CLIENT_CREDENTIAL, RECIPE_FAILURE_CODES, Failure, activity
from .vonkctl import Reply


class Progress(StrEnum):
    PROGRESSING = "progressing"
    WAITING = "waiting"
    STALLED = "stalled"


class Disposition(StrEnum):
    WAIT = (
        "wait"  # the platform or a dependency is not ready: no attempt, never a verdict
    )
    OBSERVE = "observe"  # an uncertain outcome: look again, spending one attempt
    FAIL = "fail"  # a definite outcome: record it now, with its evidence
    SKIP = "skip"  # not the sweep's to test any more: record the reason now, no verdict


# -- vendored published vocabulary (tests/fixtures/sweep-vocabulary.json) -----------

#: RunSwitchCode.PHASE_RETRY: the load's own phase keeps being retried. Own work, not a wait.
PHASE_RETRY = "run-switch.phase-retry"
#: ProfileReasonCode.FAILURE_REPEATED: the Controller stopped retrying a failed application.
FAILURE_REPEATED = "profile.failure_repeated"
#: ProgressAlarmCode.STALLED: the Controller saw the same wait repeat with nothing moving.
#: A platform problem, never the recipe's: the lane is released and the sweep pauses.
STALLED_CODE = "progress.stalled"
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
#: Standard temporary HTTP answers (the Controller's TemporaryHTTPError is 429/503).
TEMPORARY_HTTP = frozenset({408, 429, 502, 503, 504})

# -- the sweep's own outcome classes ----------------------------------------------

#: The smoke saw a temporary HTTP answer: the gateway or engine is busy, not wrong.
SMOKE_BUSY = "smoke-busy"
#: A shared load ended and nothing the Controller said names which recipe caused it.
UNATTRIBUTED = "unattributed"
WAIT_CLASSES = frozenset({SMOKE_BUSY, CLIENT_CREDENTIAL})
#: Uncertain outcomes. ``network`` is a Controller operation that ended failed on a transport
#: error after its own retries: the cause is uncertain, so it is observed once more.
OBSERVE_CLASSES = frozenset({"copy-stalled", "network", "smoke-timeout", UNATTRIBUTED})
#: The Controller reports the lane's application as stalled (``STALLED_CODE``): the lane is
#: requeued with an ``observe`` row that spends no attempt (never through ``disposition``).
PLATFORM_STALLED = "platform-stalled"
#: The sweep's own smoke code raised (a bug, a source tree changed under it): a tool fault,
#: never the recipe's. The lane is requeued with an ``observe`` row that spends no attempt.
SWEEP_CRASH = "sweep-crash"
#: A job recipe's smoke job ended failed and the Controller names no platform cause.
JOB_FAILED = "job-failed"
FAIL_CLASSES = frozenset({"smoke-assertion", "recipe-data", JOB_FAILED})
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
    """

    signature: tuple[Any, ...]
    waiting_on: str | None = None


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
    waiting_on: str | None = None
    if state in ("queued", "backoff"):
        waiting_on = f"application {state}"
    elif state == "needs-operator":
        waiting_on = f"operator: {document.get('status_reason') or 'needs-operator'}"
    elif progress.get("admission_pending"):
        waiting_on = "admission" + (f" ({held[0]})" if held else "")
    elif held:
        waiting_on = held[0]
    elif document.get("next_attempt_at") or progress.get("retry_due_at"):
        waiting_on = "controller retry"
    return Signal(tuple(activity(document)), waiting_on)


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
    waiting = f"operation {state}" if state in ("queued", "backoff") else None
    return Signal(signature, waiting)


def _comparable(value: Any) -> Any:
    """Signatures are stored as JSON: compare tuples and lists alike."""
    if isinstance(value, (list, tuple)):
        return [_comparable(item) for item in value]
    return value


def progress_verdict(
    signal: Signal | None,
    watch: MutableMapping[str, Any],
    *,
    window: float,
    now: float,
) -> Progress:
    """PROGRESSING, WAITING or STALLED, remembering the last movement in ``watch``.

    ``None`` is an unreadable observation: unknown, so it waits and restarts the window.
    A wait on the Controller restarts the window too, so it never ages into a stall. Only
    the work's own silence for a whole ``window`` is ``STALLED``, the one verdict that
    ends work.
    """
    signature = _comparable(signal.signature) if signal is not None else None
    if signal is None or signal.waiting_on is not None:
        watch["progressed_at"] = now
        watch["waiting_on"] = (
            signal.waiting_on if signal is not None else "observation unavailable"
        )
        if signature is not None:
            watch["signature"] = signature
        return Progress.WAITING
    watch.pop("waiting_on", None)
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
    if failure.klass in FAIL_CLASSES or failure.code in RECIPE_FAILURE_CODES:
        return Disposition.FAIL
    if failure.klass in WAIT_CLASSES:
        return Disposition.WAIT
    if failure.klass == "capacity":
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
        or any(document.get(name) is not None for name in _HTTP_FIELDS)
        or code not in RECIPE_FAILURE_CODES
    ):
        return Disposition.WAIT
    field = refused_field(
        f"{document.get('detail') or ''} {document.get('error') or ''}"
    )
    if field is not None and field not in RECIPE_PROFILE_FIELDS:
        return Disposition.WAIT
    return Disposition.FAIL


def http_wait(status: object) -> bool:
    return isinstance(status, int) and status in TEMPORARY_HTTP


def preparation_blockers(
    blockers: Sequence[Mapping[str, Any]],
) -> list[Mapping[str, Any]]:
    return [b for b in blockers if str(b.get("code") or "") in PREPARATION_CODES]


def retry_ended(document: Mapping[str, Any]) -> bool:
    """Did the Controller stop retrying this failed application (a definite end)?"""
    if FAILURE_REPEATED in _blocker_codes(document):
        return True
    progress = document.get("progress")
    adapter = progress.get("switch_adapter") if isinstance(progress, Mapping) else None
    failures = (
        adapter.get("assignment_failures") if isinstance(adapter, Mapping) else None
    )
    return any(
        isinstance(item, Mapping) and item.get("terminal") is True
        for item in failures or []
    )
