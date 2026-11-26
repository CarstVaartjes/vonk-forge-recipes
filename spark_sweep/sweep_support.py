"""Constants, records and helpers of the sweep run loop."""

from __future__ import annotations

import json
import signal
import threading
import time
import traceback
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import Future
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from recipe_tooling.qualification.identity import execution_key
from recipe_tooling.qualification.memory_need.constants import SPARK_MEMORY_BYTES
from recipe_tooling.qualification.sweep_settings import SETTINGS
from vonk_forge_contracts import (
    QualificationIdentity,
    QualificationStatus,
    QualificationStatusView,
)

from . import policy
from .catalog import (
    Recipe,
)
from .definitions import Definitions
from .lifecycle import (
    RecipeStatus,
    RetestReason,
    SkipReasonKind,
)
from .prefetch import PrefetchConfig
from .profile_alias import profile_alias
from .smoke import (
    HttpConfig,
    SmokeResult,
)
from .source_serving import (
    SOURCE_WAIT_SECONDS,
)

WAIT_EVENT_SECONDS = 60.0  # a wait says what it waits for at least this often
# Clearing the fleet and reviewing again, for a capacity refusal (an OBSERVE outcome),
# before it counts against the recipe.
MAX_REVIEW_FREEING = 2
TAKEOVER_RETRY_SECONDS = 30.0
SKEW_CHECK_INTERVAL = 300.0
INFRA_BASE = 30.0
INFRA_CAP = 600.0
# Ceiling of the growing re-plan delay after a wait receipt ended without a successor.
ENDED_RECEIPT_REPLAN_CAP = 3600.0
# Identical platform-exit failures of one content identity that end the retrying (a FAIL).
IDENTICAL_EXIT_LIMIT = 3
RETEST_SECONDS = 24 * 3600.0
ENDPOINT_RECHECK_SECONDS = 20.0
# A results row carries the Controller document it ended on, bounded.
DOCUMENT_EVIDENCE_LIMIT = 65536


def retest_reason(
    view: QualificationStatusView | None,
    current: Callable[[], QualificationIdentity],
    *,
    unpublishable: bool = False,
    revalidate_passes: bool = True,
    recheck: bool = False,
) -> RetestReason | None:
    """Why a recorded verdict asks for a new run, from the derived status alone.

    The one due-rule: `run` requeues by it and `plan` counts by it.
    """
    if view is None:
        return None
    if unpublishable:
        return RetestReason.UNPUBLISHABLE_PASS
    if (
        view.status == QualificationStatus.BLOCKED_PLATFORM
        and view.last_failure is not None
        and execution_key(view.last_failure.identity) != execution_key(current())
    ):
        return RetestReason.DEFERRED_OLDER_REVISION
    why = {
        QualificationStatus.RECHECK: RetestReason.PASSED_OLDER_REVISION,
        QualificationStatus.FIX_TO_VERIFY: RetestReason.FAILED_OLDER_REVISION,
    }.get(view.status)
    if why == RetestReason.PASSED_OLDER_REVISION and not revalidate_passes:
        why = None
    return why or (RetestReason.OPERATOR_RECHECK if recheck else None)


@dataclass(frozen=True)
class ProfilePruneOutcome:
    waiting_on: str | None = None


class SweepProfileOwnershipError(RuntimeError):
    """Destructive profile edits may not adopt another requester's assignments."""


def _interrupt(_signum: int, _frame: Any) -> None:
    raise KeyboardInterrupt


def _install_signal_handlers() -> dict[int, Any]:
    """SIGINT and SIGTERM raise KeyboardInterrupt, even when started with SIGINT ignored.

    A process started in the background of a non-interactive shell inherits SIGINT as
    ignored, and Python then installs no handler of its own: Ctrl-C and ``kill -INT``
    would do nothing for as long as it ran.
    """
    if threading.current_thread() is not threading.main_thread():
        return {}
    previous: dict[int, Any] = {}
    for number in (signal.SIGINT, signal.SIGTERM):
        previous[number] = signal.signal(number, _interrupt)
    return previous


def _restore_signal_handlers(previous: Mapping[int, Any]) -> None:
    for number, handler in previous.items():
        signal.signal(number, handler)


class Clock(Protocol):
    def now(self) -> float: ...
    def sleep(self, seconds: float) -> None: ...


class RealClock:
    def now(self) -> float:
        return time.time()

    def sleep(self, seconds: float) -> None:
        time.sleep(seconds)


FLEET_SKIP = SkipReasonKind.FLEET
#: Skipped because the recipe left the complete catalog; undone when it comes back.
CATALOG_SKIP = SkipReasonKind.CATALOG
#: Older state files hold job recipes skipped as "platform cannot load yet" (the removed
#: ``--job-recipes`` gate); reconcile returns every such entry to pending.
PLATFORM_SKIP = SkipReasonKind.PLATFORM


def _fleet_skip(entry: Mapping[str, Any]) -> bool:
    """A skip whose typed reason depends on the fleet."""
    return (
        entry.get("status") == RecipeStatus.SKIPPED
        and entry.get("reason_kind") == FLEET_SKIP
    )


DEFAULT_BOOST = SETTINGS.default_boost


@dataclass
class SweepConfig:
    state_dir: Path
    authority_id: str
    sweep_profile: int = 10
    pin_profile: int | None = 13
    sparks: tuple[str, ...] = ()
    reserve_bytes: int = 8_000_000_000
    default_spark_memory: int = SPARK_MEMORY_BYTES
    poll_seconds: float = 10.0
    catalog_seconds: float = 60.0
    status_seconds: float = 15.0
    idle_alert_seconds: float = 600.0
    owner_hold_seconds: float = 1800.0
    owner_poll_seconds: float = 15.0
    retry_delay: float = 60.0
    defer_delay: float = 120.0
    http: HttpConfig = field(default_factory=HttpConfig)
    prefetch: PrefetchConfig = field(default_factory=PrefetchConfig)
    timeouts: policy.TimeoutPolicy = field(default_factory=policy.TimeoutPolicy)
    boost: tuple[str, ...] = DEFAULT_BOOST
    recent_days: float = 3.0
    seed: frozenset[str] = frozenset()
    # `--recheck`: slugs or keys to run once on the deployed release, whatever their status.
    recheck: frozenset[str] = frozenset()
    only: tuple[str, ...] = ()
    limit: int = 0
    variants_last: bool = False
    revalidate_passes: bool = True
    restore_owner: int | None = None
    stop_at_end: bool = True
    watch_seconds: float = 0.0
    allow_version_skew: bool = False
    release_seconds: float = 300.0
    retry_failed: bool = False
    publish_evidence: bool = False
    # The stall window while bytes are copied to a Spark (the learned window covers the rest).
    copy_stall_seconds: float = 600.0
    # Preparation waits use this shorter bound, capped by the lane budget.
    # Expiry re-plans without spending an attempt.
    blocked_release_seconds: float = 300.0
    # How long recipes wait for the catalog to serve this sweep's source.
    source_wait_seconds: float = SOURCE_WAIT_SECONDS


def selects(only: Sequence[str], key: str) -> bool:
    """The ``--only`` rule: a recipe is selected when its key contains one of the words."""
    return not only or any(word in key for word in only)


#: The traceback frames a sweep crash keeps as evidence: where it raised, and its callers.
CRASH_FRAMES = 3


def crash_evidence(error: BaseException) -> dict[str, Any]:
    """What a crash of the sweep's own code says: type, message, filename, last frames."""
    frames = [
        line.strip()
        for line in traceback.format_tb(error.__traceback__)[-CRASH_FRAMES:]
    ]
    return {
        "type": type(error).__name__,
        "message": str(error)[:500],
        "filename": str(getattr(error, "filename", None) or "") or None,
        "frames": [frame[:400] for frame in frames],
    }


def alias_for(recipe: Recipe, definitions: Definitions) -> str:
    """The name the lane serves under: the reviewed service alias, else the slug.

    Always a valid profile alias, so a spelling the contract refuses (capitals,
    a ``/``) never reaches the Controller.
    """
    return profile_alias(definitions.alias(recipe.key) or recipe.slug)


def _blocker_text(document: Any) -> str:
    """The codes and details a refused review gives, as one line of evidence."""
    if not isinstance(document, Mapping):
        return "no review document"
    found = [
        f"{r.get('code', '')}: {r.get('detail', '')}"
        for r in document.get("reasons", [])
        if isinstance(r, Mapping) and r.get("severity") == "error"
    ]
    found += [
        f"{b.get('code', '')}: {b.get('detail', '')}"
        for d in document.get("admission_decisions", [])
        if isinstance(d, Mapping) and d.get("allowed") is False
        for b in d.get("blockers", [])
        if isinstance(b, Mapping)
    ]
    if not found and (error := document.get("error") or document.get("code")):
        found.append(str(error))
    return "; ".join(found)[:400] or "no blocker named"


def _bounded_document(document: Mapping[str, Any]) -> Any:
    """A Controller document as evidence: whole when small, else a bounded excerpt."""
    text = json.dumps(document, sort_keys=True, default=str)
    if len(text) <= DOCUMENT_EVIDENCE_LIMIT:
        return document
    return {"excerpt": text[:DOCUMENT_EVIDENCE_LIMIT], "truncated_bytes": len(text)}


def _done_future(result: SmokeResult) -> Future[SmokeResult]:
    future: Future[SmokeResult] = Future()
    future.set_result(result)
    return future
