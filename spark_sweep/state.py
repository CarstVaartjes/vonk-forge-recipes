"""The resumable record: one JSON state file and one JSON-lines results log.

The state file is rewritten atomically after every change, so a crash or a
reboot resumes exactly where the sweep stopped: finished recipes are skipped,
an in-flight load is adopted, and download operations are re-found by their
deterministic request keys. The results log has the shape the platform's
``vonk-fleet-qualify-campaign`` writes (one line per outcome, latest wins),
extended with the hardware facts of the run.
"""

from __future__ import annotations

import fcntl
import json
import os
import socket
import tempfile
import time
import uuid
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from vonk_forge_contracts.sweep import SweepAdoptionReason, SweepAdoptionResult

from .lifecycle import (
    DownloadIntentKind,
    LifecycleState,
    RecipeStatus,
    ReleaseAuthority,
    SkipReasonKind,
)
from .policy import LOAD_RULES, SMOKE_RULES

SCHEMA = 2
EVENT_LIMIT = 60
# The live results log is rotated (never trimmed) once it is this large.
RESULT_ROTATE_BYTES = 4 * 1024 * 1024
TERMINAL = frozenset({"passed", "failed", "deferred", "skipped"})


def _fresh() -> dict[str, Any]:
    return {
        "schema": SCHEMA,
        # Folded into every request key: a fresh state directory never replays an old run's requests.
        "nonce": uuid.uuid4().hex,
        "recipes": {},  # key -> operational bookkeeping, never qualification proof
        "qualification_evidence": {},  # adopted and newly observed evidence
        "slots": {},  # key -> a recipe currently loading, serving or smoking
        "cleanup": {},  # assignment alias -> retry metadata; never occupies a lane
        "loads": {},  # request key -> in-flight profile application
        "load_seq": 0,
        "downloads": {},  # key -> {operation_id, request_key, ...}
        "download_history": {},  # key -> complete prior accepted attempt records
        "pins": [],  # recipe keys in the pin profile
        "learned": {},  # engine -> [[model bytes, load seconds]]
        "rate": {"ema": 0.0, "samples": 0},
        "owner": {"baseline": {}, "hold_until": 0.0, "export": {}},
        "models_done": [],  # digests whose download finished, whatever a lagging listing says
        "model_removals": {},  # model content digest -> durable fleet-removal request/receipt
        "took_over": False,
        "model_failures": {},  # model digest -> failure cluster
        "release": {},  # authenticated deployed API identity; publication is separate
        "release_history": [],
        "publication": {},  # accepted signed publication, not a deployed component
        "publication_history": [],
        "platform": {},  # latest authenticated API/worker observation (unknowns retained)
        "cleanup_loads": {},  # original cleanup request -> per-effect observation
        # Every load the sweep submitted itself (request key, application id): never an owner's load.
        "own_loads": [],
        "own_aliases": [],  # workload names the sweep put in its profile: never an owner's
        "own_seq": 0,  # numbers the loads the sweep submits itself: one request key per attempt
        "infra": {},  # source -> {message, since, count, until}: problems that are not the recipes
        "mode": "single",
        "events": [],
    }


def _superseded(reason: object) -> bool:
    text = str(reason or "").lower()
    return any(
        marker in text
        for marker in (
            "reconciled by profile retry",
            "profile intent was superseded",
            "changed during admission",
        )
    )


def _adopt_evidence(entry: dict[str, Any]) -> None:
    if isinstance(entry.get("evidence"), str):
        entry.setdefault("evidence_bundle", entry.pop("evidence"))


def _adopt(
    data: dict[str, Any],
    now: float,
    record_result: Callable[[SweepAdoptionResult], None],
) -> dict[str, Any]:
    """Normalize saved sweep records once; runtime reads only the current shape.

    Receipts, accepted content and remote effect scopes are never reconstructed
    from the current catalog. An unbound lane still goes through canonical
    requeue/cleanup, which retains its original request for observation.
    """
    if "qualification_evidence" not in data:
        from .qualification import adopt_state_evidence

        data["qualification_evidence"] = {
            key: evidence.model_dump(mode="json")
            for key, evidence in adopt_state_evidence(data).items()
        }
    merged = _fresh()
    for name, value in data.items():
        if isinstance(value, dict) and isinstance(merged.get(name), dict):
            merged[name].update(value)
        else:
            merged[name] = value
    data = merged
    if data["schema"] == SCHEMA:
        return data
    load = data.pop("load", None)
    if load:
        data["loads"].setdefault(load["request_key"], load)
    for record in data["downloads"].values():
        intent = record.get("request_intent")
        if (
            record.get("accepted_intent") is None
            and record.get("operation_id")
            and isinstance(intent, dict)
            and isinstance(intent.get("recipe_key"), str)
        ):
            # Old sweep submissions used these saved CLI arguments. A retry still
            # requires the Controller's exact terminal receipt before any effect.
            parent = record.get("retry_of_operation_id")
            record["accepted_intent"] = (
                {"kind": DownloadIntentKind.RETRY, "operation_id": parent}
                if parent
                else {
                    "kind": DownloadIntentKind.SELECTOR,
                    "selector": intent["recipe_key"],
                    "force": False,
                }
            )
    release = data["release"]
    if release and release.get("authority") != ReleaseAuthority.DEPLOYED_CONTROLLER:
        data["release_history"].append({**release, "authority": None})
        release.clear()
    for row in data["release_history"]:
        if (
            isinstance(row, dict)
            and row.get("authority") != ReleaseAuthority.DEPLOYED_CONTROLLER
        ):
            row["authority"] = None
    for app in data.get("apps", {}).values():
        if app.pop("superseded", False) or (
            app.get("state") in (LifecycleState.FAILED, LifecycleState.CANCELLED)
            and (
                _superseded(app.get("reason"))
                or app.get("cause") == LifecycleState.SUPERSEDED
            )
        ):
            app["state"] = LifecycleState.SUPERSEDED
    for key, entry in data["recipes"].items():
        _adopt_evidence(entry)
        if (
            entry.get("status") == RecipeStatus.SKIPPED
            and "reason_kind" not in entry
            and str(entry.get("reason", "")).startswith("not testable on this fleet")
        ):
            entry["reason_kind"] = SkipReasonKind.FLEET
        if (
            entry.get("release")
            and entry.get("release_authority") != ReleaseAuthority.DEPLOYED_CONTROLLER
        ):
            entry["release_authority"] = None
        if entry.get("status") != RecipeStatus.FAILED:
            continue
        evidence = entry.get("evidence")
        evidence = evidence if isinstance(evidence, dict) else {}
        why = None
        if _superseded(evidence.get("reason")) or _superseded(entry.get("error")):
            why = SweepAdoptionReason.SUPERSESSION
        elif (
            entry.get("failure_class") == "timeout"
            and int(entry.get("rules", 1)) < LOAD_RULES
        ):
            why = SweepAdoptionReason.LOAD_RULES
        elif (
            entry.get("phase") == "smoke"
            and entry.get("failure_class") == "smoke-assertion"
            and int(entry.get("smoke_rules", 1)) < SMOKE_RULES
        ):
            why = SweepAdoptionReason.SMOKE_RULES
        elif (
            entry.get("phase") == "review"
            and entry.get("failure_class") == "capacity"
            and not evidence.get("freeing_attempts")
        ):
            why = SweepAdoptionReason.CAPACITY
        if why is None:
            continue
        # Keep all prior failure evidence in state as well as the append-only log.
        previous = dict(entry)
        record_result(SweepAdoptionResult(recipe=key, reason=why, previous=previous))
        entry.update(
            status=RecipeStatus.PENDING,
            attempts=0,
            previous=previous,
            requeued_because=why,
        )
        for field in (
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
            entry.pop(field, None)
        data["events"].append({"at": now, "message": f"adopted {key}: {why}"})
    del data["events"][:-EVENT_LIMIT]
    data["schema"] = SCHEMA
    return data


def write_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


class StateLocked(RuntimeError):
    """Another sweep process holds this state directory."""


class StateLock:
    """An exclusive lock on a state directory: pid and host, released when the process ends.

    Two sweeps on one state would fight over the same profile and requests. The lock
    is an ``flock`` on ``run.lock``, so a crashed process never leaves a stale lock
    behind; the file's content only says who holds it.
    """

    def __init__(self, directory: Path) -> None:
        self.path = directory / "run.lock"
        self._handle: Any = None

    def acquire(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        handle = self.path.open("a+", encoding="utf-8")
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            handle.seek(0)
            holder = handle.read().strip() or "an unknown process"
            handle.close()
            raise StateLocked(
                f"another sweep is running on {self.path.parent} ({holder}); "
                "stop it first (its status page shows what it is doing)"
            ) from None
        handle.seek(0)
        handle.truncate()
        handle.write(f"pid {os.getpid()} on {socket.gethostname()}")
        handle.flush()
        self._handle = handle

    def release(self) -> None:
        if self._handle is not None:
            self._handle.close()  # closing releases the flock
            self._handle = None


class State:
    def __init__(
        self,
        path: Path,
        data: dict[str, Any] | None = None,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self.path = path
        self.data = data if data is not None else _fresh()
        self.clock = clock
        self.adoptions: list[SweepAdoptionResult] = []

    @classmethod
    def load(cls, path: Path, clock: Callable[[], float] = time.time) -> State:
        if not path.exists():
            return cls(path, None, clock)
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(data, dict) or data.get("schema") not in (1, SCHEMA):
                raise ValueError("unreadable state schema")
            for name, fresh in _fresh().items():
                if (
                    name in data
                    and isinstance(fresh, (dict, list))
                    and not isinstance(data[name], type(fresh))
                ):
                    raise ValueError(f"unreadable state field: {name}")
            legacy = data.get("load")
            if legacy is not None and (
                not isinstance(legacy, dict)
                or not isinstance(legacy.get("request_key"), str)
            ):
                raise ValueError("unreadable legacy request")
            for name in (
                "slots",
                "recipes",
                "downloads",
                "loads",
                "cleanup_loads",
                "infra",
            ):
                if any(
                    not isinstance(value, dict) for value in data.get(name, {}).values()
                ):
                    raise ValueError(f"unreadable state entry: {name}")
            for slot in data.get("slots", {}).values():
                if (
                    slot.get("phase") not in ("loading", "smoking", "finished")
                    or not isinstance(slot.get("request_key"), str)
                    or not isinstance(slot.get("alias"), str)
                    or not isinstance(slot.get("node_ids"), list)
                    or any(not isinstance(node, str) for node in slot["node_ids"])
                    or not isinstance(slot.get("started_at"), (int, float))
                ):
                    raise ValueError("unreadable lane record")
            owner = data.get("owner", {})
            if not isinstance(owner.get("baseline", {}), dict):
                raise TypeError("unreadable owner observations")
            if any(
                not isinstance(value, dict)
                or not isinstance(value.get("id"), str)
                or not isinstance(value.get("state"), str)
                for value in owner.get("baseline", {}).values()
            ):
                raise ValueError("unreadable owner observation")
            if any(
                not isinstance(value, dict)
                or not isinstance(value.get("request_key"), str)
                for value in data.get("own_loads", [])
            ):
                raise ValueError("unreadable owned request")
        except (TypeError, ValueError, UnicodeError) as error:
            # Preserve unreadable bytes, then rebuild disposable sweep bookkeeping.
            backup = path.with_name(f"{path.name}.unreadable-{uuid.uuid4().hex}")
            path.replace(backup)
            recovered = cls(path, None, clock)
            recovered.event(
                f"state rebuilt after {type(error).__name__}: {backup.name}"
            )
            return recovered
        adoptions: list[SweepAdoptionResult] = []
        adopted = cls(path, _adopt(data, clock(), adoptions.append), clock)
        adopted.adoptions = adoptions
        return adopted

    def save(self) -> None:
        live = {
            slot.get("request_key")
            for slot in self.slots.values()
            if slot.get("phase") != "finished"
        }
        if "apps" in self.data:
            self.data["apps"] = {
                key: app for key, app in self.data["apps"].items() if key in live
            }
        del self.data["release_history"][:-EVENT_LIMIT]
        del self.data["publication_history"][:-EVENT_LIMIT]
        self.data["updated_at"] = self.clock()
        write_atomic(self.path, json.dumps(self.data, indent=1, sort_keys=True))

    @property
    def nonce(self) -> str:
        return str(self.data["nonce"])

    # -- recipes -----------------------------------------------------------

    @property
    def recipes(self) -> dict[str, dict[str, Any]]:
        return self.data["recipes"]

    def entry(self, key: str) -> dict[str, Any]:
        return self.recipes.setdefault(key, {"status": "pending", "attempts": 0})

    def status(self, key: str) -> str:
        return str(self.recipes.get(key, {}).get("status", "pending"))

    def count(self, status: str) -> int:
        return sum(1 for e in self.recipes.values() if e.get("status") == status)

    def event(self, message: str) -> None:
        events = self.data["events"]
        events.append({"at": self.clock(), "message": message})
        del events[:-EVENT_LIMIT]

    # -- shorthands --------------------------------------------------------

    @property
    def slots(self) -> dict[str, dict[str, Any]]:
        return self.data["slots"]

    @property
    def downloads(self) -> dict[str, dict[str, Any]]:
        return self.data["downloads"]


class ResultsLog:
    """One JSON line per outcome, in the platform campaign's results-log shape.

    Append-only: a large log is rotated to ``results.jsonl.<UTC stamp>`` and a new one
    begins, so no outcome or its evidence is ever dropped. ``entries`` reads them all.
    """

    def __init__(self, path: Path, authority_id: str) -> None:
        self.path = path
        self.authority_id = authority_id

    def append(self, **entry: Any) -> dict[str, Any]:
        record = {
            "recorded_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
            "authority_id": self.authority_id,
            **entry,
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, sort_keys=True, default=str) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        if self.path.stat().st_size > RESULT_ROTATE_BYTES:
            stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
            os.replace(self.path, self.path.with_name(f"{self.path.name}.{stamp}"))
        return record

    def files(self) -> list[Path]:
        """The rotated logs, oldest first, then the live one."""
        rotated = sorted(self.path.parent.glob(f"{self.path.name}.*"))
        return [*rotated, self.path]

    def entries(self) -> list[dict[str, Any]]:
        lines: list[str] = []
        for path in self.files():
            try:
                lines += path.read_text(encoding="utf-8").splitlines()
            except FileNotFoundError:
                continue
        records: list[dict[str, Any]] = []
        for line in lines:
            try:
                record = json.loads(line)
            except ValueError:
                continue  # a torn final line from an interrupted write
            if (
                isinstance(record, dict)
                and record.get("authority_id") == self.authority_id
            ):
                _adopt_evidence(record)
                records.append(record)
        return records
