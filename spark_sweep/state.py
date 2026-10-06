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

SCHEMA = 1
EVENT_LIMIT = 60
RESULT_LIMIT = 2000
TERMINAL = frozenset({"passed", "failed", "deferred", "skipped"})


def _fresh() -> dict[str, Any]:
    return {
        "schema": SCHEMA,
        # Folded into every request key: a fresh state directory never replays an old run's requests.
        "nonce": uuid.uuid4().hex,
        "recipes": {},  # key -> result entry
        "slots": {},  # key -> a recipe currently loading, serving or smoking
        "cleanup": {},  # assignment alias -> retry metadata; never occupies a lane
        "loads": {},  # request key -> in-flight profile application
        "load_seq": 0,
        "downloads": {},  # key -> {operation_id, request_key, ...}
        "pins": [],  # recipe keys in the pin profile
        "learned": {},  # engine -> [[model bytes, load seconds]]
        "rate": {"ema": 0.0, "samples": 0},
        "owner": {"baseline": {}, "hold_until": 0.0, "export": {}},
        "models_done": [],  # digests whose download finished, whatever a lagging listing says
        "took_over": False,
        "model_failures": {},  # model digest -> failure cluster
        "release": {},  # the accepted Controller release last seen: sha, version, seen_at
        "release_history": [],
        # Every load the sweep submitted itself (request key, application id): never an owner's load.
        "own_loads": [],
        "own_aliases": [],  # workload names the sweep put in its profile: never an owner's
        "own_seq": 0,  # numbers the loads the sweep submits itself: one request key per attempt
        "infra": {},  # source -> {message, since, count, until}: problems that are not the recipes
        "mode": "single",
        "events": [],
    }


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
        legacy = self.data.pop("load", None)
        self.data.setdefault("loads", {})
        if legacy:
            self.data["loads"].setdefault(legacy["request_key"], legacy)

    @classmethod
    def load(cls, path: Path, clock: Callable[[], float] = time.time) -> State:
        if not path.exists():
            return cls(path, None, clock)
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(data, dict) or data.get("schema") != SCHEMA:
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
            for name in ("slots", "recipes", "downloads", "loads", "infra"):
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
        merged = _fresh()
        for name, value in data.items():
            if isinstance(value, dict) and isinstance(merged.get(name), dict):
                merged[name].update(value)
            else:
                merged[name] = value
        return cls(path, merged, clock)

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
    """One JSON line per outcome, in the platform campaign's results-log shape."""

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
        # Retain bounded current evidence; results are bookkeeping, not a permanent audit.
        if self.path.stat().st_size > 4 * 1024 * 1024:
            with self.path.open("rb") as stream:
                stream.seek(max(0, self.path.stat().st_size - 4 * 1024 * 1024))
                tail = stream.read().decode("utf-8", "replace").splitlines()
            valid = []
            for line in tail[-RESULT_LIMIT:]:
                try:
                    json.loads(line)
                except ValueError:
                    continue
                valid.append(line)
            write_atomic(self.path, "\n".join(valid) + "\n")
        return record

    def entries(self) -> list[dict[str, Any]]:
        try:
            lines = self.path.read_text(encoding="utf-8").splitlines()
        except FileNotFoundError:
            return []
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
                records.append(record)
        return records
