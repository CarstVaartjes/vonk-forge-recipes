"""Owner safety: the sweep never writes profiles 1-3 and never fights a load of theirs.

The guard watches the latest application of each owner profile. While one is
queued or running, the sweep submits nothing. A load that starts *after* the
sweep began is also a preemption (it replaces whatever the Sparks run), so the
sweep requeues its in-flight tests without blaming the recipes, and stays out
of the way for a hold period after the owner's load completes.

The sweep's own loads (its restores of an owner profile) are recognised by id or
request key, and so are the Controller's continuations of them: an application
whose ``retry_of_application_id`` lineage reaches an own application is the same
decision, and is recorded as ours so the next process knows it too.
"""

from __future__ import annotations

from collections.abc import Collection
from dataclasses import dataclass
from typing import Any

from .lifecycle import ACTIVE
from .state import State
from .vonkctl import OWNER_PROFILES, Vonkctl, VonkctlError

LINEAGE_HOPS = 16  # how far back a continuation chain is followed to an own load


@dataclass(frozen=True)
class OwnerStatus:
    paused: bool = False
    reason: str = ""
    new_activity: bool = False


class OwnerGuard:
    def __init__(
        self,
        vk: Vonkctl,
        state: State,
        hold_seconds: float,
        profiles: Collection[int] = OWNER_PROFILES,
    ) -> None:
        self.vk = vk
        self.state = state
        self.hold_seconds = hold_seconds
        self.profiles = sorted(profiles)

    def _latest(self, number: int) -> tuple[str, str, str, str] | None:
        reply = self.vk.run("profile", "progress", profile=number)
        document = reply.document
        if (
            reply.ok
            and isinstance(document, dict)
            and isinstance(document.get("id"), str)
        ):
            return (
                document["id"],
                str(document.get("state", "")),
                str(document.get("request_key", "")),
                _retry_of(document),
            )
        if reply.is_not_found:
            return None
        raise VonkctlError(reply.argv, reply, "owner application is unreadable")

    def _own(self, app_id: str, request: str) -> dict[str, Any] | None:
        return next(
            (
                item
                for item in self.state.data["own_loads"]
                if item.get("application_id") == app_id
                or (request and item["request_key"] == request)
            ),
            None,
        )

    def _is_ours(
        self, number: int, app_id: str, request: str, retry_of: str, now: float
    ) -> bool:
        if self._own(app_id, request) is not None:
            return True
        # A Controller continuation (new id, new request key) of an own load is ours too.
        ancestor, seen = retry_of, {app_id}
        for _ in range(LINEAGE_HOPS):
            if not ancestor or ancestor in seen:
                return False
            seen.add(ancestor)
            origin = self._own(ancestor, "")
            if origin is None:
                reply = self.vk.run(
                    "profile", "progress", "--application", ancestor, profile=number
                )
                document = reply.document
                if not reply.ok or not isinstance(document, dict):
                    return False  # an unreadable lineage is not proof: the guard holds
                origin = self._own(ancestor, str(document.get("request_key", "")))
                if origin is None:
                    ancestor = _retry_of(document)
                    continue
            self.state.data["own_loads"].append(
                {
                    "request_key": request or origin["request_key"],
                    "application_id": app_id,
                    "kind": origin.get("kind", ""),
                    "profile": number,
                    "at": now,
                    "continuation_of": ancestor,
                }
            )
            return True
        return False

    def check(self, now: float) -> OwnerStatus:
        owner = self.state.data["owner"]
        baseline: dict[str, dict[str, str]] = owner["baseline"]
        paused: list[str] = []
        new_activity = False
        for number in self.profiles:
            try:
                latest = self._latest(number)
            except VonkctlError as error:
                paused.append(
                    f"profile {number} ownership is unreadable: {str(error)[:160]}"
                )
                continue
            known = baseline.get(str(number))
            if latest is None:
                baseline.setdefault(
                    str(number), {"id": "", "state": ""}
                )  # none yet: any later one is new
                continue
            app_id, app_state, request, retry_of = latest
            if self._is_ours(number, app_id, request, retry_of, now):
                # The sweep's own load (a restore of the owner's profile, a clearing load): not an
                # owner's. While it runs we wait for it; afterwards no hold, no preemption.
                if app_state in ACTIVE:
                    paused.append(
                        f"profile {number} application {app_id} is loading (our own load)"
                    )
                baseline[str(number)] = {"id": app_id, "state": app_state}
                continue
            changed = known is not None and known["id"] != app_id
            if app_state in ACTIVE:
                paused.append(f"profile {number} application {app_id} is loading")
                if changed:
                    new_activity = True
            elif known is not None and (changed or known["state"] in ACTIVE):
                # An owner application that appeared or settled while we ran.
                owner["hold_until"] = max(
                    float(owner["hold_until"]), now + self.hold_seconds
                )
                new_activity = new_activity or changed
            baseline[str(number)] = {"id": app_id, "state": app_state}
        if not paused and now < float(owner["hold_until"]):
            paused.append(f"holding after an owner load until {owner['hold_until']}")
        return OwnerStatus(bool(paused), "; ".join(paused), new_activity)


def _retry_of(document: dict[str, Any]) -> str:
    value = document.get("retry_of_application_id")
    return value if isinstance(value, str) else ""
