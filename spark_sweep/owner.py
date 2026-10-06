"""Owner safety: the sweep never writes profiles 1-3 and never fights a load of theirs.

The guard watches the latest application of each owner profile. While one is
queued or running, the sweep submits nothing. A load that starts *after* the
sweep began is also a preemption (it replaces whatever the Sparks run), so the
sweep requeues its in-flight tests without blaming the recipes, and stays out
of the way for a hold period after the owner's load completes.
"""

from __future__ import annotations

from collections.abc import Collection
from dataclasses import dataclass

from .lifecycle import ACTIVE
from .state import State
from .vonkctl import OWNER_PROFILES, Vonkctl, VonkctlError


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

    def _latest(self, number: int) -> tuple[str, str, str] | None:
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
            )
        if isinstance(document, dict) and document.get("code") in {
            "not_found",
            "profile.application_not_found",
        }:
            return None
        raise VonkctlError(reply.argv, reply, "owner application is unreadable")

    def _is_ours(self, app_id: str, request: str) -> bool:
        return any(
            item.get("application_id") == app_id or item["request_key"] == request
            for item in self.state.data["own_loads"]
        )

    def check(self, now: float) -> OwnerStatus:
        owner = self.state.data["owner"]
        baseline: dict[str, dict[str, str]] = owner["baseline"]
        paused: list[str] = []
        new_activity = False
        for number in self.profiles:
            try:
                latest = self._latest(number)
            except VonkctlError:
                paused.append(f"profile {number} ownership is unreadable")
                continue
            known = baseline.get(str(number))
            if latest is None:
                baseline.setdefault(
                    str(number), {"id": "", "state": ""}
                )  # none yet: any later one is new
                continue
            app_id, app_state, request = latest
            if self._is_ours(app_id, request):
                # The sweep's own load (a restore of the owner's profile, a clearing load): not an
                # owner's. While it runs we wait for it; afterwards no hold, no preemption.
                if app_state in ACTIVE:
                    paused.append(f"profile {number} is loading (our own load)")
                baseline[str(number)] = {"id": app_id, "state": app_state}
                continue
            changed = known is not None and known["id"] != app_id
            if app_state in ACTIVE:
                paused.append(f"profile {number} is loading")
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
            paused.append("holding after an owner load")
        return OwnerStatus(bool(paused), "; ".join(paused), new_activity)
