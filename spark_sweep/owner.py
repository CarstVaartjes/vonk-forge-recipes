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

from .state import State
from .vonkctl import OWNER_PROFILES, Vonkctl

ACTIVE = frozenset({"queued", "running"})


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

    def _latest(self, number: int) -> tuple[str, str] | None:
        reply = self.vk.run("profile", "progress", profile=number)
        document = reply.document
        if (
            reply.ok
            and isinstance(document, dict)
            and isinstance(document.get("id"), str)
        ):
            return document["id"], str(document.get("state", ""))
        return None  # no application yet, or unreadable: nothing to defer to

    def check(self, now: float) -> OwnerStatus:
        owner = self.state.data["owner"]
        baseline: dict[str, dict[str, str]] = owner["baseline"]
        paused: list[str] = []
        new_activity = False
        for number in self.profiles:
            latest = self._latest(number)
            known = baseline.get(str(number))
            if latest is None:
                baseline.setdefault(
                    str(number), {"id": "", "state": ""}
                )  # none yet: any later one is new
                continue
            app_id, app_state = latest
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
