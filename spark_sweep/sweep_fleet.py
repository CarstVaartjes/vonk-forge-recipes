"""Fleet and profile: preflight, claim, enrolment and lanes (methods of ``Sweep``)."""

# pyright: reportAttributeAccessIssue=false
# The methods reach the state set in ``Sweep.__init__`` and the sibling mixins through ``self``.
from __future__ import annotations

from dataclasses import replace
from typing import Any

from .catalog import (
    Recipe,
    fetch_fleet,
)
from .prefetch import SWEEP_LABEL
from .sweep_support import (
    SweepProfileOwnershipError,
)
from .vonkctl import OWNER_PROFILES, VonkctlError


class SweepFleet:
    def preflight(self) -> None:
        """Read-only facts before anything is changed: fleet, owner baseline, owner profile export."""
        self.fleet = fetch_fleet(self.vk, self.cfg.default_spark_memory)
        if not any(s.online for s in self.fleet.sparks):
            raise VonkctlError(
                ("fleet",), None, "no Spark is online; observing enrollment"
            )
        usable = self.sparks()
        if usable and not self.prefetcher.config.pin_spark:
            self.prefetcher.config = replace(
                self.prefetcher.config, pin_spark=usable[0].name
            )
        self.claim_profile(self.cfg.sweep_profile, "Hardware sweep")
        if self.cfg.prefetch.pin_profile is not None:
            self.claim_profile(
                self.cfg.prefetch.pin_profile, "Hardware sweep pins (never loaded)"
            )
        if not self.state.data["owner"]["export"]:
            for number in sorted(
                OWNER_PROFILES
            ):  # sequential: three one-off start exports
                reply = self.vk.run("profile", "export", profile=number)
                if reply.ok and isinstance(reply.document, dict):
                    self.state.data["owner"]["export"][str(number)] = reply.document
        self.owner_status = self.guard.check(self.clock.now())
        self.state.save()

    def claim_profile(self, number: int, name: str) -> None:
        """Use a profile only if it is empty or already ours: never adopt someone's saved profile."""
        reply = self.vk.run("profile", "export", profile=number)
        if not reply.ok or not isinstance(reply.document, dict):
            raise VonkctlError(
                reply.argv, reply, "profile ownership is unreadable; retry observation"
            )
        definition = reply.document
        labels = definition.get("labels") or {}
        if labels.get(SWEEP_LABEL[0]) == SWEEP_LABEL[1]:
            return
        if definition.get("assignments"):
            raise SweepProfileOwnershipError(
                f"profile {number} already holds assignments that are not the sweep's; "
                "choose another number"
            )
        self.vk.call(
            "profile",
            "configure",
            "--name",
            name,
            "--label",
            "=".join(SWEEP_LABEL),
            "--yes",
            profile=number,
        )

    # ---------------------------------------------------------------- catalog

    def enrolled(self) -> list[Any]:
        """The Sparks the fleet listing enrolls (the chosen ones), online or not: the topology."""
        if self.cfg.sparks:
            picked = [self.fleet.by_ref(ref) for ref in self.cfg.sparks]
            return [s for s in picked if s is not None]
        return list(self.fleet.sparks)

    def sparks(self) -> list[Any]:
        """The lanes that can take a placement now: enrolled and online."""
        return [s for s in self.enrolled() if s.online]

    def lanes_short(self, recipe: Recipe) -> int:
        """How many more online lanes a recipe waits for (0: it can be placed now)."""
        return max(0, recipe.node_count - len(self.sparks()))

    @property
    def catalog_loaded(self) -> bool:
        """Nothing is done, or stuck, before the recipe library has been read once."""
        return self.recipe_list.passes > 0
