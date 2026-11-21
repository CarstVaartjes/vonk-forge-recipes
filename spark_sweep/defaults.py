"""The sweep's operator defaults, in one place; every value has a CLI flag that overrides it."""

from __future__ import annotations

from dataclasses import dataclass

from .vonkctl import OWNER_PROFILES


@dataclass(frozen=True)
class SweepDefaults:
    # Equal to AUTHORITY_ID in tools/build-qualification-authority (a test checks it), so the
    # results log is read by `vonk-fleet-qualify-campaign status` for that authority.
    authority_id: str = "nl-family-aware-20260924"
    # Selector words of the owner's favourite recipes; matching recipes are tested first.
    priority: tuple[str, ...] = ("glm-5-3",)
    # Profile numbers the sweep must never use.
    protected_profiles: frozenset[int] = OWNER_PROFILES


DEFAULTS = SweepDefaults()
