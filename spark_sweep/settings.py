"""The sweep's three fixed settings, read from ``policy/sweep.json`` (their one owner).

``authority_id`` names the qualification authority the results log belongs to (the
catalog build derives the same authority from the same file), ``default_boost`` is the
keyword the queue prioritises by default, and ``owner_profiles`` are the Controller
profiles a sweep never writes.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, ConfigDict

SETTINGS_PATH = Path(__file__).resolve().parents[1] / "policy" / "sweep.json"


class SweepSettings(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    authority_id: str
    default_boost: tuple[str, ...]
    owner_profiles: frozenset[int]


SETTINGS = SweepSettings.model_validate_json(SETTINGS_PATH.read_bytes())
