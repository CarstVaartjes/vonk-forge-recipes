"""The contracts version that introduced each serving kind and recipe-api mode.

A deployed platform reads recipes with the contracts version it pins. A feature
newer than that version is a read failure there (strict ``Literal``), or for an
api mode a recipe it serves without facts. Recipes CI refuses a recipe whose
features need a newer contracts version than the pinned platform reports
(``tools/check-platform-contracts``).

This module is the one owner of those minimums. A feature missing here fails
``tests/test_introduced_versions.py``, so a new kind or mode cannot ship
without saying when it arrived.
"""

from __future__ import annotations

from collections.abc import Mapping

from .api import RecipeApi
from .recipe import RecipeDefinition, ServingKind

# Features that predate this table need nothing: every deployed platform reads them.
ORIGINAL = "0.0.0"

SERVING_KIND_SINCE: Mapping[ServingKind, str] = {
    "openai.health": ORIGINAL,
    "openai.chat": ORIGINAL,
    "openai.vision": ORIGINAL,
    "openai.tools": ORIGINAL,
    "openai.completion": ORIGINAL,
    "openai.embedding": ORIGINAL,
    "openai.systemone": ORIGINAL,
    "openai.transcription": "4.4.0",
    "openai.speech": "4.4.0",
    "image-job.output": ORIGINAL,
    "audio-job.output": ORIGINAL,
    "video-job.output": ORIGINAL,
    "mesh-job.output": ORIGINAL,
    "artifact-job.output": ORIGINAL,
}

API_MODE_SINCE: Mapping[str, str] = {
    "chat": ORIGINAL,
    "decision": ORIGINAL,
    "audio_transcription": "4.3.0",
    "audio_speech": "4.3.0",
}


def parse_version(text: str) -> tuple[int, int, int]:
    """``MAJOR.MINOR.PATCH`` as a comparable tuple; anything else raises."""

    parts = text.split(".")
    if len(parts) != 3 or not all(p.isascii() and p.isdigit() for p in parts):
        raise ValueError(f"not a MAJOR.MINOR.PATCH version: {text!r}")
    major, minor, patch = (int(p) for p in parts)
    return major, minor, patch


def features_used(recipe: RecipeDefinition, api: RecipeApi | None) -> dict[str, str]:
    """Each serving kind and api mode the documents use, with its minimum version."""

    used = {
        f"serving kind {check.kind}": SERVING_KIND_SINCE[check.kind]
        for check in recipe.validation.serving.checks
    }
    if api is not None:
        blocks = [api.api, *(choice.api for choice in api.option_choices)]
        used |= {
            f"api mode {block.mode}": API_MODE_SINCE[block.mode] for block in blocks
        }
    return used


def unreadable_features(
    recipe: RecipeDefinition, api: RecipeApi | None, platform_version: str
) -> list[str]:
    """The features a platform on ``platform_version`` cannot read (sorted)."""

    have = parse_version(platform_version)
    return sorted(
        f"{feature} needs contracts {since}"
        for feature, since in features_used(recipe, api).items()
        if parse_version(since) > have
    )
