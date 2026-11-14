"""The contracts version that introduced each closed-vocabulary value a document carries.

The deployed Controller reads recipes and Model displays with the contracts
version its ``public-contracts.lock`` pins. A closed vocabulary (a strict
``Literal`` or ``StrEnum``) refuses a value newer than that version, so the
document is not imported until the platform reads it. Recipes CI refuses such a
document up front (``tools/check-platform-contracts``).

Gated vocabularies, one table each:

* ``ServingKind`` (a recipe's serving checks). Their assertions and the
  ``http-multipart`` request come only with their kind, so they need no row.
* ``CapabilityName`` (a Model display's capabilities), a closed ``StrEnum``
  since contracts 4.1.0.
* The ``kind`` of a reasoning effort in a recipe's API facts (base block and
  every option choice). A platform that cannot read one drops the route's whole
  API document (``_entry_api`` fails open), so every capability fact the route
  published before would vanish with it.

A recipe-api ``mode`` is not gated: a new mode arrives with a new kind of
route, which published no facts before.

A value missing from a table fails ``tests/test_introduced_versions.py``, so a
new kind or capability cannot ship without saying when it arrived. This module
is used only by recipes CI, so it lives here and not in the published wheel.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import get_args

from vonk_forge_contracts import (
    CapabilityName,
    ChatApi,
    ModelDisplay,
    RecipeApi,
    RecipeDefinition,
)
from vonk_forge_contracts.api import ApiReasoningNone
from vonk_forge_contracts.recipe import ServingKind

# Values that predate this table need nothing: every deployed platform reads them.
ORIGINAL = "0.0.0"

SERVING_KIND_SINCE: Mapping[str, str] = {
    **{kind: ORIGINAL for kind in get_args(ServingKind)},
    "openai.transcription": "4.4.0",
    "openai.speech": "4.4.0",
}

CAPABILITY_SINCE: Mapping[CapabilityName, str] = {
    name: ORIGINAL for name in CapabilityName
}

# ``levels`` documents gained their ``kind`` in 4.7.0; older readers ignore it.
EFFORT_KIND_SINCE: Mapping[str, str] = {
    "levels": ORIGINAL,
    "text": "4.7.0",
}


def parse_version(text: str) -> tuple[int, int, int]:
    """``MAJOR.MINOR.PATCH`` as a comparable tuple; anything else raises."""

    parts = text.split(".")
    if len(parts) != 3 or not all(p.isascii() and p.isdigit() for p in parts):
        raise ValueError(f"not a MAJOR.MINOR.PATCH version: {text!r}")
    major, minor, patch = (int(p) for p in parts)
    return major, minor, patch


def capability_features(display: ModelDisplay) -> dict[str, str]:
    """Each capability a Model display lists, with the version that introduced it."""

    # Authored displays are validated strictly, so every name is a CapabilityName.
    return {
        f"capability {name.value}": CAPABILITY_SINCE[name]
        for name in display.capabilities
        if isinstance(name, CapabilityName)
    }


def api_features(api: RecipeApi) -> dict[str, str]:
    """Each effort kind a recipe's API facts use, with the version that introduced it."""

    used: dict[str, str] = {}
    for block in [api.api, *(choice.api for choice in api.option_choices)]:
        if not isinstance(block, ChatApi) or isinstance(
            block.reasoning, ApiReasoningNone
        ):
            continue
        effort = block.reasoning.effort
        if effort is not None:
            used[f"reasoning effort {effort.kind}"] = EFFORT_KIND_SINCE[effort.kind]
    return used


def features_used(
    recipe: RecipeDefinition, models: Sequence[ModelDisplay], api: RecipeApi | None
) -> dict[str, str]:
    """Each gated value the documents use, with the version that introduced it."""

    used = {
        f"serving kind {check.kind}": SERVING_KIND_SINCE[check.kind]
        for check in recipe.validation.serving.checks
    }
    for display in models:
        used |= capability_features(display)
    if api is not None:
        used |= api_features(api)
    return used


def unreadable_capabilities(display: ModelDisplay, platform_version: str) -> list[str]:
    """The capabilities of one Model display a platform cannot read (sorted)."""

    have = parse_version(platform_version)
    return sorted(
        f"{feature} needs contracts {since}"
        for feature, since in capability_features(display).items()
        if parse_version(since) > have
    )


def unreadable_features(
    recipe: RecipeDefinition,
    models: Sequence[ModelDisplay],
    api: RecipeApi | None,
    platform_version: str,
) -> list[str]:
    """The values a platform on ``platform_version`` cannot read (sorted)."""

    have = parse_version(platform_version)
    return sorted(
        f"{feature} needs contracts {since}"
        for feature, since in features_used(recipe, models, api).items()
        if parse_version(since) > have
    )
