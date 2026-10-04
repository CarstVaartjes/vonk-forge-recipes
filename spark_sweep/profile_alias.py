"""A recipe's name in the sweep profile: a valid, unique profile alias.

The Controller's profile contract takes a lowercase identifier for an
assignment name (the client-facing model name). A reviewed service alias is the
upstream's own spelling (``Qwen3-Coder-Next-FP8``, ``nvidia/...``) and is not
one, so the sweep derives the profile alias the way the Controller derives its
default (lowercase, ``/`` and other separators to ``-``) instead of sending it
as written. The contract's own pattern lives in ``vonk_forge_contracts``; the
tests check every alias produced here against it.
"""

from __future__ import annotations

import hashlib
import re

_MAX_LENGTH = 63
_SUFFIX_LENGTH = 5
_FALLBACK = "assignment"


def profile_alias(name: str) -> str:
    """The valid profile alias for a served or reviewed model name."""
    value = re.sub(r"[^a-z0-9._-]+", "-", name.lower())
    value = value.strip("-_.") or _FALLBACK
    return value[:_MAX_LENGTH].rstrip("-_.") or _FALLBACK


def unique_profile_alias(alias: str, recipe_key: str) -> str:
    """The same alias made distinct for one recipe, stable across restarts.

    Two variants of one model share a service alias; a profile cannot hold two
    running assignments with the same name.
    """
    suffix = hashlib.sha1(recipe_key.encode()).hexdigest()[:_SUFFIX_LENGTH]
    head = alias[: _MAX_LENGTH - _SUFFIX_LENGTH - 1].rstrip("-_.") or _FALLBACK
    return f"{head}-{suffix}"
