"""Canonical content identity for validated public contract documents."""
from __future__ import annotations

import hashlib
import json

from .model import ModelDefinition
from .recipe import RecipeDefinition


def content_sha256(document: ModelDefinition | RecipeDefinition) -> str:
    """Return the stable digest of one validated contract document.

    Callers must pass a validated ``ModelDefinition`` or ``RecipeDefinition``
    instance; raw dictionaries are rejected so parsing and semantic
    normalization cannot be bypassed. ``model_dump(mode="json")`` includes
    declared defaults and omits no fields. Object keys are sorted, while list
    order remains the document's semantic order. JSON is compact UTF-8 with
    non-ASCII characters preserved before SHA-256 hashing.
    """

    if not isinstance(document, (ModelDefinition, RecipeDefinition)):
        raise TypeError("content_sha256 requires a validated ModelDefinition or RecipeDefinition")
    normalized = document.model_dump(mode="json", exclude_unset=False, exclude_none=False)
    payload = json.dumps(normalized, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()
