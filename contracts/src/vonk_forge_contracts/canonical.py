"""Content identity and tolerant reading of published contract documents."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping

from .model import ModelDefinition
from .recipe import RecipeDefinition


def canonical_json(document: Mapping[str, object]) -> bytes:
    """Return the canonical bytes of one published JSON document.

    Object keys are sorted, list order is kept, and the JSON is compact UTF-8
    with non-ASCII characters preserved.
    """

    return json.dumps(
        document,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def document_sha256(document: Mapping[str, object]) -> str:
    """Return the content identity of one published Model or Recipe document.

    The digest covers the document exactly as published, not a re-serialized
    copy, so it never depends on which contract release a reader has: a field
    a newer minor release adds changes the digest only because the document
    changed. Hash a document once when it is published or imported and keep
    that digest with it; never recompute it from a parsed model.
    """

    if not isinstance(document, Mapping):
        raise TypeError("document_sha256 requires the published JSON object")
    return hashlib.sha256(canonical_json(document)).hexdigest()


def read_model(document: Mapping[str, object]) -> ModelDefinition:
    """Parse a published Model, ignoring fields a newer minor contract added."""

    return ModelDefinition.model_validate(document, extra="ignore")


def read_recipe(document: Mapping[str, object]) -> RecipeDefinition:
    """Parse a published Recipe, ignoring fields a newer minor contract added."""

    return RecipeDefinition.model_validate(document, extra="ignore")
