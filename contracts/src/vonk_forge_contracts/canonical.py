"""Content identity and tolerant reading of published contract documents."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from typing import Any, cast

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


def execution_sha256(
    recipe: Mapping[str, object], models: Mapping[str, Mapping[str, object]]
) -> str:
    """Hash executable inputs, resolving exact Model references before display removal.

    Document hashes still authenticate every published byte. This independent
    fingerprint is only for qualification equivalence, never asset verification.
    Unknown/new fields remain covered by default.
    """
    import copy

    read_recipe(recipe)
    visiting: set[str] = set()
    resolved: dict[str, str] = {}

    def resolve(digest: str) -> str:
        if digest in resolved:
            return resolved[digest]
        if digest in visiting:
            raise ValueError("cyclic Model dependencies")
        model = models.get(digest)
        if model is None or document_sha256(model) != digest:
            raise ValueError("Model reference is missing or has a different digest")
        read_model(model)
        visiting.add(digest)
        value: dict[str, Any] = copy.deepcopy(dict(model))
        # The selected files, sources, formats and capabilities remain covered.
        value.pop("metadata", None)
        identity = value["identity"]
        for field in ("family", "model", "version", "variant"):
            identity.pop(field, None)
        rewrite(value)
        visiting.remove(digest)
        resolved[digest] = document_sha256(value)
        return resolved[digest]

    def rewrite(value: object) -> None:
        if isinstance(value, dict):
            if value.get("kind") == "model" and "content_sha256" in value:
                digest = str(value["content_sha256"])
                model = models.get(digest)
                if model is None:
                    raise ValueError("Model reference is missing")
                identity = cast(Mapping[str, object], model["identity"])
                if any(
                    value.get(key) != identity[key] for key in ("publisher", "slug")
                ):
                    raise ValueError("Model reference identity mismatch")
                value["content_sha256"] = resolve(digest)
            for item in value.values():
                rewrite(item)
        elif isinstance(value, list):
            for item in value:
                rewrite(item)

    value: dict[str, Any] = copy.deepcopy(dict(recipe))
    # Alignment is deliberately retained: it can change model behaviour.
    for field in ("title", "description", "tags"):
        value["metadata"].pop(field, None)
    value.pop("release", None)
    value["provenance"].pop("attribution", None)
    rewrite(value)
    return document_sha256(value)
