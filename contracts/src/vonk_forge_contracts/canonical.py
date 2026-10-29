"""Content identity and tolerant reading of published contract documents."""

from __future__ import annotations

import copy
import hashlib
import json
from collections.abc import Mapping, MutableMapping
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


# The one list of authored display fields the qualification fingerprint ignores,
# as leaf paths. Everything else, including fields added later, stays covered.
RECIPE_DISPLAY_FIELDS: tuple[tuple[str, ...], ...] = (
    ("metadata", "title"),
    ("metadata", "description"),
    ("metadata", "tags"),
    ("release", "version"),
    ("release", "released_at"),
    ("provenance", "attribution"),
)
MODEL_DISPLAY_FIELDS: tuple[tuple[str, ...], ...] = (
    ("metadata", "description"),
    ("metadata", "tags"),
    ("identity", "family", "title"),
    ("identity", "model", "title"),
)


def _drop(value: dict[str, Any], path: tuple[str, ...]) -> None:
    for step in path[:-1]:
        value = value[step]
    value.pop(path[-1], None)


def qualification_fingerprint_sha256(
    recipe: Mapping[str, object],
    models: Mapping[str, Mapping[str, object]],
    *,
    resolved: MutableMapping[str, str] | None = None,
    authenticated: bool = False,
) -> str:
    """Hash what a Recipe executes, ignoring only the display fields above.

    Each exact Model reference is resolved and authenticated by its document
    digest before the Model's display fields are removed, so a label edit keeps
    the fingerprint while a changed file, source, format or variant does not.
    This is a qualification-equivalence key, never an asset verification digest.
    Raises ValueError when a reference is missing, mismatched or cyclic.

    A caller that holds many recipes passes ``resolved`` (Model digest to its
    stripped digest) so each Model is stripped and hashed once, and
    ``authenticated=True`` when ``models`` is keyed by the digest it computed
    itself, which makes re-checking each key redundant.
    """

    read_recipe(recipe)
    visiting: set[str] = set()
    if resolved is None:
        resolved = {}

    def resolve(digest: str) -> str:
        if digest in resolved:
            return resolved[digest]
        if digest in visiting:
            raise ValueError("cyclic Model dependencies")
        model = models.get(digest)
        if model is None or (not authenticated and document_sha256(model) != digest):
            raise ValueError("Model reference is missing or has a different digest")
        if not authenticated:
            read_model(model)
        visiting.add(digest)
        value: dict[str, Any] = copy.deepcopy(dict(model))
        for path in MODEL_DISPLAY_FIELDS:
            _drop(value, path)
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
    for path in RECIPE_DISPLAY_FIELDS:
        _drop(value, path)
    rewrite(value)
    return document_sha256(value)
