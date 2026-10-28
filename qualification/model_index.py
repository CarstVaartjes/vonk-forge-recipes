"""The one reader of the authored Model documents, and the execution fingerprint.

Every consumer (the sweep, the memory gate, the catalog build, the change gate)
asks this module for the Models of a catalog and for a Recipe's execution
fingerprint. Documents are keyed by the digest of their own bytes, so an index
that is out of date can only miss a Model, never return the wrong bytes. An
unreadable or invalid Model file is a miss, not an error.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from vonk_forge_contracts import ModelDefinition, document_sha256, execution_sha256

ROOT = Path(__file__).resolve().parents[1]


@dataclass
class ModelIndex:
    documents: dict[str, dict[str, Any]]  # document digest -> authored document
    by_selector: dict[str, dict[str, Any]]  # "publisher/slug" -> authored document
    unreadable: tuple[str, ...] = ()
    # Fingerprints by accepted recipe, for callers that cannot afford to reread it.
    accepted: dict[tuple[str, str, tuple[str, ...]], str] = field(
        default_factory=dict, repr=False
    )
    _fingerprints: dict[tuple[str, tuple[str, ...]], str | None] = field(
        default_factory=dict, repr=False
    )

    @classmethod
    def from_documents(
        cls, documents: Iterable[Mapping[str, Any]], unreadable: tuple[str, ...] = ()
    ) -> ModelIndex:
        by_digest: dict[str, dict[str, Any]] = {}
        by_selector: dict[str, dict[str, Any]] = {}
        for document in documents:
            by_digest[document_sha256(document)] = dict(document)
            identity = document["identity"]
            by_selector[f"{identity['publisher']}/{identity['slug']}"] = dict(document)
        return cls(by_digest, by_selector, unreadable)

    def fingerprint(
        self, recipe: Mapping[str, Any], content_sha256: str | None = None
    ) -> str | None:
        """The recipe's execution fingerprint, or None when a Model is unavailable.

        Memoized by (recipe digest, Model digests); the Recipe is hashed only
        when the caller does not already know its digest.
        """
        digest = content_sha256 or document_sha256(recipe)
        models = tuple(sorted({m["model"]["content_sha256"] for m in recipe["models"]}))
        key = (digest, models)
        if key not in self._fingerprints:
            try:
                self._fingerprints[key] = execution_sha256(recipe, self.documents)
            except (ValueError, KeyError, TypeError):
                self._fingerprints[key] = None
        return self._fingerprints[key]


def _read_models(directory: Path) -> ModelIndex:
    documents: list[dict[str, Any]] = []
    unreadable: list[str] = []
    for path in sorted(directory.glob("*.json")):
        try:
            document = json.loads(path.read_text(encoding="utf-8"))
            ModelDefinition.model_validate(document)
        except (OSError, ValueError):
            unreadable.append(path.name)
        else:
            documents.append(document)
    return ModelIndex.from_documents(documents, tuple(unreadable))


_cache: dict[Path, tuple[tuple[int, int], ModelIndex]] = {}


def model_index(root: Path = ROOT) -> ModelIndex:
    """The Models under ``root``, read once until the models directory changes."""
    directory = root / "models"
    try:
        stat = os.stat(directory)
    except OSError:
        return ModelIndex({}, {})
    signature = (stat.st_mtime_ns, stat.st_ino)
    cached = _cache.get(directory)
    if cached is None or cached[0] != signature:
        cached = (signature, _read_models(directory))
        _cache[directory] = cached
    return cached[1]
