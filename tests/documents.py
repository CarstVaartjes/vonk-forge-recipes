"""Tests get documents, displays and API facts through the repository accessor."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from repo_root import ROOT

from recipe_tooling.qualification.documents import (
    has_recipe_api,
    model,
    model_display,
    recipe,
    recipe_api,
    recipe_display,
    recipe_document,
)

__all__ = [
    "has_recipe_api",
    "json_document",
    "model",
    "model_display",
    "model_document",
    "recipe",
    "recipe_api",
    "recipe_display",
    "recipe_document",
    "repository_bytes",
]


def json_document(raw: str | bytes) -> Any:
    """A document (or a JSON-valued argument) parsed from text a tool produced."""
    return json.loads(raw)


def model_document(slug: str, root: Path = ROOT) -> dict[str, Any]:
    """The authored Model document as written, like ``recipe_document``."""
    return json.loads((root / "models" / f"{slug}.json").read_text(encoding="utf-8"))


def repository_bytes(path: str | Path, root: Path = ROOT) -> bytes:
    """The bytes of one repository file, for comparing with what a package carries."""
    return (root / path).read_bytes()
