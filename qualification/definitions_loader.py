"""Assemble the qualification definitions from their source files.

``qualification/definitions.json`` holds the shared parts (schema version,
fixtures, special fixtures, service case templates). Each recipe owns one file,
``qualification/recipes/<slug>.json``, where ``<slug>`` is the recipe id without
its publisher prefix::

    {"id": "<publisher>/<slug>", "recipes": {...}, "service_recipes": {...}}

Either section may be omitted. ``load_definitions`` is the only reader; it
returns the same document the single file used to hold, with both maps sorted
by recipe id. Entries still in the shared file are read too. It fails loudly
on a recipe id defined twice (in two files or in a file and the shared file), on
a duplicate JSON key, or on a file name that does not match its id.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

QUALIFICATION_ROOT = Path(__file__).resolve().parent
PER_RECIPE_SECTIONS = ("recipes", "service_recipes")


def _read(path: Path) -> dict[str, Any]:
    def no_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        keys = [key for key, _ in pairs]
        duplicates = sorted({key for key in keys if keys.count(key) > 1})
        if duplicates:
            raise ValueError(f"{path}: duplicate keys {duplicates}")
        return dict(pairs)

    document = json.loads(
        path.read_text(encoding="utf-8"), object_pairs_hook=no_duplicate_keys
    )
    if not isinstance(document, dict):
        raise TypeError(f"{path}: expected a JSON object")
    return document


def recipe_file_name(recipe_id: str) -> str:
    """The file name that owns ``recipe_id`` (``publisher/slug`` -> ``slug.json``)."""
    publisher, separator, slug = recipe_id.partition("/")
    if not separator or not publisher or not slug or "/" in slug:
        raise ValueError(f"recipe id must be <publisher>/<slug>: {recipe_id!r}")
    return f"{slug}.json"


def load_definitions(root: Path = QUALIFICATION_ROOT) -> dict[str, Any]:
    document = _read(root / "definitions.json")
    for section in PER_RECIPE_SECTIONS:
        # Entries not yet moved out of the shared file are still honoured.
        document.setdefault(section, {})
    for path in sorted((root / "recipes").glob("*.json")):
        entry = _read(path)
        recipe_id = entry.get("id")
        if not isinstance(recipe_id, str) or recipe_file_name(recipe_id) != path.name:
            raise ValueError(f"{path}: 'id' must be <publisher>/{path.stem}")
        unknown = set(entry) - {"id", *PER_RECIPE_SECTIONS}
        if unknown:
            raise ValueError(f"{path}: unknown keys {sorted(unknown)}")
        for section in PER_RECIPE_SECTIONS:
            if section in entry:
                if recipe_id in document[section]:
                    raise ValueError(f"{recipe_id} is defined twice in {section}")
                document[section][recipe_id] = entry[section]
    for section in PER_RECIPE_SECTIONS:
        document[section] = dict(sorted(document[section].items()))
    return document
