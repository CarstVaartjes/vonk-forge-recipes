"""The repository evidence files: path, read, merge, atomic write and the authored catalog."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

from vonk_forge_contracts import (
    QualificationMeasuredRoleMemory,
    RecipeDefinition,
    RecipeQualificationEvidence,
    document_sha256,
    read_recipe,
)


def write_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        Path(temporary).replace(path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def evidence_path(root: Path, recipe_key: str) -> Path:
    # Validate before constructing a path; authored keys cannot traverse directories.
    key = RecipeQualificationEvidence(recipe_key=recipe_key).recipe_key
    return root / "qualification" / "evidence" / f"{key}.json"


def read_evidence(root: Path, recipe_key: str) -> RecipeQualificationEvidence:
    path = evidence_path(root, recipe_key)
    if not path.exists():
        return RecipeQualificationEvidence(recipe_key=recipe_key)
    evidence = RecipeQualificationEvidence.model_validate_json(path.read_text())
    if evidence.recipe_key != recipe_key:
        raise ValueError(f"evidence key does not match {path.name}")
    return evidence


def write_evidence(root: Path, evidence: RecipeQualificationEvidence) -> bool:
    path = evidence_path(root, evidence.recipe_key)
    payload = evidence.model_dump_json(indent=2, exclude_none=True) + "\n"
    if path.exists() and path.read_text() == payload:
        return False
    write_atomic(path, payload)
    return True


def merge_evidence(
    left: RecipeQualificationEvidence, right: RecipeQualificationEvidence
) -> RecipeQualificationEvidence:
    if left.recipe_key != right.recipe_key:
        raise ValueError("cannot merge different recipes")
    unique = {r.model_dump_json(): r for r in (*left.records, *right.records)}
    ordered = sorted(unique.values(), key=lambda r: (r.tested_at, r.model_dump_json()))
    rows: dict[tuple[str, str], QualificationMeasuredRoleMemory] = {}
    for row in (*left.measured_memory, *right.measured_memory):
        rows[row.key] = (
            merge_measurement(rows[row.key], row) if row.key in rows else row
        )
    return left.model_copy(
        update={
            "records": tuple(ordered),
            "measured_memory": tuple(rows[key] for key in sorted(rows)),
        }
    )


def merge_measurement(
    left: QualificationMeasuredRoleMemory, right: QualificationMeasuredRoleMemory
) -> QualificationMeasuredRoleMemory:
    """One (content, role) measured twice: the larger use, the smaller budget.

    Both are the conservative side for a reader asking whether a declaration
    covers what hardware used under no smaller budget; no budget is smallest.
    """
    a, b = left.budget_bytes, right.budget_bytes
    return left.model_copy(
        update={
            "steady_bytes": max(left.steady_bytes, right.steady_bytes),
            "budget_bytes": None if a is None or b is None else min(a, b),
        }
    )


def catalog_recipes(root: Path) -> dict[str, tuple[RecipeDefinition, str]]:
    """Each authored recipe by key, with its content digest (the catalog's `content_sha256`)."""
    directory = root / "recipes"
    if not directory.is_dir():
        raise ValueError("evidence retention requires the authored recipe catalog")
    recipes = {}
    for path in directory.glob("*.json"):
        document = json.loads(path.read_text())
        recipe = read_recipe(document)
        key = f"{recipe.identity.publisher}/{recipe.identity.slug}"
        recipes[key] = (recipe, document_sha256(document))
    return recipes


def catalog_recipe_keys(root: Path) -> set[str]:
    return set(catalog_recipes(root))
