"""The kit lock document."""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from recipe_tooling.kit_pins.model import GITHUB_REFERENCE, LOCK, MANIFEST, Pin

# ---------------------------------------------------------------------------
# The lock
# ---------------------------------------------------------------------------


def lock_document(
    kit_repo: str, kit_commit: str, pins: Iterable[Pin]
) -> dict[str, Any]:
    dependencies: dict[str, Any] = {}
    for pin in pins:
        entry: dict[str, Any] = {
            "datasource": pin.datasource,
            "repo": pin.repo,
            "ref": pin.ref,
            "kind": pin.kind,
            pin.key: pin.immutable,
        }
        if pin.overridden:
            entry["overridden"] = pin.overridden
        dependencies[pin.name] = entry
    return {
        "dependencies": dependencies,
        "kit": {"commit": kit_commit, "repo": kit_repo},
    }


def lock_text(document: dict[str, Any]) -> str:
    return json.dumps(document, indent=2, sort_keys=True, ensure_ascii=False) + "\n"


def read_lock(adapter: Path) -> dict[str, Any] | None:
    path = adapter / LOCK
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def locked_base(adapter: Path) -> tuple[str, str] | None:
    """``(repo, commit)`` of the lock's ``base`` dependency, for readers of the old ``source``."""
    lock = read_lock(adapter)
    entry = None if lock is None else lock.get("dependencies", {}).get("base")
    return None if entry is None else (entry["repo"], entry["commit"])


def kit_of(recipe: dict[str, Any]) -> tuple[str, str] | None:
    reference = recipe.get("provenance", {}).get("source_reference")
    match = GITHUB_REFERENCE.match(reference) if isinstance(reference, str) else None
    return None if match is None else (match.group(1), match.group(2))


def adapter_of(root: Path, recipe: dict[str, Any]) -> Path | None:
    path = recipe.get("execution", {}).get("build", {}).get("context", {}).get("path")
    return None if not path else root / path


def manifest_of(adapter: Path) -> dict[str, Any]:
    path = adapter / MANIFEST
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
