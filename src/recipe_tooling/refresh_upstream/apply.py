"""Applying edits and verifying the result."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Any

APP_AUTH: Any = None


# ---------------------------------------------------------------------------
# Applying edits and verifying the result
# ---------------------------------------------------------------------------


def apply_edits(root: Path, edits: dict[str, bytes | None]) -> dict[str, bytes | None]:
    """Write edits; returns what is needed to undo them."""
    undo: dict[str, bytes | None] = {}
    for name, data in edits.items():
        path = root / name
        undo[name] = path.read_bytes() if path.is_file() else None
        if data is None:
            path.unlink(missing_ok=True)
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
    return undo


def run(
    command: list[str],
    *,
    cwd: Path,
    check: bool = True,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    app_env = (
        APP_AUTH.environment()
        if APP_AUTH is not None and command[0] in {"gh", "git"}
        else {}
    )
    result = subprocess.run(
        command,
        cwd=cwd,
        text=True,
        capture_output=True,
        check=False,
        env={**os.environ, **(env or {}), **app_env},
    )
    if check and result.returncode != 0:
        raise RuntimeError(
            f"{' '.join(command[:3])} failed ({result.returncode}): "
            f"{(result.stderr or result.stdout)[-1500:]}"
        )
    return result


def verify_steps() -> list[tuple[str, list[str]]]:
    """The data checks of ``validate.yml`` (tests assert none is missing here)."""
    return [
        ("model references", [sys.executable, "tools/check-recipe-model-references"]),
        ("model families", [sys.executable, "tools/check-model-families"]),
        (
            "build context sources",
            [sys.executable, "tools/check-build-context-sources"],
        ),
        ("vendored upstream files", [sys.executable, "tools/check-vendored-upstream"]),
        ("kit-declared pins", [sys.executable, "tools/kit-pins", "check"]),
        (
            "model construction",
            [
                sys.executable,
                "tools/check-model-construction-device",
                "--base",
                "origin/main",
            ],
        ),
        (
            "writable state paths",
            [
                sys.executable,
                "tools/check-writable-state-env",
                "--base",
                "origin/main",
            ],
        ),
        (
            "document reads",
            [
                sys.executable,
                "tools/check-document-reads",
                "--base",
                "origin/main",
            ],
        ),
        (
            "estimate gates",
            [
                sys.executable,
                "tools/check-estimate-gates",
                "--base",
                "origin/main",
            ],
        ),
        (
            "full test suite",
            [
                sys.executable,
                "-m",
                "pytest",
                "-q",
                "-x",
                "-m",
                "not lane",
                "-p",
                "no:cacheprovider",
            ],
        ),
    ]


def verify(root: Path, recipe_id: str) -> str | None:
    """Run the producer checks; returns a failure description."""
    steps = verify_steps()
    for name, command in steps:
        result = run(command, cwd=root, check=False)
        if result.returncode != 0:
            return f"{name} failed:\n{(result.stdout + result.stderr)[-2500:]}"
    return None
