"""The gh and git edge."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from recipe_tooling.refresh_upstream import apply as refresh_apply
from recipe_tooling.refresh_upstream import model as refresh_model

# ---------------------------------------------------------------------------
# GitHub plumbing
# ---------------------------------------------------------------------------


def gh(*arguments: str, check: bool = True) -> str:
    result = refresh_apply.run(["gh", *arguments], cwd=refresh_model.ROOT, check=check)
    return result.stdout


def gh_json(*arguments: str) -> Any:
    output = gh(*arguments)
    return json.loads(output) if output.strip() else []


def ensure_labels() -> None:
    for name, color, description in (
        (
            refresh_model.LABEL_MECHANICAL,
            "0e8a16",
            "Automatic upstream refresh that needed no judgement",
        ),
        (
            refresh_model.LABEL_NEEDS_REVIEW,
            "d93f0b",
            "Upstream moved and a script could not vouch for it",
        ),
        (
            refresh_model.LABEL_MEMORY_ESTIMATE,
            "fbca04",
            "Declared memory disagrees with the estimate; the try decides",
        ),
    ):
        gh(
            "label",
            "create",
            name,
            "--color",
            color,
            "--description",
            description,
            "--force",
            check=False,
        )


def git(*arguments: str, cwd: Path, check: bool = True) -> str:
    return refresh_apply.run(["git", *arguments], cwd=cwd, check=check).stdout.strip()
