"""Require a licence entry for every upstream an adapter vendors files from.

``upstream-licences.json`` at the repository root records, per upstream repository,
its SPDX licence and the path of its licence text, or ``unlicensed-upstream`` with
the issue that asks the creator for a licence. It lives outside every build context,
so recording or changing a licence never changes a recipe's identity or image key.

Which upstream an adapter vendors from is ``check-vendored-upstream``'s own
classification (``vendored_upstream.upstream_files``). A missing entry fails only for
an adapter directory this change adds; for an existing adapter (a refresh that moves
its upstream, say) it is a note, so a refresh is never blocked. Licences are
informative: the entry records the status, it never gates a recipe.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, StrictStr, field_validator

from recipe_tooling.adapter_sources import materialize_or_exit
from recipe_tooling.paths import REPO_ROOT
from recipe_tooling.vendored_upstream import adapter_dirs, upstream_files

LICENCES = "upstream-licences.json"
UNLICENSED = "unlicensed-upstream"


class _Entry(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)


class LicensedUpstream(_Entry):
    license: StrictStr = Field(pattern=r"^\S+$")
    texts: list[StrictStr] = Field(min_length=1)

    @field_validator("license")
    @classmethod
    def _not_unlicensed(cls, value: str) -> str:
        if value == UNLICENSED:
            raise ValueError(f"'{UNLICENSED}' takes a request issue, not texts")
        return value


class UnlicensedUpstream(_Entry):
    license: Literal["unlicensed-upstream"]
    request: StrictStr = Field(
        pattern=r"^https://github\.com/[^/\s]+/[^/\s]+/issues/\d+$"
    )


class UpstreamLicences(_Entry):
    upstreams: dict[StrictStr, UnlicensedUpstream | LicensedUpstream]


def vendoring_upstreams(root: Path) -> dict[str, list[str]]:
    """Upstream repository -> the adapters that vendor files from it."""
    found: dict[str, list[str]] = {}
    for adapter in adapter_dirs(root):
        name = adapter.relative_to(root / "adapters").as_posix()
        for repo in upstream_files(adapter):
            found.setdefault(repo, []).append(name)
    return found


def check(root: Path, added: set[str]) -> tuple[list[str], list[str]]:
    """Problems and notes; ``added``: adapter directories (under adapters/) this change adds."""
    licences = UpstreamLicences.model_validate_json(
        (root / LICENCES).read_text(encoding="utf-8")
    ).upstreams
    problems: list[str] = []
    notes: list[str] = []
    for repo, adapters in sorted(vendoring_upstreams(root).items()):
        entry = licences.get(repo)
        if entry is None:
            missing = (
                f"{repo}: vendored by {', '.join(adapters)} but has no entry in {LICENCES}; "
                f"add its SPDX licence and licence text (licenses/{repo}/), or "
                f"'{UNLICENSED}' with the licence-request issue"
            )
            (problems if added.intersection(adapters) else notes).append(missing)
        elif isinstance(entry, LicensedUpstream):
            problems.extend(
                f"{repo}: licence text {text} is missing"
                for text in entry.texts
                if not (root / text).is_file()
            )
    return problems, notes


def added_adapters(root: Path, base: str) -> set[str]:
    """Adapter directories whose Dockerfile this change adds (``base...HEAD``)."""
    completed = subprocess.run(
        [
            "git",
            "-C",
            str(root),
            "diff",
            "--name-only",
            "--diff-filter=A",
            f"{base}...HEAD",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    return {
        name.removeprefix("adapters/").removesuffix("/Dockerfile")
        for name in completed.stdout.split()
        if name.startswith("adapters/") and name.endswith("/Dockerfile")
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default="origin/main")
    arguments = parser.parse_args(argv)
    materialize_or_exit(REPO_ROOT)
    problems, notes = check(REPO_ROOT, added_adapters(REPO_ROOT, arguments.base))
    for note in notes:
        print(f"note: {note}")
    for problem in problems:
        print(problem, file=sys.stderr)
    return 1 if problems else 0
