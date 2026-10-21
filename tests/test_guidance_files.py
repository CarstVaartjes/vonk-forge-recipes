"""Keep repository guidance in the two owner-selected entry points."""

from __future__ import annotations

import re
import subprocess
from collections.abc import Iterable
from pathlib import Path, PurePosixPath
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parents[1]
GUIDANCE_FILES = frozenset(
    {
        "AGENTS.md",
        "README.md",
        "docs/recipe-authoring.md",
        "contracts/README.md",
        "contracts/src/vonk_forge_contracts/README.md",
    }
)


def unexpected_guidance(paths: Iterable[str]) -> list[str]:
    return sorted(
        name
        for name in paths
        if PurePosixPath(name).suffix.lower() == ".md"
        and PurePosixPath(name).parts[0] != "adapters"
        and name not in GUIDANCE_FILES
    )


def test_repository_guidance_allowlist() -> None:
    paths = subprocess.run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.split("\0")
    present = [name for name in paths if name and (ROOT / name).exists()]
    assert not (unexpected := unexpected_guidance(present)), (
        "Move guidance into AGENTS.md or docs/recipe-authoring.md: "
        + ", ".join(unexpected)
    )


def test_guard_rejects_nested_dated_and_package_guidance() -> None:
    rejected = [
        "docs/extra.md",
        "docs/gb10-memory-headroom.md",
        "docs/audits/incident-2026-10-21.md",
        "tools/README.md",
        "contracts/new-guide.MD",
    ]
    assert unexpected_guidance(
        [*GUIDANCE_FILES, "adapters/a/README.md", *rejected]
    ) == sorted(rejected)


def test_guidance_relative_links_resolve() -> None:
    for name in sorted(GUIDANCE_FILES):
        source = ROOT / name
        for target in re.findall(r"\[[^\]]*\]\(([^)]+)\)", source.read_text()):
            url = urlsplit(target)
            if url.scheme or url.netloc or not url.path:
                continue
            assert (source.parent / unquote(url.path)).exists(), (
                f"{name}: broken link {target}"
            )
