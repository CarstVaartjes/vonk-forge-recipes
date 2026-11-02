"""Workflow and action YAML never writes git refs with GITHUB_TOKEN.

Scope: every ``*.yml``/``*.yaml`` under ``.github`` (workflows and composite
actions), whose shell runs with GITHUB_TOKEN. GitHub refuses a ref that
GITHUB_TOKEN points at a commit whose workflow files differ from main's, so a
publish that lags a newer workflow change would fail. Releases are addressed
by tag name and carry their commit in the signed bundle; undrafting a release
creates a new contract version's tag. ``tools/refresh-upstream`` pushes its PR
branches with the refresh GitHub App's token, a different authority, and is
outside this guard.
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REF_WRITE = re.compile(r"\bgit\s+(?:push|tag)\b|\bgit/refs\b|\bupdateRef\b")


def ref_write_problems(sources: list[tuple[str, str]]) -> list[str]:
    return sorted(name for name, text in sources if REF_WRITE.search(text))


def test_ci_writes_no_git_refs() -> None:
    sources = [
        (str(path.relative_to(ROOT)), path.read_text())
        for path in sorted((ROOT / ".github").rglob("*.y*ml"))
    ]
    assert sources
    assert ref_write_problems(sources) == []


def test_ref_write_detection() -> None:
    assert ref_write_problems(
        [
            ("force-tag.yml", 'git tag --force --annotate "$TAG" "$SHA"'),
            ("push.yml", 'git push --force origin "refs/tags/$TAG"'),
            ("api.yml", "gh api -X PATCH repos/o/r/git/refs/tags/v1"),
            ("release.yml", 'gh release create "$TAG" --target "$SHA"'),
        ]
    ) == ["api.yml", "force-tag.yml", "push.yml"]
