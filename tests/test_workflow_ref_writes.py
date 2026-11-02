"""CI never writes git refs: GITHUB_TOKEN cannot move one across workflow changes."""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# A pushed or API-written ref is refused once the commits it spans change a
# workflow file, which blocks every later publish. Releases are addressed by
# tag name and carry their commit in the signed bundle; the Releases API
# creates a new contract version's tag.
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
