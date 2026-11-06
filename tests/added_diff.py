"""Unified ``-U0`` diff text for tests of the tools that judge added lines.

The tools take a diff as text; a test builds it here and needs no repository.
"""

from __future__ import annotations


def added_diff(
    files: dict[str, str], *, old_lines: dict[str, int] | None = None
) -> str:
    """A diff that adds every line of each file after its first ``old_lines`` lines."""
    skipped = old_lines or {}
    parts: list[str] = []
    for name, text in files.items():
        total = len(text.splitlines())
        start = skipped.get(name, 0) + 1
        if start > total:
            continue
        count = total - start + 1
        parts.append(
            f"diff --git a/{name} b/{name}\n--- /dev/null\n+++ b/{name}\n"
            f"@@ -0,0 +{start},{count} @@\n"
        )
    return "".join(parts)
