#!/usr/bin/env python3
"""Apply unified diffs to an installed package tree, failing closed.

Each hunk's exact old text (context plus removed lines) must be found in the
target file, at the hunk's line or, when the base moved, the nearest exact
match. A hunk that matches nowhere, or whose replacement is already present,
stops the build; nothing is skipped silently.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

HUNK = re.compile(r"^@@ -(\d+)(?:,\d+)? \+\d+(?:,\d+)? @@")


def parse(text: str) -> list[tuple[str, list[tuple[int, list[str]]]]]:
    files: list[tuple[str, list[tuple[int, list[str]]]]] = []
    lines = text.split("\n")
    i = 0
    while i < len(lines):
        line = lines[i]
        if (
            line.startswith("--- ")
            and i + 1 < len(lines)
            and lines[i + 1].startswith("+++ ")
        ):
            if lines[i].startswith("--- /dev/null"):
                raise SystemExit("patch creates a file; only edits are supported")
            target = lines[i + 1][4:].split("\t")[0]
            files.append((target.removeprefix("b/"), []))
            i += 2
            continue
        match = HUNK.match(line)
        if match and files:
            start = int(match.group(1))
            body: list[str] = []
            i += 1
            while i < len(lines) and (
                lines[i][:1] in {" ", "+", "-", "\\"} or lines[i] == ""
            ):
                if lines[i] == "" and (
                    i + 1 >= len(lines) or not lines[i + 1][:1] in {" ", "+", "-", "\\"}
                ):
                    break
                body.append(lines[i])
                i += 1
            files[-1][1].append((start, body))
            continue
        i += 1
    return files


def apply_hunk(source: list[str], start: int, body: list[str], label: str) -> list[str]:
    old = [b[1:] for b in body if b[:1] in {" ", "-"}]
    new = [b[1:] for b in body if b[:1] in {" ", "+"}]
    limit = len(source) - len(old) + 1
    hits = [n for n in range(max(limit, 0)) if source[n : n + len(old)] == old]
    if not hits:
        raise SystemExit(f"{label}: hunk at line {start} does not match")
    at = min(hits, key=lambda n: abs(n - (start - 1)))
    return source[:at] + new + source[at + len(old) :]


def main() -> None:
    root = Path(sys.argv[1])
    for patch in sys.argv[2:]:
        for name, hunks in parse(Path(patch).read_text()):
            path = root / name
            source = path.read_text().split("\n")
            for start, body in hunks:
                source = apply_hunk(source, start, body, f"{patch}:{name}")
            path.write_text("\n".join(source))
            print(f"patched {name} ({len(hunks)} hunks) from {Path(patch).name}")


if __name__ == "__main__":
    main()
