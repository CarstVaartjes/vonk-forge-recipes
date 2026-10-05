"""File one GitHub issue per failed recipe, labelled ``hardware-test`` (opt-in)."""

from __future__ import annotations

import json
import subprocess
from collections.abc import Callable, Mapping, Sequence
from typing import Any

LABEL = "hardware-test"
Gh = Callable[[Sequence[str]], tuple[int, str]]


def gh_runner(argv: Sequence[str]) -> tuple[int, str]:
    done = subprocess.run(["gh", *argv], capture_output=True, text=True, check=False)
    return done.returncode, done.stdout.strip() or done.stderr.strip()


def title_for(key: str, entry: Mapping[str, Any]) -> str:
    return f"Hardware sweep: {key.partition('/')[2]} failed at {entry.get('phase')} ({entry.get('failure_class')})"


def body_for(key: str, entry: Mapping[str, Any]) -> str:
    timings = entry.get("timings") or {}
    lines = [
        f"Recipe `{key}` failed on the real Sparks.",
        "",
        f"- phase / class: `{entry.get('phase')}` / `{entry.get('failure_class')}`",
        f"- root-cause cluster: `{entry.get('cluster')}` (`{entry.get('signature')}`)",
        f"- recipe document digest: `{entry.get('content_sha256')}`",
        f"- attempts: {entry.get('attempts')}",
        f"- timings: {json.dumps(timings, sort_keys=True)}",
        "",
        "Reason:",
        "",
        "```",
        str(entry.get("error") or "")[:1500],
        "```",
    ]
    if entry.get("inherited_from"):
        lines += [
            "",
            f"Not attempted separately: it shares a model with `{entry['inherited_from']}`, whose download failed.",
        ]
    bundle = entry.get("evidence_bundle") or (
        entry["evidence"] if isinstance(entry.get("evidence"), str) else None
    )
    if bundle:
        lines += [
            "",
            f"Diagnostics bundle (local, from `vonkctl fleet evidence`): `{bundle}`",
        ]
    return "\n".join(lines) + "\n"


def file_issues(
    recipes: dict[str, dict[str, Any]],
    repo: str,
    gh: Gh = gh_runner,
    dry_run: bool = False,
) -> list[dict[str, str]]:
    """Create the missing issues; record each URL on its entry so a rerun never duplicates."""
    filed: list[dict[str, str]] = []
    if not dry_run:
        gh(
            [
                "label",
                "create",
                LABEL,
                "--repo",
                repo,
                "--description",
                "Failed on the real Sparks",
                "--force",
            ]
        )
    for key, entry in sorted(recipes.items()):
        if entry.get("status") != "failed" or entry.get("issue"):
            continue
        title = title_for(key, entry)
        if dry_run:
            filed.append({"recipe": key, "title": title, "url": "(dry run)"})
            continue
        code, out = gh(
            [
                "issue",
                "list",
                "--repo",
                repo,
                "--label",
                LABEL,
                "--state",
                "open",
                "--search",
                f'"{title}" in:title',
                "--json",
                "url,title",
            ]
        )
        existing = []
        if code == 0:
            try:
                existing = [i for i in json.loads(out) if i.get("title") == title]
            except ValueError:
                existing = []
        if existing:
            entry["issue"] = existing[0]["url"]
        else:
            code, out = gh(
                [
                    "issue",
                    "create",
                    "--repo",
                    repo,
                    "--title",
                    title,
                    "--body",
                    body_for(key, entry),
                    "--label",
                    LABEL,
                ]
            )
            if code != 0:
                filed.append(
                    {"recipe": key, "title": title, "url": f"failed: {out[:120]}"}
                )
                continue
            entry["issue"] = out.splitlines()[-1]
        filed.append({"recipe": key, "title": title, "url": entry["issue"]})
    return filed
