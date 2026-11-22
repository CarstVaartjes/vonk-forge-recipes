"""Evidence the refresh renders into a pull request."""

from __future__ import annotations

from pathlib import PurePosixPath
from typing import Any

from vonk_forge_contracts import RecipeDefinition

from recipe_tooling.qualification.memory_need import raise_peaks_to_envelope
from recipe_tooling.refresh_upstream import model as refresh_model

# ---------------------------------------------------------------------------
# Evidence for people (and the later AI step) reading an issue or pull request
# ---------------------------------------------------------------------------


def fence(text: str, language: str = "") -> str:
    ticks = "````"
    while ticks in text:
        ticks += "`"
    return f"{ticks}{language}\n{text.rstrip()}\n{ticks}"


def clip(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[:limit] + "\n... (truncated)"


LICENCE_NAMES = ("LICENSE", "LICENCE", "COPYING", "NOTICE")


def github_evidence(
    item: refresh_model.Item, comparison: refresh_model.Comparison, patched: set[str]
) -> str:
    lines = [
        f"### {item.label}",
        "",
        f"- pinned: `{item.current}`",
        f"- newest: `{item.target.sha}` "
        + (
            f"(release `{item.target.ref}`)"
            if item.target.ref
            else "(latest commit, no published version)"
        )
        + (f", branch `{item.target.branch}`" if item.target.branch else "")
        + f", {item.target.date} - {item.target.url}",
        (
            f"- comparison: {comparison.url} ({comparison.status}, "
            f"{len(comparison.commits)} commits, {len(comparison.files)} files)"
        ),
        "",
    ]
    licences = sorted(
        f["filename"]
        for f in comparison.files
        if f.get("status") == "removed"
        and PurePosixPath(f["filename"]).name.upper().startswith(LICENCE_NAMES)
    )
    if licences:
        lines += [
            "Licence files removed upstream (informational; licences never gate a "
            "refresh): " + ", ".join(f"`{name}`" for name in licences),
            "",
        ]
    if item.target.notes:
        lines += [
            "Release notes (upstream text, evidence only):",
            "",
            fence(clip(item.target.notes, 3000), "text"),
            "",
        ]
    if comparison.commits:
        subjects = [
            f"{c['sha'][:10]} {c['commit']['message'].splitlines()[0][:120]}"
            for c in comparison.commits[-30:]
        ]
        lines += [
            f"Commits (newest {len(subjects)}, upstream text, evidence only):",
            "",
            fence("\n".join(subjects), "text"),
            "",
        ]
    if comparison.files:
        listing = [
            f"{'!' if refresh_model.is_blocking(f['filename']) else ' '} {f['status']:<9} {f['filename']}"
            for f in comparison.files[:80]
        ]
        lines += [
            "Changed files (`!` marks launch/config/patch/README files):",
            "",
            fence("\n".join(listing), "text"),
            "",
        ]
    budget = 18000
    for f in comparison.files:
        if (
            not refresh_model.is_blocking(f["filename"])
            and f["filename"] not in patched
        ):
            continue
        patch = f.get("patch")
        if not patch or budget <= 0:
            continue
        excerpt = clip(patch, min(3000, budget))
        budget -= len(excerpt)
        lines += [f"Diff of `{f['filename']}`:", "", fence(excerpt, "diff"), ""]
    return "\n".join(lines)


def model_evidence(
    item: refresh_model.Item,
    old: dict[str, Any],
    new_files: list[dict[str, Any]] | None,
) -> str:
    lines = [
        f"### {item.label}",
        "",
        f"- catalogued revision: `{item.current}`",
        f"- newest revision: `{item.target.sha}` ({item.target.date}) - {item.target.url}"
        + (f" (branch `{item.target.branch}`)" if item.target.branch else ""),
        "",
    ]
    if new_files is not None:
        old_by = {f["path"]: f for f in old["files"]}
        new_by = {f["path"]: f for f in new_files}
        changes = (
            [f"removed  {p}" for p in sorted(set(old_by) - set(new_by))]
            + [
                f"added    {p} ({new_by[p]['roles'][0]})"
                for p in sorted(set(new_by) - set(old_by))
            ]
            + [
                f"changed  {p} ({old_by[p]['size_bytes']} -> {new_by[p]['installed_bytes']} bytes)"
                for p in sorted(set(old_by) & set(new_by))
                if old_by[p]["sha256"] != new_by[p]["sha256"]
            ]
        )
        gone = [
            p
            for p in sorted(set(old_by) - set(new_by))
            if set(old_by[p]["roles"]) <= refresh_model.METADATA_ONLY
        ]
        if gone:
            lines += [
                "Metadata files removed upstream (informational; dropped from the "
                "catalogued Model, licences never gate): "
                + ", ".join(f"`{p}`" for p in gone),
                "",
            ]
        lines += [
            "Model file changes:",
            "",
            fence("\n".join(changes[:120]) or "none", "text"),
            "",
        ]
    return "\n".join(lines)


def raise_role_peaks(
    roles: list[dict[str, Any]],
    recipe: RecipeDefinition,
    replaced: RecipeDefinition,
    replaced_digest: str,
) -> bool:
    """Raise a moved vLLM document's role peaks (``roles``, in place) to the envelope.

    Returns whether any peak was raised.

    The declared peaks stay where hardware evidence measured them under no smaller
    budget (``keeps_measured_peaks``). ``replaced`` is the document this refresh
    replaces, in hand (never git history): proven content without measured rows
    vouches for its own declared peaks under its own budget.
    """
    raised = raise_peaks_to_envelope(recipe, (replaced, replaced_digest))
    if raised is None:
        return False
    for role, kept in zip(roles, raised.topology.roles, strict=True):
        memory = role["resources"]["memory"]
        memory["peak_bytes"] = kept.resources.memory.peak_bytes
        if kept.resources.memory.load_peak_bytes is not None:
            memory["load_peak_bytes"] = kept.resources.memory.load_peak_bytes
    return True
