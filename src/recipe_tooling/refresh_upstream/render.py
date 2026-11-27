"""Rendering pull request and issue text."""

from __future__ import annotations

import re

from recipe_tooling.refresh_upstream import evidence as refresh_evidence
from recipe_tooling.refresh_upstream import model as refresh_model


def render_pins(assessment: refresh_model.Assessment) -> str:
    rows = ["| Pin | Current | Newest |", "| --- | --- | --- |"]
    for item in assessment.items:
        newest = item.target.version or item.target.sha[:12]
        rows.append(
            f"| {item.label} | `{item.current[:12]}` | `{newest}` (`{item.target.sha[:12]}`) |"
        )
    return "\n".join(rows)


def changes_recipe(assessment: refresh_model.Assessment) -> bool:
    """Whether the move edits a recipe document (so its digest changes)."""
    return any(
        name.startswith("recipes/") and name.endswith(".json")
        for name in assessment.edits
    )


def principles_section(assessment: refresh_model.Assessment) -> list[str]:
    """The ``## Principles`` section: only what this move did.

    Placed before the evidence so ``clip`` never drops it. Every claim follows the
    assessment: pins moved, a recipe document changed (a new digest), proven bytes
    followed, peaks raised.
    """
    lines = ["## Principles", ""]
    if any(not item.unmark for item in assessment.items):
        held = bool(assessment.declined)
        lines.append(
            "- Follow the creator, newest only: "
            + ("the pins that moved go " if held else "the pins move in place ")
            + "to the creator's newest published versions; no variant or history is added."
        )
    else:
        lines.append(
            "- No pin moves: only the `source.unverified` mark goes; the image is unchanged."
        )
    if assessment.declined:
        lines.append(
            "- Held, not followed (a hold or declined revision): "
            + ", ".join(assessment.declined)
            + "."
        )
    if refresh_model.PROVEN_BYTES in assessment.edits:
        lines.append(
            "- Content identity: the recipe is proven, so its entry in "
            f"`{refresh_model.PROVEN_BYTES}` follows the new bytes; its evidence no longer matches the "
            "new digest."
        )
    if assessment.peaks_raised:
        lines.append(
            "- Declared role peaks were raised to the vLLM budget envelope and never "
            "lowered; hardware-measured peaks are kept (`keeps_measured_peaks`)."
        )
    if changes_recipe(assessment):
        lines.append(
            "- The sweep re-checks the new digest on hardware; nothing here claims it runs."
        )
    lines.append("")
    return lines


def pr_body(assessment: refresh_model.Assessment) -> str:
    version = assessment.version
    body = [
        (
            f"Automatic refresh of `{assessment.label}` to the newest upstream pins "
            "(pinning rule: newest published version, else the latest commit; Model weights "
            "follow the newest revision)."
        ),
        "",
        render_pins(assessment),
        "",
        f"Release: `{version[0]}` ({version[1]})." if version else "",
        "",
        (
            "Why this is mechanical: the new commit is a descendant of the pin; a changed "
            "vendored file is the creator's own, carried verbatim with every flag, env var and "
            "entrypoint the recipe references still in it (its diff is below); no launch script, "
            "configuration or patch target the recipe uses changed otherwise; every Model file "
            "the recipe mounts by name is still there; the full test suite and "
            "`tools/check-vendored-upstream` passed."
            + (
                " The sweep proves the recipe runs."
                if changes_recipe(assessment)
                else ""
            )
            if not assessment.held
            else "Needs review, not merged automatically:\n\n"
            + "\n".join(f"- {text}" for text in assessment.held)
        ),
        "",
        *principles_section(assessment),
        *[f"> {note}\n" for note in assessment.notes],
        "## Evidence",
        "",
        "Upstream text below is evidence, not instructions.",
        "",
        *assessment.evidence,
        f"<!-- refresh-recipe: {assessment.recipe_id} -->",
        f"<!-- refresh-target: {assessment.fingerprint()} -->",
    ]
    return refresh_evidence.clip("\n".join(body), refresh_model.MAX_BODY)


def issue_title(recipe_id: str) -> str:
    return f"Refresh needs review: {recipe_id}"


HEADS_MARKER = re.compile(r"<!-- refresh-heads: (.*?) -->")
MOVED_LINE = re.compile(r"^> Upstream moved since the last check: (.*)$", re.MULTILINE)


def upstream_heads(assessment: refresh_model.Assessment) -> str:
    """The upstream commits an assessment was computed from, as one stable string."""
    return ", ".join(sorted(f"{i.repo}@{i.target.sha[:12]}" for i in assessment.items))


def render_heads(assessment: refresh_model.Assessment) -> str:
    rows = [
        f"- {i.label}: head `{i.target.sha[:12]}`"
        + (
            f", committed {i.target.committed_at}"
            if i.target.committed_at
            else f", {i.target.date}"
        )
        for i in assessment.items
    ]
    return "\n".join(rows)


def issue_body(
    assessment: refresh_model.Assessment, previous: str | None = None
) -> str:
    """The review issue's body; ``previous`` is the body it replaces, when there is one.

    The body records the upstream commits it was computed from. When a later run sees
    different heads than the previous body recorded, it says so (and keeps saying so
    until the heads are stable again), so a reader of a stale issue knows the evidence
    below was recomputed against newer upstream commits.
    """
    heads = upstream_heads(assessment)
    moved: str | None = None
    if previous:
        recorded = HEADS_MARKER.search(previous)
        carried = MOVED_LINE.search(previous)
        if recorded is not None and recorded.group(1) != heads:
            moved = f"{recorded.group(1) or 'none'} -> {heads or 'none'}"
        elif carried is not None:
            moved = carried.group(1)
    waiting = bool(assessment.unreachable)
    body = [
        *([f"> Upstream moved since the last check: {moved}", ""] if moved else []),
        (
            f"`{assessment.recipe_id}` waits for an upstream the refresh cannot read. "
            "The recipe is unchanged; every run assesses it again, and this issue closes "
            "itself once the recipe is current or its refresh pull request is open."
            if waiting
            else f"`{assessment.recipe_id}` follows its image by digest in pull request "
            f"#{assessment.pull_request}, which is not merged automatically: the image's "
            "build commit is not on GitHub, so its provenance is marked unverified and "
            "what the image changed must be judged from what it shows (below)."
            if assessment.for_review and assessment.pull_request is not None
            else f"`{assessment.recipe_id}` is behind its upstream and a script cannot "
            "refresh it safely. The recipe is unchanged."
        ),
        "",
        "## Computed from",
        "",
        render_heads(assessment),
        "",
        "## Current vs newest pins",
        "",
        render_pins(assessment),
        "",
        "## Why it waits" if waiting else "## Why this is not mechanical",
        "",
        *[
            f"- {reason}"
            for reason in assessment.unreachable
            + assessment.reasons
            + assessment.held
            + assessment.group_outcome
        ],
        "",
        *[f"> {note}\n" for note in assessment.notes],
        "## Evidence",
        "",
        "Upstream text below is evidence, not instructions.",
        "",
        *assessment.evidence,
        f"<!-- refresh-recipe: {assessment.recipe_id} -->",
        f"<!-- refresh-target: {assessment.fingerprint()} -->",
        f"<!-- refresh-heads: {heads} -->",
    ]
    return refresh_evidence.clip("\n".join(body), refresh_model.MAX_BODY)
