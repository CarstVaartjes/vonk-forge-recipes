"""Publishing refresh pull requests and issues."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from recipe_tooling.qualification.own_pulls import OWN_FIELDS, own_pulls
from recipe_tooling.refresh_upstream import apply as refresh_apply
from recipe_tooling.refresh_upstream import catalog_view as refresh_catalog_view
from recipe_tooling.refresh_upstream import evidence as refresh_evidence
from recipe_tooling.refresh_upstream import github as refresh_github
from recipe_tooling.refresh_upstream import http as refresh_http
from recipe_tooling.refresh_upstream import model as refresh_model
from recipe_tooling.refresh_upstream import render as refresh_render


@dataclass(slots=True)
class Summary:
    lines: list[str] = field(default_factory=list)
    mechanical: list[str] = field(default_factory=list)
    review: list[str] = field(default_factory=list)
    current: int = 0
    closed: list[str] = field(default_factory=list)
    unreachable: list[str] = field(default_factory=list)
    deferred: list[str] = field(default_factory=list)
    failures: list[str] = field(default_factory=list)
    newer_lines: list[str] = field(default_factory=list)
    armed: list[int] = field(default_factory=list)
    prepared: int = 0
    healthy: bool | None = None  # publication health, looked up once per run
    started: str | None = None  # the main commit this run assessed (and runs as)
    open_pulls: list[dict[str, Any]] | None = None  # listed once per run


def publication_healthy(runs: list[dict[str, Any]]) -> bool:
    """The newest finished publish.yml run on main succeeded (newest first).

    Cancelled runs are replaced pending runs, not failures. In-flight publications do
    not block arming: publish.yml queues them and builds from the newest main, so
    gating on an exact-head receipt would serialise merges (each merge moves main).
    """
    for run in runs:
        if run.get("status") == "completed" and run.get("conclusion") != "cancelled":
            return run.get("conclusion") == "success"
    return False


def main_publication_healthy() -> bool:
    runs = refresh_github.gh_json(
        "run",
        "list",
        "--workflow",
        "publish.yml",
        "--branch",
        "main",
        "--limit",
        "20",
        "--json",
        "status,conclusion",
    )
    return publication_healthy(runs)


def file_overlaps(
    number: int, pull_requests: list[dict[str, Any]], armed: Iterable[int] = ()
) -> list[int]:
    """Armed PRs (open with auto-merge, or armed earlier in this run) sharing a file with PR ``number``.

    An auto-merge PR whose checks failed or were cancelled holds no lane; if it merges
    anyway (a rerun, or a non-required check failed), a conflicting sibling is
    regenerated on main.

    Recipe files are per recipe, so refreshes only collide when two recipes share a
    Model file; the later PR then waits for the next run and is regenerated on main.
    """

    def paths(pr: dict[str, Any]) -> set[str]:
        return {f.get("path", "") for f in pr.get("files") or []}

    own = next((paths(pr) for pr in pull_requests if pr.get("number") == number), set())
    taken = set(armed)
    return sorted(
        pr["number"]
        for pr in pull_requests
        if pr.get("number") != number
        and (
            pr.get("number") in taken
            or (
                pr.get("autoMergeRequest")
                and check_state(pr.get("statusCheckRollup"))
                not in refresh_model.PARKED_CHECK_STATES
            )
        )
        and paths(pr) & own
    )


def arm_mechanical_pr(number: int, recipe_id: str, summary: Summary) -> bool:
    """Arm a prepared mechanical PR while publication is healthy.

    Every prepared PR is armed, not one per run: required checks gate each merge and
    publication coalesces back-to-back merges. Only a shared file with another armed PR
    (a real merge conflict) or an unpublished main defers it.
    """
    if number in summary.armed:
        return True
    if summary.healthy is None:
        summary.healthy = main_publication_healthy()
    if not summary.healthy:
        summary.deferred.append(
            f"{recipe_id}: prepared PR #{number}; the latest publish.yml run on main did not succeed"
        )
        return False

    # Listed once per run; a PR prepared after the listing is not in it, so list again.
    if summary.open_pulls is None or not any(
        pr.get("number") == number for pr in summary.open_pulls
    ):
        summary.open_pulls = with_all_files(
            refresh_github.gh_json(
                "pr",
                "list",
                "--state",
                "open",
                "--limit",
                "1000",
                "--json",
                f"number,autoMergeRequest,statusCheckRollup,files,changedFiles,{OWN_FIELDS}",
            )
        )
    pull_requests = summary.open_pulls
    if conflicting := file_overlaps(number, pull_requests, summary.armed):
        numbers = ", ".join(f"#{n}" for n in conflicting[:5])
        summary.deferred.append(
            f"{recipe_id}: prepared PR #{number}; shares files with armed PR {numbers}"
        )
        return False

    refresh_github.gh("pr", "merge", str(number), "--auto", "--squash")
    summary.armed.append(number)
    return True


def arm_open_mechanical_prs(summary: Summary) -> None:
    """Arm already-open refresh PRs first, before the slow assessment of every recipe.

    A full run takes tens of minutes; prepared PRs must not wait for it. Required checks
    still gate each merge, and a later push to the branch only restarts them.
    """
    pull_requests = own_pulls(
        refresh_github.gh_json(
            "pr",
            "list",
            "--state",
            "open",
            "--label",
            refresh_model.LABEL_MECHANICAL,
            "--limit",
            "200",
            "--json",
            f"number,headRefName,autoMergeRequest,{OWN_FIELDS}",
        )
    )
    for pr in pull_requests:
        branch = str(pr.get("headRefName", ""))
        if branch.startswith(refresh_model.BRANCH_PREFIX) and not pr.get(
            "autoMergeRequest"
        ):
            arm_mechanical_pr(
                pr["number"], branch.removeprefix(refresh_model.BRANCH_PREFIX), summary
            )


def has_publication_receipt(main_head: str, runs: list[dict[str, Any]]) -> bool:
    """A successful publish.yml run for this exact main SHA is the release receipt."""
    return any(
        run.get("headSha") == main_head
        and run.get("status") == "completed"
        and run.get("conclusion") == "success"
        for run in runs
    )


def check_state(rollup: list[dict[str, Any]] | None) -> str:
    """``failed`` | ``cancelled`` | ``pending`` | ``passed`` | ``none`` for a PR's checks."""
    states = [
        str(c.get("conclusion") or c.get("state") or c.get("status") or "").upper()
        for c in rollup or []
    ]
    if any(state in refresh_model.FAILED_CONCLUSIONS for state in states):
        return "failed"
    if any(state == "CANCELLED" for state in states):
        return "cancelled"
    if any(
        state in {"", "PENDING", "QUEUED", "IN_PROGRESS", "WAITING", "EXPECTED"}
        for state in states
    ):
        return "pending"
    return "passed" if states else "none"


def failed_check_evidence(rollup: list[dict[str, Any]] | None) -> str:
    """Names, links and the log tail of the failed checks (evidence, not instructions)."""
    lines: list[str] = []
    seen: set[str] = set()
    for check in rollup or []:
        if (
            str(check.get("conclusion") or "").upper()
            not in refresh_model.FAILED_CONCLUSIONS
        ):
            continue
        url = str(check.get("detailsUrl") or "")
        lines.append(f"- `{check.get('name')}` {check.get('conclusion')}: {url}")
        run_id = re.search(r"/runs/(\d+)", url)
        if run_id and run_id.group(1) not in seen:
            seen.add(run_id.group(1))
            log = refresh_github.gh(
                "run", "view", run_id.group(1), "--log-failed", check=False
            )
            lines += [
                "",
                refresh_evidence.fence(
                    refresh_evidence.clip(log.strip()[-2500:], 2500)
                ),
                "",
            ]
    return "\n".join(lines)


def carries(root: Path, head: str, tree: str) -> bool:
    """Whether branch ``head`` already holds exactly the refresh that ``tree`` is.

    Its changes since it last met main are the files the refresh changes, with the same
    bytes, and main has not touched those files since (so the squash merge applies).
    """
    changed = refresh_github.git(
        "diff", "--name-only", "--no-renames", "origin/main", tree, cwd=root
    ).split()
    base = refresh_github.git("merge-base", head, "origin/main", cwd=root)
    carried = refresh_github.git(
        "diff", "--name-only", "--no-renames", base, head, cwd=root
    ).split()
    if not changed or sorted(carried) != sorted(changed):
        return False
    if refresh_github.git(
        "diff", "--name-only", base, "origin/main", "--", *changed, cwd=root
    ):
        return False
    return refresh_github.git(
        "ls-tree", "-r", head, "--", *changed, cwd=root
    ) == refresh_github.git("ls-tree", "-r", tree, "--", *changed, cwd=root)


def checks_moved(root: Path, head: str) -> bool:
    """Whether main changed what judges a PR (src, tools, tests, workflows, API docs) since ``head``."""
    base = refresh_github.git("merge-base", head, "origin/main", cwd=root)
    return bool(
        refresh_github.git(
            "diff",
            "--name-only",
            base,
            "origin/main",
            "--",
            *refresh_model.CHECK_PATHS,
            cwd=root,
        )
    )


def follow_proven_bytes(root: Path, edits: dict[str, bytes | None]) -> None:
    """A proven recipe that a refresh moves carries its new bytes into the proven guard.

    Owner (2026-10-26): proven recipes move to the newest upstream version and the sweep
    re-checks them (their evidence no longer matches the new digest). The guard
    (``tests/qualification/test_admissible_memory_declarations.py``) still refuses any other edit.
    """
    path = root / refresh_model.PROVEN_BYTES
    if not path.is_file():
        return
    proven = json.loads(path.read_text(encoding="utf-8"))
    moved = {
        key: hashlib.sha256(data).hexdigest()
        for key in proven
        if (data := edits.get(f"recipes/{key.split('/', 1)[1]}.json")) is not None
    }
    if any(proven[key] != digest for key, digest in moved.items()):
        proven.update(moved)
        edits[refresh_model.PROVEN_BYTES] = (
            json.dumps(proven, indent=2, sort_keys=True) + "\n"
        ).encode()


def publish_mechanical(
    root: Path, assessment: refresh_model.Assessment, summary: Summary
) -> int | None:
    """Prepare the refresh PR; a move held for review is prepared but never armed."""
    number = _publish(root, assessment, summary, arm=not assessment.held)
    assessment.pull_request = number
    return number


def _publish(
    root: Path, assessment: refresh_model.Assessment, summary: Summary, *, arm: bool
) -> int | None:
    """Prepare (or move forward) the refresh PR on the newest upstream; returns its number.

    The branch only moves forward: a new commit has the branch head and main as its
    parents and main plus this refresh as its tree, so the push is a fast-forward, never
    forced. Nothing is pushed while the branch already carries this exact refresh, so a
    PR is never pushed again only to restart its checks: failed or cancelled checks on
    the current refresh are evidence for the recipe's needs-review issue.
    """
    recipe_id = assessment.recipe_id
    branch = f"{refresh_model.BRANCH_PREFIX}{recipe_id}"
    marker = f"<!-- refresh-target: {assessment.fingerprint()} -->"
    existing = own_pulls(
        refresh_github.gh_json(
            "pr",
            "list",
            "--head",
            branch,
            "--state",
            "all",
            "--json",
            f"number,state,body,statusCheckRollup,{OWN_FIELDS}",
        )
    )
    if any(pr["state"] == "CLOSED" and marker in (pr["body"] or "") for pr in existing):
        summary.deferred.append(
            f"{recipe_id}: a pull request for this exact target was closed unmerged"
        )
        return None
    open_pr = next((pr for pr in existing if pr["state"] == "OPEN"), None)
    if summary.started is not None:
        # A run assesses main as it was when the run started, with that commit's code,
        # and prepares PRs for up to two hours. When main has since changed the refresher
        # itself or a file this refresh rewrites (an identical refresh merged, a fix to
        # the rules), this run's result is stale: the next run (chained to every
        # publication) assesses afresh.
        refresh_github.git(
            "fetch", "-q", "origin", "+main:refs/remotes/origin/main", cwd=root
        )
        moved = refresh_github.git(
            "diff",
            "--name-only",
            summary.started,
            "origin/main",
            "--",
            "src/",
            "tools/",
            "contracts/",
            "policy/",
            *assessment.edits,
            cwd=root,
        ).split()
        if moved:
            summary.deferred.append(
                f"{recipe_id}: main changed {', '.join(moved[:5])} since this run "
                "started; the next run assesses it afresh"
            )
            return None
    # -f: whatever an earlier recipe left behind (staged edits after an early return or
    # a publish, a half-applied edit) combined with main moving at the fetch above
    # must never block this one, not even an untracked file where main now tracks one.
    refresh_github.git("checkout", "-f", "--detach", "origin/main", cwd=root)
    refresh_github.git("reset", "--hard", "origin/main", cwd=root)
    refresh_github.git("clean", "-fdq", cwd=root)
    follow_proven_bytes(root, assessment.edits)
    refresh_apply.apply_edits(root, assessment.edits)
    refresh_github.git("add", "-A", cwd=root)
    tree = refresh_github.git("write-tree", cwd=root)
    head = None
    if refresh_github.git("ls-remote", "origin", f"refs/heads/{branch}", cwd=root):
        refresh_github.git("fetch", "-q", "origin", branch, cwd=root)
        head = refresh_github.git("rev-parse", "FETCH_HEAD", cwd=root)
    state = check_state(open_pr.get("statusCheckRollup")) if open_pr else "none"
    if (
        open_pr
        and head
        and carries(root, head, tree)
        and not (
            state in refresh_model.PARKED_CHECK_STATES and checks_moved(root, head)
        )
    ):
        number = int(open_pr["number"])
        if state in refresh_model.PARKED_CHECK_STATES:
            assessment.reasons.append(
                f"refresh PR #{number} carries this exact refresh and its checks "
                f"{state}; it moves forward again when upstream or main's checks "
                f"({', '.join(refresh_model.CHECK_PATHS)}) change, or after a repair of the recipe"
            )
            assessment.evidence.append(
                f"### Failed checks of PR #{number}\n\n"
                f"{failed_check_evidence(open_pr.get('statusCheckRollup'))}\n"
            )
        else:
            if arm:
                arm_mechanical_pr(number, recipe_id, summary)
            summary.mechanical.append(
                f"{recipe_id}: PR #{number} already current"
                + ("" if arm else " (review, not armed)")
            )
        return number
    failure = refresh_apply.verify(root)
    if failure is not None:
        assessment.reasons.append(
            f"checks failed after applying the refresh: {failure}"
        )
        refresh_github.git("reset", "--hard", "origin/main", cwd=root)
        refresh_github.git("clean", "-fdq", cwd=root)
        return None
    parents = [head] if head else []
    parents.append(refresh_github.git("rev-parse", "origin/main", cwd=root))
    commit = refresh_github.git(
        "commit-tree",
        tree,
        *(flag for parent in parents for flag in ("-p", parent)),
        "-m",
        f"Refresh {assessment.label} to the newest upstream pins\n\n{refresh_render.render_pins(assessment)}",
        cwd=root,
    )
    refresh_github.git(
        "push", "-q", "origin", f"{commit}:refs/heads/{branch}", cwd=root
    )
    summary.prepared += 1
    title = f"Refresh {assessment.label} to the newest upstream pins"
    body = refresh_render.pr_body(assessment)
    label, other = (
        (refresh_model.LABEL_MECHANICAL, refresh_model.LABEL_NEEDS_REVIEW)
        if arm
        else (refresh_model.LABEL_NEEDS_REVIEW, refresh_model.LABEL_MECHANICAL)
    )
    if open_pr:
        number = open_pr["number"]
        refresh_github.gh(
            "pr",
            "edit",
            str(number),
            "--add-label",
            label,
            "--remove-label",
            other,
            "--title",
            title,
            "--body",
            body,
        )
    else:
        url = refresh_github.gh(
            "pr",
            "create",
            "--head",
            branch,
            "--base",
            "main",
            "--title",
            title,
            "--body",
            body,
            "--label",
            label,
        ).strip()
        number = int(url.rsplit("/", 1)[-1])
    if arm:
        arm_mechanical_pr(number, recipe_id, summary)
    # App-created PRs trigger normal pull_request validation without human approval.
    summary.mechanical.append(
        f"{recipe_id}: PR #{number}" + ("" if arm else " (review, not armed)")
    )
    return int(number)


def sharing_groups(catalog_: refresh_catalog_view.Catalog) -> dict[str, list[str]]:
    """Recipe -> every recipe (itself included, sorted) built from the same context.

    Recipes sharing an adapter share its pins, lock and Dockerfile, so a refresh edits
    them all or none: a PR that moved the shared files for one recipe would leave the
    others inconsistent (the kit lock check fails) and never merge.
    """
    by_context: dict[str, list[str]] = {}
    for recipe_id, recipe in catalog_.recipes.items():
        context = recipe["execution"]["build"]["context"]["path"].rstrip("/")
        by_context.setdefault(context, []).append(recipe_id)
    return {
        recipe_id: sorted(ids)
        for ids in by_context.values()
        if len(ids) > 1
        for recipe_id in ids
    }


def combine_group(
    members: list[str], assessed: dict[str, refresh_model.Assessment]
) -> refresh_model.Assessment | str:
    """One assessment that refreshes all ``members`` together, or why that is not possible.

    A member that is current has nothing of its own to move (a sibling's own Model
    revision moved, not the shared pins), so it never holds the others back.
    """
    parts: list[refresh_model.Assessment] = []
    for member in members:
        part = assessed.get(member)
        if part is not None and not part.drifted and not part.unreachable:
            continue
        if part is None or not part.prepared:
            return f"{member} sharing the build context is not mechanical right now"
        parts.append(part)
    if not parts:
        return "no member of the shared build context drifted"
    edits: dict[str, bytes | None] = {}
    for part in parts:
        for name, data in part.edits.items():
            if name in edits and edits[name] != data:
                return f"{part.recipe_id} and another recipe edit {name} differently"
            edits[name] = data
    lead = parts[0]
    return refresh_model.Assessment(
        recipe_id=members[0],  # the group's branch (close_stale_refresh_prs)
        items=[item for part in parts for item in part.items],
        evidence=[text for part in parts for text in part.evidence],
        notes=list(dict.fromkeys(note for part in parts for note in part.notes)),
        held=list(dict.fromkeys(text for part in parts for text in part.held)),
        declined=list(dict.fromkeys(text for part in parts for text in part.declined)),
        peaks_raised=any(part.peaks_raised for part in parts),
        edits=edits,
        version=lead.version,
        members=[part.recipe_id for part in parts],  # the recipes that move
    )


def publish_group(
    root: Path,
    combined: refresh_model.Assessment,
    result: refresh_model.Assessment,
    summary: Summary,
) -> int | None:
    """Publish a group's PR; ``result``'s issue gets what publishing found out.

    That is the reasons and the evidence publishing added (a failed check's log tail),
    counted on the combined assessment itself: a current member adds nothing to it.
    """
    seen = len(combined.evidence)
    number = publish_mechanical(root, combined, summary)
    result.pull_request = number
    result.reasons.extend(combined.reasons)
    result.evidence.extend(combined.evidence[seen:])
    return number


def explain_members(
    members: list[str], assessed: dict[str, refresh_model.Assessment], why: list[str]
) -> None:
    """Show every prepared member of a group that did not land why it waits.

    The group is refreshed together or not at all, so a member (a single sharing its
    dual's build context) has no pull request of its own and its issue would keep a
    stale body. The outcome rides in ``group_outcome`` for the issue to render; the
    member stays prepared, so the group's pull request is not judged stale by it.
    """
    for member in members:
        part = assessed.get(member)
        if part is None or not part.prepared:
            continue  # members with reasons of their own report those
        others = ", ".join(f"`{m}`" for m in members if m != member)
        part.group_outcome = [
            f"refreshed together with {others} or not at all: {text}" for text in why
        ]


def supersede_member_prs(
    members: list[str], lead: str, number: int, summary: Summary
) -> None:
    """Close the separate refresh PRs of recipes now refreshed together with ``lead``."""
    for member in members:
        if member == lead:
            continue
        for pr in own_pulls(
            refresh_github.gh_json(
                "pr",
                "list",
                "--head",
                f"{refresh_model.BRANCH_PREFIX}{member}",
                "--state",
                "open",
                "--json",
                f"number,{OWN_FIELDS}",
            )
        ):
            refresh_github.gh(
                "pr",
                "close",
                str(pr["number"]),
                "--delete-branch",
                "--comment",
                f"Superseded by #{number}: recipes that share a build context are "
                "refreshed together in one pull request.",
            )
            summary.closed.append(
                f"{member}: PR #{pr['number']} superseded by #{number}"
            )


def close_stale_refresh_prs(
    assessed: dict[str, refresh_model.Assessment],
    summary: Summary,
    groups: dict[str, list[str]] | None = None,
    published: set[str] | None = None,
) -> None:
    """Close open refresh PRs the detector no longer regenerates.

    A refresh PR of a recipe that is current again, gone, or no longer mechanical (it
    needs review) would otherwise sit on an old upstream commit forever. The review
    issue, when there is one, is linked in the comment.
    """
    pulls = [
        pr
        for label in (
            refresh_model.LABEL_MECHANICAL,
            refresh_model.LABEL_NEEDS_REVIEW,
        )  # held moves are PRs too
        for pr in own_pulls(
            refresh_github.gh_json(
                "pr",
                "list",
                "--state",
                "open",
                "--label",
                label,
                "--limit",
                "200",
                "--json",
                f"number,headRefName,body,{OWN_FIELDS}",
            )
        )
    ]
    issues: list[dict[str, Any]] = (
        refresh_github.gh_json(
            "issue",
            "list",
            "--label",
            refresh_model.LABEL_NEEDS_REVIEW,
            "--state",
            "open",
            "--limit",
            "500",
            "--json",
            "number,title",
        )
        if pulls
        else []
    )
    for pr in pulls:
        branch = str(pr.get("headRefName", ""))
        if not branch.startswith(refresh_model.BRANCH_PREFIX):
            continue
        recipe_id = branch.removeprefix(refresh_model.BRANCH_PREFIX)
        result = assessed.get(recipe_id)
        together = (groups or {}).get(recipe_id)
        if together is not None and together[0] != recipe_id:
            note = (
                f"it shares its build context with `{together[0]}`, whose pull request "
                "refreshes them together; a pull request for one of them alone leaves "
                "the shared pins inconsistent."
            )
        elif any(  # a group's branch lives while any member still moves
            member in (published or set())
            or (
                (part := assessed.get(member)) is not None
                and (part.unreachable or part.prepared)
            )
            for member in together or [recipe_id]
        ):
            continue
        elif result is None:
            note = "The recipe no longer exists."
        elif not result.drifted:
            note = "The recipe is current again."
        else:
            link = next(
                (
                    i
                    for i in issues
                    if i["title"] == refresh_render.issue_title(recipe_id)
                ),
                None,
            )
            note = "The refresh is no longer mechanical: " + (
                f"see #{link['number']}."
                if link
                else f"{result.reasons[0][:200]} (an issue follows on the next run)."
            )
        refresh_github.gh(
            "pr",
            "close",
            str(pr["number"]),
            "--delete-branch",
            "--comment",
            f"Closed by the refresh detector, not regenerated: {note}",
        )
        summary.closed.append(f"{recipe_id}: PR #{pr['number']}")


def review_issue(
    title: str, state: str, label: str = refresh_model.LABEL_NEEDS_REVIEW
) -> dict[str, Any] | None:
    """The newest issue labelled ``label`` (default ``refresh:needs-review``) titled ``title`` in ``state``."""
    issues = refresh_github.gh_json(
        "issue",
        "list",
        "--label",
        label,
        "--state",
        state,
        "--limit",
        "1000",
        "--json",
        "number,title,body",
    )
    found = [i for i in issues if i["title"] == title]
    return max(found, key=lambda i: int(i["number"])) if found else None


def sync_titled_issue(
    title: str,
    render: Callable[[str | None], str],
    label: str = refresh_model.LABEL_NEEDS_REVIEW,
    *,
    create: bool = True,
) -> None:
    """Keep the one issue titled ``title`` showing ``render(previous body)``.

    A subject has one issue for life: when it needs attention again, its closed issue
    is reopened with the new body rather than a second issue opened. ``create=False``
    only updates an open issue.
    """
    existing = review_issue(title, "open", label)
    if existing is None and create and (closed := review_issue(title, "closed", label)):
        refresh_github.gh("issue", "reopen", str(closed["number"]))
        existing = closed
    if existing is None and not create:
        return
    body = render(None if existing is None else existing["body"])
    if existing is None:
        refresh_github.gh(
            "issue", "create", "--title", title, "--body", body, "--label", label
        )
    elif (existing["body"] or "").strip() != body.strip():
        refresh_github.gh("issue", "edit", str(existing["number"]), "--body", body)


def sync_issue(
    assessment: refresh_model.Assessment, summary: Summary, *, create: bool = True
) -> None:
    """Keep the recipe's one review issue showing the current assessment."""
    sync_titled_issue(
        refresh_render.issue_title(assessment.recipe_id),
        lambda previous: refresh_render.issue_body(assessment, previous),
        create=create,
    )
    if shown := assessment.reasons or assessment.group_outcome:
        summary.review.append(f"{assessment.recipe_id}: {shown[0][:140]}")


def command_memory_estimate_issue(report: Path) -> int:
    """Keep the one memory-estimate issue current from ``check-recipe-memory-need --report``.

    A body with findings opens or refreshes the issue; an empty report closes it.
    """
    body = report.read_text(encoding="utf-8")
    refresh_github.ensure_labels()
    if body.strip():
        sync_titled_issue(
            refresh_model.MEMORY_ESTIMATE_TITLE,
            lambda _previous: body,
            refresh_model.LABEL_MEMORY_ESTIMATE,
        )
    elif open_issue := review_issue(
        refresh_model.MEMORY_ESTIMATE_TITLE, "open", refresh_model.LABEL_MEMORY_ESTIMATE
    ):
        refresh_github.gh(
            "issue",
            "close",
            str(open_issue["number"]),
            "--comment",
            "No memory estimate is missing a sourced input any more.",
        )
    return 0


def close_current_issues(
    assessed: dict[str, refresh_model.Assessment], handled: set[str], summary: Summary
) -> None:
    """Close the review issue of every recipe that no longer needs review.

    That is a recipe that is current or gone, or whose refresh pull request this run
    prepared (``handled``). A drifted recipe without one keeps its issue: a refresh
    that is mechanical on paper but was deferred, refused or failed its checks has not
    been handled, and closing its issue would only reopen it next run. A recipe has one
    issue: an older open issue with the same title is closed in favour of the newest,
    which is the one ``sync_issue`` keeps current.
    """
    issues = refresh_github.gh_json(
        "issue",
        "list",
        "--label",
        refresh_model.LABEL_NEEDS_REVIEW,
        "--state",
        "open",
        "--limit",
        "500",
        "--json",
        "number,title",
    )
    prefix = refresh_render.issue_title("")
    newest: dict[str, int] = {}
    for issue in issues:
        newest[issue["title"]] = max(newest.get(issue["title"], 0), issue["number"])
    for issue in issues:
        if not issue["title"].startswith(prefix):
            continue
        recipe_id = issue["title"][len(prefix) :]
        result = assessed.get(recipe_id)
        if issue["number"] != newest[issue["title"]]:
            note = (
                f"Superseded by #{newest[issue['title']]}, this recipe's current review "
                "issue."
            )
        elif result is None:
            note = "The recipe no longer exists."
        elif result.unreachable:
            continue
        elif recipe_id in handled:
            note = (
                "The refresh is mechanical now; an automatic pull request handles it."
            )
        elif not result.drifted:
            note = "The recipe is current again."
        else:
            continue
        refresh_github.gh(
            "issue",
            "close",
            str(issue["number"]),
            "--reason",
            "completed",
            "--comment",
            note,
        )
        summary.closed.append(recipe_id)


def recipe_footprint(
    catalog_: refresh_catalog_view.Catalog, recipe_id: str
) -> tuple[set[str], list[str]]:
    """Files and directories a refresh of the recipe may write (exact paths, prefixes)."""
    recipe = catalog_.recipes[recipe_id]
    files = {f"recipes/{recipe_id}.json"}
    for reference in recipe["models"]:
        files.add(f"models/{reference['model']['slug']}.json")
        files.add(f"display/models/{reference['model']['slug']}.json")
    context = recipe["execution"]["build"]["context"]["path"].rstrip("/") + "/"
    return files, [context]


def blocking_prs(
    footprint: tuple[set[str], list[str]], open_prs: dict[int, set[str]]
) -> list[int]:
    """Open pull requests (not the detector's own) that modify the recipe's files."""
    files, prefixes = footprint
    return sorted(
        number
        for number, changed in open_prs.items()
        if any(name in files or name.startswith(tuple(prefixes)) for name in changed)
    )


def with_all_files(pulls: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """This repository's PRs from a ``gh pr list`` that asked for ``files,changedFiles``,
    each with its complete file list: the listing returns at most 100 files per PR."""
    complete = []
    for pr in own_pulls(pulls):
        files = pr.get("files") or []
        if int(pr.get("changedFiles") or 0) > len(files):
            names = refresh_github.gh(
                "api",
                "--paginate",
                f"repos/{{owner}}/{{repo}}/pulls/{pr['number']}/files?per_page=100",
                "--jq",
                ".[].filename",
            ).split()
            files = [{"path": name} for name in names]
        complete.append({**pr, "files": files})
    return complete


def foreign_open_prs() -> dict[int, set[str]]:
    """Changed paths of every open pull request of this repository that is not ``refresh/*``.

    A fork PR is never ours to wait for: only our own PRs can block a refresh.
    """
    listed = own_pulls(
        refresh_github.gh_json(
            "pr",
            "list",
            "--state",
            "open",
            "--limit",
            "1000",
            "--json",
            f"number,headRefName,files,changedFiles,{OWN_FIELDS}",
        )
    )
    return {
        int(pr["number"]): {f.get("path", "") for f in pr["files"]}
        for pr in with_all_files(
            [
                p
                for p in listed
                if not str(p.get("headRefName", "")).startswith(
                    refresh_model.BRANCH_PREFIX
                )
            ]
        )
    }


def refresh_is_spinning(
    runs: list[dict[str, Any]],
    run_id: int,
    now: datetime,
    minutes: int = refresh_model.CHAIN_WINDOW_MINUTES,
    limit: int = refresh_model.CHAIN_MAX_RUNS,
) -> bool:
    """At least ``limit`` other refresh runs finished successfully in the last ``minutes``.

    A real chain has gaps (every merge waits for a publication, minutes long), so a burst
    means a loop. Active or queued runs, and a run that just finished ahead of this one
    in the concurrency queue, must not count against a single run.
    """
    finished = [
        datetime.fromisoformat(str(run.get("updatedAt", "")))
        for run in runs
        if run.get("databaseId") != run_id
        and run.get("status") == "completed"
        and run.get("conclusion") == "success"
    ]
    return (
        sum(1 for when in finished if now - when < timedelta(minutes=minutes)) >= limit
    )


def command_guard(arguments: argparse.Namespace) -> int:
    """Decide whether a chained (workflow_run) refresh should proceed.

    Scheduled and manual runs always proceed. A run chained from a finished
    publication proceeds unless the publication failed or the refresh is spinning
    (CHAIN_MAX_RUNS successful runs in the last CHAIN_WINDOW_MINUTES minutes).
    """
    run = True
    if arguments.event == "workflow_run":
        if arguments.upstream_conclusion != "success":
            run = False
        else:
            runs = refresh_github.gh_json(
                "run",
                "list",
                "--workflow",
                "refresh-upstream.yml",
                "--limit",
                "10",
                "--json",
                "databaseId,status,conclusion,updatedAt",
            )
            run = not refresh_is_spinning(
                runs,
                arguments.run_id,
                datetime.now(UTC),
            )
    output = os.environ.get("GITHUB_OUTPUT")
    if output:
        with Path(output).open("a", encoding="utf-8") as handle:
            handle.write(f"run={'true' if run else 'false'}\n")
    print(f"refresh {'proceeds' if run else 'skipped by the chain guard'}")
    return 0


def publish_if_stale() -> None:
    """Repair a missing publication receipt without duplicating an active main run."""
    head = refresh_github.gh(
        "api", "repos/{owner}/{repo}/commits/main", "--jq", ".sha"
    ).strip()
    runs = refresh_github.gh_json(
        "run",
        "list",
        "--workflow",
        "publish.yml",
        "--branch",
        "main",
        "--limit",
        "5",
        "--json",
        "headSha,status,conclusion,event",
    )
    if has_publication_receipt(head, runs):
        return
    if any(
        run.get("headSha") == head and run.get("status") in {"queued", "in_progress"}
        for run in runs
    ):
        return
    refresh_github.gh("workflow", "run", "publish.yml", "--ref", "main", check=False)


def write_summary(
    summary: Summary,
    http: refresh_http.Http,
    path: str | None,
    embedded: dict[str, Any],
) -> None:
    def section(title: str, rows: list[str]) -> list[str]:
        return [
            f"### {title}: {len(rows)}",
            "",
            *([f"- {row}" for row in rows] or ["None."]),
            "",
        ]

    lines = [
        "## Upstream refresh",
        "",
        (
            f"{summary.current} recipes had no drift in configured source, adapter and Model watches; "
            f"{len(summary.mechanical)} mechanical pull requests "
            f"({summary.prepared} prepared, {len(summary.armed)} armed); "
            f"{len(summary.review)} need review; {len(summary.unreachable)} unreachable. "
            f"{http.requests} upstream requests ({http.not_modified} answered 304)."
        ),
        "",
        "Embedded source inventory is evidence, not a latest-channel claim. Immutable image digests are pin-only; unresolved transitive dependencies remain unknown.",
        "",
        f"Embedded inputs by status: `{json.dumps(embedded['coverage'].get('counts_by_status', {}), sort_keys=True)}`.",
        "",
    ]
    candidate_counts = embedded["coverage"].get("counts_by_candidate_status", {})
    lines += [
        f"Embedded Git head candidates by status: `{json.dumps(candidate_counts, sort_keys=True)}`. Candidates are review evidence only.",
        "",
        "### Embedded inputs requiring review or retaining an unknown reason",
        "",
    ]
    details: list[str] = []
    for recipe in embedded.get("recipes", []):
        if not isinstance(recipe, dict):
            continue
        recipe_id = recipe.get("recipe_id", "unknown")
        for item in recipe.get("inputs", []):
            if not isinstance(item, dict):
                continue
            if item.get("status") not in {
                "mutable",
                "unknown",
                "pinned_untracked",
            } and item.get("candidate_status") not in {
                "advanced",
                "unknown",
                "no-immutable-pin",
            }:
                continue
            location = str(item.get("path", "unknown"))
            if item.get("line") is not None:
                location += f":{item['line']}"
            evidence = item.get("candidate_reason") or item.get("reason")
            if item.get("candidate_status") == "advanced":
                evidence = (
                    f"{item.get('repository')}@{item.get('revision')} -> observed candidate "
                    f"{item.get('candidate_revision')}; {evidence}"
                )
            details.append(
                f"- `{recipe_id}` `{item.get('kind')}` at `{location}` "
                f"({item.get('status')}/{item.get('candidate_status', 'no head candidate')}): {evidence}"
            )
    lines += details or ["None."]
    lines += [
        "",
        *section("Mechanical refresh pull requests", summary.mechanical),
        *section("Needs review (issues)", summary.review),
        *section("Issues closed (recipe current again)", summary.closed),
        *section("Deferred to a later run", summary.deferred),
        *section("Upstream unreachable (recipe untouched)", summary.unreachable),
        *section("Newer release line (not followed)", summary.newer_lines),
        *section("Failures", summary.failures),
    ]
    text = "\n".join(lines)
    print(text)
    if path:
        with Path(path).open("a", encoding="utf-8") as handle:
            handle.write(text + "\n")
