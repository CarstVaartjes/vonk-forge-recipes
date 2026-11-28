"""Resolving the newest upstream pin."""

from __future__ import annotations

import json
import re
from collections.abc import Iterable
from pathlib import Path
from typing import Any
from urllib.parse import quote

from recipe_tooling.refresh_upstream import catalog_view as refresh_catalog_view
from recipe_tooling.refresh_upstream import http as refresh_http
from recipe_tooling.refresh_upstream import model as refresh_model
from recipe_tooling.upstream_http import UpstreamBusy


def pick_version_tag(tags: Iterable[dict[str, Any]]) -> dict[str, Any] | None:
    """The highest plain version tag (v1.2.3); pre-release spellings are not versions."""
    best: tuple[tuple[int, ...], dict[str, Any]] | None = None
    for tag in tags:
        match = refresh_model.VERSION_TAG.match(str(tag.get("name", "")))
        if match is None:
            continue
        key = tuple(int(part) for part in match.group(1).split("."))
        if best is None or key > best[0]:
            best = (key, tag)
    return None if best is None else best[1]


def _date(value: str | None) -> str:
    return (value or "")[:10]


def resolve_github(http: refresh_http.Http, repo: str) -> refresh_model.Target:
    tag: str | None = None
    notes: str | None = None
    released: str | None = None
    try:
        release = http.github(f"repos/{repo}/releases/latest")
        tag = release.get("tag_name")
        notes = release.get("body")
        released = release.get("published_at")
    except refresh_http.NotFound:
        found = pick_version_tag(http.github(f"repos/{repo}/tags?per_page=100"))
        tag = None if found is None else found["name"]
    return _github_target(http, repo, tag, notes, released)


# ---------------------------------------------------------------------------
# Release lines
# ---------------------------------------------------------------------------
#
# An adapter may declare the upstream release line its source pin follows
# (``vendored-upstream.json`` ``source.line``, e.g. ``"0.6"``): "newest" is then the
# newest published release inside that line, even when the repository's newest
# release is a later line (TensorFold 1.x is a native binary; the Python engine our
# adapters install stays on 0.6.x). A newer line is reported, never followed.

RELEASE_LINE = re.compile(r"^\d+(?:\.\d+)*$")


def version_key(tag: str | None) -> tuple[int, ...] | None:
    match = refresh_model.VERSION_TAG.match(tag or "")
    return None if match is None else tuple(int(p) for p in match.group(1).split("."))


def line_parts(line: str) -> tuple[int, ...]:
    if not RELEASE_LINE.match(line):
        raise RuntimeError(f"release line {line!r} is not a version prefix like 0.6")
    return tuple(int(part) for part in line.split("."))


def in_line(key: tuple[int, ...], parts: tuple[int, ...]) -> bool:
    return key[: len(parts)] == parts


def release_line(directory: Path) -> str | None:
    """The release line the adapter's source pin follows, when it declares one."""
    line = (refresh_catalog_view.adapter_manifest(directory).get("source") or {}).get(
        "line"
    )
    return line if isinstance(line, str) else None


def resolve_github_line(
    http: refresh_http.Http, repo: str, line: str
) -> refresh_model.Target:
    """The newest published release of ``repo`` inside ``line`` (``0.6`` is 0.6.x)."""
    parts = line_parts(line)
    best: tuple[tuple[int, ...], dict[str, Any]] | None = None
    releases = http.github(f"repos/{repo}/releases?per_page=100")
    for release in releases:
        key = version_key(release.get("tag_name"))
        if release.get("draft") or release.get("prerelease") or key is None:
            continue
        if in_line(key, parts) and (best is None or key > best[0]):
            best = (key, release)
    if best is not None:
        release = best[1]
        return _github_target(
            http,
            repo,
            release["tag_name"],
            release.get("body"),
            release.get("published_at"),
        )
    if not releases:  # a repository that publishes only tags
        tags = http.github(f"repos/{repo}/tags?per_page=100")
        found = pick_version_tag(
            t for t in tags if (k := version_key(t.get("name"))) and in_line(k, parts)
        )
        if found is not None:
            return _github_target(http, repo, found["name"], None, None)
    raise RuntimeError(f"{repo} publishes no release in the declared {line} line")


def newer_line_note(repo: str, line: str, newest: refresh_model.Target) -> str | None:
    """A stable note when upstream's newest release is a later line than ``line``.

    It names the line (``1.0``), not the release, so the issue or pull request that
    carries it does not change with every patch release of that line.
    """
    parts = line_parts(line)
    key = version_key(newest.ref)
    if key is None or in_line(key, parts) or key[: len(parts)] < parts:
        return None
    newer = ".".join(str(part) for part in key[: len(parts)])
    return (
        f"Newer release line available: {repo} {newer} (this recipe follows the "
        f"declared {line} line; moving to another line is engine work, not a refresh)."
    )


def _github_target(
    http: refresh_http.Http,
    repo: str,
    tag: str | None,
    notes: str | None,
    released: str | None,
) -> refresh_model.Target:
    ref = tag or "HEAD"
    commit = http.github(f"repos/{repo}/commits/{quote(ref, safe='')}")
    sha = commit["sha"]
    if not refresh_model.SHA.match(sha):
        raise RuntimeError(f"{repo}: provider returned a non-Git revision")
    version = tag.removeprefix("v") if tag else None
    return refresh_model.Target(
        provider="github",
        repo=repo,
        sha=sha,
        date=_date(released or commit["commit"]["committer"]["date"]),
        ref=tag,
        version=version,
        url=commit["html_url"],
        notes=notes,
        committed_at=commit["commit"]["committer"]["date"],
        archived=_archived(http, repo),
    )


def _archived(http: refresh_http.Http, repo: str) -> bool:
    """Advisory: a failed lookup means "not known to be archived", never a failed target."""
    try:
        return http.github(f"repos/{repo}").get("archived") is True
    except (refresh_http.NotFound, RuntimeError, AttributeError):
        return False


def resolve_huggingface(http: refresh_http.Http, repo: str) -> refresh_model.Target:
    document = http.json(
        f"https://huggingface.co/api/models/{quote(repo, safe='/')}",
        "huggingface",
        anonymous=True,
    )
    sha = document["sha"]
    if not refresh_model.SHA.match(sha):
        raise RuntimeError(f"{repo}: provider returned a non-Git revision")
    gated = document.get("gated")
    if document.get("disabled") is True:
        access_status = "disabled"
    elif document.get("private") is True:
        access_status = "private"
    elif gated not in (None, False, "false", ""):
        access_status = "gated"
    elif document.get("private") is False and gated in (None, False, "false", ""):
        access_status = "public"
    else:
        access_status = "unknown"
    return refresh_model.Target(
        provider="huggingface",
        repo=repo,
        sha=sha,
        date=_date(document.get("lastModified")),
        ref=None,
        version=None,
        url=f"https://huggingface.co/{repo}/tree/{sha}",
        access_status=access_status,
    )


# ---------------------------------------------------------------------------
# Branch-following pins
# ---------------------------------------------------------------------------
#
# A pin that sits on a non-default branch (a per-bpw branch of an EXL3 repository whose
# ``main`` holds only a README, a feature branch of a Docker recipe) follows that branch,
# not the default branch. The branch is derived from the pin itself: the branch whose head
# is the pinned commit, or whose history contains it when the pin is not the head. A pin
# the default branch's history contains follows the default branch.

# Branches probed for containing a pin (GitHub compare, Hugging Face history).
BRANCH_SEARCH_LIMIT = 30
HF_HISTORY_PAGES = 3  # 50 commits a page


def _github_compare_status(
    http: refresh_http.Http, repo: str, old: str, new: str
) -> str:
    document = http.github(f"repos/{repo}/compare/{old}...{new}?per_page=1")
    return str(document.get("status", "unknown"))


def follow_github_branch(
    http: refresh_http.Http, repo: str, current: str
) -> refresh_model.Target | None:
    """The non-default branch a GitHub pin follows, as a Target at that branch's head."""
    try:
        default = http.github(f"repos/{repo}")["default_branch"]
        default_head = http.github(f"repos/{repo}/commits/{quote(default, safe='')}")
        if _github_compare_status(http, repo, current, default_head["sha"]) in {
            "ahead",
            "identical",
        }:
            return None  # the default branch's own history
        chosen: tuple[str, str] | None = None
        heads = http.github(f"repos/{repo}/commits/{current}/branches-where-head")
        names = sorted(b["name"] for b in heads if b["name"] != default)
        if names:
            chosen = (names[0], current)
        else:
            candidates: list[tuple[int, str, str]] = []
            for branch in http.github(f"repos/{repo}/branches?per_page=100"):
                name = branch["name"]
                if name == default:
                    continue
                if len(candidates) >= BRANCH_SEARCH_LIMIT:
                    break
                head = branch["commit"]["sha"]
                document = http.github(
                    f"repos/{repo}/compare/{current}...{head}?per_page=1"
                )
                if document.get("status") == "ahead":
                    candidates.append((int(document.get("ahead_by", 0)), name, head))
            if candidates:
                _, name, head = min(candidates)
                chosen = (name, head)
        if chosen is None:
            return None
        name, head = chosen
        commit = http.github(f"repos/{repo}/commits/{head}")
        target = refresh_model.Target(
            provider="github",
            repo=repo,
            sha=head,
            date=_date(commit["commit"]["committer"]["date"]),
            ref=None,
            version=None,
            url=commit["html_url"],
            branch=name,
            committed_at=commit["commit"]["committer"]["date"],
        )
        return _prefer_branch_release(http, repo, target)
    except UpstreamBusy:
        raise  # not checked this run: the caller lists the item as unreachable
    except (refresh_http.NotFound, RuntimeError, KeyError, TypeError, AttributeError):
        return None


def _prefer_branch_release(
    http: refresh_http.Http, repo: str, target: refresh_model.Target
) -> refresh_model.Target:
    """The newest published version tag inside the branch, else the branch head."""
    try:
        found = pick_version_tag(http.github(f"repos/{repo}/tags?per_page=100"))
        if found is None:
            return target
        tag_sha = found["commit"]["sha"]
        if tag_sha != target.sha and _github_compare_status(
            http, repo, tag_sha, target.sha
        ) not in {"ahead", "identical"}:
            return target  # the tag is not on this branch
        commit = http.github(f"repos/{repo}/commits/{tag_sha}")
        return refresh_model.Target(
            provider="github",
            repo=repo,
            sha=tag_sha,
            date=_date(commit["commit"]["committer"]["date"]),
            ref=found["name"],
            version=found["name"].removeprefix("v"),
            url=commit["html_url"],
            branch=target.branch,
            committed_at=commit["commit"]["committer"]["date"],
        )
    except UpstreamBusy:
        raise  # not checked this run: the caller lists the item as unreachable
    except (refresh_http.NotFound, RuntimeError, KeyError, TypeError):
        return target


def follow_huggingface_branch(
    http: refresh_http.Http, repo: str, current: str, default: refresh_model.Target
) -> refresh_model.Target | None:
    """The non-default branch a Hugging Face pin follows, as a Target at its head."""
    base = f"https://huggingface.co/api/models/{quote(repo, safe='/')}"

    def get(url: str) -> Any:
        return http.json(url, "huggingface", anonymous=True)

    def history(branch: str) -> list[dict[str, Any]]:
        commits: list[dict[str, Any]] = []
        for page in range(HF_HISTORY_PAGES):
            batch = get(f"{base}/commits/{quote(branch, safe='')}?p={page}")
            commits.extend(batch)
            if len(batch) < 50:
                break
        return commits

    try:
        branches = get(f"{base}/refs")["branches"]
        others = sorted(
            (b for b in branches if b["targetCommit"] != default.sha),
            key=lambda b: b["name"],
        )
        chosen: tuple[str, str, str] | None = None
        for branch in others:
            if branch["targetCommit"] == current:
                chosen = (branch["name"], current, default.date)
                break
        if chosen is None:
            main = next((b for b in branches if b["targetCommit"] == default.sha), None)
            if main is not None and any(
                c["id"] == current for c in history(main["name"])
            ):
                return None  # the default branch's own history
            found: list[tuple[str, str, str]] = []
            for branch in others[:BRANCH_SEARCH_LIMIT]:
                commits = history(branch["name"])
                if any(c["id"] == current for c in commits):
                    found.append(
                        (
                            branch["name"],
                            branch["targetCommit"],
                            _date(commits[0].get("date")),
                        )
                    )
            if len(found) != 1:
                return None  # none, or ambiguous: leave it to review
            chosen = found[0]
        name, head, date = chosen
        return refresh_model.Target(
            provider="huggingface",
            repo=repo,
            sha=head,
            date=date,
            ref=None,
            version=None,
            url=f"https://huggingface.co/{repo}/tree/{head}",
            access_status=default.access_status,
            branch=name,
        )
    except UpstreamBusy:
        raise  # not checked this run: the caller lists the item as unreachable
    except (refresh_http.NotFound, RuntimeError, KeyError, TypeError, AttributeError):
        return None


def resolve(http: refresh_http.Http, provider: str, repo: str) -> refresh_model.Target:
    if provider == "github":
        return resolve_github(http, repo)
    return resolve_huggingface(http, repo)


# ---------------------------------------------------------------------------
# Catalog view
# ---------------------------------------------------------------------------


def canonical(document: object) -> bytes:
    return (
        json.dumps(document, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode()
