"""Creator signals: supersession and new repositories."""

from __future__ import annotations

import base64
import binascii
import re
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Any, Literal
from urllib.parse import urlsplit

from recipe_tooling.upstream_drift.model import (
    GITHUB_REPO_LINK,
    LISTING_PAGE_SIZE,
    MAX_LISTING_PAGES,
    NEW_REPO_WINDOW,
    NON_OWNER_SEGMENTS,
    NON_REPO_PAGES,
    NOTICE_HEAD_LINES,
    NOTICE_WORDS,
    SIGNAL_WORKERS,
    CreatorFinding,
    CreatorsFile,
    CreatorSignals,
    RemoteNotFound,
    SignalError,
    Watch,
)
from recipe_tooling.upstream_drift.observe import (
    ProviderClient,
    _github_commit,
    _required_string,
)


def _finding(
    kind: Literal["supersession", "new-repo"],
    source: str,
    linked: str,
    **fields: Any,
) -> CreatorFinding:
    return CreatorFinding(
        kind=kind,
        dedupe_key=f"{kind}:{source}:{linked}".lower(),
        linked_repo=linked,
        **fields,
    )


def find_supersession(
    readme: str, source_repo: str, commit: str
) -> CreatorFinding | None:
    """Return the first notice block in the README head that links another repo."""
    head = "\n".join(readme.splitlines()[:NOTICE_HEAD_LINES])
    for block in re.split(r"\n\s*\n", head):
        if not NOTICE_WORDS.search(block):
            continue
        for owner, name, page in GITHUB_REPO_LINK.findall(block):
            linked = f"{owner}/{name}"
            if (
                owner in NON_OWNER_SEGMENTS
                or page in NON_REPO_PAGES
                or linked.lower() == source_repo.lower()
            ):
                continue
            return _finding(
                "supersession",
                source_repo,
                linked,
                source_repo=source_repo,
                source_commit=commit,
                linked_commit=None,
                recipe_slugs=[],
                matched_family=None,
                detail=re.sub(r"[>#*\s]+", " ", block).strip()[:200],
            )
    return None


def _tokens(text: str) -> set[str]:
    return set(re.split(r"[^a-z0-9]+", text.lower())) - {""}


def match_new_repos(
    owner: str,
    listing: object,
    now: datetime,
    families: set[str],
    pinned: set[str],
) -> list[CreatorFinding]:
    """Flag repos created inside the window that name a carried family.

    A family matches when every token of its slug is a whole token of the repo
    name or description; the longest matching slug wins.
    """
    if not isinstance(listing, list):
        raise TypeError("repository listing is not a list")
    found: list[CreatorFinding] = []
    for item in listing:
        if not isinstance(item, dict) or item.get("fork"):
            continue
        name, created = item.get("name"), item.get("created_at")
        if not isinstance(name, str) or not isinstance(created, str):
            continue
        if now - datetime.fromisoformat(created) > NEW_REPO_WINDOW:
            continue
        linked = f"{owner}/{name}"
        if linked.lower() in pinned:
            continue
        description = item.get("description")
        tokens = _tokens(
            f"{name} {description if isinstance(description, str) else ''}"
        )
        matches = [f for f in families if _tokens(f) <= tokens]
        if matches:
            family = max(matches, key=lambda f: (len(_tokens(f)), f))
            found.append(
                _finding(
                    "new-repo",
                    owner,
                    linked,
                    source_repo=None,
                    source_commit=None,
                    linked_commit=None,
                    recipe_slugs=[],
                    matched_family=family,
                    detail=f"created {created}: {description or '-'}"[:200],
                )
            )
    return found


def _recent_repos(
    client: ProviderClient, owner: str, now: datetime
) -> tuple[list[object], bool]:
    """Newest-first listing, paged until a repo falls outside the window.

    The flag is True when MAX_LISTING_PAGES ended the paging instead.
    """
    repos: list[object] = []
    for page in range(1, MAX_LISTING_PAGES + 1):
        batch = client.json(
            f"https://api.github.com/users/{owner}/repos?sort=created&direction=desc"
            f"&per_page={LISTING_PAGE_SIZE}&page={page}&type=owner",
            "github",
        )
        if not isinstance(batch, list):
            raise TypeError("repository listing is not a list")
        repos.extend(batch)
        oldest = (
            batch[-1].get("created_at")
            if batch and isinstance(batch[-1], dict)
            else None
        )
        if (
            len(batch) < LISTING_PAGE_SIZE
            or not isinstance(oldest, str)
            or now - datetime.fromisoformat(oldest) > NEW_REPO_WINDOW
        ):
            return repos, False
    return repos, True


def _readme_notice(client: ProviderClient, repository: str) -> CreatorFinding | None:
    commit, _, _ = _github_commit(client, repository, "HEAD")
    document = client.json(
        f"https://api.github.com/repos/{repository}/readme?ref={commit}", "github"
    )
    content = _required_string(document, "content", "readme")
    try:
        text = base64.b64decode(content).decode("utf-8", errors="replace")
    except binascii.Error as error:
        raise RuntimeError(f"readme is not base64: {error}") from error
    return find_supersession(text, repository, commit)


def _creator_owners(root: Path) -> list[str]:
    creators = CreatorsFile.model_validate_json(
        (root / "policy/creators.json").read_text(encoding="utf-8")
    )
    owners: dict[str, str] = {}
    for creator in creators.creators:
        parts = urlsplit(creator.url)
        segments = [part for part in parts.path.split("/") if part]
        if parts.hostname == "github.com" and segments:
            owners.setdefault(segments[0].lower(), segments[0])
    return sorted(owners.values(), key=str.lower)


def scan_creator_signals(
    root: Path, watches: list[Watch], client: ProviderClient, now: datetime
) -> CreatorSignals:
    """Report creator findings; a failure is recorded per item and never hides the rest."""
    repositories = sorted({w.repository for w in watches if w.provider == "github"})
    pinned = {repository.lower() for repository in repositories}
    family_models: dict[str, set[str]] = {}
    for watch in watches:
        if watch.family:
            family_models.setdefault(watch.family, set()).add(watch.entity)
    try:
        owners = _creator_owners(root)
    except (OSError, ValueError) as error:
        owners = []
        owner_error = SignalError(
            subject="policy/creators.json", detail=str(error)[:256]
        )
    else:
        owner_error = None

    def head(repository: str) -> str | None:
        try:
            return _github_commit(client, repository, "HEAD")[0]
        except RemoteNotFound, RuntimeError, KeyError, TypeError:
            return None

    def notice(repository: str) -> list[CreatorFinding] | SignalError:
        try:
            found = _readme_notice(client, repository)
        except (RemoteNotFound, RuntimeError, KeyError, TypeError) as error:
            return SignalError(subject=repository, detail=str(error)[:256])
        if found is None:
            return []
        slugs = sorted(
            w.entity.rsplit("/", 1)[1]
            for w in watches
            if w.entity.startswith("recipe/")
            and w.repository.lower() == repository.lower()
        )
        return [
            found.model_copy(
                update={"recipe_slugs": slugs, "linked_commit": head(found.linked_repo)}
            )
        ]

    def recent(owner: str) -> list[CreatorFinding | SignalError] | SignalError:
        try:
            repos, capped = _recent_repos(client, owner, now)
            found = match_new_repos(
                owner,
                repos,
                now,
                set(family_models),
                pinned,
            )
        except (RemoteNotFound, RuntimeError, TypeError, ValueError, KeyError) as error:
            return SignalError(subject=owner, detail=str(error)[:256])
        matched: list[CreatorFinding | SignalError] = [
            f.model_copy(
                update={
                    "recipe_slugs": sorted(
                        w.entity.rsplit("/", 1)[1]
                        for w in watches
                        if w.entity.startswith("recipe/")
                        and family_models[str(f.matched_family)].intersection(w.models)
                    ),
                    "linked_commit": head(f.linked_repo),
                }
            )
            for f in found
        ]
        if capped:
            matched.append(
                SignalError(
                    subject=owner,
                    detail=f"listing stopped at {MAX_LISTING_PAGES} pages; older repos in the window were not seen",
                )
            )
        return matched

    # Independent sources: one bounded pool; each item keeps its own outcome.
    with ThreadPoolExecutor(max_workers=SIGNAL_WORKERS) as pool:
        jobs = [pool.submit(notice, repository) for repository in repositories]
        jobs += [pool.submit(recent, owner) for owner in owners]
        outcomes = [job.result() for job in jobs]
    result = CreatorSignals()
    if owner_error is not None:
        result.signal_errors.append(owner_error)
    for outcome in outcomes:
        for item in [outcome] if isinstance(outcome, SignalError) else outcome:
            if isinstance(item, SignalError):
                result.signal_errors.append(item)
            else:
                result.findings.append(item)
    return result
