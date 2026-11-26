"""Resolving a declared pin to the newest upstream."""

from __future__ import annotations

import re
from collections.abc import Iterable

from recipe_tooling.kit_pins.declarations import read_declared
from recipe_tooling.kit_pins.model import IMAGE_DIGEST, SHA, Declaration, Pin
from recipe_tooling.kit_pins.upstream import GitHub, Upstream
from recipe_tooling.upstream_http import UpstreamBusy


def resolve(
    source: Upstream, declaration: Declaration, repo: str, ref: str
) -> tuple[str, str, str]:
    """``(immutable value, date, kind)`` of a declared ref, resolved once.

    A full commit id is already immutable: it is its own pin and needs no provider read,
    so a repository the provider no longer serves cannot block it.
    """
    if declaration.datasource in ("git-ref", "hf-revision") and SHA.match(ref):
        return ref, "", "commit"
    if declaration.datasource == "git-ref":
        commit, date = source.github.resolve(repo, ref)
        return commit, date, source.github.kind(repo, ref)
    if declaration.datasource == "hf-revision":
        hub = source.hf()
        revision, date = hub.resolve(repo, ref)
        return revision, date, hub.kind(repo, ref)
    registry = source.oci()
    return (
        registry.resolve(repo, ref),
        "",
        "digest" if IMAGE_DIGEST.match(ref) else "tag",
    )


def derive(
    kit_repo: str,
    kit_commit: str,
    wanted: Iterable[Declaration],
    source: Upstream,
) -> list[Pin]:
    """Resolve what the kit declares at ``kit_commit`` to immutable values."""
    pins: list[Pin] = []
    for declaration in wanted:
        repo, ref, reason = read_declared(source, kit_repo, kit_commit, declaration)
        immutable, date, kind = resolve(source, declaration, repo, ref)
        pins.append(
            Pin(
                declaration.name,
                repo,
                ref,
                immutable,
                date,
                kind,
                reason,
                declaration.datasource,
            )
        )
    return pins


def newest_release(source: GitHub, repo: str) -> str | None:
    """The repository's newest version tag, for information in issues only."""
    try:
        tags = source.api(f"repos/{repo}/tags?per_page=100")
    except UpstreamBusy:
        raise
    except (RuntimeError, KeyError):
        return None
    best: tuple[tuple[int, ...], str] | None = None
    for tag in tags:
        match = re.match(r"^v?(\d+(?:\.\d+)*)$", str(tag.get("name", "")))
        if match is None:
            continue
        key = tuple(int(part) for part in match.group(1).split("."))
        if best is None or key > best[0]:
            best = (key, tag["name"])
    return None if best is None else best[1]


def newest_information(source: Upstream, pin: Pin) -> str | None:
    """What the dependency's own channel offers now, for information only.

    The pin follows the kit's declaration: this is never a candidate.
    """
    if pin.datasource == "git-ref":
        newest = newest_release(source.github, pin.repo)
        return f"{pin.repo} newest release `{newest}`" if newest else None
    if pin.datasource == "hf-revision" and source.hub is not None:
        head = source.hub.head(pin.repo)
        return f"{pin.repo} newest revision `{head[:12]}`" if head else None
    return None
