"""Whether the Controller's catalog serves the recipe content this sweep's source authored.

A sweep is started from a checkout of recipes main, often right after a merge. The
Controller serves that content only once the publish workflow has released it and the
Controller has imported the release, minutes later. Until then the catalog still serves
the older content, whose verdicts are already recorded: a run that trusts it finds
nothing to do and ends having tested nothing.

The comparison is by content identity: the ``document_sha256`` the library publishes for
each recipe document, hashed once here from the authored file. A commit says nothing
about content (a docs-only commit publishes nothing and changes no recipe).

The wait is bounded (publish plus import), and it ends early when the catalog moves to
another library without serving the source (the Controller took a newer release, or the
checkout is behind main). Either way the sweep then tests what the catalog serves.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from qualification.documents import recipe_key_digest

# Publish (about 14 min on 2026-11-08) plus the Controller's import (under 2 min).
SOURCE_WAIT_SECONDS = 15 * 60.0


class SourceServing(StrEnum):
    """One sweep process's view of its source in the catalog (never persisted)."""

    UNREAD = "unread"  # no complete library pass yet
    UNSERVED = "unserved"  # selected recipes wait for the catalog to serve the source
    SERVED = "served"  # the sweep tests what the catalog serves, newer content included


class SourceEvent(StrEnum):
    UNSERVED_PASS = "unserved-pass"  # a complete pass serves other content
    SERVED_PASS = "served-pass"  # a complete pass serves the source
    CATALOG_MOVED = "catalog-moved"  # another library, still not the source
    WAIT_EXPIRED = "wait-expired"  # not served within SOURCE_WAIT_SECONDS


# The edges; any other (state, event) keeps the state. SERVED is final for the process.
EDGES: Mapping[tuple[SourceServing, SourceEvent], SourceServing] = {
    (SourceServing.UNREAD, SourceEvent.UNSERVED_PASS): SourceServing.UNSERVED,
    (SourceServing.UNREAD, SourceEvent.SERVED_PASS): SourceServing.SERVED,
    (SourceServing.UNSERVED, SourceEvent.UNSERVED_PASS): SourceServing.UNSERVED,
    (SourceServing.UNSERVED, SourceEvent.SERVED_PASS): SourceServing.SERVED,
    (SourceServing.UNSERVED, SourceEvent.CATALOG_MOVED): SourceServing.SERVED,
    (SourceServing.UNSERVED, SourceEvent.WAIT_EXPIRED): SourceServing.SERVED,
}


def advance(state: SourceServing, event: SourceEvent) -> SourceServing:
    """The one transition."""
    return EDGES.get((state, event), state)


@dataclass(frozen=True)
class Unserved:
    recipe: str
    served: str
    authored: str


def authored_contents(root: Path) -> dict[str, str]:
    """``publisher/slug`` to the content digest of each authored recipe, read once at start.

    A file that cannot be read is not judged (the catalog decides for that recipe).
    """
    contents: dict[str, str] = {}
    for path in sorted((root / "recipes").glob("*.json")):
        try:
            key, digest = recipe_key_digest(path.stem, root)
        except (OSError, ValueError, KeyError, TypeError):
            continue
        contents[key] = digest
    return contents


def unserved(
    served: Mapping[str, str],
    authored: Mapping[str, str],
    selected: Callable[[str], bool],
) -> tuple[Unserved, ...]:
    """Selected recipes both sides name whose served content is not the authored content.

    A recipe on one side only is not judged: the catalog lists only recipes this
    Controller can run and leaves out rows it cannot read, so an absence has other
    causes, each shown on its own (catalog skips, catalog diagnostics).
    """
    return tuple(
        Unserved(key, served[key], authored[key])
        for key in sorted(served.keys() & authored.keys())
        if selected(key) and served[key] != authored[key]
    )


def digests(differing: tuple[Unserved, ...]) -> list[dict[str, str]]:
    return [
        {"recipe": u.recipe, "served": u.served, "authored": u.authored}
        for u in differing
    ]


def describe(differing: tuple[Unserved, ...], library_commit: str) -> str:
    shown = "; ".join(
        f"{item.recipe}: served {item.served[:12]}, source {item.authored[:12]}"
        for item in differing[:3]
    )
    more = f" and {len(differing) - 3} more" if len(differing) > 3 else ""
    return (
        f"library {library_commit[:12] or 'unknown'} does not serve this sweep's "
        f"source for {len(differing)} recipe(s) ({shown}{more}); they wait for the "
        "library publish and the Controller's import (`vonkctl recipe sync-status`)"
    )
