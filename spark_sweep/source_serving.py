"""Whether the Controller's catalog serves the recipe content this sweep's source authored.

A sweep is started from a checkout of recipes main, often right after a merge. The
Controller serves that content only once the publish workflow has released it and the
Controller has imported the release, minutes later. Until then the catalog still serves
the older content, whose verdicts are already recorded: a run that trusts it finds
nothing to do and ends having tested nothing.

The comparison is by content identity: the ``document_sha256`` the library publishes for
each recipe document, hashed once here from the authored file. A commit says nothing
about content (a docs-only commit publishes nothing and changes no recipe).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from qualification.documents import recipe, recipe_digest


class SourceServing(StrEnum):
    """One sweep process's view of its source in the catalog (never persisted)."""

    UNREAD = "unread"  # no complete library pass yet
    UNSERVED = "unserved"  # a complete pass serves other content for a selected recipe
    SERVED = "served"  # a complete pass served the source; later library changes are followed


# The edges. SERVED is final for the process: after the catalog has caught up with the
# source, a newer library (a merge during a long sweep) is tested as the catalog serves it.
EDGES: Mapping[SourceServing, frozenset[SourceServing]] = {
    SourceServing.UNREAD: frozenset(
        {SourceServing.UNREAD, SourceServing.UNSERVED, SourceServing.SERVED}
    ),
    SourceServing.UNSERVED: frozenset({SourceServing.UNSERVED, SourceServing.SERVED}),
    SourceServing.SERVED: frozenset({SourceServing.SERVED}),
}


@dataclass(frozen=True)
class Unserved:
    recipe: str
    served: str
    authored: str


def authored_contents(root: Path) -> dict[str, str]:
    """``publisher/slug`` to the content digest of each authored recipe, read once at start."""
    contents: dict[str, str] = {}
    for path in sorted((root / "recipes").glob("*.json")):
        identity = recipe(path.stem, root).identity
        contents[f"{identity.publisher}/{identity.slug}"] = recipe_digest(
            path.stem, root
        )
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


def advance(
    state: SourceServing, *, complete_pass: bool, differing: tuple[Unserved, ...]
) -> SourceServing:
    """The one transition: only a complete library pass decides."""
    if not complete_pass:
        return state
    target = SourceServing.UNSERVED if differing else SourceServing.SERVED
    return target if target in EDGES[state] else state


def describe(differing: tuple[Unserved, ...], library_commit: str) -> str:
    shown = "; ".join(
        f"{item.recipe}: served {item.served[:12]}, source {item.authored[:12]}"
        for item in differing[:3]
    )
    more = f" and {len(differing) - 3} more" if len(differing) > 3 else ""
    return (
        f"WAITING FOR THE CATALOG: library {library_commit[:12] or 'unknown'} does not "
        f"serve this sweep's source for {len(differing)} recipe(s) ({shown}{more}); "
        "these wait for the library publish and the Controller's import "
        "(`vonkctl recipe sync-status`). If the catalog is the newer one, update "
        "the sweep's source checkout."
    )
