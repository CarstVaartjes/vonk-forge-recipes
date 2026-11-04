"""Pull requests this repository's own tooling may act on.

`gh pr list --head <branch>` and label listings match by branch name or label
only, and the repository is public: a fork can open a PR from a branch of the
same name. Fail closed here, the one place where ownership decides whether we
merge, close or arm a PR: anything without the fields below is not ours.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

OWNER = "CarstVaartjes"
#: Add to the `--json` fields of every `gh pr list`; `own_pulls` needs them.
OWN_FIELDS = "isCrossRepository,headRepositoryOwner"


def own_pulls(pulls: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep only PRs whose head branch lives in this repository."""
    return [
        pr
        for pr in pulls
        if pr.get("isCrossRepository") is False
        and (pr.get("headRepositoryOwner") or {}).get("login") == OWNER
    ]
