"""The upstream a command line run reads, and the withdrawn-kit check."""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path
from typing import Any

from recipe_tooling import vendored_upstream
from recipe_tooling.kit_pins.image_scan import scan_download
from recipe_tooling.kit_pins.lock import adapter_of, kit_of, manifest_of
from recipe_tooling.kit_pins.upstream import (
    GitHub,
    HuggingFace,
    Registry,
    Upstream,
    _hub_api,
    _hub_inventory,
    fetch,
)
from recipe_tooling.paths import REPO_ROOT

ROOT = REPO_ROOT


def withdrawn(root: Path, recipe: dict[str, Any]) -> bool:
    """GitHub no longer serves the recipe's kit commit, and the adapter says so.

    The manifest's ``source.unverified`` (tools/check-vendored-upstream) names the kit
    commit: its declaring files cannot be re-read, so the committed lock stands as the
    record instead of failing every check on a creator's deleted repository.
    """
    kit, adapter = kit_of(recipe), adapter_of(root, recipe)
    if kit is None or adapter is None:
        return False
    mark = (manifest_of(adapter).get("source") or {}).get("unverified")
    return mark is not None and mark == vendored_upstream.unverified_mark(*kit)


def cli_source() -> Upstream:
    sys.path.insert(0, str(ROOT / "tools"))
    vendored = vendored_upstream
    token = vendored._token()
    cache = Path(
        os.environ.get(
            "VONK_UPSTREAM_CACHE", Path.home() / ".cache" / "vonk-forge-upstream"
        )
    )

    def api(path: str) -> Any:
        # Commits are immutable, so a fetch by commit sha is cached; refs are not.
        match = re.match(
            r"^repos/([^/]+/[^/]+)/contents/(.+)\?ref=([0-9a-f]{40})$", path
        )
        file = None
        if match:
            file = (
                cache
                / "kit"
                / f"{match.group(1).replace('/', '__')}@{match.group(3)}"
                / (match.group(2).replace("/", "__") + ".json")
            )
            if file.is_file():
                return json.loads(file.read_text())
        document = vendored._api(path, token)
        if file is not None:
            file.parent.mkdir(parents=True, exist_ok=True)
            file.write_text(json.dumps(document, sort_keys=True))
        return document

    def images(repo: str, commit: str, scope: str) -> list[dict[str, Any]]:
        # A commit is immutable, so its scan is cached; the archive itself is not kept.
        name = f"{repo.replace('/', '__')}@{commit}{scope.replace('/', '__')}.json"
        file = cache / "kit-images" / name
        if file.is_file():
            return json.loads(file.read_text())
        found = scan_download(repo, commit, scope)
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_text(json.dumps(found, sort_keys=True))
        return found

    return Upstream(
        GitHub(api),
        HuggingFace(_hub_api, _hub_inventory),
        Registry(fetch),
        images,
    )
