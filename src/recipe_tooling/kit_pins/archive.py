"""The base archive a lock mirrors and declares."""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable
from pathlib import Path
from typing import Any

from recipe_tooling import adapter_mirror
from recipe_tooling.adapter_sources import AdapterSourceError, load
from recipe_tooling.kit_pins.model import (
    ARCHIVE_ROUNDS,
    FETCH_ERRORS,
    MAX_ARCHIVE_BYTES,
    KitError,
)
from recipe_tooling.upstream_http import fetch as fetch_url

# Writing the lock


def archive_url(repo: str, commit: str) -> str:
    return f"https://codeload.github.com/{repo}/tar.gz/{commit}"


def declare_archive(root: Path, archive: Path, origin: str) -> str:
    """Mirror a new vendored archive and declare it; return its SHA-256.

    The file stays in the working tree for the tools that read it; the manifest's
    generated ignore block keeps it out of commits. On failure the file is removed,
    so a rerun fetches and declares it again instead of finishing with an
    undeclared archive.
    """

    relative = archive.relative_to(root).as_posix()
    try:
        adapter_mirror.add([relative], root=root, origin=origin)
        return next(
            sha256
            for sha256, source in load(root).sources.items()
            if relative in source.paths
        )
    except (AdapterSourceError, OSError, ValueError, StopIteration) as error:
        archive.unlink(missing_ok=True)
        raise KitError(
            f"cannot mirror {archive.name}: {error}; "
            "rerun `tools/kit-pins lock <recipe>`"
        ) from error


def download_archive(
    repo: str, commit: str, sleep: Callable[[float], None] = time.sleep
) -> bytes:
    url = archive_url(repo, commit)
    request = urllib.request.Request(url, headers={"User-Agent": "vonk-forge-recipes"})
    for attempt in range(ARCHIVE_ROUNDS):
        try:
            data = fetch_url(
                request,
                lambda response: response.read(MAX_ARCHIVE_BYTES + 1),
                timeout=120,
            )
            break
        except FETCH_ERRORS as error:
            if attempt == ARCHIVE_ROUNDS - 1:
                raise KitError(f"cannot fetch {url}: {error}") from error
            sleep(2.0**attempt)
    else:
        raise KitError(f"cannot fetch {url}: no attempt was made")
    if len(data) > MAX_ARCHIVE_BYTES:
        raise KitError(f"{url} is larger than {MAX_ARCHIVE_BYTES} bytes")
    return data


def recipe_text(recipe: dict[str, Any]) -> str:
    return json.dumps(recipe, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
