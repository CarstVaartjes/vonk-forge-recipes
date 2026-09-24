"""Materialize the build inputs from a published catalog's source commit."""

from __future__ import annotations

import os
import re
import subprocess
import tarfile
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path, PurePosixPath
from typing import Any

_COMMIT = re.compile(r"[0-9a-f]{40}\Z")
_ARCHIVE_TIMEOUT_SECONDS = 120


def _archive_roots(recipes: list[Any]) -> list[str]:
    paths: set[str] = set()
    for recipe in recipes:
        if recipe.execution.mode != "build":
            raise ValueError(
                "qualification stack identity requires a build execution: "
                f"{recipe.identity.publisher}/{recipe.identity.slug}"
            )
        build = recipe.execution.build
        paths.update(
            (
                build.context.path,
                build.dockerfile,
                *(patch.path for patch in build.patches),
            )
        )

    normalized: list[str] = []
    for value in sorted(paths, key=lambda item: (item.count("/"), item)):
        path = PurePosixPath(value)
        if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
            raise ValueError(
                f"published build input is not repository-relative: {value}"
            )
        if not any(
            value == parent or value.startswith(parent.rstrip("/") + "/")
            for parent in normalized
        ):
            normalized.append(value)
    return normalized


@contextmanager
def pinned_catalog_source_tree(
    repository_root: Path, source_commit: str, recipes: list[Any]
) -> Iterator[Path]:
    """Yield an isolated source tree containing only declared catalog build inputs.

    The working checkout may contain source edits newer than the accepted catalog
    release. Hashing an archive of the catalog's pinned ``source_commit`` keeps
    runtime-stack identities tied to the published source bytes.
    """

    if _COMMIT.fullmatch(source_commit) is None:
        raise ValueError("catalog source_commit must be a full Git SHA")
    roots = _archive_roots(recipes)
    if not roots:
        raise ValueError("published catalog has no build input roots")

    environment = os.environ.copy()
    for name in (
        "GIT_DIR",
        "GIT_WORK_TREE",
        "GIT_INDEX_FILE",
        "GIT_COMMON_DIR",
        "GIT_OBJECT_DIRECTORY",
        "GIT_ALTERNATE_OBJECT_DIRECTORIES",
    ):
        environment.pop(name, None)
    environment.update(
        {
            "GIT_NO_REPLACE_OBJECTS": "1",
            "GIT_NO_LAZY_FETCH": "1",
            "GIT_OPTIONAL_LOCKS": "0",
            "GIT_TERMINAL_PROMPT": "0",
        }
    )

    with tempfile.TemporaryDirectory(prefix="vonk-qualification-source-") as temp:
        temporary_root = Path(temp)
        archive_path = temporary_root / "published-source.tar"
        tree_root = temporary_root / "tree"
        tree_root.mkdir()
        try:
            with archive_path.open("wb") as archive_file:
                subprocess.run(
                    [
                        "git",
                        "--no-replace-objects",
                        "archive",
                        "--format=tar",
                        source_commit,
                        "--",
                        *roots,
                    ],
                    cwd=repository_root,
                    env=environment,
                    check=True,
                    stdout=archive_file,
                    stderr=subprocess.PIPE,
                    timeout=_ARCHIVE_TIMEOUT_SECONDS,
                )
        except subprocess.TimeoutExpired as error:
            raise ValueError(
                "timed out materializing published build inputs from "
                f"{source_commit} after {_ARCHIVE_TIMEOUT_SECONDS} seconds"
            ) from error
        except subprocess.CalledProcessError as error:
            detail = (error.stderr or b"").decode("utf-8", errors="replace").strip()
            raise ValueError(
                f"cannot read published build inputs from {source_commit}: {detail}"
            ) from error
        except OSError as error:
            raise ValueError(
                f"cannot archive published build inputs: {error}"
            ) from error

        try:
            with tarfile.open(archive_path, mode="r:") as archive:
                archive.extractall(path=tree_root, filter="data")
        except (OSError, tarfile.TarError, ValueError) as error:
            raise ValueError(
                f"published source archive {source_commit} is invalid: {error}"
            ) from error
        yield tree_root
