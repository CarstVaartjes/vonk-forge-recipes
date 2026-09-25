"""Exact source identities for recipe execution stacks.

The catalog package digest is recipe-specific and includes model metadata. This
module derives the separate build/source identity used for stack grouping by
hashing the complete declared runtime/build settings and the bytes of every
regular file selected by the build context, Dockerfile, and patches.
"""

from __future__ import annotations

import hashlib
import json
import stat
from pathlib import Path, PurePosixPath
from typing import Any


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _repository_path(repository_root: Path, relative_name: str) -> Path:
    relative = PurePosixPath(relative_name)
    if relative.is_absolute() or any(
        part in {"", ".", ".."} for part in relative.parts
    ):
        raise ValueError(
            f"build source path is not repository-relative: {relative_name}"
        )
    path = repository_root
    for part in relative.parts:
        path = path / part
        if path.is_symlink():
            raise ValueError(
                f"build source must not traverse a symlink: {relative_name}"
            )
    return path


def _file_record(repository_root: Path, path: Path) -> dict[str, str]:
    if path.is_symlink():
        raise ValueError(
            "selected build source must not be a symlink: "
            f"{path.relative_to(repository_root).as_posix()}"
        )
    mode = stat.S_IMODE(path.stat().st_mode)
    return {
        "path": path.relative_to(repository_root).as_posix(),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "mode": f"{mode:04o}",
    }


def _build_sources(repository_root: Path, build: Any) -> list[dict[str, str]]:
    """Return sorted path/content digests for the complete selected build closure."""

    context_name = build.context.path
    context = _repository_path(repository_root, context_name)
    if not context.is_dir():
        raise ValueError(f"declared build context is not a directory: {context_name}")

    files: dict[str, Path] = {}
    for path in context.rglob("*"):
        if path.is_symlink():
            raise ValueError(
                "build context must not contain selected symlinks: "
                f"{path.relative_to(repository_root).as_posix()}"
            )
        if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc":
            files[path.relative_to(repository_root).as_posix()] = path

    for relative_name in (
        build.dockerfile,
        *(patch.path for patch in build.patches),
    ):
        path = _repository_path(repository_root, relative_name)
        if path.is_symlink() or not path.is_file():
            raise ValueError(
                f"declared build input is not a regular file: {relative_name}"
            )
        files[PurePosixPath(relative_name).as_posix()] = path

    return [_file_record(repository_root, path) for _, path in sorted(files.items())]


def execution_stack_identity(
    repository_root: Path, recipe: Any, build: Any
) -> dict[str, Any]:
    """Bind execution settings and source bytes into stable full-length hashes."""

    build_sources = _build_sources(repository_root, build)
    build_document = build.model_dump(mode="json", exclude_none=False)
    runtime_document = recipe.runtime.model_dump(mode="json", exclude_none=False)
    source_digest = _canonical_sha256(
        {
            "context_path": build.context.path,
            "files": build_sources,
        }
    )
    stack_digest = _canonical_sha256(
        {
            "runtime": runtime_document,
            "build": build_document,
            "build_source_sha256": source_digest,
        }
    )
    topology_digest = _canonical_sha256(
        {
            "topology": recipe.topology.model_dump(mode="json", exclude_none=False),
            "failure_policy": (
                None
                if recipe.runtime.lifecycle.failure is None
                else recipe.runtime.lifecycle.failure.model_dump(
                    mode="json", exclude_none=False
                )
            ),
        }
    )
    return {
        "runtime_stack_sha256": stack_digest,
        "build_source_sha256": source_digest,
        "topology_sha256": topology_digest,
        "build_source_file_count": len(build_sources),
    }
