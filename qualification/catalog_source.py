"""Read exact published recipe packages and safely materialize their inputs."""

from __future__ import annotations

import hashlib
import importlib.machinery
import importlib.util
import io
import os
import re
import subprocess
import tarfile
import tempfile
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from functools import lru_cache
from pathlib import Path, PurePosixPath
from typing import Any

from vonk_forge_contracts import content_sha256

_COMMIT = re.compile(r"[0-9a-f]{40}\Z")
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_GIT_OBJECT_TIMEOUT_SECONDS = 10
_PACKAGE_MEDIA_TYPE = "application/vnd.vonk-forge.recipe-package.v2+tar+gzip"


def pinned_git_blob(repository_root: Path, commit: str, path: str) -> bytes:
    """Read one immutable Git blob after validating its commit and path."""

    if _COMMIT.fullmatch(commit) is None:
        raise ValueError("catalog commit must be a full Git SHA")
    relative_path = PurePosixPath(path)
    if (
        not path
        or relative_path.as_posix() != path
        or relative_path.is_absolute()
        or any(part in {"", ".", ".."} for part in relative_path.parts)
    ):
        raise ValueError(f"catalog blob path is not repository-relative: {path}")
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
    object_name = f"{commit}:{relative_path.as_posix()}"
    try:
        result = subprocess.run(
            ["git", "--no-replace-objects", "cat-file", "blob", object_name],
            cwd=repository_root,
            env=environment,
            check=False,
            capture_output=True,
            timeout=_GIT_OBJECT_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired as error:
        raise ValueError(
            f"timed out reading catalog blob {object_name} after "
            f"{_GIT_OBJECT_TIMEOUT_SECONDS} seconds"
        ) from error
    except OSError as error:
        raise ValueError(f"cannot read catalog blob {object_name}: {error}") from error
    if result.returncode != 0:
        detail = result.stderr.decode("utf-8", errors="replace").strip()
        raise ValueError(f"cannot read catalog blob {object_name}: {detail}")
    return result.stdout


@lru_cache(maxsize=1)
def _canonical_package_validator() -> Any:
    """Load the recipe package validator owned by the catalog producer."""

    repository_root = Path(__file__).resolve().parents[1]
    source = repository_root / "tools/build-catalog-index"
    loader = importlib.machinery.SourceFileLoader(
        "vonk_qualification_catalog_index", str(source)
    )
    spec = importlib.util.spec_from_loader(loader.name, loader)
    if spec is None:
        raise ValueError("cannot load the canonical catalog package validator")
    module = importlib.util.module_from_spec(spec)
    loader.exec_module(module)
    validator = getattr(module, "validate_recipe_archive", None)
    if not callable(validator):
        raise TypeError("canonical catalog package validator is unavailable")
    return validator


def _checked_package_payload(
    package: Mapping[str, Any], recipe: Any, payload: bytes
) -> tuple[str, bytes]:
    package_path = package.get("path")
    if not isinstance(package_path, str):
        raise TypeError("catalog recipe package path must be a string")
    if package_path != f"packages/{recipe.identity.slug}.tar.gz":
        raise ValueError("published recipe package path does not match its identity")
    expected_bytes = package.get("expected_bytes")
    package_sha256 = package.get("sha256")
    if (
        type(expected_bytes) is not int
        or expected_bytes <= 0
        or not isinstance(package_sha256, str)
        or _SHA256.fullmatch(package_sha256) is None
        or package.get("media_type") != _PACKAGE_MEDIA_TYPE
        or package.get("minimum_consumer_schema") != 2
    ):
        raise ValueError("published recipe package metadata is invalid")
    if package.get("recipe_content_sha256") != content_sha256(recipe):
        raise ValueError("catalog package recipe digest differs from its Recipe")

    if len(payload) != expected_bytes:
        raise ValueError(
            f"catalog recipe package size differs from its metadata: {package_path}"
        )
    if hashlib.sha256(payload).hexdigest() != package_sha256:
        raise ValueError(
            f"catalog recipe package digest differs from its metadata: {package_path}"
        )
    return package_path, payload


@contextmanager
def _materialized_package_tree(
    package_path: str,
    payload: bytes,
    recipe: Any,
    entities: dict[str, dict[str, Any]],
) -> Iterator[Path]:
    try:
        _canonical_package_validator()(
            payload,
            recipe.model_dump(mode="json", exclude_unset=False, exclude_none=False),
            entities,
        )
    except SystemExit as error:
        raise ValueError(
            f"catalog recipe package failed canonical validation: {error}"
        ) from error

    with tempfile.TemporaryDirectory(prefix="vonk-qualification-package-") as temp:
        tree_root = Path(temp) / "tree"
        tree_root.mkdir()
        try:
            with tarfile.open(fileobj=io.BytesIO(payload), mode="r:gz") as archive:
                archive.extractall(path=tree_root, filter="data")
        except (OSError, tarfile.TarError, ValueError) as error:
            raise ValueError(
                f"published recipe package archive is invalid: {package_path}"
            ) from error
        yield tree_root


@contextmanager
def pinned_catalog_package_tree(
    repository_root: Path,
    catalog_commit: str,
    package: Mapping[str, Any],
    recipe: Any,
    entities: dict[str, dict[str, Any]],
) -> Iterator[Path]:
    """Yield source files from the digest-verified package blob in a catalog.

    The catalog's ``source_commit`` can be stale or inconsistent with a released
    archive. Never use it as source-byte authority: validate the exact package
    bytes pinned at ``catalog_commit`` and hash the selected regular members.
    """

    package_path = package.get("path")
    if not isinstance(package_path, str):
        raise TypeError("published recipe package path is invalid")
    if package_path != f"packages/{recipe.identity.slug}.tar.gz":
        raise ValueError("published recipe package path does not match its identity")
    payload = pinned_git_blob(repository_root, catalog_commit, package_path)
    package_path, payload = _checked_package_payload(package, recipe, payload)
    with _materialized_package_tree(
        package_path, payload, recipe, entities
    ) as tree_root:
        yield tree_root


@contextmanager
def current_catalog_package_tree(
    repository_root: Path,
    package: Mapping[str, Any],
    recipe: Any,
    entities: dict[str, dict[str, Any]],
) -> Iterator[Path]:
    """Yield source files from the current index's digest-verified package."""

    package_path = package.get("path")
    if not isinstance(package_path, str):
        raise TypeError("current recipe package path is invalid")
    if package_path != f"packages/{recipe.identity.slug}.tar.gz":
        raise ValueError("current recipe package path does not match its identity")
    relative_path = PurePosixPath(package_path)
    if (
        relative_path.as_posix() != package_path
        or relative_path.is_absolute()
        or any(part in {"", ".", ".."} for part in relative_path.parts)
    ):
        raise ValueError(f"current recipe package path is unsafe: {package_path}")
    local_package = repository_root
    for part in relative_path.parts:
        local_package = local_package / part
        if local_package.is_symlink():
            raise ValueError(
                f"current recipe package path traverses a symlink: {package_path}"
            )
    if not local_package.is_file():
        raise ValueError(
            f"current recipe package is not a regular file: {package_path}"
        )
    payload = local_package.read_bytes()
    package_path, payload = _checked_package_payload(package, recipe, payload)
    with _materialized_package_tree(
        package_path, payload, recipe, entities
    ) as tree_root:
        yield tree_root
