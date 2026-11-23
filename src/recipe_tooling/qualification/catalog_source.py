"""Safely materialize the recipe packages of a generated catalog.

``tools/build-catalog-index`` writes ``catalog-index.json`` and
``packages/<slug>.tar.gz``; the qualification tools read a package only after
its size and SHA-256 match the index metadata.
"""

from __future__ import annotations

import hashlib
import io
import re
import tarfile
import tempfile
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path, PurePosixPath
from typing import Any

_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_PACKAGE_MEDIA_TYPE = "application/vnd.vonk-forge.recipe-package.v2+tar+gzip"


def _checked_package_payload(
    package: Mapping[str, Any], recipe: Any, recipe_sha256: str, payload: bytes
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
    if package.get("recipe_content_sha256") != recipe_sha256:
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
) -> Iterator[Path]:
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
def package_tree(
    package: Mapping[str, Any],
    recipe: Any,
    recipe_sha256: str,
    payload: bytes,
) -> Iterator[Path]:
    """Yield the source files of verified package bytes for ``recipe``.

    ``recipe_sha256`` is the document digest the catalog index records.
    """

    package_path, payload = _checked_package_payload(
        package, recipe, recipe_sha256, payload
    )
    with _materialized_package_tree(package_path, payload) as tree_root:
        yield tree_root


@contextmanager
def current_catalog_package_tree(
    catalog_root: Path,
    package: Mapping[str, Any],
    recipe: Any,
    recipe_sha256: str,
) -> Iterator[Path]:
    """Yield source files from the generated index's digest-verified package."""

    package_path = package.get("path")
    if not isinstance(package_path, str):
        raise TypeError("current recipe package path is invalid")
    if package_path != f"packages/{recipe.identity.slug}.tar.gz":
        raise ValueError("current recipe package path does not match its identity")
    relative_path = PurePosixPath(package_path)
    local_package = catalog_root
    for part in relative_path.parts:
        local_package = local_package / part
        if local_package.is_symlink():
            raise ValueError(
                f"current recipe package path traverses a symlink: {package_path}"
            )
    if not local_package.is_file():
        raise ValueError(
            f"current recipe package is not a regular file: {package_path}; "
            "run tools/build-catalog-index"
        )
    with package_tree(
        package, recipe, recipe_sha256, local_package.read_bytes()
    ) as tree:
        yield tree
