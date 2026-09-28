"""Read exact published recipe releases and safely materialize their packages.

A qualification authority binds a signed GitHub release of this repository by
tag and by the digests of its catalog and qualification indexes. The release
assets are the published bytes: they are read from a local file whose digest
matches (such as the checkout's generated package), from the
``.artifacts/releases/<tag>/`` cache, or downloaded from GitHub, and are never
returned unless they match the expected SHA-256.
"""

from __future__ import annotations

import hashlib
import io
import os
import re
import tarfile
import tempfile
import time
import urllib.error
import urllib.request
from collections.abc import Iterable, Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path, PurePosixPath
from typing import Any

from vonk_forge_contracts import content_sha256

RELEASE_REPOSITORY = "CarstVaartjes/vonk-forge-recipes"
RELEASE_CHECKSUMS = "SHA256SUMS"
RELEASE_CACHE = Path(__file__).resolve().parents[1] / ".artifacts" / "releases"
_TAG = re.compile(r"v[0-9]+\.[0-9]+\.[0-9]+\Z")
_ASSET = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z")
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_CHECKSUM_LINE = re.compile(r"([0-9a-f]{64})  ([A-Za-z0-9][A-Za-z0-9._-]{0,127})\Z")
_PACKAGE_MEDIA_TYPE = "application/vnd.vonk-forge.recipe-package.v2+tar+gzip"
_DOWNLOAD_TIMEOUT_SECONDS = 120
_DOWNLOAD_ATTEMPTS = 4


class ReleaseAssetUnavailable(ValueError):
    """A release asset could not be fetched now; retrying later may succeed."""


def _transient(error: OSError) -> bool:
    if isinstance(error, urllib.error.HTTPError):
        return error.code >= 500 or error.code == 429
    return True  # timeouts, resets and DNS failures


def _download(tag: str, name: str) -> bytes:
    url = f"https://github.com/{RELEASE_REPOSITORY}/releases/download/{tag}/{name}"
    for attempt in range(_DOWNLOAD_ATTEMPTS):
        try:
            with urllib.request.urlopen(
                url, timeout=_DOWNLOAD_TIMEOUT_SECONDS
            ) as response:
                return response.read()
        except OSError as error:
            if not _transient(error) or attempt + 1 == _DOWNLOAD_ATTEMPTS:
                raise ReleaseAssetUnavailable(
                    f"release asset {tag}/{name} is unavailable: {error}"
                ) from error
            time.sleep(2**attempt)
    raise AssertionError("unreachable")


def release_asset(
    tag: str, name: str, sha256: str, *, candidates: Iterable[Path] = ()
) -> bytes:
    """Return the bytes of one release asset, proven by ``sha256``."""

    if (
        _TAG.fullmatch(tag) is None
        or _ASSET.fullmatch(name) is None
        or _SHA256.fullmatch(sha256) is None
    ):
        raise ValueError(f"release asset reference is invalid: {tag}/{name}")
    cached = RELEASE_CACHE / tag / name
    for path in (*candidates, cached):
        try:
            payload = path.read_bytes()
        except OSError:
            continue
        if hashlib.sha256(payload).hexdigest() == sha256:
            return payload
    payload = _download(tag, name)
    if hashlib.sha256(payload).hexdigest() != sha256:
        raise ValueError(f"release asset {tag}/{name} differs from its pinned digest")
    cached.parent.mkdir(parents=True, exist_ok=True)
    temporary = cached.with_name(f".{name}.{os.getpid()}.tmp")
    temporary.write_bytes(payload)
    os.replace(temporary, cached)
    return payload


def release_checksums(release: Mapping[str, Any]) -> dict[str, str]:
    """Return the asset digests of an accepted release record.

    ``release`` is ``qualification/catalog-release.json``: it pins the tag and
    the SHA-256 of that release's signed ``SHA256SUMS``.
    """

    tag, digest = release.get("release_tag"), release.get("sha256sums_sha256")
    if not isinstance(tag, str) or not isinstance(digest, str):
        raise TypeError("accepted release must pin release_tag and sha256sums_sha256")
    text = release_asset(tag, RELEASE_CHECKSUMS, digest).decode("ascii")
    checksums: dict[str, str] = {}
    for line in text.splitlines():
        match = _CHECKSUM_LINE.fullmatch(line)
        if match is None or match.group(2) in checksums:
            raise ValueError(f"release {tag} SHA256SUMS is malformed")
        checksums[match.group(2)] = match.group(1)
    return checksums


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
    payload: bytes,
) -> Iterator[Path]:
    """Yield the source files of verified package bytes for ``recipe``."""

    package_path, payload = _checked_package_payload(package, recipe, payload)
    with _materialized_package_tree(package_path, payload) as tree_root:
        yield tree_root


@contextmanager
def pinned_catalog_package_tree(
    release_tag: str,
    package: Mapping[str, Any],
    recipe: Any,
    *,
    catalog_root: Path | None = None,
) -> Iterator[Path]:
    """Yield source files from the digest-verified package of a release.

    The catalog's ``source_commit`` is never source-byte authority: the exact
    published package bytes are. A generated package under ``catalog_root``
    with the same digest is the same bytes and avoids a download.
    """

    package_path = package.get("path")
    sha256 = package.get("sha256")
    if not isinstance(package_path, str) or not isinstance(sha256, str):
        raise TypeError("published recipe package metadata is invalid")
    if package_path != f"packages/{recipe.identity.slug}.tar.gz":
        raise ValueError("published recipe package path does not match its identity")
    name = PurePosixPath(package_path).name
    candidates = () if catalog_root is None else (catalog_root / "packages" / name,)
    payload = release_asset(release_tag, name, sha256, candidates=candidates)
    with package_tree(package, recipe, payload) as tree_root:
        yield tree_root


@contextmanager
def current_catalog_package_tree(
    catalog_root: Path,
    package: Mapping[str, Any],
    recipe: Any,
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
    with package_tree(package, recipe, local_package.read_bytes()) as tree:
        yield tree
