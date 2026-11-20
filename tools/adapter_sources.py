"""Fetch, verify and place the build-context files declared in adapter-sources.json.

Each declared blob is downloaded once into a content-addressed cache, mirror
first and then its upstream origin. Its SHA-256 and size are checked while it
streams in (the only place they are checked): a cache entry exists only after
that check, so a present entry is used as is and a wrong-sized one is fetched
again. A declared repository path that does not already hold a file of the
declared size is then written from the cache. A failed download is retried for
a fixed number of rounds with jittered backoff, then the tool fails with each
URL's evidence. Each socket operation has a timeout; the total download time of
a slow but live transfer is not capped.

``adapter-sources.json`` declares each blob once by content: the SHA-256 is the
key, the mirror serves it at ``mirror + sha256`` and ``origin`` optionally names
where upstream published the same bytes. One blob may sit at several paths (a
vendor overlay several adapters share). Only repository tooling reads the file;
packages carry the placed bytes, so no Controller or Spark ever sees it.
"""

from __future__ import annotations

import hashlib
import http.client
import os
import random
import shutil
import tempfile
import time
import urllib.request
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from contextlib import AbstractContextManager
from pathlib import Path
from typing import Annotated, BinaryIO

from pydantic import BaseModel, ConfigDict, Field, StrictInt, StrictStr, model_validator
from vonk_forge_contracts.recipe import RelativePath, Sha256

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = "adapter-sources.json"
CACHE_ENVIRONMENT = "VONK_ADAPTER_SOURCE_CACHE"
ROUNDS = 4
BACKOFF_BASE_SECONDS = 1.0
BACKOFF_CAP_SECONDS = 30.0
TIMEOUT_SECONDS = 60
PARALLEL_FETCHES = 4
CHUNK_BYTES = 1 << 20

Opener = Callable[[str], AbstractContextManager[BinaryIO]]
HttpsUrl = Annotated[StrictStr, Field(max_length=2048, pattern=r"^https://\S+$")]


class _Manifest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)


class AdapterSource(_Manifest):
    size_bytes: StrictInt = Field(gt=0)
    origin: HttpsUrl | None = None
    paths: tuple[RelativePath, ...] = Field(min_length=1)


class AdapterSources(_Manifest):
    mirror: HttpsUrl
    sources: dict[Sha256, AdapterSource]

    @model_validator(mode="after")
    def one_content_per_path(self) -> AdapterSources:
        if not self.mirror.endswith("/"):
            raise ValueError("mirror must end with '/'")
        paths = [path for source in self.sources.values() for path in source.paths]
        if len(paths) != len(set(paths)):
            raise ValueError("a path is declared more than once")
        if any(".git" in path.split("/") for path in paths):
            raise ValueError("a declared path is inside .git")
        return self

    def urls(self, sha256: str) -> tuple[str, ...]:
        """Where the blob is fetched from, mirror first."""
        origin = self.sources[sha256].origin
        mirror = f"{self.mirror}{sha256}"
        return (mirror,) if origin is None else (mirror, origin)


class AdapterSourceError(RuntimeError):
    """A declared blob could not be obtained; the message holds every URL's evidence."""


def open_url(url: str) -> AbstractContextManager[BinaryIO]:
    request = urllib.request.Request(url, headers={"User-Agent": "vonk-forge-recipes"})
    return urllib.request.urlopen(request, timeout=TIMEOUT_SECONDS)


def default_cache() -> Path:
    configured = os.environ.get(CACHE_ENVIRONMENT)
    if configured:
        return Path(configured)
    return Path.home() / ".cache" / "vonk-forge-recipes" / "adapter-sources"


def load(root: Path = ROOT) -> AdapterSources:
    return AdapterSources.model_validate_json((root / MANIFEST).read_bytes())


def _download(url: str, sha256: str, size: int, cache: Path, opener: Opener) -> str:
    """Stream ``url`` into the cache; return the refusal reason, or "" when stored."""
    descriptor, temporary = tempfile.mkstemp(dir=cache, prefix=f".{sha256}.")
    try:
        digest = hashlib.sha256()
        received = 0
        with os.fdopen(descriptor, "wb") as output, opener(url) as response:
            while received <= size and (chunk := response.read(CHUNK_BYTES)):
                digest.update(chunk)
                output.write(chunk)
                received += len(chunk)
        if received != size or digest.hexdigest() != sha256:
            return f"got sha256:{digest.hexdigest()} ({received} bytes)"
        os.replace(temporary, cache / sha256)
        return ""
    finally:
        Path(temporary).unlink(missing_ok=True)


def fetch(
    sources: AdapterSources,
    sha256: str,
    cache: Path,
    opener: Opener,
    sleep: Callable[[float], None],
    urls: tuple[str, ...] | None = None,
) -> str:
    """Store one blob in the cache; return "" or what every attempt observed."""
    size = sources.sources[sha256].size_bytes
    urls = sources.urls(sha256) if urls is None else urls
    evidence: list[str] = []
    for round_number in range(ROUNDS):
        if round_number:
            ceiling = min(BACKOFF_CAP_SECONDS, BACKOFF_BASE_SECONDS * 2**round_number)
            sleep(random.uniform(0, ceiling))
        for url in urls:
            try:
                refusal = _download(url, sha256, size, cache, opener)
            except (OSError, http.client.HTTPException) as error:
                refusal = f"{type(error).__name__}: {error}"
            if not refusal:
                return ""
            evidence.append(f"{url}: {refusal}")
    return f"sha256:{sha256} ({size} bytes) unavailable: " + "; ".join(evidence)


def _holds(path: Path, size: int) -> bool:
    """A placed or cached file of the declared size is the declared blob."""
    return path.is_file() and path.stat().st_size == size


def materialize(
    root: Path = ROOT,
    *,
    cache: Path | None = None,
    opener: Opener = open_url,
    sleep: Callable[[float], None] = time.sleep,
) -> int:
    """Place every declared path not yet placed; return how many were written."""
    sources = load(root)
    cache = default_cache() if cache is None else cache
    unplaced = {
        sha256: [
            root / path
            for path in source.paths
            if not _holds(root / path, source.size_bytes)
        ]
        for sha256, source in sources.sources.items()
    }
    missing = [
        sha256
        for sha256, targets in unplaced.items()
        if targets and not _holds(cache / sha256, sources.sources[sha256].size_bytes)
    ]
    if missing:
        cache.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=PARALLEL_FETCHES) as pool:
        outcomes = pool.map(
            lambda sha256: fetch(sources, sha256, cache, opener, sleep), missing
        )
        failures = [outcome for outcome in outcomes if outcome]
    if failures:
        raise AdapterSourceError("\n".join(failures))
    written = 0
    for sha256, targets in unplaced.items():
        for target in targets:
            target.parent.mkdir(parents=True, exist_ok=True)
            _place(root, cache / sha256, target)
            written += 1
    return written


STAGING_PREFIX = ".adapter-source-"


def _place(root: Path, blob: Path, target: Path) -> None:
    """Copy ``blob`` to ``target`` by rename, so no reader sees a partial file.

    The copy is staged at the repository root: on the same filesystem as every
    declared path (so the rename is atomic) and outside every build context (so a
    concurrent package build never lists the staging file). A leftover from a
    crash is ignored by ``.gitignore`` and never read.
    """
    descriptor, staging = tempfile.mkstemp(dir=root, prefix=STAGING_PREFIX)
    try:
        with os.fdopen(descriptor, "wb") as output, blob.open("rb") as source:
            shutil.copyfileobj(source, output)
        os.chmod(staging, 0o644)  # what a checkout writes; mkstemp makes 0600
        os.replace(staging, target)
    finally:
        Path(staging).unlink(missing_ok=True)


def materialize_or_exit(root: Path = ROOT) -> None:
    """Place the declared files before a tool reads the tree; exit with the evidence."""
    try:
        materialize(root)
    except AdapterSourceError as error:
        raise SystemExit(f"adapter sources: {error}") from error
