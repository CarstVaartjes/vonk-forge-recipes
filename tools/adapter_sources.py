"""Fetch, verify and place the build-context files declared in adapter-sources.json.

Each declared blob is downloaded once into a content-addressed cache, mirror
first and then its upstream origin. Its SHA-256 and size are checked while it
streams in (the only place they are checked): a cache entry exists only after
that check, so a present entry is used as is and a wrong-sized one is fetched
again. Every declared repository path is then written from the cache. A
download that fails is retried a bounded number of times with jittered backoff,
and the tool then fails with each URL's evidence; it never waits unbounded.
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
from typing import BinaryIO

from vonk_forge_contracts.adapter_sources import AdapterSources

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
) -> str:
    """Store one blob in the cache; return "" or what every attempt observed."""
    size = sources.sources[sha256].size_bytes
    evidence: list[str] = []
    for round_number in range(ROUNDS):
        if round_number:
            ceiling = min(BACKOFF_CAP_SECONDS, BACKOFF_BASE_SECONDS * 2**round_number)
            sleep(random.uniform(0, ceiling))
        for url in sources.urls(sha256):
            try:
                refusal = _download(url, sha256, size, cache, opener)
            except (OSError, http.client.HTTPException) as error:
                refusal = f"{type(error).__name__}: {error}"
            if not refusal:
                return ""
            evidence.append(f"{url}: {refusal}")
    return f"sha256:{sha256} ({size} bytes) unavailable: " + "; ".join(evidence)


def materialize(
    root: Path = ROOT,
    *,
    cache: Path | None = None,
    opener: Opener = open_url,
    sleep: Callable[[float], None] = time.sleep,
) -> int:
    """Write every declared path from the verified cache; return the paths written."""
    sources = load(root)
    cache = default_cache() if cache is None else cache
    missing = [
        sha256
        for sha256, source in sources.sources.items()
        if not (cache / sha256).is_file()
        or (cache / sha256).stat().st_size != source.size_bytes
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
    for sha256, source in sources.sources.items():
        for path in source.paths:
            target = root / path
            target.parent.mkdir(parents=True, exist_ok=True)
            descriptor, temporary = tempfile.mkstemp(
                dir=target.parent, prefix=f".{target.name}."
            )
            os.close(descriptor)
            try:
                shutil.copyfile(cache / sha256, temporary)
                os.replace(temporary, target)
            finally:
                Path(temporary).unlink(missing_ok=True)
            written += 1
    return written
