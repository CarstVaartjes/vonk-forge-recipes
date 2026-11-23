"""Declared adapter sources are fetched once by digest, verified at ingress and placed."""

from __future__ import annotations

import hashlib
import io
import json
import threading
import urllib.error
from pathlib import Path

import pytest
from pydantic import ValidationError

from recipe_tooling.adapter_sources import (
    AdapterSourceError,
    AdapterSources,
    materialize,
)

MIRROR = "https://mirror.test/adapter-sources/"
ORIGIN = "https://origin.test/archive.tar.gz"
BLOB = b"creator source archive"
DIGEST = hashlib.sha256(BLOB).hexdigest()
PATHS = ["adapters/a/vendor/kit.tar.gz", "adapters/b/vendor/kit.tar.gz"]


def no_sleep(_: float) -> None:
    pass


class FakeWeb:
    def __init__(
        self,
        responses: dict[str, bytes | OSError],
        barrier: threading.Barrier | None = None,
    ) -> None:
        self.responses = responses
        self.barrier = barrier
        self.requests: list[str] = []

    def __call__(self, url: str) -> io.BytesIO:
        self.requests.append(url)
        if self.barrier is not None:
            # Returns only once every party is in flight at the same time.
            self.barrier.wait()
        response = self.responses[url]
        if isinstance(response, OSError):
            raise response
        return io.BytesIO(response)


def declare(
    root: Path, blobs: dict[bytes, list[str]], origin: str | None = None
) -> None:
    sources = {
        hashlib.sha256(blob).hexdigest(): {
            "size_bytes": len(blob),
            "origin": origin,
            "paths": paths,
        }
        for blob, paths in blobs.items()
    }
    (root / "adapter-sources.json").write_text(
        json.dumps({"mirror": MIRROR, "sources": sources})
    )


def test_shared_blob_is_fetched_once_and_placed_at_every_path(tmp_path: Path) -> None:
    declare(tmp_path, {BLOB: PATHS})
    web = FakeWeb({MIRROR + DIGEST: BLOB})

    assert materialize(tmp_path, cache=tmp_path / "a", opener=web, sleep=no_sleep) == 2
    # A placed tree needs no cache and no network, even with an empty cache.
    assert materialize(tmp_path, cache=tmp_path / "b", opener=web, sleep=no_sleep) == 0

    assert web.requests == [MIRROR + DIGEST]
    assert [(tmp_path / path).read_bytes() for path in PATHS] == [BLOB, BLOB]
    assert [(tmp_path / path).stat().st_mode & 0o777 for path in PATHS] == [0o644] * 2
    assert not list(tmp_path.glob(".adapter-source-*"))  # staged, then renamed


def test_wrong_mirror_bytes_are_refused_and_origin_serves(tmp_path: Path) -> None:
    declare(tmp_path, {BLOB: PATHS}, origin=ORIGIN)
    web = FakeWeb({MIRROR + DIGEST: b"replaced asset", ORIGIN: BLOB})

    materialize(tmp_path, cache=tmp_path / "cache", opener=web, sleep=no_sleep)

    assert web.requests == [MIRROR + DIGEST, ORIGIN]
    assert (tmp_path / PATHS[0]).read_bytes() == BLOB


def test_unavailable_blob_fails_after_four_rounds_with_evidence(tmp_path: Path) -> None:
    declare(tmp_path, {BLOB: PATHS}, origin=ORIGIN)
    web = FakeWeb(
        {MIRROR + DIGEST: urllib.error.URLError("unreachable"), ORIGIN: BLOB + b"tail"}
    )
    sleeps: list[float] = []
    cache = tmp_path / "cache"

    with pytest.raises(AdapterSourceError) as raised:
        materialize(tmp_path, cache=cache, opener=web, sleep=sleeps.append)

    assert len(web.requests) == 8
    assert len(sleeps) == 3
    assert "unreachable" in str(raised.value)
    assert f"({len(BLOB) + 4} bytes)" in str(raised.value)
    assert not (tmp_path / PATHS[0]).exists()
    assert list(cache.iterdir()) == []


def test_missing_blobs_are_fetched_four_at_a_time(tmp_path: Path) -> None:
    blobs = [f"archive {number}".encode() for number in range(4)]
    declare(
        tmp_path, {blob: [f"adapters/{i}/kit.tar.gz"] for i, blob in enumerate(blobs)}
    )
    responses: dict[str, bytes | OSError] = {
        MIRROR + hashlib.sha256(blob).hexdigest(): blob for blob in blobs
    }
    # Serial fetching would leave the first request waiting forever; the timeout
    # breaks the barrier and fails the run instead.
    web = FakeWeb(responses, barrier=threading.Barrier(4, timeout=10))

    assert (
        materialize(tmp_path, cache=tmp_path / "cache", opener=web, sleep=no_sleep) == 4
    )


def test_one_failing_blob_leaves_the_others_cached(tmp_path: Path) -> None:
    other = b"other archive"
    other_url = MIRROR + hashlib.sha256(other).hexdigest()
    declare(tmp_path, {BLOB: PATHS[:1], other: PATHS[1:]})
    cache = tmp_path / "cache"
    failing = FakeWeb(
        {MIRROR + DIGEST: BLOB, other_url: urllib.error.URLError("unreachable")}
    )

    with pytest.raises(AdapterSourceError):
        materialize(tmp_path, cache=cache, opener=failing, sleep=no_sleep)
    recovered = FakeWeb({other_url: other})
    assert materialize(tmp_path, cache=cache, opener=recovered, sleep=no_sleep) == 2

    assert recovered.requests == [other_url]
    assert (tmp_path / PATHS[1]).read_bytes() == other


def test_a_path_has_one_content() -> None:
    other = hashlib.sha256(b"other").hexdigest()
    with pytest.raises(ValidationError):
        AdapterSources.model_validate_json(
            json.dumps(
                {
                    "mirror": MIRROR,
                    "sources": {
                        DIGEST: {"size_bytes": 1, "paths": PATHS[:1]},
                        other: {"size_bytes": 1, "paths": PATHS[:1]},
                    },
                }
            )
        )
