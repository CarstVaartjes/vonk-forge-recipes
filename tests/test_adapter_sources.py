"""Declared adapter sources are fetched once by digest, verified at ingress and placed."""

from __future__ import annotations

import hashlib
import io
import json
import urllib.error
from pathlib import Path

import pytest
from pydantic import ValidationError
from vonk_forge_contracts.adapter_sources import AdapterSources

from tools.adapter_sources import ROOT, ROUNDS, AdapterSourceError, load, materialize

MIRROR = "https://mirror.test/adapter-sources/"
ORIGIN = "https://origin.test/archive.tar.gz"
BLOB = b"creator source archive"
DIGEST = hashlib.sha256(BLOB).hexdigest()
PATHS = ["adapters/a/vendor/kit.tar.gz", "adapters/b/vendor/kit.tar.gz"]


class FakeWeb:
    def __init__(self, responses: dict[str, bytes | OSError]) -> None:
        self.responses = responses
        self.requests: list[str] = []

    def __call__(self, url: str) -> io.BytesIO:
        self.requests.append(url)
        response = self.responses[url]
        if isinstance(response, OSError):
            raise response
        return io.BytesIO(response)


def declare(root: Path) -> None:
    (root / "adapter-sources.json").write_text(
        json.dumps(
            {
                "mirror": MIRROR,
                "sources": {
                    DIGEST: {"size_bytes": len(BLOB), "origin": ORIGIN, "paths": PATHS}
                },
            }
        )
    )


def test_shared_blob_is_fetched_once_and_placed_at_every_path(tmp_path: Path) -> None:
    declare(tmp_path)
    web = FakeWeb({MIRROR + DIGEST: BLOB})
    cache = tmp_path / "cache"

    assert materialize(tmp_path, cache=cache, opener=web, sleep=lambda _: None) == 2
    assert materialize(tmp_path, cache=cache, opener=web, sleep=lambda _: None) == 2

    assert web.requests == [MIRROR + DIGEST]
    assert [(tmp_path / path).read_bytes() for path in PATHS] == [BLOB, BLOB]


def test_wrong_mirror_bytes_are_refused_and_origin_serves(tmp_path: Path) -> None:
    declare(tmp_path)
    web = FakeWeb({MIRROR + DIGEST: b"replaced asset", ORIGIN: BLOB})

    materialize(tmp_path, cache=tmp_path / "cache", opener=web, sleep=lambda _: None)

    assert web.requests == [MIRROR + DIGEST, ORIGIN]
    assert (tmp_path / PATHS[0]).read_bytes() == BLOB


def test_unavailable_blob_fails_bounded_with_evidence(tmp_path: Path) -> None:
    declare(tmp_path)
    web = FakeWeb(
        {
            MIRROR + DIGEST: urllib.error.URLError("unreachable"),
            ORIGIN: BLOB + b"tail",
        }
    )
    sleeps: list[float] = []
    cache = tmp_path / "cache"

    with pytest.raises(AdapterSourceError) as raised:
        materialize(tmp_path, cache=cache, opener=web, sleep=sleeps.append)

    assert len(web.requests) == 2 * ROUNDS and len(sleeps) == ROUNDS - 1
    assert "unreachable" in str(raised.value)
    assert f"{len(BLOB) + 4} bytes" in str(raised.value)
    assert not (tmp_path / PATHS[0]).exists() and list(cache.iterdir()) == []


def test_a_path_has_one_content() -> None:
    other = hashlib.sha256(b"other").hexdigest()
    with pytest.raises(ValidationError, match="more than once"):
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


def test_repository_manifest_is_valid() -> None:
    assert load(ROOT).mirror.startswith("https://")
