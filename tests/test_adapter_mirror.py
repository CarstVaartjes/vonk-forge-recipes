"""The mirror is written before it is declared, and a lost blob is healed from origin."""

from __future__ import annotations

import hashlib
import io
import json
import subprocess
from pathlib import Path

import pytest

from tools.adapter_mirror import add, reconcile
from tools.adapter_sources import AdapterSourceError, load

MIRROR = "https://github.com/owner/repo/releases/download/adapter-sources/"
ORIGIN = "https://origin.test/kit.tar.gz"
BLOB = b"creator source archive"
DIGEST = hashlib.sha256(BLOB).hexdigest()


class FakeGh:
    def __init__(self, root: Path, assets: dict[str, str | None]) -> None:
        self.root = root
        self.assets = assets
        self.calls: list[list[str]] = []
        self.upload_failures = 0
        self.fresh = False

    def __call__(self, arguments: list[str]) -> str:
        self.calls.append(arguments)
        if arguments[1] == "view":
            listing = [{"name": n, "digest": d} for n, d in self.assets.items()]
            return json.dumps({"assets": listing})
        if self.upload_failures:
            self.upload_failures -= 1
            raise subprocess.CalledProcessError(1, "gh", stderr="boom")
        # The manifest must not name a blob before its asset exists.
        if self.fresh:
            assert DIGEST not in (self.root / "adapter-sources.json").read_text()
        name = Path(arguments[3]).name
        self.assets[name] = f"sha256:{name}"
        return ""

    @property
    def uploads(self) -> list[str]:
        return [c[3] for c in self.calls if c[1] == "upload"]


def repo(tmp_path: Path, declared: dict | None = None) -> Path:
    (tmp_path / "adapters/a").mkdir(parents=True, exist_ok=True)
    for name in ("one", "two"):
        (tmp_path / f"adapters/a/{name}.tar.gz").write_bytes(BLOB)
    document = {"mirror": MIRROR, "sources": declared or {}}
    (tmp_path / "adapter-sources.json").write_text(json.dumps(document))
    return tmp_path


def test_add_uploads_once_then_declares_every_path(tmp_path: Path) -> None:
    root = repo(tmp_path)
    gh = FakeGh(root, {})
    gh.fresh = True
    paths = ["adapters/a/one.tar.gz", "adapters/a/two.tar.gz"]

    assert add(paths, root=root, origin=ORIGIN, cache=tmp_path / "c", gh=gh) == [DIGEST]
    assert add(paths, root=root, cache=tmp_path / "c", gh=gh) == []

    entry = load(root).sources[DIGEST]
    assert (entry.size_bytes, entry.origin, entry.paths) == (
        len(BLOB),
        ORIGIN,
        tuple(paths),
    )
    assert [Path(u).name for u in gh.uploads] == [DIGEST]


def test_add_moves_a_path_to_its_newest_content(tmp_path: Path) -> None:
    root = repo(tmp_path)
    gh = FakeGh(root, {})
    add(["adapters/a/one.tar.gz"], root=root, cache=tmp_path / "c", gh=gh)
    (root / "adapters/a/one.tar.gz").write_bytes(b"newer")
    add(["adapters/a/one.tar.gz"], root=root, cache=tmp_path / "c", gh=gh)

    assert [e.paths for e in load(root).sources.values()] == [
        ("adapters/a/one.tar.gz",)
    ]


def test_failed_upload_leaves_the_manifest_untouched(tmp_path: Path) -> None:
    root = repo(tmp_path)
    gh = FakeGh(root, {})
    gh.upload_failures = 3

    with pytest.raises(AdapterSourceError, match="boom"):
        add(["adapters/a/one.tar.gz"], root=root, cache=tmp_path / "c", gh=gh)
    assert load(root).sources == {}


def test_tampered_asset_is_refused_not_replaced(tmp_path: Path) -> None:
    root = repo(tmp_path)
    gh = FakeGh(root, {DIGEST: "sha256:" + "0" * 64})

    with pytest.raises(AdapterSourceError, match="reports sha256:0"):
        add(["adapters/a/one.tar.gz"], root=root, cache=tmp_path / "c", gh=gh)
    assert gh.uploads == []


def declared(origin: str | None) -> dict:
    return {
        DIGEST: {
            "size_bytes": len(BLOB),
            "origin": origin,
            "paths": ["adapters/a/one.tar.gz"],
        }
    }


def test_reconcile_heals_a_lost_blob_from_origin_after_a_retry(tmp_path: Path) -> None:
    root = repo(tmp_path, declared(ORIGIN))
    gh = FakeGh(root, {})
    answers = [OSError("down"), BLOB, BLOB]

    def opener(url: str) -> io.BytesIO:
        assert url == ORIGIN
        answer = answers.pop(0)
        if isinstance(answer, OSError):
            raise answer
        return io.BytesIO(answer)

    problems = reconcile(
        root, cache=tmp_path / "c", opener=opener, sleep=lambda _: None, gh=gh
    )

    assert problems == [] and DIGEST in gh.assets


def test_reconcile_reports_what_it_cannot_heal_and_never_raises(tmp_path: Path) -> None:
    root = repo(tmp_path, declared(None))
    gh = FakeGh(root, {})
    assert "no origin" in reconcile(root, cache=tmp_path / "c", gh=gh)[0]

    root = repo(tmp_path, declared(ORIGIN))

    def wrong(_: str) -> io.BytesIO:
        return io.BytesIO(b"other bytes")

    problems = reconcile(
        root, cache=tmp_path / "c", opener=wrong, sleep=lambda _: None, gh=gh
    )
    assert "sha256:" in problems[0] and ORIGIN in problems[0] and gh.uploads == []


def test_reconcile_skips_present_assets_without_any_download(tmp_path: Path) -> None:
    root = repo(tmp_path, declared(ORIGIN))
    gh = FakeGh(root, {DIGEST: f"sha256:{DIGEST}"})

    assert reconcile(root, cache=tmp_path / "c", opener=None, gh=gh) == []  # type: ignore[arg-type]
