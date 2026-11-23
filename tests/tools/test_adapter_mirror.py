"""The mirror is written before it is declared, and a lost blob is healed from origin."""

from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
from typing import Any

import pytest
from repo_root import ROOT

from recipe_tooling.adapter_mirror import (
    IGNORE_BEGIN,
    IGNORE_END,
    _ignore_line,
    add,
    reconcile,
    undeclare,
)
from recipe_tooling.adapter_sources import AdapterSourceError, load

MIRROR = "https://github.com/owner/repo/releases/download/adapter-sources/"
ORIGIN = "https://origin.test/kit.tar.gz"
BLOB = b"creator source archive"
DIGEST = hashlib.sha256(BLOB).hexdigest()


class FakeGh:
    def __init__(self, root: Path, assets: dict[str, str | None]) -> None:
        # name -> digest; a value of "partial" leaves a non-uploaded asset
        self.root = root
        self.assets = assets
        self.calls: list[list[str]] = []
        self.upload_failures = 0
        self.fresh = False

    def __call__(self, arguments: list[str]) -> str:
        self.calls.append(arguments)
        if arguments[1] == "view":
            listing = [
                {"name": n, "digest": d, "state": "uploaded"}
                if d != "partial"
                else {"name": n, "digest": None, "state": "starter"}
                for n, d in self.assets.items()
            ]
            return json.dumps({"assets": listing})
        if self.upload_failures:
            self.upload_failures -= 1
            raise AdapterSourceError("boom")
        # The manifest must not name a blob before its asset exists.
        if self.fresh:
            assert DIGEST not in (self.root / "adapter-sources.json").read_text()
        name = Path(arguments[3]).name
        self.assets[name] = f"sha256:{name}"
        return ""

    @property
    def uploads(self) -> list[str]:
        return [c[3] for c in self.calls if c[1] == "upload"]


def repo(tmp_path: Path, declared: dict[str, Any] | None = None) -> Path:
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


def declared(origin: str | None) -> dict[str, Any]:
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

    assert problems == []
    assert DIGEST in gh.assets


def test_reconcile_reports_a_blob_with_no_origin(tmp_path: Path) -> None:
    root = repo(tmp_path, declared(None))

    problems = reconcile(root, cache=tmp_path / "c", gh=FakeGh(root, {}))

    assert "no origin" in problems[0]


def test_reconcile_heals_a_no_origin_blob_from_the_restored_cache(
    tmp_path: Path,
) -> None:
    root = repo(tmp_path, declared(None))
    (tmp_path / "c").mkdir()
    (tmp_path / "c" / DIGEST).write_bytes(BLOB)
    gh = FakeGh(root, {})

    assert reconcile(root, cache=tmp_path / "c", gh=gh) == []
    assert DIGEST in gh.assets


def test_reconcile_refuses_wrong_origin_bytes(tmp_path: Path) -> None:
    root = repo(tmp_path, declared(ORIGIN))
    gh = FakeGh(root, {})

    def wrong(_: str) -> io.BytesIO:
        return io.BytesIO(b"other bytes")

    problems = reconcile(
        root, cache=tmp_path / "c", opener=wrong, sleep=lambda _: None, gh=gh
    )

    assert ORIGIN in problems[0]
    assert gh.uploads == []


def test_reconcile_reports_a_failed_upload(tmp_path: Path) -> None:
    root = repo(tmp_path, declared(ORIGIN))
    gh = FakeGh(root, {})
    gh.upload_failures = 3

    problems = reconcile(
        root,
        cache=tmp_path / "c",
        opener=lambda _: io.BytesIO(BLOB),
        sleep=lambda _: None,
        gh=gh,
    )

    assert "upload of" in problems[0]
    assert "boom" in problems[0]


def test_reconcile_reports_a_half_uploaded_asset_as_missing(tmp_path: Path) -> None:
    root = repo(tmp_path, declared(None))

    problems = reconcile(
        root, cache=tmp_path / "c", gh=FakeGh(root, {DIGEST: "partial"})
    )

    assert "no origin" in problems[0]


def test_add_reuploads_a_half_uploaded_asset(tmp_path: Path) -> None:
    root = repo(tmp_path)
    gh = FakeGh(root, {DIGEST: "partial"})

    assert add(["adapters/a/one.tar.gz"], root=root, cache=tmp_path / "c", gh=gh) == [
        DIGEST
    ]


def test_a_bad_asset_under_another_digest_does_not_block_an_add(
    tmp_path: Path,
) -> None:
    root = repo(tmp_path)
    other = "1" * 64
    gh = FakeGh(root, {other: "sha256:" + "0" * 64})

    add(["adapters/a/one.tar.gz"], root=root, cache=tmp_path / "c", gh=gh)

    assert DIGEST in load(root).sources


def test_add_replaces_a_truncated_cache_entry_before_uploading(tmp_path: Path) -> None:
    root = repo(tmp_path)
    (tmp_path / "c").mkdir()
    (tmp_path / "c" / DIGEST).write_bytes(BLOB[:5])
    gh = FakeGh(root, {})
    uploaded: list[bytes] = []
    original = gh.__call__

    def spy(arguments: list[str]) -> str:
        if arguments[1] == "upload":
            uploaded.append(Path(arguments[3]).read_bytes())
        return original(arguments)

    add(["adapters/a/one.tar.gz"], root=root, cache=tmp_path / "c", gh=spy)

    assert uploaded == [BLOB]


def test_reconcile_skips_present_assets_without_any_download(tmp_path: Path) -> None:
    root = repo(tmp_path, declared(ORIGIN))
    gh = FakeGh(root, {DIGEST: f"sha256:{DIGEST}"})

    assert reconcile(root, cache=tmp_path / "c", opener=None, gh=gh) == []  # type: ignore[arg-type]


def test_also_declares_paths_without_files_and_ignores_every_declared_path(
    tmp_path: Path,
) -> None:
    root = repo(tmp_path)
    (root / ".gitignore").write_text("*.pyc\n")
    gh = FakeGh(root, {})
    paths = ["adapters/a/one.tar.gz"]
    also = ("adapters/b/gone.tar.gz",)

    add(paths, root=root, also=also, cache=tmp_path / "c", gh=gh)
    add(paths, root=root, also=also, cache=tmp_path / "c", gh=gh)

    assert load(root).sources[DIGEST].paths == (*paths, *also)
    assert (root / ".gitignore").read_text() == (
        "*.pyc\n\n# adapter-sources: declared paths (generated by tools/adapter-sources)\n"
        "/adapters/a/one.tar.gz\n/adapters/b/gone.tar.gz\n# adapter-sources: end\n"
    )
    (root / "adapters/a/other").write_bytes(b"different")
    with pytest.raises(AdapterSourceError, match="one blob"):
        add(
            ["adapters/a/one.tar.gz", "adapters/a/other"],
            root=root,
            also=also,
            cache=tmp_path / "c",
            gh=gh,
        )


def test_undeclare_drops_a_superseded_path_and_its_empty_entry(tmp_path: Path) -> None:
    root = repo(tmp_path)
    gh = FakeGh(root, {})
    add(
        ["adapters/a/one.tar.gz", "adapters/a/two.tar.gz"],
        root=root,
        cache=tmp_path / "c",
        gh=gh,
    )

    undeclare(["adapters/a/one.tar.gz", "adapters/a/never.tar.gz"], root=root)
    assert load(root).sources[DIGEST].paths == ("adapters/a/two.tar.gz",)
    undeclare(["adapters/a/two.tar.gz"], root=root)
    assert load(root).sources == {}
    assert ".tar.gz" not in (root / ".gitignore").read_text()


def test_the_repository_ignore_block_is_exactly_the_declared_paths() -> None:
    root = ROOT
    text = (root / ".gitignore").read_text()
    block = (
        text.split(f"{IGNORE_BEGIN}\n", 1)[1].split(IGNORE_END, 1)[0].splitlines()
        if IGNORE_BEGIN in text
        else []
    )
    declared = sorted(p for s in load(root).sources.values() for p in s.paths)
    assert block == [_ignore_line(path) for path in declared]
