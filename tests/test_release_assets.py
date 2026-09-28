"""The signed release asset set is complete and self-consistent.

Control planes verify the signature over SHA256SUMS and then trust only the
digests it lists, so an asset the manifest omits, a digest that differs from
the bytes written, or a package digest that disagrees with the index would be
rejected (or, worse, never checked) by every consumer.
"""

from __future__ import annotations

import hashlib
import json
import runpy
from pathlib import Path, PurePosixPath

import pytest

ROOT = Path(__file__).resolve().parents[1]
TOOL = runpy.run_path(str(ROOT / "tools/build-catalog-index"))
SOURCE_COMMIT = "0123456789abcdef0123456789abcdef01234567"


def test_release_manifest_covers_every_asset_and_matches_the_index(
    tmp_path: Path,
) -> None:
    release = tmp_path / "release"
    TOOL["write_release"](release, source_commit=SOURCE_COMMIT)

    lines = (release / "SHA256SUMS").read_text(encoding="utf-8").splitlines()
    listed = {name: digest for digest, name in (line.split("  ") for line in lines)}
    assert [line.split("  ")[1] for line in lines] == sorted(listed)
    written = {path.name for path in release.iterdir()} - {"SHA256SUMS"}
    assert set(listed) == written
    for name, digest in listed.items():
        assert hashlib.sha256((release / name).read_bytes()).hexdigest() == digest

    index = json.loads((release / "catalog-index.json").read_text(encoding="utf-8"))
    assert index["source_commit"] == SOURCE_COMMIT
    packages = {
        PurePosixPath(row["package"]["path"]).name: row["package"]["sha256"]
        for row in index["recipes"]
    }
    assert packages
    assert {name: listed[name] for name in packages} == packages
    assert set(listed) == {
        "catalog-index.json",
        "qualification-index.json",
        *packages,
    }


def test_included_files_are_listed_and_cannot_shadow_an_asset(tmp_path: Path) -> None:
    report = tmp_path / "coverage.md"
    report.write_text("coverage\n", encoding="utf-8")
    digests = TOOL["write_release"](
        tmp_path / "release", source_commit=SOURCE_COMMIT, include=(report,)
    )
    assert digests["coverage.md"] == hashlib.sha256(b"coverage\n").hexdigest()

    shadow = tmp_path / "catalog-index.json"
    shadow.write_text("{}", encoding="utf-8")
    with pytest.raises(SystemExit, match="not unique: catalog-index.json"):
        TOOL["write_release"](
            tmp_path / "shadowed", source_commit=SOURCE_COMMIT, include=(shadow,)
        )


def test_checkout_outputs_drop_packages_of_removed_recipes(tmp_path: Path) -> None:
    stale = tmp_path / "packages" / "removed-recipe.tar.gz"
    stale.parent.mkdir()
    stale.write_bytes(b"stale")
    document = TOOL["write_outputs"](tmp_path, source_commit=SOURCE_COMMIT)
    assert not stale.exists()
    assert {path.name for path in (tmp_path / "packages").iterdir()} == {
        PurePosixPath(row["package"]["path"]).name for row in document["recipes"]
    }
    assert (tmp_path / "qualification" / "qualification-index.json").is_file()
    index = json.loads((tmp_path / "catalog-index.json").read_text(encoding="utf-8"))
    assert index["source_commit"] == SOURCE_COMMIT


def test_release_refuses_to_mix_with_existing_files(tmp_path: Path) -> None:
    (tmp_path / "stale.tar.gz").write_bytes(b"stale")
    with pytest.raises(SystemExit, match="release directory must be empty"):
        TOOL["write_release"](tmp_path, source_commit=SOURCE_COMMIT)
