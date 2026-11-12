"""The signed release asset set is complete and self-consistent.

Control planes verify the signature over SHA256SUMS and then trust only the
digests it lists, so an asset the manifest omits, a digest that differs from
the bytes written, or a package digest that disagrees with the index would be
rejected (or, worse, never checked) by every consumer.
"""

from __future__ import annotations

import copy
import hashlib
import json
import runpy
import shutil
from pathlib import Path, PurePosixPath

import pytest

ROOT = Path(__file__).resolve().parents[1]
TOOL = runpy.run_path(str(ROOT / "tools/build-catalog-index"))
SOURCE_COMMIT = "0123456789abcdef0123456789abcdef01234567"
SOURCE_DATE_EPOCH = 0


def test_release_manifest_covers_every_asset_and_matches_the_index(
    tmp_path: Path,
    built_catalog: tuple[dict, Path],
) -> None:
    release = tmp_path / "release"
    report = tmp_path / "coverage.md"
    report.write_text("coverage\n", encoding="utf-8")
    digest = built_catalog[0]["recipes"][0]["content_sha256"]
    digests = TOOL["write_release"](
        release,
        source_commit=SOURCE_COMMIT,
        source_date_epoch=SOURCE_DATE_EPOCH,
        include=(report,),
        prebuilt_images={digest: {"build_key": "a" * 64}},
    )
    # An included file is published and listed like every other asset.
    assert digests["coverage.md"] == hashlib.sha256(b"coverage\n").hexdigest()

    lines = (release / "SHA256SUMS").read_text(encoding="utf-8").splitlines()
    listed = {name: digest for digest, name in (line.split("  ") for line in lines)}
    assert [line.split("  ")[1] for line in lines] == sorted(listed)
    written = {path.name for path in release.iterdir()} - {"SHA256SUMS"}
    assert set(listed) == written
    for name, digest in listed.items():
        assert hashlib.sha256((release / name).read_bytes()).hexdigest() == digest

    index = json.loads((release / "catalog-index.json").read_text(encoding="utf-8"))
    assert index["source_commit"] == SOURCE_COMMIT
    assert all("prebuilt_image" not in row for row in index["recipes"])
    packages = {
        PurePosixPath(row["package"]["path"]).name: row["package"]["sha256"]
        for row in index["recipes"]
    }
    assert packages
    assert {name: listed[name] for name in packages} == packages
    assert set(listed) == {
        "catalog-index.json",
        "qualification-index.json",
        "coverage.md",
        *packages,
    }


@pytest.fixture
def session_build(
    monkeypatch: pytest.MonkeyPatch, built_catalog: tuple[dict, Path]
) -> dict:
    """``build`` answers with the session's real build output.

    These cases test what the writers do with a build's index and packages, so
    they reuse the one full build instead of repeating it.
    """
    catalog, packages = built_catalog

    def build(*, package_dir: Path, **_: object) -> dict:
        package_dir.mkdir(parents=True, exist_ok=True)
        for path in packages.glob("*.tar.gz"):
            shutil.copyfile(path, package_dir / path.name)
        return copy.deepcopy(catalog)

    monkeypatch.setitem(TOOL["write_release"].__globals__, "build", build)
    return catalog


def test_an_included_file_cannot_shadow_an_asset(
    tmp_path: Path, session_build: dict
) -> None:
    shadow = tmp_path / "catalog-index.json"
    shadow.write_text("{}", encoding="utf-8")
    with pytest.raises(SystemExit, match="not unique: catalog-index.json"):
        TOOL["write_release"](
            tmp_path / "shadowed",
            source_commit=SOURCE_COMMIT,
            source_date_epoch=SOURCE_DATE_EPOCH,
            include=(shadow,),
        )


def test_checkout_outputs_drop_packages_of_removed_recipes(
    tmp_path: Path, session_build: dict
) -> None:
    stale = tmp_path / "packages" / "removed-recipe.tar.gz"
    stale.parent.mkdir()
    stale.write_bytes(b"stale")
    document = TOOL["write_outputs"](
        tmp_path, source_commit=SOURCE_COMMIT, source_date_epoch=SOURCE_DATE_EPOCH
    )
    assert not stale.exists()
    assert {path.name for path in (tmp_path / "packages").iterdir()} == {
        PurePosixPath(row["package"]["path"]).name for row in document["recipes"]
    }
    assert (tmp_path / "qualification" / "qualification-index.json").is_file()
    index = json.loads((tmp_path / "catalog-index.json").read_text(encoding="utf-8"))
    assert index == session_build


def test_release_refuses_to_mix_with_existing_files(tmp_path: Path) -> None:
    (tmp_path / "stale.tar.gz").write_bytes(b"stale")
    with pytest.raises(SystemExit, match="release directory must be empty"):
        TOOL["write_release"](
            tmp_path, source_commit=SOURCE_COMMIT, source_date_epoch=SOURCE_DATE_EPOCH
        )
