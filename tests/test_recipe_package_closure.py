from __future__ import annotations

import io
import json
import gzip
import hashlib
import runpy
import tarfile
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
TOOL = runpy.run_path(str(ROOT / "tools/build-catalog-index"))


def _job_row(tmp_path: Path) -> tuple[dict, bytes]:
    catalog = TOOL["build"](package_dir=tmp_path)
    row = next(row for row in catalog["recipes"] if row["document"]["interfaces"][0]["adapter"] != "openai")
    return row, (tmp_path / Path(row["package"]["path"]).name).read_bytes()


def _rewrite(payload: bytes, names: list[tuple[str, bytes]], *, repair_manifest: bool = False) -> bytes:
    manifest = None
    if repair_manifest:
        with tarfile.open(fileobj=io.BytesIO(payload), mode="r:gz") as source:
            manifest = json.load(source.extractfile("manifest.json"))
        names = [(name, body) for name, body in names if name != "manifest.json"]
        manifest["files"] = [
            {"path": name, "size": len(body), "sha256": hashlib.sha256(body).hexdigest()}
            for name, body in names
            if name != "manifest.json"
        ]
        names = [("manifest.json", json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()), *names]
    output = io.BytesIO()
    with gzip.GzipFile(fileobj=output, mode="wb", mtime=0) as compressed:
        with tarfile.open(fileobj=compressed, mode="w", format=tarfile.PAX_FORMAT) as archive:
            for name, body in names:
                info = tarfile.TarInfo(name)
                info.size = len(body)
                info.mode = 0o644
                archive.addfile(info, io.BytesIO(body))
    return output.getvalue()


def test_archive_has_one_entrypoint_and_real_closure(tmp_path: Path) -> None:
    row, payload = _job_row(tmp_path)
    TOOL["validate_recipe_archive"](payload, row["document"])
    with tarfile.open(fileobj=io.BytesIO(payload), mode="r:gz") as archive:
        names = [(member.name, archive.extractfile(member).read()) for member in archive.getmembers() if member.isfile()]
    assert [name for name, _ in names].count("recipe.json") == 1


@pytest.mark.parametrize("bad_names", [["recipe.json", "nested/recipe.json"], ["../recipe.json"]])
def test_archive_rejects_duplicate_or_traversal_entrypoints(tmp_path: Path, bad_names: list[str]) -> None:
    row, payload = _job_row(tmp_path)
    with tarfile.open(fileobj=io.BytesIO(payload), mode="r:gz") as archive:
        body = archive.extractfile("recipe.json").read()
    malformed = _rewrite(payload, [(name, body) for name in bad_names])
    with pytest.raises(SystemExit):
        TOOL["validate_recipe_archive"](malformed, row["document"])


def test_archive_rejects_missing_model_source_and_fixture(tmp_path: Path) -> None:
    row, payload = _job_row(tmp_path)
    with tarfile.open(fileobj=io.BytesIO(payload), mode="r:gz") as archive:
        entries = [(member.name, archive.extractfile(member).read()) for member in archive.getmembers() if member.isfile() and member.name != "models/" + row["document"]["models"][0]["model"]["slug"] + ".json"]
    with pytest.raises(SystemExit, match="Model snapshot"):
        TOOL["validate_recipe_archive"](_rewrite(payload, entries, repair_manifest=True), row["document"])

    with tarfile.open(fileobj=io.BytesIO(payload), mode="r:gz") as archive:
        entries = [(member.name, archive.extractfile(member).read()) for member in archive.getmembers() if member.isfile()]
    without_source = [item for item in entries if item[0] != row["document"]["execution"]["build"]["dockerfile"]]
    with pytest.raises(SystemExit, match="source closure"):
        TOOL["validate_recipe_archive"](_rewrite(payload, without_source, repair_manifest=True), row["document"])

    fixture = row["document"]["validation"]["serving"]["checks"][0]["request"]["fixture"]
    without_fixture = [item for item in entries if item[0] != fixture]
    with pytest.raises(SystemExit, match="serving closure"):
        TOOL["validate_recipe_archive"](_rewrite(payload, without_fixture, repair_manifest=True), row["document"])
