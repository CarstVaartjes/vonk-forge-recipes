from __future__ import annotations

import io
import json
import gzip
import hashlib
import runpy
import tarfile
from pathlib import Path

import pytest

from vonk_forge_contracts import RecipeDefinition, content_sha256


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


@pytest.mark.parametrize("mutation", ["duplicate", "traversal"])
def test_archive_rejects_duplicate_or_traversal_entrypoints(tmp_path: Path, mutation: str) -> None:
    row, payload = _job_row(tmp_path)
    with tarfile.open(fileobj=io.BytesIO(payload), mode="r:gz") as archive:
        entries = [(member.name, archive.extractfile(member).read()) for member in archive.getmembers() if member.isreg() and member.name != "manifest.json"]
    if mutation == "duplicate":
        entries.append(("nested/recipe.json", next(body for name, body in entries if name == "recipe.json")))
        expected = "exactly one recipe.json entrypoint"
    else:
        entries = [("../recipe.json" if name == "recipe.json" else name, body) for name, body in entries]
        expected = "unsafe path"
    malformed = _rewrite(payload, entries, repair_manifest=True)
    with pytest.raises(SystemExit, match=expected):
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


def _rewrite_member(payload: bytes, target: str, *, member_type: bytes | None = None, body: bytes | None = None) -> bytes:
    output = io.BytesIO()
    with gzip.GzipFile(fileobj=output, mode="wb", mtime=0) as compressed:
        with tarfile.open(fileobj=compressed, mode="w", format=tarfile.PAX_FORMAT) as archive:
            with tarfile.open(fileobj=io.BytesIO(payload), mode="r:gz") as source:
                for member in source.getmembers():
                    data = source.extractfile(member).read() if member.isreg() else None
                    if member.name == target and member_type is not None:
                        member.type = member_type
                        member.linkname = "recipe.json"
                        member.size = 0
                        data = None
                    elif member.name == target and body is not None:
                        data = body
                        member.size = len(body)
                    archive.addfile(member, io.BytesIO(data) if data is not None else None)
    return output.getvalue()


def test_archive_rejects_payload_digest_and_undeclared_member(tmp_path: Path) -> None:
    row, payload = _job_row(tmp_path)
    with tarfile.open(fileobj=io.BytesIO(payload), mode="r:gz") as archive:
        recipe_body = archive.extractfile("recipe.json").read()
        members = [(member.name, archive.extractfile(member).read()) for member in archive.getmembers() if member.isreg()]
    altered = _rewrite_member(payload, "recipe.json", body=recipe_body + b" \n")
    with pytest.raises(SystemExit, match="manifest digest is stale: recipe.json"):
        TOOL["validate_recipe_archive"](altered, row["document"])
    extra = _rewrite(payload, members + [("undeclared.txt", b"extra")])
    with pytest.raises(SystemExit, match="does not exactly describe archive payloads"):
        TOOL["validate_recipe_archive"](extra, row["document"])


def test_archive_rejects_consistent_but_different_recipe(tmp_path: Path) -> None:
    row, payload = _job_row(tmp_path)
    with tarfile.open(fileobj=io.BytesIO(payload), mode="r:gz") as archive:
        entries = [(member.name, archive.extractfile(member).read()) for member in archive.getmembers() if member.isreg()]
    altered_document = json.loads(next(body for name, body in entries if name == "recipe.json"))
    altered_document["metadata"]["description"] += " altered"
    altered_body = json.dumps(altered_document, ensure_ascii=False, sort_keys=True, indent=2).encode()
    altered_entries = [(name, altered_body if name == "recipe.json" else body) for name, body in entries]
    repaired = _rewrite(payload, altered_entries, repair_manifest=True)
    with tarfile.open(fileobj=io.BytesIO(repaired), mode="r:gz") as archive:
        manifest = json.load(archive.extractfile("manifest.json"))
    manifest["recipe_content_sha256"] = content_sha256(RecipeDefinition.model_validate(altered_document))
    consistent = _rewrite_member(repaired, "manifest.json", body=json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode())
    with pytest.raises(SystemExit, match="recipe does not match the requested Recipe"):
        TOOL["validate_recipe_archive"](consistent, row["document"])


@pytest.mark.parametrize("field", ["size", "sha256"])
def test_archive_rejects_stale_manifest_member_metadata(tmp_path: Path, field: str) -> None:
    row, payload = _job_row(tmp_path)
    with tarfile.open(fileobj=io.BytesIO(payload), mode="r:gz") as archive:
        manifest = json.load(archive.extractfile("manifest.json"))
    recipe_entry = next(entry for entry in manifest["files"] if entry["path"] == "recipe.json")
    recipe_entry[field] = recipe_entry[field] + 1 if field == "size" else "0" * 64
    manifest_body = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
    malformed = _rewrite_member(payload, "manifest.json", body=manifest_body)
    with pytest.raises(SystemExit, match="manifest digest is stale: recipe.json"):
        TOOL["validate_recipe_archive"](malformed, row["document"])


@pytest.mark.parametrize("member_type", [tarfile.SYMTYPE, tarfile.LNKTYPE, tarfile.FIFOTYPE, tarfile.CHRTYPE])
def test_archive_rejects_non_regular_members(tmp_path: Path, member_type: bytes) -> None:
    row, payload = _job_row(tmp_path)
    malformed = _rewrite_member(payload, "recipe.json", member_type=member_type)
    with pytest.raises(SystemExit, match="non-regular archive member"):
        TOOL["validate_recipe_archive"](malformed, row["document"])
