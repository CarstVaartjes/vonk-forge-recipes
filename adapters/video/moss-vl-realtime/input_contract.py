"""Resolve the controller-authenticated MOSS input slots without filename conventions."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path


_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z")
_IMAGE_MEDIA_TYPES = {"image/jpeg", "image/png", "image/webp"}
_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}


def _fail(message: str) -> None:
    raise ValueError(message)


def resolve_moss_inputs(root: Path) -> tuple[Path, frozenset[str]]:
    """Return the session document and authenticated frame names from manifest v1."""

    manifest = root / "manifest.json"
    if (
        not manifest.is_file()
        or manifest.is_symlink()
        or manifest.stat().st_size > 64 * 1024
    ):
        _fail("/inputs/manifest.json is missing or unsafe")
    try:
        document = json.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        _fail(f"invalid authenticated input manifest: {exc}")
    if not isinstance(document, dict) or set(document) != {
        "schema_version",
        "total_bytes",
        "files",
    }:
        _fail("authenticated input manifest fields are invalid")
    files = document.get("files")
    if document.get("schema_version") != 1 or not isinstance(files, list) or not 2 <= len(files) <= 32:
        _fail("authenticated input manifest shape is invalid")

    sessions: list[Path] = []
    frames: set[str] = set()
    observed_names: set[str] = set()
    observed_total = 0
    for item in files:
        if not isinstance(item, dict) or set(item) != {
            "slot",
            "name",
            "media_type",
            "size_bytes",
            "sha256",
        }:
            _fail("authenticated input entry fields are invalid")
        slot = item["slot"]
        name = item["name"]
        media_type = item["media_type"]
        size_bytes = item["size_bytes"]
        sha256 = item["sha256"]
        if (
            not isinstance(name, str)
            or _NAME.fullmatch(name) is None
            or name == "manifest.json"
            or name in observed_names
            or not isinstance(size_bytes, int)
            or isinstance(size_bytes, bool)
            or size_bytes < 0
            or not isinstance(sha256, str)
            or re.fullmatch(r"[0-9a-f]{64}", sha256) is None
        ):
            _fail("authenticated input entry is invalid")
        path = root / name
        if not path.is_file() or path.is_symlink() or path.stat().st_size != size_bytes:
            _fail(f"authenticated input file is missing or changed: {name}")
        with path.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        if digest != sha256:
            _fail(f"authenticated input digest changed: {name}")
        if slot == "session":
            if media_type != "application/json" or path.suffix.lower() != ".json" or size_bytes > 1024 * 1024:
                _fail("session slot must contain one bounded JSON document")
            sessions.append(path)
        elif slot == "frames":
            if media_type not in _IMAGE_MEDIA_TYPES or path.suffix.lower() not in _IMAGE_SUFFIXES or size_bytes > 8 * 1024 * 1024:
                _fail("frames slot contains an unsupported image")
            frames.add(name)
        else:
            _fail(f"unsupported MOSS input slot: {slot}")
        observed_names.add(name)
        observed_total += size_bytes

    if len(sessions) != 1 or not 1 <= len(frames) <= 31:
        _fail("MOSS requires one session document and between 1 and 31 frames")
    if document.get("total_bytes") != observed_total or observed_total > 249 * 1024 * 1024:
        _fail("authenticated MOSS input total is invalid")
    directory_names = {path.name for path in root.iterdir()}
    if directory_names != observed_names | {"manifest.json"}:
        _fail("/inputs contains files outside the authenticated manifest")
    return sessions[0], frozenset(frames)
