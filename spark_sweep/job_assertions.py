"""The output assertions of a job recipe's qualification definition, as the sweep checks them.

``qualification/recipes/<slug>.json`` says what a job's result must be: how many
files, their names, media types, sizes and formats, and for some kinds what is
inside them. The sweep checks every kind the same way the platform's qualification
does (the media kinds in ``job_media``). What the sweep host cannot observe (a
``video-metadata`` without ``ffprobe`` installed) is reported as not evaluated with
its reason, never passed and never failed. ``tests/test_sweep_job_smoke.py`` keeps
every kind the definitions use in exactly one of the two sets.
"""

from __future__ import annotations

import io
import re
import struct
import wave
import zipfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, NoReturn

from .job_media import (
    NotObserved,
    glb_structure,
    realtime_transcript,
    strict_json,
    video_metadata,
)
from .smoke import AssertionFailed


@dataclass(frozen=True)
class Output:
    name: str
    media_type: str
    content: bytes


#: Decided by the sweep: a miss fails the recipe.
EVALUATED = frozenset(
    {
        "file-count",
        "file-names",
        "media-type",
        "media-type-counts",
        "minimum-bytes",
        "format",
        "image-metadata",
        "audio-metadata",
        "document-archive",
        "json-document",
        "video-metadata",
        "glb-structure",
        "realtime-transcript",
    }
)
#: Kinds the sweep has no evaluator for: recorded as not evaluated, never passed.
NOT_EVALUATED: frozenset[str] = frozenset()
#: The media kinds, each checking one selected output's content.
MEDIA = {
    "video-metadata": video_metadata,
    "glb-structure": glb_structure,
    "realtime-transcript": realtime_transcript,
}


def _fail(assertion: Mapping[str, Any], message: str, value: Any = None) -> NoReturn:
    raise AssertionFailed(message, value, str(assertion.get("kind")))


def _check_format(content: bytes, name: str) -> bool:
    if name == "png":
        return (
            len(content) >= 24
            and content[:8] == b"\x89PNG\r\n\x1a\n"
            and min(struct.unpack(">II", content[16:24])) >= 1
        )
    if name == "jpeg":
        return content.startswith(b"\xff\xd8") and content.endswith(b"\xff\xd9")
    if name == "wav":
        try:
            _wav(content)
        except AssertionFailed:
            return False
        return True
    if name == "mp4":
        return len(content) >= 16 and content[4:8] == b"ftyp" and b"moov" in content
    if name == "glb":  # the container header; the mesh itself is glb-structure's
        return (
            len(content) >= 12
            and content[:4] == b"glTF"
            and struct.unpack("<II", content[4:12]) == (2, len(content))
        )
    if name == "json":
        try:
            strict_json(content)
        except ValueError:
            return False
        return True
    if name == "zip":
        try:
            with zipfile.ZipFile(io.BytesIO(content)) as archive:
                return bool(archive.namelist()) and archive.testzip() is None
        except (ValueError, zipfile.BadZipFile):
            return False
    raise AssertionFailed(f"unknown format {name!r}", name, "format")


def _png(content: bytes) -> dict[str, int]:
    if (
        len(content) < 29
        or content[:8] != b"\x89PNG\r\n\x1a\n"
        or content[12:16] != b"IHDR"
    ):
        raise AssertionFailed("not a PNG with an IHDR header", None, "image-metadata")
    width, height, bit_depth, color_type = struct.unpack(">IIBB", content[16:26])
    return {
        "width": width,
        "height": height,
        "bit_depth": bit_depth,
        "color_type": color_type,
    }


def _wav(content: bytes) -> dict[str, float]:
    try:
        with wave.open(io.BytesIO(content), "rb") as source:
            frames = source.getnframes()
            rate = source.getframerate()
            metadata = {
                "channels": source.getnchannels(),
                "sample_rate": rate,
                "sample_width_bytes": source.getsampwidth(),
                "duration_seconds": frames / rate if rate else 0.0,
            }
    except (EOFError, wave.Error) as error:
        raise AssertionFailed(
            f"not a PCM WAV: {error}", None, "audio-metadata"
        ) from error
    if metadata["channels"] < 1 or rate < 1 or frames < 1:
        raise AssertionFailed("WAV has no audio", metadata, "audio-metadata")
    return metadata


def _fields(
    assertion: Mapping[str, Any], metadata: Mapping[str, Any], names: Sequence[str]
) -> None:
    for name in names:
        if name in assertion and metadata.get(name) != assertion[name]:
            _fail(assertion, f"{assertion['kind']} {name}", metadata.get(name))
    if "minimum_duration_seconds" in assertion:
        duration = metadata.get("duration_seconds", -1)
        if not (
            assertion["minimum_duration_seconds"]
            <= duration
            <= assertion.get("maximum_duration_seconds", float("inf"))
        ):
            _fail(assertion, f"{assertion['kind']} duration_seconds", duration)


def _document_archive(content: bytes, assertion: Mapping[str, Any]) -> None:
    try:
        with zipfile.ZipFile(io.BytesIO(content)) as archive:
            names = archive.namelist()
            payloads = {name: archive.read(name) for name in names}
    except (ValueError, zipfile.BadZipFile) as error:
        raise AssertionFailed(
            f"not a ZIP: {error}", None, "document-archive"
        ) from error
    if names != assertion.get("exact_names"):
        _fail(assertion, "document-archive entry names", names)
    try:
        manifest = strict_json(payloads["manifest.json"])
    except (KeyError, ValueError) as error:
        raise AssertionFailed(
            "document-archive manifest.json is unreadable", None, "document-archive"
        ) from error
    if not isinstance(manifest, dict):
        _fail(assertion, "document-archive manifest is not an object", manifest)
    for name, value in dict(assertion.get("manifest_equals") or {}).items():
        if manifest.get(name) != value:
            _fail(assertion, f"document-archive manifest {name}", manifest.get(name))
    if (
        "sampling_equals" in assertion
        and manifest.get("sampling") != assertion["sampling_equals"]
    ):
        _fail(assertion, "document-archive sampling", manifest.get("sampling"))
    documents = manifest.get("documents")
    first = (
        documents[0] if isinstance(documents, list) and len(documents) == 1 else None
    )
    if (
        not isinstance(first, dict)
        or first.get("input") != assertion.get("input_name")
        or first.get("output") != assertion.get("output_name")
    ):
        _fail(assertion, "document-archive receipt", documents)
    text = payloads.get(str(assertion.get("output_name")), b"").decode(
        "utf-8", "replace"
    )
    if re.search(str(assertion.get("text_pattern", "")), text) is None:
        _fail(assertion, "document-archive text", text[:200])


def _json_document(content: bytes, assertion: Mapping[str, Any]) -> None:
    try:
        document = strict_json(content)
    except ValueError as error:
        raise AssertionFailed(f"not JSON: {error}", None, "json-document") from error
    if not isinstance(document, dict):
        _fail(assertion, "json-document is not an object", document)
    for name in assertion.get("required_keys") or []:
        if name not in document:
            _fail(assertion, f"json-document required key {name}")
    for name, value in dict(assertion.get("equals") or {}).items():
        if document.get(name) != value:
            _fail(assertion, f"json-document {name}", document.get(name))


def evaluate(
    assertions: Sequence[Mapping[str, Any]], outputs: Sequence[Output]
) -> dict[str, str]:
    """Raise :class:`AssertionFailed` on the first miss; return each kind not evaluated, with why."""
    skipped: dict[str, str] = {}
    for assertion in assertions:
        kind = str(assertion.get("kind"))
        if kind not in EVALUATED:
            skipped[kind] = "no-evaluator"
            continue
        wanted = assertion.get("media_type")
        selected = [o for o in outputs if wanted is None or o.media_type == wanted]
        if kind == "file-count" and len(outputs) != assertion.get("exact"):
            _fail(assertion, "file-count", len(outputs))
        elif kind == "file-names" and [o.name for o in outputs] != assertion.get(
            "exact"
        ):
            _fail(assertion, "file-names", [o.name for o in outputs])
        elif kind == "media-type":
            allowed = assertion.get("allowed") or []
            bad = [o.media_type for o in outputs if o.media_type not in allowed]
            if bad:
                _fail(assertion, "media-type", bad)
        elif kind == "media-type-counts":
            counts = dict(assertion.get("counts") or {})
            actual = {t: sum(o.media_type == t for o in outputs) for t in counts}
            if actual != counts:
                _fail(assertion, "media-type-counts", actual)
        elif kind == "minimum-bytes":
            small = [
                o.name for o in outputs if len(o.content) < int(assertion["value"])
            ]
            if small:
                _fail(assertion, "minimum-bytes", small)
        elif kind == "format":
            bad = [
                o.name
                for o in outputs
                if not _check_format(o.content, str(assertion["format"]))
            ]
            if bad:
                _fail(assertion, f"format {assertion['format']}", bad)
        elif not selected:
            _fail(assertion, f"{kind} selected no output", wanted)
        elif kind == "image-metadata":
            for output in selected:
                _fields(
                    assertion,
                    _png(output.content),
                    ("width", "height", "bit_depth", "color_type"),
                )
        elif kind == "audio-metadata":
            for output in selected:
                _fields(
                    assertion,
                    _wav(output.content),
                    ("channels", "sample_rate", "sample_width_bytes"),
                )
        elif kind == "document-archive":
            for output in selected:
                _document_archive(output.content, assertion)
        elif kind == "json-document":
            for output in selected:
                _json_document(output.content, assertion)
        elif kind in MEDIA:
            try:
                for output in selected:
                    MEDIA[kind](output.content, assertion)
            except NotObserved as unobserved:
                skipped[kind] = unobserved.reason
    return skipped
