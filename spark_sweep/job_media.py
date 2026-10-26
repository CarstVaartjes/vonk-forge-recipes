"""The media assertions of a job recipe: ``video-metadata``, ``glb-structure``, ``realtime-transcript``.

Each one has the meaning the platform's qualification gives it (vonk-forge
``qualification_fixtures/outputs.py``): a miss raises :class:`AssertionFailed`.
``video-metadata`` reads the MP4 with ``ffprobe``; a sweep host without it, or an
ffprobe that cannot be run, cannot observe the video at all, which is
:class:`NotObserved` with its reason: never a pass and never the recipe's failure.
``glb-structure`` checks the container and the triangle meshes it declares, then
what its contract profile (``QualificationGlbProfile``) adds: textures or a bound
skin. It is a lighter walk than the platform's full mesh validator, as the sweep's
PNG and WAV checks are lighter than the platform's decoders.
"""

from __future__ import annotations

import json
import math
import shutil
import struct
import subprocess
import tempfile
from collections.abc import Callable, Mapping
from functools import partial
from pathlib import Path
from typing import Any, NoReturn, TypeIs

from vonk_forge_contracts.qualification_status import QualificationGlbProfile

from .smoke import AssertionFailed

FFPROBE_TIMEOUT_SECONDS = 30
#: What the platform asks ffprobe for; ``-count_frames`` yields ``nb_read_frames``.
FFPROBE_ENTRIES = (
    "stream=index,codec_type,codec_name,pix_fmt,width,height,r_frame_rate,"
    "avg_frame_rate,nb_frames,nb_read_frames,start_time,duration,sample_rate,"
    "channels:format=format_name,start_time,duration"
)


class NotObserved(Exception):
    """The sweep host cannot observe this assertion; ``reason`` says why."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


def ffprobe() -> str | None:
    return shutil.which("ffprobe")


def _fail(kind: str, message: str, value: Any = None) -> NoReturn:
    raise AssertionFailed(message, value, kind)


def _strict_json(content: bytes | str) -> Any:
    def unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        value: dict[str, Any] = {}
        for key, item in pairs:
            if key in value:
                raise ValueError(f"duplicate JSON key: {key}")
            value[key] = item
        return value

    def constant(value: str) -> Any:
        raise ValueError(f"non-finite JSON number: {value}")

    return json.loads(content, object_pairs_hook=unique, parse_constant=constant)


def _number(kind: str, value: Any, label: str) -> float:
    """An ffprobe numeric token (normally a JSON string); anything else is unavailable."""
    if isinstance(value, bool) or not isinstance(value, (str, int, float)):
        _fail(kind, f"{label} is unavailable", value)
    try:
        return float(value)
    except ValueError:
        _fail(kind, f"{label} is unavailable", value)


def _range(kind: str, assertion: Mapping[str, Any], name: str, raw: Any) -> None:
    if f"minimum_{name}" not in assertion:
        return
    value = _number(kind, raw, name)
    if not (
        assertion[f"minimum_{name}"]
        <= value
        <= assertion.get(f"maximum_{name}", math.inf)
    ):
        _fail(kind, f"{kind} {name}", value)


# -- video-metadata ----------------------------------------------------------------------


def _probe(content: bytes) -> dict[str, Any]:
    kind = "video-metadata"
    executable = ffprobe()
    if executable is None:
        raise NotObserved("ffprobe-missing")
    with tempfile.TemporaryDirectory(prefix="sweep-video-") as root:
        path = Path(root) / "output.mp4"
        path.write_bytes(content)
        argv = [executable, "-v", "error", "-count_frames", "-show_entries"]
        argv += [FFPROBE_ENTRIES, "-of", "json", str(path)]
        try:
            result = subprocess.run(
                argv,
                check=False,
                capture_output=True,
                timeout=FFPROBE_TIMEOUT_SECONDS,
            )
        except (OSError, subprocess.TimeoutExpired) as error:
            raise NotObserved("ffprobe-failed") from error
    if result.returncode != 0:
        _fail(
            kind,
            "ffprobe rejected the MP4",
            result.stderr[-500:].decode("utf-8", "replace"),
        )
    try:
        value = _strict_json(result.stdout)
    except ValueError as error:
        raise NotObserved("ffprobe-failed") from error
    streams = value.get("streams") if isinstance(value, dict) else None
    if not isinstance(streams, list) or not all(isinstance(s, dict) for s in streams):
        raise NotObserved("ffprobe-failed")
    video = [dict(s) for s in streams if s.get("codec_type") == "video"]
    if len(video) != 1:
        _fail(kind, "MP4 must contain exactly one video stream", len(video))
    container = value.get("format")
    if not isinstance(container, dict) or "mp4" not in str(
        container.get("format_name", "")
    ).split(","):
        _fail(kind, "not an MP4 container", container)
    video[0].setdefault("duration", container.get("duration"))
    return {
        "video": video[0],
        "audio": [s for s in streams if s.get("codec_type") == "audio"],
        "streams": streams,
        "format": container,
    }


def _fps(value: Any) -> float:
    numerator, _, denominator = str(value).partition("/")
    try:
        fps = float(numerator) / float(denominator)
    except (ValueError, ZeroDivisionError):
        fps = 0.0
    if fps <= 0:
        _fail("video-metadata", "MP4 frame rate is invalid", value)
    return fps


def _decoded_frames(video: Mapping[str, Any]) -> int:
    try:
        frames = int(str(video.get("nb_read_frames")))
    except ValueError as error:
        raise NotObserved("ffprobe-frame-count-unavailable") from error
    declared = video.get("nb_frames")
    if declared not in (None, "N/A"):
        try:
            declared_frames = int(str(declared))
        except ValueError as error:
            raise NotObserved("ffprobe-frame-count-unavailable") from error
        if declared_frames != frames:
            _fail("video-metadata", "MP4 declared and decoded frame counts differ")
    return frames


def video_metadata(content: bytes, assertion: Mapping[str, Any]) -> None:
    kind = "video-metadata"
    metadata = _probe(content)
    video, audio, container = metadata["video"], metadata["audio"], metadata["format"]
    for field, probed in (
        ("width", "width"),
        ("height", "height"),
        ("codec", "codec_name"),
        ("pixel_format", "pix_fmt"),
    ):
        if field in assertion and video.get(probed) != assertion[field]:
            _fail(kind, f"{kind} {field}", video.get(probed))
    fps = _fps(video.get("avg_frame_rate"))
    if "fps" in assertion and abs(fps - assertion["fps"]) > assertion.get(
        "fps_tolerance", 0.01
    ):
        _fail(kind, f"{kind} fps", fps)
    if (
        "frame_count" in assertion
        and _decoded_frames(video) != assertion["frame_count"]
    ):
        _fail(kind, f"{kind} frame_count", video.get("nb_read_frames"))
    # The platform names this range ``duration_seconds`` but reads it from the stream's
    # ``duration``, the only duration ffprobe reports.
    _range(kind, assertion, "duration_seconds", video.get("duration"))
    for label, start in (
        ("video", video.get("start_time")),
        ("container", container.get("start_time")),
    ):
        if abs(_number(kind, start, f"{label} start_time")) > 0.05:
            _fail(kind, f"{kind} {label} start_time", start)
    if (
        "stream_count" in assertion
        and len(metadata["streams"]) != assertion["stream_count"]
    ):
        _fail(kind, f"{kind} stream_count", len(metadata["streams"]))
    if "audio_streams" in assertion and len(audio) != assertion["audio_streams"]:
        _fail(kind, f"{kind} audio_streams", len(audio))
    if "audio_sample_rate" in assertion:
        _audio(assertion, video, audio)
    _range(kind, assertion, "container_duration_seconds", container.get("duration"))


def _audio(
    assertion: Mapping[str, Any],
    video: Mapping[str, Any],
    audio: list[dict[str, Any]],
) -> None:
    kind = "video-metadata"
    if len(audio) != 1:
        _fail(kind, "MP4 audio metadata is missing", len(audio))
    (stream,) = audio
    if (
        "audio_codec" in assertion
        and stream.get("codec_name") != assertion["audio_codec"]
    ):
        _fail(kind, f"{kind} audio_codec", stream.get("codec_name"))
    try:
        sample_rate = int(str(stream.get("sample_rate")))
    except ValueError as error:
        raise NotObserved("ffprobe-sample-rate-unavailable") from error
    if sample_rate != assertion["audio_sample_rate"]:
        _fail(kind, f"{kind} audio_sample_rate", sample_rate)
    if (
        "audio_channels" in assertion
        and stream.get("channels") != assertion["audio_channels"]
    ):
        _fail(kind, f"{kind} audio_channels", stream.get("channels"))
    _range(kind, assertion, "audio_duration_seconds", stream.get("duration"))
    if "maximum_av_sync_delta_seconds" in assertion:
        if "duration" not in video or "duration" not in stream:
            raise NotObserved("ffprobe-duration-unavailable")
        delta = abs(
            _number(kind, video["duration"], "video duration")
            - _number(kind, stream["duration"], "audio duration")
        )
        if delta > assertion["maximum_av_sync_delta_seconds"]:
            _fail(kind, f"{kind} av sync", delta)


# -- glb-structure -----------------------------------------------------------------------

_JSON_CHUNK, _BIN_CHUNK = 0x4E4F534A, 0x004E4942
_COMPONENT_BYTES = {5120: 1, 5121: 1, 5122: 2, 5123: 2, 5125: 4, 5126: 4}
_COMPONENTS = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4, "MAT4": 16}


def _glb_fail(message: str, value: Any = None) -> NoReturn:
    _fail("glb-structure", f"GLB {message}", value)


def _int(value: object) -> TypeIs[int]:
    return isinstance(value, int) and not isinstance(value, bool)


def _list(document: Mapping[str, Any], name: str) -> list[Any]:
    value = document.get(name)
    if not isinstance(value, list) or not value:
        _glb_fail(f"{name} must be a non-empty array", value)
    return value


def _glb_chunks(content: bytes) -> tuple[dict[str, Any], bytes]:
    if len(content) < 28:
        _glb_fail("is shorter than its header and chunks", len(content))
    magic, version, length = struct.unpack_from("<4sII", content)
    if magic != b"glTF" or version != 2 or length != len(content):
        _glb_fail("header is not GLB 2.0 of this length", [magic, version, length])
    chunks: list[tuple[int, bytes]] = []
    offset = 12
    while offset < len(content):
        if offset + 8 > len(content):
            _glb_fail("chunk header is truncated", offset)
        size, chunk = struct.unpack_from("<II", content, offset)
        end = offset + 8 + size
        if size % 4 or end > len(content):
            _glb_fail("chunk bounds or alignment are invalid", offset)
        chunks.append((chunk, content[offset + 8 : end]))
        offset = end
    if [chunk for chunk, _ in chunks] != [_JSON_CHUNK, _BIN_CHUNK]:
        _glb_fail("must be one JSON chunk followed by one BIN chunk", len(chunks))
    try:
        document = _strict_json(chunks[0][1].decode("utf-8"))
    except ValueError as error:  # UnicodeDecodeError and JSONDecodeError included
        _glb_fail(f"JSON chunk is invalid: {error}")
    if not isinstance(document, dict):
        _glb_fail("JSON chunk is not an object")
    asset = document.get("asset")
    if not isinstance(asset, dict) or asset.get("version") != "2.0":
        _glb_fail("JSON does not declare glTF 2.0", asset)
    return document, chunks[1][1]


def _glb_accessors(
    document: Mapping[str, Any], blob: bytes
) -> list[tuple[int, str, int]]:
    """Every accessor's (componentType, type, count), each within its bufferView and buffer."""
    buffers = _list(document, "buffers")
    buffer = buffers[0]
    size = buffer.get("byteLength") if isinstance(buffer, dict) else None
    if (
        len(buffers) != 1
        or not isinstance(buffer, dict)
        or "uri" in buffer
        or not _int(size)
        or not 1 <= size <= len(blob)
        or len(blob) - size > 3
    ):
        _glb_fail("must embed exactly one buffer in its BIN chunk", buffers)
    views: list[tuple[int, int, int | None]] = []
    for view in _list(document, "bufferViews"):
        start = view.get("byteOffset", 0) if isinstance(view, dict) else None
        length = view.get("byteLength") if isinstance(view, dict) else None
        stride = view.get("byteStride") if isinstance(view, dict) else None
        if (
            not _int(start)
            or not _int(length)
            or start < 0
            or length < 1
            or start + length > size
            or (
                stride is not None
                and (not _int(stride) or not 4 <= stride <= 252 or stride % 4)
            )
        ):
            _glb_fail("bufferView exceeds its buffer", view)
        views.append((start, length, stride))
    accessors: list[tuple[int, str, int]] = []
    for accessor in _list(document, "accessors"):
        if not isinstance(accessor, dict) or "sparse" in accessor:
            _glb_fail("accessor is not a dense accessor", accessor)
        index, component, shape = (
            accessor.get("bufferView"),
            accessor.get("componentType"),
            accessor.get("type"),
        )
        count, offset = accessor.get("count"), accessor.get("byteOffset", 0)
        if (
            not _int(index)
            or not 0 <= index < len(views)
            or component not in _COMPONENT_BYTES
            or shape not in _COMPONENTS
            or not _int(count)
            or count < 1
            or not _int(offset)
            or offset < 0
        ):
            _glb_fail("accessor is invalid", accessor)
        _, length, stride = views[index]
        element = _COMPONENT_BYTES[component] * _COMPONENTS[shape]
        if offset + (count - 1) * (stride or element) + element > length:
            _glb_fail("accessor exceeds its bufferView", accessor)
        accessors.append((component, shape, count))
    return accessors


def _accessor(
    accessors: list[tuple[int, str, int]], value: Any, name: str
) -> tuple[int, str, int]:
    if not _int(value) or not 0 <= value < len(accessors):
        _glb_fail(f"primitive {name} accessor index is invalid", value)
    return accessors[value]


def _glb_item(document: Mapping[str, Any], name: str, index: Any) -> dict[str, Any]:
    items = _list(document, name)
    if (
        not _int(index)
        or not 0 <= index < len(items)
        or not isinstance(items[index], dict)
    ):
        _glb_fail(f"{name} index is invalid", index)
    return items[index]


def _texture_image(document: Mapping[str, Any], texture: Mapping[str, Any]) -> int:
    """Resolve the effective image, validating a core fallback independently."""
    source = texture.get("source")
    if source is not None:
        image = _glb_item(document, "images", source)
        if "uri" in image or image.get("mimeType") not in {"image/png", "image/jpeg"}:
            _glb_fail("core texture image must be an embedded PNG or JPEG", image)
    extensions = texture.get("extensions")
    if isinstance(extensions, dict) and "EXT_texture_webp" in extensions:
        webp = extensions["EXT_texture_webp"]
        if not isinstance(webp, dict):
            _glb_fail("EXT_texture_webp must be an object", webp)
        source = webp.get("source")
        image = _glb_item(document, "images", source)
        if "uri" in image or image.get("mimeType") != "image/webp":
            _glb_fail("WebP texture image must be embedded WebP", image)
        for name in ("extensionsUsed", "extensionsRequired"):
            if "EXT_texture_webp" not in _list(document, name):
                _glb_fail(
                    "WebP textures must declare EXT_texture_webp as used and required",
                    name,
                )
    else:
        _glb_item(document, "images", source)
    if not _int(source):
        _glb_fail("texture source index is invalid", source)
    return source


def _textured(
    document: Mapping[str, Any],
    accessors: list[tuple[int, str, int]],
    primitives: list[tuple[Mapping[str, Any], int]],
    *,
    pbr_required: bool = False,
) -> None:
    for primitive, vertex_count in primitives:
        _textured_primitive(document, primitive, accessors, vertex_count, pbr_required)


def _textured_primitive(
    document: Mapping[str, Any],
    primitive: Mapping[str, Any],
    accessors: list[tuple[int, str, int]],
    vertex_count: int,
    pbr_required: bool,
) -> None:
    attributes = primitive["attributes"]
    uv = _accessor(accessors, attributes.get("TEXCOORD_0"), "TEXCOORD_0")
    if uv != (5126, "VEC2", vertex_count):
        _glb_fail("TEXCOORD_0 must be FLOAT VEC2 matching POSITION count", uv)
    material = _glb_item(document, "materials", primitive.get("material"))
    pbr = material.get("pbrMetallicRoughness")
    channels = (
        ("baseColorTexture", "metallicRoughnessTexture")
        if pbr_required
        else ("baseColorTexture",)
    )
    sources: list[int] = []
    for channel in channels:
        info = pbr.get(channel) if isinstance(pbr, dict) else None
        if not isinstance(info, dict) or info.get("texCoord", 0) != 0:
            _glb_fail(f"{channel} must use TEXCOORD_0", info)
        texture = _glb_item(document, "textures", info.get("index"))
        source = _texture_image(document, texture)
        image = _glb_item(document, "images", source)
        sources.append(source)
        view = _glb_item(document, "bufferViews", image.get("bufferView"))
        if view.get("buffer") != 0 or "target" in view or "byteStride" in view:
            _glb_fail("texture image bufferView is invalid", view)
    if pbr_required and sources[0] == sources[1]:
        _glb_fail("PBR textures must use distinct embedded images", sources)


def _skinned(
    document: Mapping[str, Any],
    accessors: list[tuple[int, str, int]],
    primitives: list[tuple[Mapping[str, Any], int]],
) -> None:
    """Every mesh node binds a skin with joints; every primitive carries its weights."""
    nodes = _list(document, "nodes")
    skins = _list(document, "skins")
    for skin in skins:
        joints = skin.get("joints") if isinstance(skin, dict) else None
        if (
            not isinstance(joints, list)
            or not joints
            or not all(_int(joint) and 0 <= joint < len(nodes) for joint in joints)
            or len(set(joints)) != len(joints)
        ):
            _glb_fail("skin joints must be distinct node indices", skin)
    mesh_nodes = [node for node in nodes if isinstance(node, dict) and "mesh" in node]
    if not mesh_nodes:
        _glb_fail("has no mesh node to bind a skin", len(nodes))
    for node in mesh_nodes:
        skin = node.get("skin")
        if not _int(skin) or not 0 <= skin < len(skins):
            _glb_fail("mesh node must bind a skin", node)
    for primitive, vertex_count in primitives:
        attributes = primitive["attributes"]
        joints = _accessor(accessors, attributes.get("JOINTS_0"), "JOINTS_0")
        if joints[0] not in (5121, 5123) or joints[1:] != ("VEC4", vertex_count):
            _glb_fail("JOINTS_0 must be unsigned VEC4 matching POSITION count", joints)
        weights = _accessor(accessors, attributes.get("WEIGHTS_0"), "WEIGHTS_0")
        if weights != (5126, "VEC4", vertex_count):
            _glb_fail("WEIGHTS_0 must be FLOAT VEC4 matching POSITION count", weights)


def _triangle_mesh(
    _document: Mapping[str, Any],
    _accessors: list[tuple[int, str, int]],
    _primitives: list[tuple[Mapping[str, Any], int]],
) -> None:
    """The triangle meshes every GLB declares are the whole profile."""


#: What each contract profile checks beyond the triangle meshes every GLB declares.
GLB_PROFILE_CHECKS: dict[
    QualificationGlbProfile,
    Callable[
        [
            Mapping[str, Any],
            list[tuple[int, str, int]],
            list[tuple[Mapping[str, Any], int]],
        ],
        None,
    ],
] = {
    QualificationGlbProfile.TRIANGLE_MESH: _triangle_mesh,
    QualificationGlbProfile.TEXTURED: _textured,
    QualificationGlbProfile.TEXTURED_PBR: partial(_textured, pbr_required=True),
    QualificationGlbProfile.SKINNED: _skinned,
}


def glb_structure(content: bytes, assertion: Mapping[str, Any]) -> None:
    value = assertion.get("profile", QualificationGlbProfile.TRIANGLE_MESH)
    try:
        check = GLB_PROFILE_CHECKS[QualificationGlbProfile(value)]
    except (KeyError, ValueError):
        _glb_fail(f"profile {value!r} is not checked by the sweep", value)
    document, blob = _glb_chunks(content)
    accessors = _glb_accessors(document, blob)
    meshes = _list(document, "meshes")
    primitives: list[tuple[Mapping[str, Any], int]] = []
    for mesh in meshes:
        for primitive in _list(mesh if isinstance(mesh, dict) else {}, "primitives"):
            if not isinstance(primitive, dict) or primitive.get("mode", 4) != 4:
                _glb_fail("primitive is not TRIANGLES", primitive)
            attributes = primitive.get("attributes")
            position = _accessor(
                accessors,
                attributes.get("POSITION") if isinstance(attributes, dict) else None,
                "POSITION",
            )
            if position[:2] != (5126, "VEC3") or position[2] < 3:
                _glb_fail("POSITION must be FLOAT VEC3 with three vertices", position)
            indices = _accessor(accessors, primitive.get("indices"), "indices")
            if (
                indices[0] not in (5121, 5123, 5125)
                or indices[1] != "SCALAR"
                or indices[2] < 3
                or indices[2] % 3
            ):
                _glb_fail("indices must be unsigned SCALAR triangle indices", indices)
            primitives.append((primitive, position[2]))
    check(document, accessors, primitives)
    if len(meshes) < assertion.get("minimum_meshes", 1):
        _glb_fail("has too few meshes", len(meshes))
    if len(primitives) < assertion.get("minimum_primitives", 1):
        _glb_fail("has too few primitives", len(primitives))


# -- realtime-transcript -----------------------------------------------------------------

_TRANSCRIPT_SHAPES = {
    "session-start": {"sequence", "elapsed_seconds", "type", "model_revision"},
    "frame-ack": {
        "sequence",
        "elapsed_seconds",
        "type",
        "event_index",
        "timestamp",
        "dropped_oldest",
    },
    "output": {"sequence", "elapsed_seconds", "type", "kind", "text"},
    "session-stop": {"sequence", "elapsed_seconds", "type"},
}
_OUTPUT_KINDS = {"silence", "round-start", "round-end", "response-start", "text"}


def _transcript_fail(message: str, value: Any = None) -> NoReturn:
    _fail("realtime-transcript", f"realtime transcript {message}", value)


def realtime_transcript(content: bytes, assertion: Mapping[str, Any]) -> None:
    frame_count = assertion.get("frame_count", 1)
    lines = content.splitlines()
    if not 3 <= len(lines) <= 4_096 or not all(lines):
        _transcript_fail("record count is invalid", len(lines))
    records: list[dict[str, Any]] = []
    for line in lines:
        try:
            record = _strict_json(line)
        except ValueError:
            record = None
        if not isinstance(record, dict):
            _transcript_fail("record is not a JSON object", line[:200])
        records.append(record)
    previous = 0.0
    for sequence, record in enumerate(records):
        shape = _TRANSCRIPT_SHAPES.get(str(record.get("type")))
        if shape is None or set(record) != shape:
            _transcript_fail("record shape is invalid", record)
        elapsed = record.get("elapsed_seconds")
        if (
            record.get("sequence") != sequence
            or isinstance(elapsed, bool)
            or not isinstance(elapsed, (int, float))
            or not previous <= elapsed <= 3_600
        ):
            _transcript_fail("ordering is invalid", record)
        previous = float(elapsed)
        if record["type"] == "output" and (
            record.get("kind") not in _OUTPUT_KINDS
            or not isinstance(record.get("text"), str)
            or len(record["text"]) > 65_536
        ):
            _transcript_fail("output record is invalid", record)
    if records[0].get("type") != "session-start" or records[0].get(
        "model_revision"
    ) != assertion.get("model_revision"):
        _transcript_fail("model revision is not the recipe's", records[0])
    frames = [record for record in records if record["type"] == "frame-ack"]
    if (
        not frames
        or len(frames) != frame_count
        or any(
            record.get("event_index") != index
            or record.get("timestamp") != float(index)
            or not isinstance(record.get("dropped_oldest"), bool)
            for index, record in enumerate(frames)
        )
    ):
        _transcript_fail("frame acknowledgements are invalid", len(frames))
    stops = [
        index
        for index, record in enumerate(records)
        if record["type"] == "session-stop"
    ]
    if (
        len(stops) != 1
        or stops[0] <= frames[0]["sequence"]
        or any(record["type"] != "output" for record in records[stops[0] + 1 :])
    ):
        _transcript_fail("terminal ordering is invalid", stops)
