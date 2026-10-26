"""A job's media outputs: a GLB's triangle meshes, an MP4's metadata, a realtime transcript."""

from __future__ import annotations

import base64
import json
import shutil
import struct
import subprocess
from pathlib import Path
from typing import Any

import pytest
from vonk_forge_contracts.qualification_status import QualificationGlbProfile

from spark_sweep import job_media
from spark_sweep.job_assertions import Output, evaluate
from spark_sweep.smoke import AssertionFailed

# -- glb-structure -----------------------------------------------------------------------

TRIANGLE_MESH = {"kind": "glb-structure", "profile": "triangle-mesh"}


def _glb(*, textured: bool = False, skinned: bool = False, **overrides: Any) -> bytes:
    """One indexed triangle: a FLOAT VEC3 POSITION and three UNSIGNED_SHORT indices.

    ``skinned`` binds it to a one-joint skin with JOINTS_0 and WEIGHTS_0.
    """
    positions = struct.pack("<9f", 0, 0, 0, 1, 0, 0, 0, 1, 0)
    indices = struct.pack("<3H", 0, 1, 2) + b"\0\0"
    uv = struct.pack("<6f", 0, 0, 1, 0, 0, 1)
    png = base64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII="
    )
    joints = struct.pack("<12B", *[0] * 12)
    weights = struct.pack("<12f", *[1, 0, 0, 0] * 3)
    blob = positions + indices + (uv + png if textured else b"")
    blob += joints + weights if skinned else b""
    document: dict[str, Any] = {
        "asset": {"version": "2.0"},
        "buffers": [{"byteLength": len(blob)}],
        "bufferViews": [
            {"buffer": 0, "byteOffset": 0, "byteLength": 36, "target": 34962},
            {"buffer": 0, "byteOffset": 36, "byteLength": 6, "target": 34963},
        ],
        "accessors": [
            {
                "bufferView": 0,
                "componentType": 5126,
                "count": 3,
                "type": "VEC3",
                "min": [0, 0, 0],
                "max": [1, 1, 0],
            },
            {"bufferView": 1, "componentType": 5123, "count": 3, "type": "SCALAR"},
        ],
        "meshes": [{"primitives": [{"attributes": {"POSITION": 0}, "indices": 1}]}],
        "nodes": [{"mesh": 0}],
        "scenes": [{"nodes": [0]}],
        "scene": 0,
    }
    if textured:
        document["bufferViews"] += [
            {"buffer": 0, "byteOffset": 44, "byteLength": len(uv)},
            {"buffer": 0, "byteOffset": 68, "byteLength": len(png)},
        ]
        document["accessors"].append(
            {"bufferView": 2, "componentType": 5126, "count": 3, "type": "VEC2"}
        )
        document["meshes"][0]["primitives"][0].update(
            attributes={"POSITION": 0, "TEXCOORD_0": 2}, material=0
        )
        document["materials"] = [
            {"pbrMetallicRoughness": {"baseColorTexture": {"index": 0}}}
        ]
        document["textures"] = [{"source": 0}]
        document["images"] = [{"bufferView": 3, "mimeType": "image/png"}]
    if skinned:
        document["bufferViews"] += [
            {"buffer": 0, "byteOffset": 44, "byteLength": 12, "target": 34962},
            {"buffer": 0, "byteOffset": 56, "byteLength": 48, "target": 34962},
        ]
        document["accessors"] += [
            {"bufferView": 2, "componentType": 5121, "count": 3, "type": "VEC4"},
            {"bufferView": 3, "componentType": 5126, "count": 3, "type": "VEC4"},
        ]
        document["meshes"][0]["primitives"][0]["attributes"].update(
            JOINTS_0=2, WEIGHTS_0=3
        )
        document["nodes"] = [{"mesh": 0, "skin": 0}, {"name": "root"}]
        document["scenes"] = [{"nodes": [0, 1]}]
        document["skins"] = [{"joints": [1]}]
    document.update(overrides)
    text = json.dumps(document).encode()
    text += b" " * (-len(text) % 4)
    body = struct.pack("<II", len(text), 0x4E4F534A) + text
    body += struct.pack("<II", len(blob), 0x004E4942) + blob
    return struct.pack("<4sII", b"glTF", 2, 12 + len(body)) + body


def _glb_output(content: bytes) -> list[Output]:
    return [Output("mesh.glb", "model/gltf-binary", content)]


def test_a_glb_with_a_triangle_mesh_holds() -> None:
    assertions = [
        {"kind": "format", "format": "glb"},
        {**TRIANGLE_MESH, "minimum_meshes": 1, "minimum_primitives": 1},
    ]
    assert evaluate(assertions, _glb_output(_glb())) == {}


def _lines_mesh() -> dict[str, Any]:
    primitive = {"attributes": {"POSITION": 0}, "indices": 1, "mode": 1}
    return {"meshes": [{"primitives": [primitive]}]}


def _vec2_position() -> dict[str, Any]:
    accessor = {"bufferView": 0, "componentType": 5126, "count": 3, "type": "VEC2"}
    index = {"bufferView": 1, "componentType": 5123, "count": 3, "type": "SCALAR"}
    return {"accessors": [accessor, index]}


@pytest.mark.parametrize(
    ("content", "assertion"),
    [
        pytest.param(_glb(), {**TRIANGLE_MESH, "minimum_primitives": 2}, id="few"),
        pytest.param(_glb(), {**TRIANGLE_MESH, "minimum_meshes": 2}, id="few-meshes"),
        pytest.param(_glb(), {**TRIANGLE_MESH, "profile": "voxel"}, id="profile"),
        pytest.param(_glb(meshes=[]), TRIANGLE_MESH, id="no-meshes"),
        pytest.param(_glb(**_lines_mesh()), TRIANGLE_MESH, id="lines"),
        pytest.param(_glb(**_vec2_position()), TRIANGLE_MESH, id="vec2-position"),
        pytest.param(_glb(asset={"version": "1.0"}), TRIANGLE_MESH, id="gltf-1"),
        pytest.param(_glb()[:-4], TRIANGLE_MESH, id="truncated"),
        pytest.param(b"glTF" + b"\0" * 40, TRIANGLE_MESH, id="not-glb"),
    ],
)
def test_a_glb_without_the_declared_triangle_meshes_fails(
    content: bytes, assertion: dict[str, Any]
) -> None:
    with pytest.raises(AssertionFailed) as missed:
        evaluate([assertion], _glb_output(content))
    assert missed.value.kind == "glb-structure"


def test_an_accessor_beyond_its_buffer_view_fails() -> None:
    index = {"bufferView": 1, "componentType": 5123, "count": 6, "type": "SCALAR"}
    accessors = [_vec2_position()["accessors"][0] | {"type": "VEC3"}, index]
    with pytest.raises(AssertionFailed, match="exceeds its bufferView"):
        evaluate([TRIANGLE_MESH], _glb_output(_glb(accessors=accessors)))


TEXTURED = {"kind": "glb-structure", "profile": "textured"}


def test_a_textured_glb_with_material_texture_and_uv_holds() -> None:
    assert evaluate([TEXTURED], _glb_output(_glb(textured=True))) == {}


@pytest.mark.parametrize(
    "overrides",
    [
        {"materials": []},
        {"materials": [{}]},
        {"textures": []},
        {"textures": [{"source": 9}]},
        {"images": []},
        {"images": [{"bufferView": 9, "mimeType": "image/png"}]},
        {"images": [{"bufferView": 3, "mimeType": "text/plain"}]},
        {"images": [{"uri": "missing.png"}]},
        {"meshes": [{"primitives": [{"attributes": {"POSITION": 0}, "indices": 1}]}]},
        {
            "meshes": [
                {
                    "primitives": [
                        {
                            "attributes": {"POSITION": 0, "TEXCOORD_0": 0},
                            "indices": 1,
                            "material": 0,
                        }
                    ]
                }
            ]
        },
    ],
)
def test_a_textured_glb_requires_bound_materials_textures_and_uv(
    overrides: dict[str, Any],
) -> None:
    with pytest.raises(AssertionFailed) as missed:
        evaluate([TEXTURED], _glb_output(_glb(textured=True, **overrides)))
    assert missed.value.kind == "glb-structure"


@pytest.mark.parametrize("content", [_glb(), _glb(textured=True)[:-4]])
def test_textured_profile_rejects_bare_mesh_and_invalid_binary(content: bytes) -> None:
    with pytest.raises(AssertionFailed):
        evaluate([TEXTURED], _glb_output(content))


TEXTURED_PBR = {"kind": "glb-structure", "profile": "textured-pbr"}


def _pbr_glb() -> bytes:
    return _glb(
        textured=True,
        materials=[
            {
                "pbrMetallicRoughness": {
                    "baseColorTexture": {"index": 0},
                    "metallicRoughnessTexture": {"index": 1},
                }
            }
        ],
        textures=[{"source": 0}, {"source": 1}],
        images=[
            {"bufferView": 3, "mimeType": "image/png"},
            {"bufferView": 3, "mimeType": "image/png"},
        ],
    )


def test_textured_pbr_checks_both_material_channels() -> None:
    assert evaluate([TEXTURED_PBR], _glb_output(_pbr_glb())) == {}
    with pytest.raises(AssertionFailed, match="metallicRoughnessTexture"):
        evaluate([TEXTURED_PBR], _glb_output(_glb(textured=True)))


@pytest.mark.parametrize(
    "info", [{"index": 9}, {"index": 0}, {"index": 1, "texCoord": 1}]
)
def test_textured_pbr_rejects_invalid_or_shared_metallic_roughness(
    info: dict[str, Any],
) -> None:
    content = _glb(
        textured=True,
        materials=[
            {
                "pbrMetallicRoughness": {
                    "baseColorTexture": {"index": 0},
                    "metallicRoughnessTexture": info,
                }
            }
        ],
        textures=[{"source": 0}, {"source": 0}],
    )
    with pytest.raises(AssertionFailed):
        evaluate([TEXTURED_PBR], _glb_output(content))


SKINNED = {"kind": "glb-structure", "profile": "skinned"}


def test_a_skinned_glb_with_a_bound_skin_and_vertex_weights_holds() -> None:
    """Catches the sweep failing SkinTokens' rigged output as an unchecked profile."""
    assert evaluate([SKINNED], _glb_output(_glb(skinned=True))) == {}


def _skinned_attributes(**attributes: int) -> dict[str, Any]:
    primitive = {"attributes": {"POSITION": 0, **attributes}, "indices": 1}
    return {"meshes": [{"primitives": [primitive]}]}


def _skinned_accessors(joint_count: int) -> dict[str, Any]:
    position = {"bufferView": 0, "componentType": 5126, "count": 3, "type": "VEC3"}
    index = {"bufferView": 1, "componentType": 5123, "count": 3, "type": "SCALAR"}
    joints = {"bufferView": 2, "componentType": 5121, "type": "VEC4"}
    weights = {"bufferView": 3, "componentType": 5126, "count": 3, "type": "VEC4"}
    return {"accessors": [position, index, joints | {"count": joint_count}, weights]}


@pytest.mark.parametrize(
    "overrides",
    [
        pytest.param({"skins": []}, id="no-skin"),
        pytest.param({"skins": [{"joints": []}]}, id="no-joints"),
        pytest.param({"skins": [{"joints": [9]}]}, id="unknown-joint"),
        pytest.param({"skins": [{"joints": [1, 1]}]}, id="duplicate-joints"),
        pytest.param({"nodes": [{"mesh": 0}, {}]}, id="unbound-mesh"),
        pytest.param({"nodes": [{}, {}]}, id="no-mesh-node"),
        pytest.param(_skinned_attributes(WEIGHTS_0=3), id="no-joints-attribute"),
        pytest.param(_skinned_attributes(JOINTS_0=2), id="no-weights-attribute"),
        pytest.param(
            _skinned_attributes(JOINTS_0=3, WEIGHTS_0=3), id="float-joints-attribute"
        ),
        pytest.param(
            _skinned_attributes(JOINTS_0=2, WEIGHTS_0=2), id="byte-weights-attribute"
        ),
        pytest.param(_skinned_accessors(2), id="joints-count"),
    ],
)
def test_a_skinned_glb_requires_a_bound_skin_with_joints_and_vertex_weights(
    overrides: dict[str, Any],
) -> None:
    with pytest.raises(AssertionFailed) as missed:
        evaluate([SKINNED], _glb_output(_glb(skinned=True, **overrides)))
    assert missed.value.kind == "glb-structure"


def test_skintokens_mesh_only_input_is_a_mesh_but_not_skinned() -> None:
    """The job's input fixture has no rig: only the job's output can pass ``skinned``."""
    fixtures = Path(__file__).resolve().parents[1] / "qualification/fixtures"
    content = (fixtures / "skintokens-rigged-figure.glb").read_bytes()
    assert evaluate([TRIANGLE_MESH], _glb_output(content)) == {}
    with pytest.raises(AssertionFailed, match="skins"):
        evaluate([SKINNED], _glb_output(content))


#: A GLB that holds for each contract profile; a new profile needs one, and a checker.
PROFILE_EXAMPLES = {
    QualificationGlbProfile.TRIANGLE_MESH: _glb(),
    QualificationGlbProfile.TEXTURED: _glb(textured=True),
    QualificationGlbProfile.TEXTURED_PBR: _pbr_glb(),
    QualificationGlbProfile.SKINNED: _glb(skinned=True),
}


@pytest.mark.parametrize("profile", list(QualificationGlbProfile))
def test_the_sweep_checks_every_glb_profile_the_contract_allows(
    profile: QualificationGlbProfile,
) -> None:
    """Catches a contract profile the sweep fails as unchecked (SkinTokens' ``skinned``)."""
    assert profile in job_media.GLB_PROFILE_CHECKS
    assertion = {"kind": "glb-structure", "profile": profile.value}
    assert evaluate([assertion], _glb_output(PROFILE_EXAMPLES[profile])) == {}


@pytest.mark.parametrize("fallback", [False, True])
def test_pixal3d_webp_export_passes_the_sweep(fallback: bool) -> None:
    """Catches the sweep rejecting the adapter's required WebP texture sources."""
    content = (
        Path(__file__).parent / "fixtures/pixal3d-trimesh-4.10.1-padded.glb"
    ).read_bytes()
    document, blob = job_media._glb_chunks(content)
    if fallback:
        document["images"].append({"bufferView": 2, "mimeType": "image/png"})
        for texture in document["textures"]:
            texture["source"] = 2
    evaluate([TEXTURED_PBR], _glb_output(_repack_glb(document, blob)))


def _repack_glb(document: dict[str, Any], blob: bytes) -> bytes:
    text = json.dumps(document).encode()
    text += b" " * (-len(text) % 4)
    body = struct.pack("<II", len(text), 0x4E4F534A) + text
    body += struct.pack("<II", len(blob), 0x004E4942) + blob
    return struct.pack("<4sII", b"glTF", 2, 12 + len(body)) + body


@pytest.mark.parametrize(
    "damage",
    ["used", "required", "index", "mime", "uri", "shared", "extension", "core"],
)
def test_sweep_rejects_broken_webp_texture_references(damage: str) -> None:
    """Extension support must preserve declaration, source and distinct-image checks."""
    content = (
        Path(__file__).parent / "fixtures/pixal3d-trimesh-4.10.1-padded.glb"
    ).read_bytes()
    document, blob = job_media._glb_chunks(content)
    if damage == "used":
        document["extensionsUsed"] = []
    elif damage == "required":
        document["extensionsRequired"] = []
    elif damage == "index":
        document["textures"][0]["extensions"]["EXT_texture_webp"]["source"] = 9
    elif damage == "mime":
        document["images"][0]["mimeType"] = "image/png"
    elif damage == "uri":
        document["images"][0]["uri"] = "external.webp"
    elif damage == "shared":
        document["textures"][1]["extensions"]["EXT_texture_webp"]["source"] = 0
    elif damage == "extension":
        document["textures"][0]["extensions"]["EXT_texture_webp"] = None
    elif damage == "core":
        document["textures"][0]["source"] = 9
    with pytest.raises(AssertionFailed):
        evaluate([TEXTURED_PBR], _glb_output(_repack_glb(document, blob)))


# -- realtime-transcript -----------------------------------------------------------------

REVISION = "06b067617677661194cf837970fe3a10f1a0e56d"
TRANSCRIPT = {
    "kind": "realtime-transcript",
    "media_type": "application/x-ndjson",
    "model_revision": REVISION,
}


def _record(record_type: str, **fields: Any) -> dict[str, Any]:
    return {"type": record_type, "elapsed_seconds": 0.5, **fields}


def _frame(index: int) -> dict[str, Any]:
    return _record(
        "frame-ack", event_index=index, timestamp=float(index), dropped_oldest=False
    )


def _transcript(*records: dict[str, Any]) -> list[Output]:
    lines = [
        json.dumps({"sequence": sequence, **record})
        for sequence, record in enumerate(records)
    ]
    content = ("\n".join(lines) + "\n").encode()
    return [Output("transcript.jsonl", "application/x-ndjson", content)]


START = _record("session-start", model_revision=REVISION)
STOP = _record("session-stop")
TEXT = _record("output", kind="text", text="a red square")


def test_a_realtime_transcript_in_order_holds() -> None:
    assert evaluate([TRANSCRIPT], _transcript(START, _frame(0), TEXT, STOP)) == {}
    two_frames = _transcript(START, _frame(0), _frame(1), STOP, TEXT)
    assert evaluate([{**TRANSCRIPT, "frame_count": 2}], two_frames) == {}


@pytest.mark.parametrize(
    ("outputs", "assertion"),
    [
        pytest.param(
            _transcript(START, _frame(0), STOP),
            {**TRANSCRIPT, "model_revision": "0" * 40},
            id="other-revision",
        ),
        pytest.param(
            _transcript(START, _frame(0), STOP),
            {**TRANSCRIPT, "frame_count": 2},
            id="frames-missing",
        ),
        pytest.param(
            _transcript(START, STOP, _frame(0)), TRANSCRIPT, id="stop-before-frame"
        ),
        pytest.param(
            _transcript(START, _frame(0), STOP, STOP), TRANSCRIPT, id="two-stops"
        ),
        pytest.param(_transcript(START, _frame(1), STOP), TRANSCRIPT, id="frame-index"),
        pytest.param(
            _transcript(START, _frame(0), _record("output", kind="x", text=""), STOP),
            TRANSCRIPT,
            id="output-kind",
        ),
        pytest.param(
            _transcript(START, _frame(0) | {"extra": 1}, STOP),
            TRANSCRIPT,
            id="record-shape",
        ),
        pytest.param(
            _transcript(START | {"elapsed_seconds": 1.0}, _frame(0), STOP),
            TRANSCRIPT,
            id="elapsed-backwards",
        ),
        pytest.param(_transcript(START, STOP), TRANSCRIPT, id="too-short"),
    ],
)
def test_a_realtime_transcript_out_of_contract_fails(
    outputs: list[Output], assertion: dict[str, Any]
) -> None:
    with pytest.raises(AssertionFailed) as missed:
        evaluate([assertion], outputs)
    assert missed.value.kind == "realtime-transcript"


def test_a_transcript_record_with_a_duplicate_key_fails() -> None:
    lines = [json.dumps({"sequence": 0, **START})]
    lines.append('{"sequence": 1, "sequence": 1, "type": "session-stop"}')
    lines.append(json.dumps({"sequence": 2, **TEXT}))
    outputs = [Output("t.jsonl", "application/x-ndjson", "\n".join(lines).encode())]
    with pytest.raises(AssertionFailed):
        evaluate([TRANSCRIPT], outputs)


# -- video-metadata ----------------------------------------------------------------------

VIDEO = {
    "kind": "video-metadata",
    "media_type": "video/mp4",
    "width": 64,
    "height": 48,
    "fps": 24,
    "fps_tolerance": 0.01,
    "frame_count": 24,
    "codec": "h264",
    "pixel_format": "yuv420p",
    "stream_count": 1,
    "audio_streams": 0,
    "minimum_duration_seconds": 0.99,
    "maximum_duration_seconds": 1.01,
}
WITH_AUDIO = {
    **VIDEO,
    "stream_count": 2,
    "audio_streams": 1,
    "audio_codec": "aac",
    "audio_channels": 1,
    "audio_sample_rate": 48000,
    "minimum_audio_duration_seconds": 0.9,
    "maximum_audio_duration_seconds": 1.1,
    "maximum_av_sync_delta_seconds": 0.1,
    "minimum_container_duration_seconds": 0.9,
    "maximum_container_duration_seconds": 1.1,
}


def _mp4(tmp_path: Path, *, audio: bool = False) -> list[Output]:
    """A one-second 64x48 H.264 MP4 at 24 fps, made here; skipped without ffmpeg/x264."""
    if shutil.which("ffmpeg") is None or job_media.ffprobe() is None:
        pytest.skip("ffmpeg and ffprobe are not installed on this host")
    path = tmp_path / "out.mp4"
    argv = ["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "testsrc=size=64x48:rate=24"]
    if audio:
        argv += ["-f", "lavfi", "-i", "sine=frequency=440:sample_rate=48000"]
        argv += ["-c:a", "aac", "-ac", "1", "-t", "1"]
    argv += ["-frames:v", "24", "-c:v", "libx264", "-pix_fmt", "yuv420p", str(path)]
    if subprocess.run(argv, capture_output=True, check=False).returncode != 0:
        pytest.skip("this host's ffmpeg cannot encode H.264 (libx264)")
    return [Output("out.mp4", "video/mp4", path.read_bytes())]


def test_a_video_whose_metadata_matches_holds(tmp_path: Path) -> None:
    assert evaluate([VIDEO], _mp4(tmp_path)) == {}


def test_a_video_with_its_audio_track_holds(tmp_path: Path) -> None:
    assert evaluate([WITH_AUDIO], _mp4(tmp_path, audio=True)) == {}


@pytest.mark.parametrize(
    "change",
    [
        {"width": 65},
        {"frame_count": 25},
        {"fps": 25},
        {"codec": "hevc"},
        {"audio_streams": 1},
        {"minimum_duration_seconds": 2.0, "maximum_duration_seconds": 3.0},
    ],
    ids=lambda change: next(iter(change)),
)
def test_a_video_whose_metadata_differs_fails(
    tmp_path: Path, change: dict[str, Any]
) -> None:
    with pytest.raises(AssertionFailed) as missed:
        evaluate([{**VIDEO, **change}], _mp4(tmp_path))
    assert missed.value.kind == "video-metadata"


def test_a_video_with_audio_out_of_sync_or_rate_fails(tmp_path: Path) -> None:
    outputs = _mp4(tmp_path, audio=True)
    for change in ({"audio_sample_rate": 44100}, {"audio_channels": 2}):
        with pytest.raises(AssertionFailed):
            evaluate([{**WITH_AUDIO, **change}], outputs)


def test_bytes_ffprobe_cannot_read_fail(tmp_path: Path) -> None:
    if job_media.ffprobe() is None:
        pytest.skip("ffprobe is not installed on this host")
    with pytest.raises(AssertionFailed, match="ffprobe rejected"):
        evaluate([VIDEO], [Output("out.mp4", "video/mp4", b"not a video")])


MP4_STUB = [Output("out.mp4", "video/mp4", b"\0\0\0\x18ftypisom moov")]


def test_without_ffprobe_a_video_is_not_evaluated_never_passed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(job_media.shutil, "which", lambda name: None)
    assertions = [{"kind": "file-count", "exact": 1}, VIDEO]
    assert evaluate(assertions, MP4_STUB) == {"video-metadata": "ffprobe-missing"}
    # What is decidable without ffprobe still fails: no MP4 selected at all.
    png = [Output("a.png", "image/png", b"\x89PNG")]
    with pytest.raises(AssertionFailed, match="selected no output"):
        evaluate([VIDEO], png)


def test_an_ffprobe_that_cannot_run_is_not_evaluated(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def timeout(*args: Any, **kwargs: Any) -> Any:
        raise subprocess.TimeoutExpired("ffprobe", 30)

    monkeypatch.setattr(job_media.shutil, "which", lambda name: "/usr/bin/ffprobe")
    monkeypatch.setattr(job_media.subprocess, "run", timeout)
    assert evaluate([VIDEO], MP4_STUB) == {"video-metadata": "ffprobe-failed"}
