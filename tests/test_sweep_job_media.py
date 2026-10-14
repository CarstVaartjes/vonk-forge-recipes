"""A job's media outputs: a GLB's triangle meshes, an MP4's metadata, a realtime transcript."""

from __future__ import annotations

import json
import shutil
import struct
import subprocess
from pathlib import Path
from typing import Any

import pytest

from spark_sweep import job_media
from spark_sweep.job_assertions import Output, evaluate
from spark_sweep.smoke import AssertionFailed

# -- glb-structure -----------------------------------------------------------------------

TRIANGLE_MESH = {"kind": "glb-structure", "profile": "triangle-mesh"}


def _glb(**overrides: Any) -> bytes:
    """One indexed triangle: a FLOAT VEC3 POSITION and three UNSIGNED_SHORT indices."""
    positions = struct.pack("<9f", 0, 0, 0, 1, 0, 0, 0, 1, 0)
    indices = struct.pack("<3H", 0, 1, 2) + b"\0\0"
    blob = positions + indices
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
        **overrides,
    }
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
        pytest.param(_glb(), {**TRIANGLE_MESH, "profile": "skinned"}, id="profile"),
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
