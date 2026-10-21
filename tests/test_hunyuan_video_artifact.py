"""Exercise artifact behavior without loading models or accessing Spark hardware."""

from __future__ import annotations

import json
import math
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from test_hunyuan_single_recipes import video_module


@pytest.mark.parametrize("frame_count", [1, 121])
def test_square_i2v_job_publishes_native_bucket(
    tmp_path: Path, monkeypatch, frame_count
):
    """The pre-fix main() rejects this 640x640 probe as an invalid 480p MP4."""
    module = video_module()
    np = __import__("numpy")
    frames = np.broadcast_to(
        np.array([[[0.5, 0.5, 0.5]]], dtype="float32"), (frame_count, 640, 640, 3)
    )

    class HunyuanVideo15ImageToVideoPipeline:
        guider = SimpleNamespace(guidance_scale=1.0)
        vae = MagicMock()
        enable_model_cpu_offload = MagicMock()
        target_size = 640
        video_processor = MagicMock()

        def __call__(self, **kwargs):
            assert kwargs["num_frames"] == frame_count
            assert kwargs["num_inference_steps"] == 8
            return SimpleNamespace(frames=[frames])

    pipe = HunyuanVideo15ImageToVideoPipeline()
    pipe.video_processor.calculate_default_height_width.return_value = (640, 640)
    factory = SimpleNamespace(from_pretrained=MagicMock(return_value=pipe))
    index = tmp_path / "model_index.json"
    index.write_text("{}")
    monkeypatch.setattr(module, "_MODEL_INDEX", index)
    monkeypatch.setattr(module, "HunyuanVideo15ImageToVideoPipeline", factory)
    monkeypatch.setattr(module, "_one_prompt", lambda path: "a moving blue square")
    monkeypatch.setattr(
        module, "_one_image", lambda path: SimpleNamespace(size=(64, 64))
    )
    monkeypatch.setattr(module.torch, "bfloat16", "bfloat16", raising=False)
    monkeypatch.setattr(module.torch, "Generator", MagicMock(), raising=False)
    monkeypatch.setattr(
        "sys.argv",
        [
            "diffusers-job",
            "--pipeline",
            "image-to-image",
            "--output-mime",
            "video/mp4",
            "--num-inference-steps",
            "8",
            "--guidance-scale",
            "1.0",
            "--resolution",
            "480",
            "--seed",
            "1",
            "--num-frames",
            str(frame_count),
            "--output-dir",
            str(tmp_path),
        ],
    )

    def export(video, path, **kwargs):
        assert video is frames
        path.write_bytes(b"encoded")

    probe = {
        "streams": [
            {
                "codec_type": "video",
                "codec_name": "h264",
                "pix_fmt": "yuv420p",
                "width": 640,
                "height": 640,
                "nb_read_frames": str(frame_count),
                "avg_frame_rate": "24/1",
            }
        ],
        "format": {
            "format_name": "mov,mp4,m4a,3gp,3g2,mj2",
            "duration": str(frame_count / 24),
        },
    }
    with (
        patch.object(module, "export_to_video", side_effect=export),
        patch.object(
            module.subprocess,
            "run",
            return_value=SimpleNamespace(stdout=json.dumps(probe), stderr=""),
        ),
    ):
        module.main()
    assert (tmp_path / "output.mp4").read_bytes() == b"encoded"
    assert not (tmp_path / ".output.tmp.mp4").exists()


@pytest.mark.parametrize(
    ("image_size", "aspect", "target", "native_size"),
    [
        ((64, 64), None, 640, (640, 640)),
        ((480, 848), None, 640, (480, 848)),
        (None, (16, 9), 640, (848, 480)),
        (None, (16, 9), 960, (1280, 720)),
    ],
)
def test_native_size_uses_pipeline_bucket(image_size, aspect, target, native_size):
    # Pinned Diffusers image_processor.py selects by area and aspect ratio.
    # The signed I2V fixture is 64x64: its native bucket is 640x640, not 480p
    # on the short edge. Both T2V recipes use this same processor and adapter.
    module = video_module()
    processor = MagicMock()
    processor.calculate_default_height_width.return_value = native_size[::-1]
    pipe = SimpleNamespace(
        video_processor=processor, target_size=target, default_aspect_ratio=aspect
    )
    image = SimpleNamespace(size=image_size) if image_size else None
    assert module._expected_size(pipe, image) == native_size
    width, height = image_size or aspect
    processor.calculate_default_height_width.assert_called_once_with(
        height=height, width=width, target_size=target
    )


@pytest.mark.parametrize("size", [(640, 640), (480, 848), (848, 480), (1280, 720)])
def test_real_121_frame_native_mp4_passes(tmp_path: Path, size):
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        pytest.skip("ffmpeg and ffprobe are needed for the real MP4 test")
    module = video_module()
    width, height = size
    output = tmp_path / "native.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            f"color=c=blue:s={width}x{height}:r=24",
            "-frames:v",
            "121",
            "-c:v",
            "libx264",
            "-threads",
            "1",
            "-pix_fmt",
            "yuv420p",
            str(output),
        ],
        check=True,
        capture_output=True,
        timeout=30,
    )
    metrics = module.FrameMetrics(121, (121, height, width, 3), 0.0722, 0, 0, 0)
    module._validate_video(output, size, metrics)
    with pytest.raises(SystemExit, match="unexpected MP4 stream contract") as error:
        module._validate_video(output, (width + 16, height), metrics)
    message = str(error.value)
    for fact in (
        '"nb_read_frames": "121"',
        '"codec_name": "h264"',
        '"pix_fmt": "yuv420p"',
        '"mean_luminance":',
        '"nan_count": 0',
        "bytes=",
    ):
        assert fact in message


@pytest.mark.parametrize("value", [float("nan"), float("inf"), 0.0])
def test_bad_decoded_frames_fail_before_lossy_export(tmp_path: Path, value: float):
    module = video_module()
    frames = module.np.full((121, 16, 16, 3), 0.5, dtype="float32")
    if value == 0:
        frames[:] = value
    else:
        frames[0, 0, 0, 0] = value
    temporary = tmp_path / ".output.tmp.mp4"
    temporary.write_bytes(b"stale")
    with (
        patch.object(module, "export_to_video") as export,
        pytest.raises(SystemExit, match="invalid decoded RGB frames") as error,
    ):
        module._write_video(frames, tmp_path, (16, 16))
    export.assert_not_called()
    assert not temporary.exists()
    assert not (tmp_path / "output.mp4").exists()
    message = str(error.value)
    assert '"frames": 121' in message
    assert f'"nan_count": {int(math.isnan(value))}' in message
    assert f'"inf_count": {int(math.isinf(value))}' in message
    assert f'"black_frames": {121 if value == 0 else 0}' in message
    assert '"mean_luminance":' in message


def test_dark_finite_frames_publish_only_after_probe(tmp_path: Path):
    module = video_module()
    frames = module.np.full((121, 16, 16, 3), 1 / 255, dtype="float32")
    temporary = tmp_path / ".output.tmp.mp4"

    def export(video, path, **kwargs):
        assert video is frames
        assert kwargs == {"fps": 24, "macro_block_size": 1}
        path.write_bytes(b"encoded")

    with (
        patch.object(module, "export_to_video", side_effect=export),
        patch.object(module, "_validate_video") as validate,
    ):
        module._write_video(frames, tmp_path, (16, 16))
    assert validate.call_args is not None
    metrics = validate.call_args.args[2]
    assert metrics.mean_luminance == pytest.approx(1 / 255)
    assert metrics.nan_count == metrics.inf_count == metrics.black_frames == 0
    assert (tmp_path / "output.mp4").read_bytes() == b"encoded"
    assert not temporary.exists()


@pytest.mark.parametrize(
    "probe",
    [
        subprocess.CalledProcessError(1, "ffprobe", stderr="moov atom not found"),
        subprocess.TimeoutExpired("ffprobe", 120),
        FileNotFoundError("ffprobe"),
        SimpleNamespace(stdout="invalid-json", stderr=""),
        SimpleNamespace(stdout=json.dumps({"streams": []}), stderr=""),
    ],
)
def test_probe_failure_keeps_evidence_and_does_not_publish(tmp_path: Path, probe):
    module = video_module()
    frames = module.np.full((121, 16, 16, 3), 0.5, dtype="float32")

    def export(video, path, **kwargs):
        path.write_bytes(b"bad-container")

    with (
        patch.object(module, "export_to_video", side_effect=export),
        patch.object(module.subprocess, "run", side_effect=probe)
        if isinstance(probe, Exception)
        else patch.object(module.subprocess, "run", return_value=probe),
        pytest.raises(SystemExit) as error,
    ):
        module._write_video(frames, tmp_path, (16, 16))
    message = str(error.value)
    assert "bytes=13" in message
    assert '"mean_luminance": 0.5' in message
    assert '"nan_count": 0' in message
    assert "probe=" in message
    if isinstance(probe, subprocess.CalledProcessError):
        assert "moov atom not found" in message
    assert not (tmp_path / ".output.tmp.mp4").exists()
    assert not (tmp_path / "output.mp4").exists()


@pytest.mark.parametrize(
    ("section", "field", "value"),
    [
        ("stream", "nb_read_frames", "120"),
        ("stream", "codec_name", "hevc"),
        ("stream", "pix_fmt", "yuv444p"),
        ("stream", "avg_frame_rate", "30/1"),
        ("format", "format_name", "matroska,webm"),
        ("format", "duration", "4.0"),
        ("format", "duration", "nan"),
    ],
)
def test_wrong_encoded_properties_report_measured_facts(
    tmp_path: Path, section: str, field: str, value: str
):
    module = video_module()
    stream = {
        "codec_type": "video",
        "codec_name": "h264",
        "pix_fmt": "yuv420p",
        "width": 640,
        "height": 640,
        "nb_read_frames": "121",
        "avg_frame_rate": "24/1",
    }
    container = {"format_name": "mov,mp4,m4a,3gp,3g2,mj2", "duration": "5.041667"}
    (stream if section == "stream" else container)[field] = value
    probe = json.dumps({"streams": [stream], "format": container})
    output = tmp_path / "bad.mp4"
    output.write_bytes(b"encoded")
    metrics = module.FrameMetrics(121, (121, 640, 640, 3), 0.5, 0, 0, 0)
    with (
        patch.object(
            module.subprocess,
            "run",
            return_value=SimpleNamespace(stdout=probe, stderr=""),
        ),
        pytest.raises(SystemExit, match="unexpected MP4 stream contract") as error,
    ):
        module._validate_video(output, (640, 640), metrics)
    assert f'"{field}": "{value}"' in str(error.value)
