"""Offline MOVA text/image-to-synchronized-video-and-audio job adapter."""

from __future__ import annotations

import argparse
from fractions import Fraction
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile


_ENTRYPOINT = "/opt/vonk/source/pipelines/run.py"
_MODEL_INDEX = Path("/models/model_index.json")
_OFFICIAL_RUNNER = Path("/opt/mova-source/scripts/inference_single.py")
_IMAGE_SUFFIXES = frozenset({".jpg", ".jpeg", ".png", ".webp"})
_MAX_INPUT_BYTES = 16 * 1024 * 1024
_OUTPUT_FRAMES = 193
_OUTPUT_FPS = 24
_VARIANTS = {
    "checkpoints/MOVA-360p": (352, 640),
    "checkpoints/MOVA-720p": (720, 1280),
}


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--entrypoint", required=True)
    parser.add_argument("--output-mime", required=True)
    parser.add_argument("--timeout-seconds", type=int, default=3600)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.entrypoint != _ENTRYPOINT:
        raise SystemExit("unexpected signed-bundle entrypoint")
    if args.output_mime != "video/mp4":
        raise SystemExit("MOVA emits synchronized video/audio as video/mp4 only")
    if not 1 <= args.timeout_seconds <= 3600:
        raise SystemExit("timeout must be between 1 and 3600 seconds")
    if not 0 <= args.seed < 2**63:
        raise SystemExit("seed is outside the harness contract")
    return args


def _variant() -> tuple[int, int]:
    try:
        model_index = json.loads(_MODEL_INDEX.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise SystemExit(f"cannot read the pinned MOVA model manifest: {exc}") from exc
    name = model_index.get("_name_or_path")
    resolution = _VARIANTS.get(name)
    if resolution is None:
        raise SystemExit(f"unsupported MOVA checkpoint identity: {name!r}")
    return resolution


def _reference_frame(directory: Path, height: int, width: int) -> Path:
    input_directory = Path("/inputs")
    images = (
        sorted(
            path
            for path in input_directory.iterdir()
            if path.is_file() and path.suffix.lower() in _IMAGE_SUFFIXES
        )
        if input_directory.is_dir()
        else []
    )
    if len(images) > 1:
        raise SystemExit("MOVA accepts at most one reference image")
    destination = directory / "reference.png"
    from PIL import Image

    if images:
        if images[0].stat().st_size > _MAX_INPUT_BYTES:
            raise SystemExit("the MOVA reference image exceeds 16 MiB")
        with Image.open(images[0]) as image:
            image.convert("RGB").save(destination, format="PNG")
    else:
        # Upstream explicitly defines a white first frame as its T2VA path.
        Image.new("RGB", (width, height), color="white").save(destination, format="PNG")
    return destination


def _stream_duration(stream: dict[str, object]) -> float | None:
    value = stream.get("duration")
    try:
        duration = float(value) if value is not None else None
    except (TypeError, ValueError):
        return None
    return duration if duration is not None and duration > 0 else None


def _verify_synchronized_mp4(path: Path, *, height: int, width: int) -> None:
    if not path.is_file() or path.stat().st_size < 1024:
        raise SystemExit("MOVA did not produce a non-empty MP4 artifact")
    probe = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_streams",
            "-show_format",
            "-of",
            "json",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    document = json.loads(probe.stdout)
    media_format = document.get("format")
    if not isinstance(media_format, dict) or "mp4" not in str(
        media_format.get("format_name", "")
    ).split(","):
        raise SystemExit("MOVA output must use the MP4 container")
    streams = document.get("streams")
    if not isinstance(streams, list):
        raise SystemExit("ffprobe did not return an MP4 stream inventory")
    video = next((item for item in streams if item.get("codec_type") == "video"), None)
    audio = next((item for item in streams if item.get("codec_type") == "audio"), None)
    if not isinstance(video, dict) or not isinstance(audio, dict):
        raise SystemExit("MOVA output must contain both video and audio streams")
    if video.get("height") != height or video.get("width") != width:
        raise SystemExit("MOVA output video dimensions do not match the recipe")
    try:
        frame_rate = float(Fraction(str(video.get("avg_frame_rate"))))
        frame_count = int(str(video.get("nb_frames")))
    except (TypeError, ValueError, ZeroDivisionError) as exc:
        raise SystemExit("MOVA output must expose a bounded frame contract") from exc
    if abs(frame_rate - _OUTPUT_FPS) > 0.01 or frame_count != _OUTPUT_FRAMES:
        raise SystemExit("MOVA output must contain exactly 193 frames at 24 fps")
    if audio.get("sample_rate") != "48000":
        raise SystemExit("MOVA output audio must retain the native 48 kHz sample rate")
    video_duration = _stream_duration(video)
    audio_duration = _stream_duration(audio)
    if video_duration is None or audio_duration is None:
        raise SystemExit("MOVA output streams must expose bounded durations")
    expected_duration = _OUTPUT_FRAMES / _OUTPUT_FPS
    if abs(video_duration - expected_duration) > 1.0 / _OUTPUT_FPS:
        raise SystemExit("MOVA output video duration does not match 193 frames at 24 fps")
    if abs(audio_duration - expected_duration) > 0.25:
        raise SystemExit("MOVA output audio duration is outside the synchronized bound")
    if abs(video_duration - audio_duration) > max(0.25, 1.0 / 24.0):
        raise SystemExit("MOVA output audio and video durations are not synchronized")


def main() -> None:
    args = _arguments()
    height, width = _variant()
    prompt = os.environ.get(
        "VONK_PROMPT",
        "A quiet alpine lake at sunrise, with natural wind and water sounds.",
    ).strip()
    if not prompt or len(prompt) > 4096:
        raise SystemExit("prompt must contain between 1 and 4096 characters")
    if not _OFFICIAL_RUNNER.is_file():
        raise SystemExit("the pinned official MOVA inference runner is missing")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    if any(args.output_dir.iterdir()):
        raise SystemExit("MOVA requires a fresh output directory for atomic publication")
    final_output = args.output_dir / "mova.mp4"
    with tempfile.TemporaryDirectory(prefix="vonk-mova-") as temporary:
        temporary_path = Path(temporary)
        reference = _reference_frame(temporary_path, height, width)
        staged_output = temporary_path / "mova.mp4"
        command = [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nnodes=1",
            "--nproc-per-node=1",
            str(_OFFICIAL_RUNNER),
            "--ckpt_path",
            "/models",
            "--prompt",
            prompt,
            "--ref_path",
            str(reference),
            "--output_path",
            str(staged_output),
            "--num_frames",
            str(_OUTPUT_FRAMES),
            "--fps",
            str(_OUTPUT_FPS),
            "--height",
            str(height),
            "--width",
            str(width),
            "--seed",
            str(args.seed),
            "--num_inference_steps",
            "50",
            "--cfg_scale",
            "5",
            "--sigma_shift",
            "5",
            "--cp_size",
            "1",
            "--offload",
            "cpu",
            "--remove_video_dit",
        ]
        environment = dict(os.environ)
        environment.update(
            {
                "HF_HUB_OFFLINE": "1",
                "HF_HUB_DISABLE_TELEMETRY": "1",
                "TRANSFORMERS_OFFLINE": "1",
            }
        )
        subprocess.run(command, check=True, env=environment, timeout=args.timeout_seconds)
        _verify_synchronized_mp4(staged_output, height=height, width=width)
        temporary_final = args.output_dir / ".mova.mp4.complete"
        shutil.copyfile(staged_output, temporary_final)
        os.replace(temporary_final, final_output)


if __name__ == "__main__":
    main()
