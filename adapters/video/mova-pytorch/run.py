"""Offline MOVA text/image-to-synchronized-video-and-audio job adapter."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from fractions import Fraction
from pathlib import Path

_ENTRYPOINT = "/opt/vonk/source/pipelines/run.py"
_MODEL_INDEX = Path("/models/model_index.json")
_OFFICIAL_RUNNER = Path("/opt/mova-source/scripts/inference_single.py")
_INPUT_ROOT = Path("/inputs")
_IMAGE_SUFFIXES = frozenset({".jpg", ".jpeg", ".png", ".webp"})
_MAX_INPUT_BYTES = 16 * 1024 * 1024
_MAX_MANIFEST_BYTES = 256 * 1024
_MAX_PROMPT_BYTES = 16 * 1024
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


def _input_manifest() -> dict[str, object] | None:
    path = _INPUT_ROOT / "manifest.json"
    if not path.exists():
        return None
    if (
        path.is_symlink()
        or not path.is_file()
        or path.stat().st_size > _MAX_MANIFEST_BYTES
    ):
        raise SystemExit("input manifest must be a bounded regular file")
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise SystemExit("input manifest must contain UTF-8 JSON") from error
    if not isinstance(document, dict) or set(document) != {
        "schema_version",
        "total_bytes",
        "files",
    }:
        raise SystemExit("input manifest shape is invalid")
    files = document["files"]
    if document["schema_version"] != 1 or not isinstance(files, list):
        raise SystemExit("input manifest version is invalid")
    names: set[str] = set()
    total = 0
    for item in files:
        if not isinstance(item, dict) or set(item) != {
            "slot",
            "name",
            "media_type",
            "size_bytes",
            "sha256",
        }:
            raise SystemExit("input manifest file declaration is invalid")
        name = item["name"]
        size = item["size_bytes"]
        if (
            not isinstance(name, str)
            or Path(name).name != name
            or name == "manifest.json"
            or name in names
            or not isinstance(item["slot"], str)
            or type(size) is not int
            or size < 0
        ):
            raise SystemExit("input manifest file identity is invalid")
        candidate = _INPUT_ROOT / name
        if (
            candidate.is_symlink()
            or not candidate.is_file()
            or candidate.stat().st_size != size
        ):
            raise SystemExit(f"staged input does not match its manifest: {name}")
        if hashlib.sha256(candidate.read_bytes()).hexdigest() != item["sha256"]:
            raise SystemExit(f"staged input digest does not match its manifest: {name}")
        names.add(name)
        total += size
    if document["total_bytes"] != total:
        raise SystemExit("input manifest total_bytes does not match staged files")
    return document


def _slot_files(manifest: dict[str, object], slot: str) -> list[Path]:
    files = manifest["files"]
    assert isinstance(files, list)
    return [_INPUT_ROOT / item["name"] for item in files if item["slot"] == slot]


def _prompt(manifest: dict[str, object] | None) -> str:
    if manifest is None:
        value = os.environ.get("VONK_PROMPT", "")
    else:
        files = _slot_files(manifest, "prompt")
        if len(files) != 1:
            raise SystemExit("MOVA requires exactly one prompt text file")
        if files[0].stat().st_size > _MAX_PROMPT_BYTES:
            raise SystemExit("the MOVA prompt exceeds 16 KiB")
        try:
            value = files[0].read_text(encoding="utf-8")
        except UnicodeDecodeError as error:
            raise SystemExit("the MOVA prompt must be UTF-8 text") from error
    prompt = value.strip()
    if not prompt or len(prompt) > 4096:
        raise SystemExit("prompt must contain between 1 and 4096 characters")
    return prompt


def _reference_frame(
    directory: Path,
    height: int,
    width: int,
    manifest: dict[str, object] | None,
) -> Path:
    input_directory = _INPUT_ROOT
    if manifest is not None:
        images = _slot_files(manifest, "reference-image")
    else:
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
        if images[0].suffix.lower() not in _IMAGE_SUFFIXES:
            raise SystemExit("the MOVA reference image has an unsupported extension")
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
            "-count_frames",
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
    videos = [item for item in streams if item.get("codec_type") == "video"]
    audios = [item for item in streams if item.get("codec_type") == "audio"]
    if len(streams) != 2 or len(videos) != 1 or len(audios) != 1:
        raise SystemExit("MOVA output must contain exactly one video and audio stream")
    video = videos[0]
    audio = audios[0]
    if video.get("codec_name") != "h264" or audio.get("codec_name") != "aac":
        raise SystemExit("MOVA output codecs must be H.264 video and AAC audio")
    if video.get("height") != height or video.get("width") != width:
        raise SystemExit("MOVA output video dimensions do not match the recipe")
    try:
        frame_rate = float(Fraction(str(video.get("avg_frame_rate"))))
        frame_count = int(str(video.get("nb_read_frames", video.get("nb_frames"))))
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
        raise SystemExit(
            "MOVA output video duration does not match 193 frames at 24 fps"
        )
    if abs(audio_duration - expected_duration) > 0.25:
        raise SystemExit("MOVA output audio duration is outside the synchronized bound")
    if abs(video_duration - audio_duration) > max(0.25, 1.0 / 24.0):
        raise SystemExit("MOVA output audio and video durations are not synchronized")


def main() -> None:
    args = _arguments()
    manifest = _input_manifest()
    prompt = _prompt(manifest)
    height, width = _variant()
    if not _OFFICIAL_RUNNER.is_file():
        raise SystemExit("the pinned official MOVA inference runner is missing")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    if any(args.output_dir.iterdir()):
        raise SystemExit(
            "MOVA requires a fresh output directory for atomic publication"
        )
    final_output = args.output_dir / "mova.mp4"
    with tempfile.TemporaryDirectory(prefix="vonk-mova-") as temporary:
        temporary_path = Path(temporary)
        reference = _reference_frame(temporary_path, height, width, manifest)
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
        subprocess.run(
            command, check=True, env=environment, timeout=args.timeout_seconds
        )
        _verify_synchronized_mp4(staged_output, height=height, width=width)
        temporary_final = args.output_dir / ".mova.mp4.complete"
        shutil.copyfile(staged_output, temporary_final)
        os.replace(temporary_final, final_output)


if __name__ == "__main__":
    main()
