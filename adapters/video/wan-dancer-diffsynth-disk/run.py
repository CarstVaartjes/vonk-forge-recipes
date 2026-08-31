"""Vonk job adapter for the pinned DiffSynth Wan-Dancer disk canary."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import tempfile
from fractions import Fraction
from pathlib import Path
from typing import Any

INPUT_ROOT = Path("/inputs")
MODEL_ROOT = Path("/models")
RUNTIME_ROOT = Path("/opt/diffsynth-studio")
GENERATOR = Path("/opt/vonk/source/generate.py")
MAX_REQUEST_BYTES = 64 * 1024
MAX_MANIFEST_BYTES = 256 * 1024
MAX_PROMPT_BYTES = 16 * 1024
MAX_IMAGE_BYTES = 16 * 1024 * 1024
MAX_AUDIO_BYTES = 64 * 1024 * 1024
MAX_TOTAL_INPUT_BYTES = 512 * 1024 * 1024
OUTPUT_FPS = 30
OUTPUT_TRIM_SECONDS = 0.2
OUTPUT_AUDIO_SAMPLE_RATE = "44100"
MAX_GENERATION_PIXELS = 921_600
ALLOWED_KEYS = {
    "reference_image",
    "music",
    "prompt",
    "style",
    "height",
    "width",
    "num_inference_steps_global",
    "num_inference_steps_local",
    "cfg_scale",
}
CONTROL_KEYS = ALLOWED_KEYS - {"reference_image", "music", "prompt"}
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}
AUDIO_SUFFIXES = {".wav", ".flac", ".mp3", ".m4a", ".ogg"}
SLOT_CONTRACTS = {
    "prompt": ({"text/plain"}, {".txt"}, MAX_PROMPT_BYTES),
    "reference-image": (
        {"image/jpeg", "image/png", "image/webp"},
        IMAGE_SUFFIXES,
        MAX_IMAGE_BYTES,
    ),
    "music": (
        {"audio/flac", "audio/mpeg", "audio/mp4", "audio/ogg", "audio/wav"},
        AUDIO_SUFFIXES,
        MAX_AUDIO_BYTES,
    ),
    "controls": ({"application/json"}, {".json"}, MAX_REQUEST_BYTES),
}
REQUIRED_MODEL_FILES = {
    "global_model.safetensors",
    "local_model.safetensors",
    "models_t5_umt5-xxl-enc-bf16.pth",
    "models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth",
    "Wan2.1_VAE.pth",
    "google/umt5-xxl/tokenizer.json",
    "google/umt5-xxl/tokenizer_config.json",
    "google/umt5-xxl/spiece.model",
}
STYLES = {"chinese-classical", "k-pop", "street", "tap", "latin"}


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--entrypoint",
        required=True,
        choices=("/opt/vonk/source/run.py",),
    )
    parser.add_argument("--output-mime", required=True)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--timeout-seconds", required=True, type=int)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args()


def _manifest() -> dict[str, Any] | None:
    path = INPUT_ROOT / "manifest.json"
    if not path.exists():
        return None
    if (
        path.is_symlink()
        or not path.is_file()
        or path.stat().st_size > MAX_MANIFEST_BYTES
    ):
        raise ValueError("input manifest must be a bounded regular file")
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("input manifest must contain UTF-8 JSON") from error
    if not isinstance(document, dict) or set(document) != {
        "schema_version",
        "total_bytes",
        "files",
    }:
        raise ValueError("input manifest shape is invalid")
    files = document["files"]
    if document["schema_version"] != 1 or not isinstance(files, list):
        raise ValueError("input manifest version is invalid")
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
            raise ValueError("input manifest file declaration is invalid")
        name = item["name"]
        size = item["size_bytes"]
        slot = item["slot"]
        media_type = item["media_type"]
        if (
            not isinstance(name, str)
            or Path(name).name != name
            or name == "manifest.json"
            or name in names
            or not isinstance(slot, str)
            or type(size) is not int
            or size < 0
        ):
            raise ValueError("input manifest file identity is invalid")
        contract = SLOT_CONTRACTS.get(slot)
        if contract is None:
            raise ValueError(f"input manifest declares unsupported slot: {slot}")
        media_types, suffixes, maximum_bytes = contract
        if (
            not isinstance(media_type, str)
            or media_type not in media_types
            or Path(name).suffix.lower() not in suffixes
            or size > maximum_bytes
        ):
            raise ValueError(f"input manifest violates the {slot} slot contract")
        candidate = INPUT_ROOT / name
        if (
            candidate.is_symlink()
            or not candidate.is_file()
            or candidate.stat().st_size != size
        ):
            raise ValueError(f"staged input does not match its manifest: {name}")
        if hashlib.sha256(candidate.read_bytes()).hexdigest() != item["sha256"]:
            raise ValueError(f"staged input digest does not match its manifest: {name}")
        names.add(name)
        total += size
    if document["total_bytes"] != total:
        raise ValueError("input manifest total_bytes does not match staged files")
    if total > MAX_TOTAL_INPUT_BYTES:
        raise ValueError("input manifest exceeds the total input contract")
    return document


def _slot_files(manifest: dict[str, Any], slot: str) -> list[Path]:
    return [
        INPUT_ROOT / item["name"] for item in manifest["files"] if item["slot"] == slot
    ]


def _read_request(path: Path, *, allowed_keys: set[str]) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ValueError("Wan-Dancer controls must be a regular JSON file")
    if path.stat().st_size > MAX_REQUEST_BYTES:
        raise ValueError("Wan-Dancer controls exceed 64 KiB")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("Wan-Dancer controls must contain UTF-8 JSON") from error
    if not isinstance(value, dict) or not set(value).issubset(allowed_keys):
        raise ValueError("Wan-Dancer controls contain unsupported fields")
    return value


def _request(manifest: dict[str, Any] | None = None) -> dict[str, Any]:
    if manifest is None:
        value = _read_request(INPUT_ROOT / "request.json", allowed_keys=ALLOWED_KEYS)
        required = {"reference_image", "music", "prompt"}
        if not required.issubset(value):
            raise ValueError("request.json requires reference_image, music, and prompt")
        return value

    controls = _slot_files(manifest, "controls")
    if len(controls) > 1:
        raise ValueError("Wan-Dancer accepts at most one controls JSON file")
    value = _read_request(controls[0], allowed_keys=CONTROL_KEYS) if controls else {}
    references = _slot_files(manifest, "reference-image")
    music = _slot_files(manifest, "music")
    prompts = _slot_files(manifest, "prompt")
    if len(references) != 1 or len(music) != 1 or len(prompts) != 1:
        raise ValueError(
            "Wan-Dancer requires one prompt, reference image, and music file"
        )
    if prompts[0].stat().st_size > MAX_PROMPT_BYTES:
        raise ValueError("Wan-Dancer prompt exceeds 16 KiB")
    try:
        prompt = prompts[0].read_text(encoding="utf-8")
    except UnicodeDecodeError as error:
        raise ValueError("Wan-Dancer prompt must be UTF-8 text") from error
    value.update(
        {
            "reference_image": references[0].name,
            "music": music[0].name,
            "prompt": prompt,
        }
    )
    return value


def _input_file(
    value: object,
    field: str,
    suffixes: set[str],
    maximum_bytes: int = 512 * 1024 * 1024,
) -> Path:
    if not isinstance(value, str) or not value or len(value) > 256:
        raise ValueError(f"{field} must be a non-empty relative input path")
    relative = Path(value)
    if relative.is_absolute() or ".." in relative.parts or relative == Path("."):
        raise ValueError(f"{field} must stay inside /inputs")
    candidate = INPUT_ROOT / relative
    if candidate.is_symlink() or not candidate.is_file():
        raise ValueError(f"{field} must name a regular input file")
    try:
        candidate.resolve(strict=True).relative_to(INPUT_ROOT.resolve(strict=True))
    except ValueError as error:
        raise ValueError(f"{field} escapes /inputs") from error
    if candidate.suffix.lower() not in suffixes:
        raise ValueError(f"{field} has an unsupported extension")
    if candidate.stat().st_size > maximum_bytes:
        raise ValueError(f"{field} exceeds its bounded input contract")
    return candidate


def _integer(value: object, field: str, default: int, low: int, high: int) -> int:
    if value is None:
        return default
    if type(value) is not int or not low <= value <= high:
        raise ValueError(f"{field} must be an integer between {low} and {high}")
    return value


def _number(
    value: object, field: str, default: float, low: float, high: float
) -> float:
    if value is None:
        return default
    if type(value) not in (int, float) or not low <= float(value) <= high:
        raise ValueError(f"{field} must be between {low} and {high}")
    return float(value)


def _generation_dimensions(request: dict[str, Any]) -> tuple[int, int]:
    height = _integer(request.get("height"), "height", 1280, 512, 1280)
    width = _integer(request.get("width"), "width", 720, 288, 1280)
    if height % 16 or width % 16:
        raise ValueError("height and width must be multiples of 16")
    if height * width > MAX_GENERATION_PIXELS:
        raise ValueError(
            "height and width must define a canvas of at most 921600 pixels"
        )
    return height, width


def _probe_duration(music: Path, timeout: int) -> float:
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(music),
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    duration = float(result.stdout.strip())
    if not 1.0 <= duration <= 5.0:
        raise ValueError("music duration must be between 1 and 5 seconds")
    return duration


def _expected_output(
    image: Path,
    *,
    target_height: int,
    target_width: int,
    music_duration: float,
) -> tuple[int, int, int, float]:
    from PIL import Image, UnidentifiedImageError

    try:
        with Image.open(image) as source:
            source_width, source_height = source.size
            source.verify()
    except (OSError, UnidentifiedImageError, ValueError) as error:
        raise ValueError("reference_image must contain a valid image") from error
    if source_width < 1 or source_height < 1:
        raise ValueError("reference_image dimensions are invalid")
    scale = min(target_width / source_width, target_height / source_height)
    output_width = round(source_width * scale)
    output_height = round(source_height * scale)
    if output_width < 2 or output_height < 2 or output_width % 2 or output_height % 2:
        raise ValueError(
            "reference_image aspect ratio produces unsupported odd output dimensions"
        )
    trimmed_duration = music_duration - OUTPUT_TRIM_SECONDS
    frame_count = int(trimmed_duration * OUTPUT_FPS)
    if frame_count < 1:
        raise ValueError("music duration produces no output video frames")
    return output_width, output_height, frame_count, frame_count / OUTPUT_FPS


def _run(command: list[str], work: Path, timeout: int) -> None:
    environment = {
        **os.environ,
        "VONK_WORK_DIR": str(work),
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "HF_DATASETS_OFFLINE": "1",
        "DIFFSYNTH_SKIP_DOWNLOAD": "True",
        "DIFFSYNTH_DISK_MAP_BUFFER_SIZE": str(256 * 1024 * 1024),
        "PYTHONPATH": str(RUNTIME_ROOT),
    }
    subprocess.run(
        command,
        cwd=work,
        env=environment,
        check=True,
        timeout=timeout,
    )


def _require_output(path: Path, label: str) -> None:
    if path.is_symlink() or not path.is_file() or path.stat().st_size == 0:
        raise RuntimeError(f"{label} stage did not produce a non-empty regular MP4")


def _mux_output(
    video: Path,
    music: Path,
    destination: Path,
    *,
    canvas_height: int,
    canvas_width: int,
    output_height: int,
    output_width: int,
    output_frames: int,
    timeout: int,
) -> None:
    crop_x = (canvas_width - output_width) // 2
    crop_y = (canvas_height - output_height) // 2
    duration = output_frames / OUTPUT_FPS
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-v",
            "error",
            "-i",
            str(video),
            "-i",
            str(music),
            "-map",
            "0:v:0",
            "-map",
            "1:a:0",
            "-vf",
            (
                f"crop={output_width}:{output_height}:{crop_x}:{crop_y},"
                f"fps={OUTPUT_FPS},format=yuv420p"
            ),
            "-frames:v",
            str(output_frames),
            "-t",
            f"{duration:.9f}",
            "-c:v",
            "libx264",
            "-c:a",
            "aac",
            "-ar",
            OUTPUT_AUDIO_SAMPLE_RATE,
            "-movflags",
            "+faststart",
            str(destination),
        ],
        check=True,
        timeout=timeout,
    )
    _require_output(destination, "mux")


def _positive_duration(value: object, label: str) -> float:
    try:
        duration = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError) as error:
        raise RuntimeError(
            f"Wan-Dancer output does not expose {label} duration"
        ) from error
    if duration <= 0:
        raise RuntimeError(f"Wan-Dancer output does not expose {label} duration")
    return duration


def _verify_output(
    path: Path,
    timeout: int,
    *,
    expected_width: int,
    expected_height: int,
    expected_frames: int,
    expected_duration: float,
) -> None:
    result = subprocess.run(
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
        timeout=timeout,
    )
    document = json.loads(result.stdout)
    media_format = document.get("format")
    format_names = (
        set(str(media_format.get("format_name", "")).split(","))
        if isinstance(media_format, dict)
        else set()
    )
    if "mp4" not in format_names:
        raise RuntimeError("Wan-Dancer output must use the MP4 container")
    streams = document.get("streams", [])
    if not isinstance(streams, list):
        raise TypeError("Wan-Dancer output stream inventory is invalid")
    videos = [stream for stream in streams if stream.get("codec_type") == "video"]
    audios = [stream for stream in streams if stream.get("codec_type") == "audio"]
    if len(streams) != 2 or len(videos) != 1 or len(audios) != 1:
        raise RuntimeError(
            "Wan-Dancer output must contain exactly one video and audio stream"
        )
    video = videos[0]
    audio = audios[0]
    if video.get("codec_name") != "h264":
        raise RuntimeError("Wan-Dancer output must contain one H.264 video stream")
    if audio.get("codec_name") != "aac":
        raise RuntimeError("Wan-Dancer output audio codec must be AAC")
    if audio.get("sample_rate") != OUTPUT_AUDIO_SAMPLE_RATE:
        raise RuntimeError("Wan-Dancer output audio sample rate must be 44100 Hz")
    if video.get("width") != expected_width or video.get("height") != expected_height:
        raise RuntimeError(
            "Wan-Dancer output dimensions do not match the reference image aspect"
        )
    try:
        frame_rate = float(Fraction(str(video.get("avg_frame_rate"))))
    except (ValueError, ZeroDivisionError) as error:
        raise RuntimeError(
            "Wan-Dancer output does not expose a valid frame rate"
        ) from error
    if abs(frame_rate - OUTPUT_FPS) > 0.01:
        raise RuntimeError("Wan-Dancer output must be 30 fps")
    try:
        frame_count = int(str(video.get("nb_read_frames", video.get("nb_frames"))))
    except (TypeError, ValueError) as error:
        raise RuntimeError("Wan-Dancer output does not expose a frame count") from error
    if frame_count != expected_frames:
        raise RuntimeError(
            f"Wan-Dancer output must contain exactly {expected_frames} frames"
        )
    video_duration = _positive_duration(video.get("duration"), "video")
    encoded_duration = expected_frames / OUTPUT_FPS
    if abs(video_duration - encoded_duration) > 1.0 / OUTPUT_FPS:
        raise RuntimeError(
            "Wan-Dancer output video duration is outside its frame bound"
        )
    audio_duration = _positive_duration(audio.get("duration"), "audio")
    if abs(audio_duration - expected_duration) > 0.1:
        raise RuntimeError(
            "Wan-Dancer output audio duration is outside its music bound"
        )
    format_duration = _positive_duration(
        media_format.get("duration") if isinstance(media_format, dict) else None,
        "container",
    )
    if abs(format_duration - expected_duration) > 0.1:
        raise RuntimeError("Wan-Dancer MP4 duration is outside its music bound")
    if abs(video_duration - audio_duration) > 0.1:
        raise RuntimeError("Wan-Dancer output audio and video are not synchronized")


def main() -> None:
    args = _arguments()
    if args.output_mime != "video/mp4":
        raise SystemExit("Wan-Dancer emits video/mp4 only")
    if not 0 <= args.seed < 2**63:
        raise SystemExit("seed is outside the supported range")
    if not 60 <= args.timeout_seconds <= 86400:
        raise SystemExit("timeout-seconds must be between 60 and 86400")
    missing = sorted(
        path for path in REQUIRED_MODEL_FILES if not (MODEL_ROOT / path).is_file()
    )
    if missing:
        raise SystemExit(f"incomplete Wan-Dancer snapshot; missing: {missing}")

    manifest = _manifest()
    request = _request(manifest)
    image = _input_file(
        request["reference_image"], "reference_image", IMAGE_SUFFIXES, MAX_IMAGE_BYTES
    )
    music = _input_file(request["music"], "music", AUDIO_SUFFIXES, MAX_AUDIO_BYTES)
    prompt = request["prompt"]
    if not isinstance(prompt, str) or not prompt.strip() or len(prompt) > 4096:
        raise ValueError("prompt must contain 1..4096 characters")
    style = request.get("style", "k-pop")
    if not isinstance(style, str) or style not in STYLES:
        raise ValueError(f"style must be one of {sorted(STYLES)}")
    height, width = _generation_dimensions(request)
    global_steps = _integer(
        request.get("num_inference_steps_global"),
        "num_inference_steps_global",
        48,
        1,
        100,
    )
    local_steps = _integer(
        request.get("num_inference_steps_local"),
        "num_inference_steps_local",
        24,
        1,
        100,
    )
    cfg_scale = _number(request.get("cfg_scale"), "cfg_scale", 5.0, 1.0, 20.0)
    music_duration = _probe_duration(music, min(args.timeout_seconds, 60))
    output_width, output_height, output_frames, output_duration = _expected_output(
        image,
        target_height=height,
        target_width=width,
        music_duration=music_duration,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix="wan-dancer-", dir=args.output_dir
    ) as value:
        work = Path(value)
        prompt_path = work / "prompt.txt"
        prompt_path.write_text(prompt.strip(), encoding="utf-8")
        global_video = work / "global.mp4"
        local_video = work / "local.mp4"
        generated = work / "final.mp4"

        global_command = [
            "python3",
            str(GENERATOR),
            "--stage",
            "global",
            "--reference-image",
            str(image),
            "--music",
            str(music),
            "--prompt-file",
            str(prompt_path),
            "--style",
            style,
            "--seed",
            str(args.seed),
            "--height",
            str(height),
            "--width",
            str(width),
            "--steps",
            str(global_steps),
            "--cfg-scale",
            str(cfg_scale),
            "--output",
            str(global_video),
        ]
        _run(global_command, work, args.timeout_seconds)
        _require_output(global_video, "global")

        local_command = [
            "python3",
            str(GENERATOR),
            "--stage",
            "local",
            "--reference-image",
            str(image),
            "--music",
            str(music),
            "--prompt-file",
            str(prompt_path),
            "--style",
            style,
            "--global-video",
            str(global_video),
            "--output-frames",
            str(output_frames),
            "--seed",
            str(args.seed),
            "--height",
            str(height),
            "--width",
            str(width),
            "--steps",
            str(local_steps),
            "--cfg-scale",
            str(cfg_scale),
            "--output",
            str(local_video),
        ]
        _run(local_command, work, args.timeout_seconds)
        _require_output(local_video, "local")
        _mux_output(
            local_video,
            music,
            generated,
            canvas_height=height,
            canvas_width=width,
            output_height=output_height,
            output_width=output_width,
            output_frames=output_frames,
            timeout=args.timeout_seconds,
        )
        _verify_output(
            generated,
            min(args.timeout_seconds, 60),
            expected_width=output_width,
            expected_height=output_height,
            expected_frames=output_frames,
            expected_duration=output_duration,
        )
        temporary = args.output_dir / ".wan-dancer-low-memory.partial.mp4"
        destination = args.output_dir / "wan-dancer-low-memory.mp4"
        shutil.copyfile(generated, temporary)
        os.replace(temporary, destination)


if __name__ == "__main__":
    main()
