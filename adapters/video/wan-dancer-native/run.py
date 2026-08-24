"""Vonk job adapter for the official two-stage Wan-Dancer pipeline."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any


INPUT_ROOT = Path("/inputs")
MODEL_ROOT = Path("/models")
SOURCE_ROOT = Path("/opt/wan-dancer")
MAX_REQUEST_BYTES = 64 * 1024
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
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}
AUDIO_SUFFIXES = {".wav", ".flac", ".mp3", ".m4a", ".ogg"}
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
STYLE_PROMPTS = {
    "chinese-classical": ("古典舞_global.txt", "古典舞_local.txt"),
    "k-pop": ("kpop_global.txt", "kpop_local.txt"),
    "street": ("街舞_global.txt", "街舞_local.txt"),
    "tap": ("踢踏舞_global.txt", "踢踏舞_local.txt"),
    "latin": ("拉丁舞_global.txt", "拉丁舞_local.txt"),
}


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


def _request() -> dict[str, Any]:
    path = INPUT_ROOT / "request.json"
    if path.is_symlink() or not path.is_file():
        raise ValueError("Wan-Dancer requires a regular /inputs/request.json")
    if path.stat().st_size > MAX_REQUEST_BYTES:
        raise ValueError("request.json exceeds 64 KiB")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("request.json must contain UTF-8 JSON") from error
    if not isinstance(value, dict) or not set(value).issubset(ALLOWED_KEYS):
        raise ValueError("request.json contains unsupported fields")
    required = {"reference_image", "music", "prompt"}
    if not required.issubset(value):
        raise ValueError("request.json requires reference_image, music, and prompt")
    return value


def _input_file(value: object, field: str, suffixes: set[str]) -> Path:
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
    return candidate


def _integer(value: object, field: str, default: int, low: int, high: int) -> int:
    if value is None:
        return default
    if type(value) is not int or not low <= value <= high:
        raise ValueError(f"{field} must be an integer between {low} and {high}")
    return value


def _number(value: object, field: str, default: float, low: float, high: float) -> float:
    if value is None:
        return default
    if type(value) not in (int, float) or not low <= float(value) <= high:
        raise ValueError(f"{field} must be between {low} and {high}")
    return float(value)


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


def _run(command: list[str], work: Path, timeout: int) -> None:
    environment = {
        **os.environ,
        "VONK_WORK_DIR": str(work),
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "HF_DATASETS_OFFLINE": "1",
        "PYTHONPATH": str(SOURCE_ROOT),
    }
    subprocess.run(
        command,
        cwd=SOURCE_ROOT,
        env=environment,
        check=True,
        timeout=timeout,
    )


def _one_mp4(folder: Path, label: str) -> Path:
    candidates = sorted(path for path in folder.glob("*.mp4") if path.is_file())
    if len(candidates) != 1 or candidates[0].stat().st_size == 0:
        raise RuntimeError(f"{label} stage did not produce exactly one non-empty MP4")
    return candidates[0]


def _verify_output(path: Path, timeout: int) -> None:
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "stream=codec_type,codec_name,avg_frame_rate",
            "-of",
            "json",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    streams = json.loads(result.stdout).get("streams", [])
    videos = [stream for stream in streams if stream.get("codec_type") == "video"]
    audios = [stream for stream in streams if stream.get("codec_type") == "audio"]
    if len(videos) != 1 or videos[0].get("codec_name") != "h264":
        raise RuntimeError("Wan-Dancer output must contain one H.264 video stream")
    if videos[0].get("avg_frame_rate") != "30/1":
        raise RuntimeError("Wan-Dancer output must be 30 fps")
    if len(audios) != 1:
        raise RuntimeError("Wan-Dancer output must contain the input music")


def main() -> None:
    args = _arguments()
    if args.output_mime != "video/mp4":
        raise SystemExit("Wan-Dancer emits video/mp4 only")
    if not 0 <= args.seed < 2**63:
        raise SystemExit("seed is outside the supported range")
    if not 60 <= args.timeout_seconds <= 86400:
        raise SystemExit("timeout-seconds must be between 60 and 86400")
    missing = sorted(path for path in REQUIRED_MODEL_FILES if not (MODEL_ROOT / path).is_file())
    if missing:
        raise SystemExit(f"incomplete Wan-Dancer snapshot; missing: {missing}")

    request = _request()
    image = _input_file(request["reference_image"], "reference_image", IMAGE_SUFFIXES)
    music = _input_file(request["music"], "music", AUDIO_SUFFIXES)
    prompt = request["prompt"]
    if not isinstance(prompt, str) or not prompt.strip() or len(prompt) > 4096:
        raise ValueError("prompt must contain 1..4096 characters")
    style = request.get("style", "k-pop")
    if not isinstance(style, str) or style not in STYLE_PROMPTS:
        raise ValueError(f"style must be one of {sorted(STYLE_PROMPTS)}")
    height = _integer(request.get("height"), "height", 1280, 512, 1280)
    width = _integer(request.get("width"), "width", 720, 288, 1280)
    if height % 16 or width % 16:
        raise ValueError("height and width must be multiples of 16")
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
    _probe_duration(music, min(args.timeout_seconds, 60))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="wan-dancer-", dir=args.output_dir) as value:
        work = Path(value)
        prompt_root = SOURCE_ROOT / "gen_video/prompt"
        global_style, local_style = STYLE_PROMPTS[style]
        global_prompt_path = work / "global-prompt.txt"
        local_prompt_path = work / "local-prompt.txt"
        global_prompt_path.write_text(
            prompt.strip()
            + ", "
            + (prompt_root / global_style).read_text(encoding="utf-8").strip(),
            encoding="utf-8",
        )
        local_prompt_path.write_text(
            prompt.strip()
            + ", "
            + (prompt_root / local_style).read_text(encoding="utf-8").strip(),
            encoding="utf-8",
        )
        global_folder = work / "global"
        final_folder = work / "final"
        global_folder.mkdir()
        final_folder.mkdir()

        global_command = [
            "python3",
            str(SOURCE_ROOT / "gen_video/gen_video_global.py"),
            "--image_path",
            str(image),
            "--prompt_path",
            str(global_prompt_path),
            "--music_path",
            str(music),
            "--seed",
            str(args.seed),
            "--height",
            str(height),
            "--width",
            str(width),
            "--output_folder",
            str(global_folder),
            "--timestamp",
            "vonk",
            "--num_inference_steps",
            str(global_steps),
            "--cfg_scale",
            str(cfg_scale),
        ]
        _run(global_command, work, args.timeout_seconds)
        global_video = _one_mp4(global_folder, "global")

        local_command = [
            "python3",
            str(SOURCE_ROOT / "gen_video/gen_video_local.py"),
            "--image_path",
            str(image),
            "--prompt_path",
            str(local_prompt_path),
            "--music_path",
            str(music),
            "--global_video_path",
            str(global_video),
            "--seed",
            str(args.seed),
            "--height",
            str(height),
            "--width",
            str(width),
            "--output_folder",
            str(final_folder),
            "--timestamp",
            "vonk",
            "--num_inference_steps",
            str(local_steps),
            "--cfg_scale",
            str(cfg_scale),
        ]
        _run(local_command, work, args.timeout_seconds)
        generated = _one_mp4(final_folder, "local")
        _verify_output(generated, min(args.timeout_seconds, 60))
        temporary = args.output_dir / ".wan-dancer.partial.mp4"
        destination = args.output_dir / "wan-dancer.mp4"
        shutil.copyfile(generated, temporary)
        os.replace(temporary, destination)


if __name__ == "__main__":
    main()
