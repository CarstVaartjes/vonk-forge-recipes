from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path

INPUT_ROOT = Path("/inputs")
MAX_PROMPT_BYTES = 16 * 1024
WIDTH = 768
HEIGHT = 512
FRAME_COUNT = 97
FPS = 24
AUDIO_SAMPLE_RATE = 24_000


def _verify_synchronized_mp4(output: Path, timeout_seconds: int) -> None:
    if not output.is_file() or output.stat().st_size == 0:
        raise SystemExit("LTX FP4 pipeline did not produce a non-empty MP4")
    probe = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-count_frames",
            "-show_entries",
            (
                "format=format_name:"
                "stream=codec_type,codec_name,width,height,avg_frame_rate,"
                "nb_read_frames,sample_rate,channels,duration"
            ),
            "-of",
            "json",
            str(output),
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=min(timeout_seconds, 60),
    )
    document = json.loads(probe.stdout)
    if "mp4" not in document.get("format", {}).get("format_name", "").split(","):
        raise SystemExit("LTX FP4 output must use an MP4 container")
    streams = document.get("streams", [])
    videos = [stream for stream in streams if stream.get("codec_type") == "video"]
    audios = [stream for stream in streams if stream.get("codec_type") == "audio"]
    if len(videos) != 1 or len(audios) != 1:
        raise SystemExit("LTX FP4 output must contain exactly one video and one audio stream")
    video, audio = videos[0], audios[0]
    if (
        video.get("codec_name") != "h264"
        or video.get("width") != WIDTH
        or video.get("height") != HEIGHT
        or video.get("avg_frame_rate") != f"{FPS}/1"
        or video.get("nb_read_frames") != str(FRAME_COUNT)
    ):
        raise SystemExit("LTX FP4 output video properties changed")
    if (
        audio.get("codec_name") != "aac"
        or audio.get("sample_rate") != str(AUDIO_SAMPLE_RATE)
        or audio.get("channels") != 2
    ):
        raise SystemExit("LTX FP4 output audio properties changed")
    expected_duration = FRAME_COUNT / FPS
    try:
        video_duration = float(video["duration"])
        audio_duration = float(audio["duration"])
    except (KeyError, TypeError, ValueError) as error:
        raise SystemExit("LTX FP4 output durations are unavailable") from error
    if abs(video_duration - expected_duration) > 1e-5:
        raise SystemExit("LTX FP4 output video duration changed")
    synchronization_tolerance = max(1 / FPS, 1024 / AUDIO_SAMPLE_RATE) + 1e-5
    if abs(audio_duration - expected_duration) > synchronization_tolerance:
        raise SystemExit("LTX FP4 output audio and video durations are not synchronized")


def _load_prompt() -> str:
    if not INPUT_ROOT.is_dir() or INPUT_ROOT.is_symlink():
        raise SystemExit("/inputs must be a directory containing prompt.txt")
    entries = list(INPUT_ROOT.iterdir())
    if len(entries) != 1:
        raise SystemExit("exactly one regular UTF-8 .txt prompt file is required")
    prompt_path = entries[0]
    if (
        prompt_path.suffix.lower() != ".txt"
        or prompt_path.is_symlink()
        or not prompt_path.is_file()
    ):
        raise SystemExit("exactly one regular UTF-8 .txt prompt file is required")
    size = prompt_path.stat().st_size
    if not 1 <= size <= MAX_PROMPT_BYTES:
        raise SystemExit("prompt.txt must contain 1..16384 UTF-8 bytes")
    try:
        prompt = prompt_path.read_text(encoding="utf-8").strip()
    except UnicodeDecodeError as error:
        raise SystemExit("prompt.txt must contain valid UTF-8") from error
    if not 1 <= len(prompt) <= 4096 or "\x00" in prompt:
        raise SystemExit("prompt.txt must contain 1..4096 non-NUL characters")
    return prompt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--entrypoint", required=True)
    parser.add_argument("--output-mime", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--timeout-seconds", type=int, default=3600)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    if args.entrypoint != "/opt/vonk/source/pipelines/run.py":
        raise SystemExit("unexpected pipeline entrypoint")
    if args.output_mime != "video/mp4":
        raise SystemExit("this candidate currently emits video/mp4")

    prompt = _load_prompt()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    temporary = args.output_dir / ".ltx2.partial.mp4"
    destination = args.output_dir / "ltx2.mp4"
    command = [
        "python3",
        "-m",
        "ltx_pipelines.ti2vid_two_stages",
        "--checkpoint-path",
        "/models/ltx-2-19b-dev-fp4.safetensors",
        "--distilled-lora",
        "/models/ltx-2-19b-distilled-lora-384.safetensors",
        "0.8",
        "--spatial-upsampler-path",
        "/models/ltx-2-spatial-upscaler-x2-1.0.safetensors",
        "--gemma-root",
        "/models/text_encoder",
        "--prompt",
        prompt,
        "--output-path",
        str(temporary),
        "--seed",
        str(args.seed),
        "--height",
        str(HEIGHT),
        "--width",
        str(WIDTH),
        "--num-frames",
        str(FRAME_COUNT),
        "--frame-rate",
        str(FPS),
        "--offload",
        "cpu",
        "--max-batch-size",
        "1",
    ]
    try:
        subprocess.run(command, check=True, timeout=args.timeout_seconds)
        _verify_synchronized_mp4(temporary, args.timeout_seconds)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
