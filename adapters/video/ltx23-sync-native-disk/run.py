from __future__ import annotations

import argparse
import json
import os
import subprocess
import tempfile
from pathlib import Path

MODEL_ROOT = Path("/models")
INPUT_ROOT = Path("/inputs")
MAX_PROMPT_BYTES = 16 * 1024
RUNTIME_SPEC = Path(os.environ.get("VONK_RUNTIME_SPEC", "/run/vonk/runtime.json"))
GEMMA_FILES = {
    "text_encoder/config.json": "config.json",
    "text_encoder/generation_config.json": "generation_config.json",
    "text_encoder/model.safetensors.index.json": "model.safetensors.index.json",
    "tokenizer/added_tokens.json": "added_tokens.json",
    "tokenizer/chat_template.jinja": "chat_template.jinja",
    "tokenizer/preprocessor_config.json": "preprocessor_config.json",
    "tokenizer/processor_config.json": "processor_config.json",
    "tokenizer/special_tokens_map.json": "special_tokens_map.json",
    "tokenizer/tokenizer.json": "tokenizer.json",
    "tokenizer/tokenizer.model": "tokenizer.model",
    "tokenizer/tokenizer_config.json": "tokenizer_config.json",
    **{
        f"text_encoder/model-{index:05d}-of-00011.safetensors": (
            f"model-{index:05d}-of-00011.safetensors"
        )
        for index in range(1, 12)
    },
}
TARGET_FILENAMES = frozenset(
    {
        "ltx-2-19b-dev.safetensors",
        "ltx-2-19b-distilled.safetensors",
        "ltx-2-19b-distilled-fp8.safetensors",
        "ltx-2.3-22b-distilled-1.1.safetensors",
    }
)
WIDTH = 768
HEIGHT = 512
FRAME_COUNT = 65
FPS = 24


def _runtime_artifacts() -> list[dict[str, str]]:
    try:
        document = json.loads(RUNTIME_SPEC.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise SystemExit(f"invalid Vonk runtime contract: {error}") from error
    artifacts = document.get("artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        raise SystemExit("Vonk runtime contract has no artifacts")
    result: list[dict[str, str]] = []
    for artifact in artifacts:
        if not isinstance(artifact, dict) or any(
            not isinstance(artifact.get(key), str)
            for key in ("kind", "repository", "revision", "path")
        ):
            raise SystemExit("Vonk runtime contract artifact is invalid")
        path = Path(artifact["path"])
        try:
            relative = path.relative_to(MODEL_ROOT)
        except ValueError as error:
            raise SystemExit("Vonk runtime artifact escapes the model root") from error
        digest = relative.parts[1] if len(relative.parts) == 2 else ""
        if (
            artifact["kind"] != "http.file"
            or not artifact["revision"].startswith("sha256:")
            or relative.parts[:1] != ("sha256",)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise SystemExit("Vonk runtime artifact authority is not immutable")
        result.append(artifact)
    return result


def _single_artifact(repository_suffix: str) -> Path:
    matches = [
        Path(artifact["path"])
        for artifact in _runtime_artifacts()
        if artifact["repository"].endswith(f"/{repository_suffix}")
    ]
    if len(matches) != 1:
        raise SystemExit(
            f"artifact {repository_suffix!r} must occur exactly once; found {len(matches)}"
        )
    mount = matches[0]
    if mount.is_file():
        return mount
    files = sorted(
        path
        for path in mount.rglob("*")
        if path.is_file() and path.name != ".vonk-manifest.json"
    )
    if len(files) != 1:
        raise SystemExit(
            f"artifact {repository_suffix!r} must contain exactly one file; "
            f"found {len(files)}"
        )
    return files[0]


def _target_checkpoint() -> Path:
    matches = [
        filename
        for filename in TARGET_FILENAMES
        if any(
            artifact["repository"].endswith(f"/{filename}")
            for artifact in _runtime_artifacts()
        )
    ]
    if len(matches) != 1:
        raise SystemExit(
            "runtime contract must contain exactly one supported immutable LTX checkpoint"
        )
    return _single_artifact(matches[0])


def _spatial_upscaler(target: Path) -> Path:
    prefix = (
        "ltx-2.3-spatial-upscaler-"
        if target.name.startswith("ltx-2.3-")
        else "ltx-2-spatial-upscaler-"
    )
    matches = [
        artifact["repository"].rsplit("/", 1)[-1]
        for artifact in _runtime_artifacts()
        if artifact["repository"].rsplit("/", 1)[-1].startswith(prefix)
        and artifact["repository"].endswith(".safetensors")
    ]
    if len(matches) != 1:
        raise SystemExit(
            f"checkpoint {target.name!r} requires exactly one {prefix!r} "
            f"artifact; found {len(matches)}"
        )
    return _single_artifact(matches[0])


def _link_gemma(root: Path) -> None:
    for repository_suffix, filename in GEMMA_FILES.items():
        source = _single_artifact(repository_suffix)
        (root / filename).symlink_to(source)


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


def _pipeline_command(
    target: Path, gemma_root: Path, output: Path, seed: int, prompt: str
) -> list[str]:
    common = [
        "--gemma-root",
        str(gemma_root),
        "--spatial-upsampler-path",
        str(_spatial_upscaler(target)),
        "--prompt",
        prompt,
        "--output-path",
        str(output),
        "--seed",
        str(seed),
        "--height",
        "512",
        "--width",
        "768",
        "--num-frames",
        "65",
        "--frame-rate",
        "24",
        "--offload",
        "disk",
        "--max-batch-size",
        "1",
    ]
    name = target.name
    if name == "ltx-2-19b-dev.safetensors":
        return [
            "python3",
            "-m",
            "ltx_pipelines.ti2vid_two_stages",
            "--checkpoint-path",
            str(target),
            "--distilled-lora",
            str(_single_artifact("ltx-2-19b-distilled-lora-384.safetensors")),
            "0.8",
            *common,
        ]
    if name in {
        "ltx-2-19b-distilled.safetensors",
        "ltx-2-19b-distilled-fp8.safetensors",
        "ltx-2.3-22b-distilled-1.1.safetensors",
    }:
        command = [
            "python3",
            "-m",
            "ltx_pipelines.distilled",
            "--distilled-checkpoint-path",
            str(target),
            *common,
        ]
        if name == "ltx-2-19b-distilled-fp8.safetensors":
            command.extend(("--quantization", "fp8-scaled-mm"))
        return command
    raise SystemExit(f"unsupported immutable LTX checkpoint: {name}")


def _expected_audio_sample_rate(target: Path) -> int:
    return 48_000 if target.name.startswith("ltx-2.3-") else 24_000


def _verify_synchronized_mp4(
    output: Path, timeout_seconds: int, *, audio_sample_rate: int
) -> None:
    if not output.is_file() or output.stat().st_size == 0:
        raise SystemExit("LTX pipeline did not produce a non-empty MP4")
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
        raise SystemExit("LTX output must use an MP4 container")
    streams = document.get("streams", [])
    videos = [stream for stream in streams if stream.get("codec_type") == "video"]
    audios = [stream for stream in streams if stream.get("codec_type") == "audio"]
    if len(videos) != 1 or len(audios) != 1:
        raise SystemExit(
            "LTX output must contain exactly one video and one audio stream"
        )
    video, audio = videos[0], audios[0]
    if (
        video.get("codec_name") != "h264"
        or video.get("width") != WIDTH
        or video.get("height") != HEIGHT
        or video.get("avg_frame_rate") != f"{FPS}/1"
        or video.get("nb_read_frames") != str(FRAME_COUNT)
    ):
        raise SystemExit("LTX output video properties changed")
    if (
        audio.get("codec_name") != "aac"
        or audio.get("sample_rate") != str(audio_sample_rate)
        or audio.get("channels") != 2
    ):
        raise SystemExit("LTX output audio properties changed")
    expected_duration = FRAME_COUNT / FPS
    try:
        video_duration = float(video["duration"])
        audio_duration = float(audio["duration"])
    except (KeyError, TypeError, ValueError) as error:
        raise SystemExit("LTX output durations are unavailable") from error
    if abs(video_duration - expected_duration) > 1e-5:
        raise SystemExit("LTX output video duration changed")
    synchronization_tolerance = max(1 / FPS, 1024 / audio_sample_rate) + 1e-5
    if abs(audio_duration - expected_duration) > synchronization_tolerance:
        raise SystemExit("LTX output audio and video durations are not synchronized")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--entrypoint", required=True)
    parser.add_argument("--output-mime", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--timeout-seconds", type=int, default=3600)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    if args.entrypoint != "/opt/vonk/source/run.py":
        raise SystemExit("unexpected pipeline entrypoint")
    if args.output_mime != "video/mp4":
        raise SystemExit("LTX synchronized generation emits video/mp4")
    if not 1 <= args.timeout_seconds <= 3600:
        raise SystemExit("timeout-seconds must be between 1 and 3600")

    prompt = _load_prompt()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    temporary = args.output_dir / ".ltx-synchronized.partial.mp4"
    destination = args.output_dir / "ltx-synchronized.mp4"
    with tempfile.TemporaryDirectory(prefix="vonk-ltx-gemma-") as gemma_dir:
        gemma_root = Path(gemma_dir)
        _link_gemma(gemma_root)
        env = os.environ.copy()
        env.update(
            {
                "HF_HUB_OFFLINE": "1",
                "TRANSFORMERS_OFFLINE": "1",
                "HF_HOME": "/tmp/vonk-hf",
            }
        )
        target = _target_checkpoint()
        try:
            subprocess.run(
                _pipeline_command(target, gemma_root, temporary, args.seed, prompt),
                check=True,
                env=env,
                timeout=args.timeout_seconds,
            )
            _verify_synchronized_mp4(
                temporary,
                args.timeout_seconds,
                audio_sample_rate=_expected_audio_sample_rate(target),
            )
            os.replace(temporary, destination)
        finally:
            temporary.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
