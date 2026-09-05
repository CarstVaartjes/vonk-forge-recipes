from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import json
import os
import re
import subprocess
import tempfile
from pathlib import Path

MODEL_ROOT = Path("/models")
INPUT_ROOT = Path("/inputs")
MAX_PROMPT_BYTES = 16 * 1024
RUNTIME_SPEC = Path("/run/vonk/runtime.json")
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
LTX_PIPELINES_VERSION = "1.3.0"
PIPELINE_OUTPUT_FIELDS = (
    "video",
    "audio",
    "num_frames",
    "tiling_config",
    "keyframes",
    "video_latent",
)


def _verify_ltx_runtime_contract() -> None:
    try:
        installed_version = importlib.metadata.version("ltx-pipelines")
        pipeline_output = importlib.import_module(
            "ltx_pipelines.utils.types"
        ).PipelineOutput
    except (ImportError, importlib.metadata.PackageNotFoundError) as error:
        raise SystemExit(f"LTX native runtime is incomplete: {error}") from error
    if installed_version != LTX_PIPELINES_VERSION:
        raise SystemExit(
            "LTX native runtime version changed: "
            f"expected {LTX_PIPELINES_VERSION}, found {installed_version}"
        )
    if tuple(getattr(pipeline_output, "_fields", ())) != PIPELINE_OUTPUT_FIELDS:
        raise SystemExit("LTX PipelineOutput contract changed")


def _materialized_path(mount_target: str, relative_path: str) -> Path:
    target = Path(mount_target)
    try:
        target_relative = target.relative_to(Path("/models"))
    except ValueError as error:
        raise SystemExit("Vonk model mount target escapes /models") from error
    candidate = MODEL_ROOT / target_relative / relative_path
    try:
        candidate.relative_to(MODEL_ROOT)
    except ValueError as error:
        raise SystemExit("Vonk model file escapes the model root") from error
    return candidate


def _runtime_artifacts() -> list[dict[str, object]]:
    try:
        document = json.loads(RUNTIME_SPEC.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise SystemExit(f"invalid Vonk runtime contract: {error}") from error
    if not isinstance(document, dict) or document.get("schema_version") != 2:
        raise SystemExit("Vonk runtime contract must use schema version 2")
    artifacts = document.get("artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        raise SystemExit("Vonk runtime contract has no artifacts")
    result: list[dict[str, object]] = []
    identities: set[tuple[str, str]] = set()
    materialized: set[Path] = set()
    for artifact in artifacts:
        allowed = {
            "id",
            "selection_id",
            "file_id",
            "path",
            "sha256",
            "size_bytes",
            "roles",
            "mount",
            "model",
            "distribution_object",
        }
        if not isinstance(artifact, dict) or set(artifact) - allowed:
            raise SystemExit("Vonk runtime contract artifact is invalid")
        required = (
            "selection_id",
            "file_id",
            "path",
            "sha256",
            "size_bytes",
            "roles",
            "mount",
            "model",
            "distribution_object",
        )
        if any(key not in artifact for key in required):
            raise SystemExit("Vonk runtime contract selected file is incomplete")
        selection_id = artifact["selection_id"]
        file_id = artifact["file_id"]
        relative = artifact["path"]
        digest = artifact["sha256"]
        size = artifact["size_bytes"]
        mount = artifact["mount"]
        model = artifact["model"]
        receipt = artifact["distribution_object"]
        if (
            not isinstance(selection_id, str)
            or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,255}", selection_id) is None
            or not isinstance(file_id, str)
            or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,255}", file_id) is None
            or not isinstance(relative, str)
            or not relative
            or relative.startswith("/")
            or "\\" in relative
            or any(part in {"", ".", ".."} for part in relative.split("/"))
            or not isinstance(digest, str)
            or re.fullmatch(r"[0-9a-f]{64}", digest) is None
            or type(size) is not int
            or size < 0
            or not isinstance(mount, dict)
            or not isinstance(mount.get("target"), str)
            or mount.get("read_only") is not True
            or set(mount) != {"target", "read_only"}
            or not isinstance(model, dict)
            or set(model) != {"publisher", "slug", "content_sha256"}
            or not isinstance(model.get("content_sha256"), str)
            or re.fullmatch(r"[0-9a-f]{64}", model["content_sha256"]) is None
            or not isinstance(receipt, dict)
            or set(receipt) != {"name", "sha256", "bytes", "kind"}
            or receipt.get("kind") != "model"
            or receipt.get("name") != relative
            or receipt.get("sha256") != digest
            or receipt.get("bytes") != size
        ):
            raise SystemExit("Vonk runtime contract selected file is invalid")
        target = mount["target"]
        if (
            not (target == "/models" or target.startswith("/models/"))
            or "//" in target
            or "\\" in target
            or any(
                part in {"", ".", ".."} for part in target.removeprefix("/").split("/")
            )
        ):
            raise SystemExit("Vonk model mount target is invalid")
        identity = (selection_id, file_id)
        if identity in identities:
            raise SystemExit("Vonk runtime contract repeats a selected file")
        path = _materialized_path(target, relative)
        try:
            path.resolve(strict=True).relative_to(MODEL_ROOT.resolve())
        except (FileNotFoundError, OSError, ValueError) as error:
            raise SystemExit(
                "Vonk selected model file escapes the model root"
            ) from error
        if path in materialized or not path.is_file() or path.is_symlink():
            raise SystemExit("Vonk selected model file is not a regular file")
        if path.stat().st_size != size:
            raise SystemExit("Vonk selected model file size changed")
        identities.add(identity)
        materialized.add(path)
        normalized = dict(artifact)
        normalized["materialized_path"] = path
        result.append(normalized)
    return result


def _single_artifact(
    relative_path: str, artifacts: list[dict[str, object]] | None = None
) -> Path:
    artifacts = _runtime_artifacts() if artifacts is None else artifacts
    matches = [
        artifact["materialized_path"]
        for artifact in artifacts
        if artifact["path"] == relative_path
    ]
    if len(matches) != 1:
        raise SystemExit(
            f"selected file {relative_path!r} must occur exactly once; found {len(matches)}"
        )
    return matches[0]


def _target_checkpoint(artifacts: list[dict[str, object]] | None = None) -> Path:
    artifacts = _runtime_artifacts() if artifacts is None else artifacts
    matches = [
        artifact["path"]
        for artifact in artifacts
        if Path(artifact["path"]).name in TARGET_FILENAMES
    ]
    if len(matches) != 1:
        raise SystemExit(
            "runtime contract must contain exactly one supported immutable LTX checkpoint"
        )
    return _single_artifact(matches[0], artifacts)


def _spatial_upscaler(
    target: Path, artifacts: list[dict[str, object]] | None = None
) -> Path:
    artifacts = _runtime_artifacts() if artifacts is None else artifacts
    prefix = (
        "ltx-2.3-spatial-upscaler-"
        if target.name.startswith("ltx-2.3-")
        else "ltx-2-spatial-upscaler-"
    )
    matches = [
        artifact["path"]
        for artifact in artifacts
        if Path(artifact["path"]).name.startswith(prefix)
        and Path(artifact["path"]).name.endswith(".safetensors")
    ]
    if len(matches) != 1:
        raise SystemExit(
            f"checkpoint {target.name!r} requires exactly one {prefix!r} "
            f"artifact; found {len(matches)}"
        )
    return _single_artifact(matches[0], artifacts)


def _link_gemma(root: Path, artifacts: list[dict[str, object]] | None = None) -> None:
    artifacts = _runtime_artifacts() if artifacts is None else artifacts
    for relative_path, filename in GEMMA_FILES.items():
        source = _single_artifact(relative_path, artifacts)
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
    target: Path,
    gemma_root: Path,
    output: Path,
    seed: int,
    prompt: str,
    artifacts: list[dict[str, object]] | None = None,
) -> list[str]:
    artifacts = _runtime_artifacts() if artifacts is None else artifacts
    common = [
        "--gemma-root",
        str(gemma_root),
        "--spatial-upsampler-path",
        str(_spatial_upscaler(target, artifacts)),
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
        "cpu",
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
            str(
                _single_artifact("ltx-2-19b-distilled-lora-384.safetensors", artifacts)
            ),
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
    _verify_ltx_runtime_contract()
    artifacts = _runtime_artifacts()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    temporary = args.output_dir / ".ltx-synchronized.partial.mp4"
    destination = args.output_dir / "ltx-synchronized.mp4"
    with tempfile.TemporaryDirectory(prefix="vonk-ltx-gemma-") as gemma_dir:
        gemma_root = Path(gemma_dir)
        _link_gemma(gemma_root, artifacts)
        env = os.environ.copy()
        env.update(
            {
                "HF_HUB_OFFLINE": "1",
                "TRANSFORMERS_OFFLINE": "1",
                "HF_HOME": "/tmp/vonk-hf",
            }
        )
        target = _target_checkpoint(artifacts)
        try:
            subprocess.run(
                _pipeline_command(
                    target, gemma_root, temporary, args.seed, prompt, artifacts
                ),
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
