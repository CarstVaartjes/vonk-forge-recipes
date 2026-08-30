"""Bounded offline adapter for the official HunyuanVideo-Foley inference code."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import wave
from pathlib import Path

_SOURCE = Path("/opt/hunyuan-video-foley/infer.py")
_ENTRYPOINTS = {
    "/opt/vonk/source/pipelines/hunyuan_foley_xl.py": (
        "xl",
        "hunyuanvideo_foley_xl.pth",
    ),
    "/opt/vonk/source/pipelines/hunyuan_foley_xxl.py": (
        "xxl",
        "hunyuanvideo_foley.pth",
    ),
}
_VIDEO_SUFFIXES = frozenset({".mkv", ".mov", ".mp4", ".webm"})
_MAX_VIDEO_BYTES = 536805376


def _slot_files(path: Path, slot: str) -> list[Path]:
    try:
        document = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise SystemExit("missing or invalid signed input manifest") from error
    files = document.get("files") if isinstance(document, dict) else None
    if not isinstance(files, list):
        raise SystemExit("signed input manifest files are invalid")
    selected: list[Path] = []
    for item in files:
        if not isinstance(item, dict) or item.get("slot") != slot:
            continue
        name = item.get("name")
        if not isinstance(name, str) or Path(name).name != name:
            raise SystemExit("signed input manifest contains an unsafe name")
        candidate = path / name
        if candidate.is_symlink() or not candidate.is_file():
            raise SystemExit("signed input file is unavailable")
        selected.append(candidate)
    return selected


def _one_prompt(path: Path) -> str:
    candidates = _slot_files(path, "prompt")
    if len(candidates) != 1 or candidates[0].suffix.lower() != ".txt":
        raise SystemExit("Foley inference requires one prompt text file")
    try:
        prompt = candidates[0].read_text(encoding="utf-8").strip()
    except UnicodeDecodeError as error:
        raise SystemExit("Foley prompt must be UTF-8") from error
    if not 1 <= len(prompt.encode("utf-8")) <= 65536:
        raise SystemExit("Foley prompt must contain 1..65536 UTF-8 bytes")
    return prompt


def _one_video(path: Path) -> Path:
    candidates = sorted(_slot_files(path, "video"))
    if (
        len(candidates) != 1
        or candidates[0].suffix.lower() not in _VIDEO_SUFFIXES
        or candidates[0].stat().st_size > _MAX_VIDEO_BYTES
    ):
        raise SystemExit(
            "Foley inference requires exactly one supported video in /inputs"
        )
    return candidates[0]


def _validate_wav(path: Path) -> None:
    try:
        with wave.open(str(path), "rb") as audio:
            if (
                audio.getnchannels() not in {1, 2}
                or audio.getframerate() <= 0
                or audio.getsampwidth() not in {1, 2, 3, 4}
                or audio.getnframes() <= 0
            ):
                raise ValueError("unexpected WAV stream contract")
    except (OSError, EOFError, ValueError, wave.Error) as error:
        raise SystemExit("upstream Foley inference produced an invalid WAV artifact") from error


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--entrypoint", required=True, choices=tuple(_ENTRYPOINTS))
    parser.add_argument("--output-mime", required=True)
    parser.add_argument("--timeout-seconds", type=int, default=3600)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    if args.output_mime != "audio/wav":
        raise SystemExit(
            "HunyuanVideo-Foley only emits audio/wav as its primary artifact"
        )
    model_size, checkpoint = _ENTRYPOINTS[args.entrypoint]
    model_root = Path("/models/target")
    required = [checkpoint, "synchformer_state_dict.pth", "vae_128d_48k.pth"]
    missing = [name for name in required if not (model_root / name).is_file()]
    if missing:
        raise SystemExit(
            f"pinned Foley snapshot is missing required files: {', '.join(missing)}"
        )
    auxiliary = [Path("/models/siglip2/config.json"), Path("/models/clap/config.json")]
    missing_auxiliary = [str(path) for path in auxiliary if not path.is_file()]
    if missing_auxiliary:
        raise SystemExit(
            f"pinned Foley auxiliary snapshot is missing: {', '.join(missing_auxiliary)}"
        )
    input_root = Path("/inputs")
    video = _one_video(input_root)
    prompt = _one_prompt(input_root)
    if not _SOURCE.is_file():
        raise SystemExit("pinned HunyuanVideo-Foley source is missing")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    staging = args.output_dir / ".staging"
    shutil.rmtree(staging, ignore_errors=True)
    staging.mkdir()

    env = dict(os.environ)
    env["PYTHONHASHSEED"] = str(args.seed)
    command = [
        sys.executable,
        str(_SOURCE),
        "--model_path",
        str(model_root),
        "--model_size",
        model_size,
        "--single_video",
        str(video),
        "--single_prompt",
        prompt,
        "--output_dir",
        str(staging),
        "--guidance_scale",
        "4.5",
        "--num_inference_steps",
        "50",
        "--device",
        "cuda",
        "--gpu_id",
        "0",
        "--enable_offload",
    ]
    try:
        subprocess.run(
            command,
            cwd=_SOURCE.parent,
            env=env,
            check=True,
            timeout=args.timeout_seconds,
        )
        outputs = list(staging.glob("*_generated.wav"))
        if len(outputs) != 1:
            raise SystemExit("upstream Foley inference did not produce one WAV artifact")
        _validate_wav(outputs[0])
        os.replace(outputs[0], args.output_dir / "output.wav")
    except subprocess.TimeoutExpired as error:
        raise SystemExit("upstream Foley inference timed out") from error
    finally:
        shutil.rmtree(staging, ignore_errors=True)


if __name__ == "__main__":
    main()
