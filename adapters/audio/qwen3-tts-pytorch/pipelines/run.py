"""Bounded offline Qwen3-TTS job: one speech.wav from text, a voice or a reference clip."""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import tempfile
import time
import wave
from pathlib import Path

from speech_mode import Mode, select_mode

_WORKER = Path(__file__).with_name("synthesize.py")
_INPUT_ROOT = Path("/inputs")
_SLOTS = ("text", "voice", "reference_audio", "reference_text")
_MODEL_DIRS = {
    Mode.DESIGN: Path("/models/voicedesign"),
    Mode.CLONE_WITH_TRANSCRIPT: Path("/models/base"),
    Mode.CLONE_SPEAKER_ONLY: Path("/models/base"),
}
_OUTPUT_NAME = "speech.wav"


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--language", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args()


def _staged_files() -> dict[str, Path]:
    """The declared input slots that carry a file, from the signed manifest."""
    try:
        document = json.loads((_INPUT_ROOT / "manifest.json").read_text("utf-8"))
        entries = document["files"]
    except (OSError, ValueError, KeyError, TypeError) as error:
        raise SystemExit("missing or invalid signed input manifest") from error
    found: dict[str, Path] = {}
    for item in entries:
        slot, name = item.get("slot"), item.get("name")
        if slot not in _SLOTS:
            raise SystemExit(f"input slot is not declared: {slot!r}")
        if not isinstance(name, str) or Path(name).name != name:
            raise SystemExit("signed input manifest contains an unsafe name")
        path = _INPUT_ROOT / name
        if path.is_symlink() or not path.is_file():
            raise SystemExit(f"signed input file is unavailable: {slot}")
        if slot in found:
            raise SystemExit(f"input slot {slot} carries more than one file")
        found[slot] = path
    return found


def _text(path: Path) -> str:
    try:
        value = path.read_text(encoding="utf-8").strip()
    except UnicodeDecodeError as error:
        raise SystemExit(f"{path.name} must be UTF-8 text") from error
    if not value:
        raise SystemExit(f"{path.name} is empty")
    return value


def _check_language(model_dir: Path, language: str) -> None:
    """Accept what the selected model's own configuration supports, plus Auto."""
    try:
        config = json.loads((model_dir / "config.json").read_text("utf-8"))
        supported = {"auto", *config["talker_config"]["codec_language_id"]}
    except (OSError, ValueError, KeyError, TypeError) as error:
        raise SystemExit(f"cannot read the language list of {model_dir}") from error
    if language.lower() not in supported:
        raise SystemExit(
            f"language {language!r} is not supported; choose one of {sorted(supported)}"
        )


def _job_deadline() -> float:
    try:
        seconds = int(os.environ["VONK_JOB_TIMEOUT_SECONDS"])
    except (KeyError, ValueError) as error:
        raise SystemExit("job must declare VONK_JOB_TIMEOUT_SECONDS") from error
    if seconds < 1:
        raise SystemExit("job timeout must be positive")
    return time.monotonic() + seconds


def _generate(command: list[str], cwd: Path, deadline: float) -> None:
    # The worker is a process tree (CUDA, compilers): expiry stops the whole group.
    with subprocess.Popen(command, cwd=cwd, start_new_session=True) as process:
        try:
            code = process.wait(timeout=max(deadline - time.monotonic(), 0))
        except subprocess.TimeoutExpired as error:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
            raise SystemExit("Qwen3-TTS exceeded the declared job budget") from error
        if code:
            raise SystemExit(f"Qwen3-TTS worker failed with exit status {code}")


def _validate_wav(path: Path) -> None:
    try:
        with wave.open(str(path), "rb") as audio:
            if (
                audio.getnchannels() != 1
                or audio.getsampwidth() != 2
                or audio.getframerate() <= 0
                or audio.getnframes() <= 0
            ):
                raise ValueError("unexpected WAV stream contract")
    except (OSError, EOFError, ValueError, wave.Error) as error:
        raise SystemExit("Qwen3-TTS produced an invalid WAV artifact") from error


def main() -> None:
    deadline = _job_deadline()
    args = _arguments()
    slots = _staged_files()
    if "text" not in slots:
        raise SystemExit("input slot text is required")
    mode = select_mode(slots.keys() - {"text"})
    model_dir = _MODEL_DIRS[mode]
    _check_language(model_dir, args.language)

    command = [
        sys.executable,
        str(_WORKER),
        "--mode",
        mode.value,
        "--model-dir",
        str(model_dir),
        "--language",
        args.language,
        "--text",
        _text(slots["text"]),
    ]
    if mode is Mode.DESIGN:
        command += ["--voice", _text(slots["voice"])]
    else:
        command += ["--reference-audio", str(slots["reference_audio"])]
    if mode is Mode.CLONE_WITH_TRANSCRIPT:
        command += ["--reference-text", _text(slots["reference_text"])]

    # The platform owns tmp/ and cache/; our unique workspace holds the unpublished file.
    scratch = args.output_dir / "tmp"
    scratch.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="vonk-qwen3-tts-", dir=scratch) as work:
        staged = Path(work) / _OUTPUT_NAME
        _generate([*command, "--output", str(staged)], Path(work), deadline)
        _validate_wav(staged)
        os.replace(staged, args.output_dir / _OUTPUT_NAME)


if __name__ == "__main__":
    main()
