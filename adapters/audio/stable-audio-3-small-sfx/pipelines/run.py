"""Offline job adapter for the creator's Stable Audio 3 small-sfx text-to-sound-effects model."""

from __future__ import annotations

import argparse
import json
import math
import os
import signal
import subprocess
import sys
import tempfile
import time
import wave
from pathlib import Path

_WORKER = "/opt/vonk/source/pipelines/generate.py"
_INPUT_ROOT = Path("/inputs")
_MIN_SECONDS = 1.0
_MAX_SECONDS = 120.0
_OUTPUT_NAME = "sound.wav"


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-mime", required=True)
    parser.add_argument("--duration-seconds", type=float, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_mime != "audio/wav":
        raise SystemExit("Stable Audio 3 small-sfx emits audio/wav only")
    if not math.isfinite(args.duration_seconds) or not (
        _MIN_SECONDS <= args.duration_seconds <= _MAX_SECONDS
    ):
        raise SystemExit("duration-seconds must be between 1 and 120")
    return args


def _prompt_file() -> Path:
    manifest = _INPUT_ROOT / "manifest.json"
    try:
        document = json.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise SystemExit("missing or invalid signed input manifest") from error
    files = document.get("files") if isinstance(document, dict) else None
    if not isinstance(files, list):
        raise SystemExit("signed input manifest files are invalid")
    prompts: list[Path] = []
    for item in files:
        name = item.get("name") if isinstance(item, dict) else None
        if not isinstance(name, str) or Path(name).name != name:
            raise SystemExit("signed input manifest contains an unsafe name")
        if item.get("slot") == "prompt":
            prompts.append(_INPUT_ROOT / name)
    if len(prompts) != 1:
        raise SystemExit("sound-effect generation requires one prompt text file")
    prompt = prompts[0]
    try:
        text = prompt.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as error:
        raise SystemExit("the prompt must be UTF-8 text") from error
    if not text.strip() or "\x00" in text:
        raise SystemExit("the prompt must contain non-blank text without NUL")
    return prompt


def _job_deadline() -> float:
    try:
        seconds = int(os.environ["VONK_JOB_TIMEOUT_SECONDS"])
    except (KeyError, ValueError) as error:
        raise SystemExit("job must declare VONK_JOB_TIMEOUT_SECONDS") from error
    return time.monotonic() + seconds


def _remaining(deadline: float) -> float:
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise SystemExit("sound generation exceeded the declared job budget")
    return remaining


def _generate(command: list[str], scratch: Path, deadline: float) -> None:
    # Own session: expiry must stop the worker and anything it started.
    with subprocess.Popen(
        command, cwd=scratch, env=dict(os.environ), start_new_session=True
    ) as process:
        try:
            code = process.wait(timeout=_remaining(deadline))
        except (subprocess.TimeoutExpired, SystemExit) as error:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
            raise SystemExit(
                "sound generation exceeded the declared job budget"
            ) from error
        if code:
            raise SystemExit(f"sound generation failed with exit code {code}")


def _validate_wav(path: Path, seconds: float) -> None:
    try:
        with wave.open(str(path), "rb") as audio:
            channels, width, rate, frames = (
                audio.getnchannels(),
                audio.getsampwidth(),
                audio.getframerate(),
                audio.getnframes(),
            )
    except (OSError, EOFError, wave.Error) as error:
        raise SystemExit("sound generation produced an invalid WAV artifact") from error
    if (
        channels != 2
        or width != 2
        or rate <= 0
        or not 0 < frames <= math.ceil(seconds * rate)
    ):
        raise SystemExit("sound generation produced a WAV outside the output contract")


def main() -> None:
    deadline = _job_deadline()
    args = _arguments()
    prompt = _prompt_file()
    # The platform owns tmp/ and cache/ under /outputs; only our workspace is fresh.
    scratch_root = args.output_dir / "tmp"
    scratch_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="vonk-sa3-", dir=scratch_root) as scratch:
        staged = Path(scratch) / _OUTPUT_NAME
        _generate(
            [
                sys.executable,
                _WORKER,
                "--prompt-file",
                str(prompt),
                "--duration-seconds",
                str(args.duration_seconds),
                "--output",
                str(staged),
            ],
            Path(scratch),
            deadline,
        )
        _validate_wav(staged, args.duration_seconds)
        _remaining(deadline)
        os.replace(staged, args.output_dir / _OUTPUT_NAME)


if __name__ == "__main__":
    main()
