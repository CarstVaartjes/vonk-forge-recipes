"""Bounded offline MOSS-TTSD job: one speech.wav from a dialogue script and optional clips."""

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

from dialogue import speakers

_WORKER = Path(__file__).with_name("synthesize.py")
_INPUT_ROOT = Path("/inputs")
_MODEL_DIR = Path("/models/ttsd")
_CODEC_DIR = Path("/models/codec")
_OUTPUT_NAME = "speech.wav"


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-new-tokens", required=True, type=int)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args()


def _staged_files() -> dict[str, list[Path]]:
    """The declared input slots that carry files, each ordered by file name."""
    try:
        document = json.loads((_INPUT_ROOT / "manifest.json").read_text("utf-8"))
        entries = document["files"]
    except (OSError, ValueError, KeyError, TypeError) as error:
        raise SystemExit("missing or invalid signed input manifest") from error
    found: dict[str, list[Path]] = {}
    for item in entries:
        slot, name = item.get("slot"), item.get("name")
        if not isinstance(name, str) or Path(name).name != name:
            raise SystemExit("signed input manifest contains an unsafe name")
        path = _INPUT_ROOT / name
        if path.is_symlink() or not path.is_file():
            raise SystemExit(f"signed input file is unavailable: {slot}")
        found.setdefault(slot, []).append(path)
    return {slot: sorted(paths, key=lambda path: path.name) for slot, paths in found.items()}


def _text(path: Path) -> str:
    try:
        value = path.read_text(encoding="utf-8").strip()
    except UnicodeDecodeError as error:
        raise SystemExit(f"{path.name} must be UTF-8 text") from error
    if not value:
        raise SystemExit(f"{path.name} is empty")
    return value


def _job_deadline() -> float:
    try:
        seconds = int(os.environ["VONK_JOB_TIMEOUT_SECONDS"])
    except (KeyError, ValueError) as error:
        raise SystemExit("job must declare VONK_JOB_TIMEOUT_SECONDS") from error
    return time.monotonic() + seconds


def _generate(command: list[str], cwd: Path, deadline: float) -> None:
    # The worker is a process tree (CUDA, compilers): expiry stops the whole group.
    with subprocess.Popen(command, cwd=cwd, start_new_session=True) as process:
        try:
            code = process.wait(timeout=max(deadline - time.monotonic(), 0))
        except subprocess.TimeoutExpired as error:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
            raise SystemExit("MOSS-TTSD exceeded the declared job budget") from error
        if code:
            raise SystemExit(f"MOSS-TTSD worker failed with exit status {code}")


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
        raise SystemExit("MOSS-TTSD produced an invalid WAV artifact") from error


def main() -> None:
    deadline = _job_deadline()
    args = _arguments()
    slots = _staged_files()
    if len(slots.get("script", [])) != 1:
        raise SystemExit("input slot script takes exactly one dialogue file")
    script = _text(slots["script"][0])
    references = slots.get("reference_audio", [])
    transcripts = [_text(path) for path in slots.get("reference_text", [])]
    request = {
        "script": script,
        "speakers": speakers(script, len(references), len(transcripts)),
        "reference_audio": [str(path) for path in references],
        "reference_text": transcripts,
        "max_new_tokens": args.max_new_tokens,
    }

    # A unique workspace under the platform's TMPDIR (/outputs/tmp) holds the unpublished file.
    with tempfile.TemporaryDirectory(prefix="vonk-moss-ttsd-") as work:
        staged = Path(work) / _OUTPUT_NAME
        request_path = Path(work) / "request.json"
        request_path.write_text(json.dumps(request), encoding="utf-8")
        command = [
            sys.executable,
            str(_WORKER),
            "--model-dir",
            str(_MODEL_DIR),
            "--codec-dir",
            str(_CODEC_DIR),
            "--request",
            str(request_path),
            "--output",
            str(staged),
        ]
        _generate(command, Path(work), deadline)
        _validate_wav(staged)
        os.replace(staged, args.output_dir / _OUTPUT_NAME)


if __name__ == "__main__":
    main()
