"""TangoFlux job adapter: one text prompt to a stereo 44.1 kHz clip.

The job reads the one ``prompt`` file, runs the creator's own steps in ``worker.py`` on the pinned
weights and FLAN-T5 text encoder (50 steps and guidance 4.5 are the creator's CLI and
``generate()`` defaults), checks the WAV the worker wrote, and publishes ``audio.wav``.

Everything the creator's code would fetch from Hugging Face is mounted instead: TangoFlux at
``/models/target`` and FLAN-T5-large at ``/models/flan-t5-large``.
"""

from __future__ import annotations

import argparse
import math
import os
import signal
import subprocess
import sys
import tempfile
import time
import wave
from argparse import Namespace
from pathlib import Path

from generation import SAMPLE_RATE, output_frames
from vonk_agent_protocol.job_inputs import RecipeJobInputManifest

ENTRYPOINT = "/opt/vonk/source/run.py"
SOURCE_ROOT = Path("/opt/tangoflux")
WORKER = Path("/opt/vonk/source/worker.py")
INPUT_ROOT = Path("/inputs")
WEIGHTS = Path("/models/target")
TEXT_ENCODER = Path("/models/flan-t5-large")
MODEL_FILES = {
    WEIGHTS: ("config.json", "tangoflux.safetensors", "vae.safetensors"),
    TEXT_ENCODER: (
        "config.json",
        "model.safetensors",
        "special_tokens_map.json",
        "spiece.model",
        "tokenizer.json",
        "tokenizer_config.json",
    ),
}
OUTPUT_MIME = "audio/wav"
OUTPUT_NAME = "audio.wav"
MAX_MANIFEST_BYTES = 262144
MAX_PROMPT_BYTES = 16384
MAX_SEED = 2**31 - 1
# The creator's CLI bounds: duration 1-30 seconds, steps 10-100.
MIN_DURATION = 1.0
MAX_DURATION = 30.0
MIN_STEPS = 10
MAX_STEPS = 100
CHANNELS = 2
SAMPLE_WIDTH_BYTES = 2
# Time kept after generation for checking and publishing the file.
PUBLISH_RESERVE_SECONDS = 30
KILL_GRACE_SECONDS = 10


def _arguments(argv: list[str] | None = None) -> Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--entrypoint", required=True)
    parser.add_argument("--output-mime", required=True)
    parser.add_argument("--duration-seconds", type=float, required=True)
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--guidance-scale", type=float, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.entrypoint != ENTRYPOINT:
        raise SystemExit("unexpected signed-bundle entrypoint")
    if args.output_mime != OUTPUT_MIME:
        raise SystemExit(f"TangoFlux emits {OUTPUT_MIME} only")
    if not MIN_DURATION <= args.duration_seconds <= MAX_DURATION:
        raise SystemExit(f"duration must be {MIN_DURATION:g}..{MAX_DURATION:g} seconds")
    if not MIN_STEPS <= args.steps <= MAX_STEPS:
        raise SystemExit(f"steps must be {MIN_STEPS}..{MAX_STEPS}")
    if not 0 <= args.seed <= MAX_SEED:
        raise SystemExit(f"seed must be 0..{MAX_SEED}")
    if not math.isfinite(args.guidance_scale) or args.guidance_scale <= 0:
        raise SystemExit("guidance scale must be a positive number")
    return args


def load_prompt(input_root: Path) -> Path:
    """The one declared ``prompt`` file, verified against the job manifest and its UTF-8 text."""

    if not input_root.is_dir() or input_root.is_symlink():
        raise SystemExit("/inputs must be a directory containing the declared inputs")
    manifest_path = input_root / "manifest.json"
    if (
        manifest_path.is_symlink()
        or not manifest_path.is_file()
        or manifest_path.stat().st_size > MAX_MANIFEST_BYTES
    ):
        raise SystemExit("a regular, bounded Vonk job input manifest is required")
    try:
        manifest = RecipeJobInputManifest.model_validate_json(
            manifest_path.read_bytes()
        )
    except (OSError, ValueError) as error:
        raise SystemExit(f"invalid Vonk job input manifest: {error}") from error
    names = [item.name for item in manifest.files]
    if {entry.name for entry in input_root.iterdir()} != {*names, "manifest.json"}:
        raise SystemExit("job input files do not match the manifest")
    if len(manifest.files) != 1 or manifest.files[0].slot != "prompt":
        raise SystemExit("TangoFlux takes exactly one file in the prompt slot")
    item = manifest.files[0]
    path = input_root / item.name
    if path.is_symlink() or not path.is_file() or path.suffix.lower() != ".txt":
        raise SystemExit(f"{item.name} must be a regular .txt file")
    payload = path.read_bytes()
    if not 1 <= len(payload) <= MAX_PROMPT_BYTES or len(payload) != item.size_bytes:
        raise SystemExit(
            f"{item.name} must match the manifest and hold 1..{MAX_PROMPT_BYTES} bytes"
        )
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as error:
        raise SystemExit(f"{item.name} is not UTF-8 text") from error
    if not text.strip():
        raise SystemExit(f"{item.name} holds no prompt")
    return path


def _job_deadline() -> float:
    """The platform's job budget as one monotonic deadline."""
    try:
        seconds = int(os.environ["VONK_JOB_TIMEOUT_SECONDS"])
    except (KeyError, ValueError) as error:
        raise SystemExit("job must declare VONK_JOB_TIMEOUT_SECONDS") from error
    if not 1 <= seconds <= 3600:
        raise SystemExit("job timeout must be between 1 and 3600 seconds")
    return time.monotonic() + seconds


def check_model_files(files: dict[Path, tuple[str, ...]] = MODEL_FILES) -> None:
    """Fail now, naming the file, instead of deep inside the creator's loader."""
    for directory, names in files.items():
        for name in names:
            if not (directory / name).is_file():
                raise SystemExit(f"model file is not mounted: {directory / name}")


def generation_command(
    prompt: Path, args: Namespace, *, output: Path, worker: Path = WORKER
) -> list[str]:
    return [
        sys.executable,
        str(worker),
        "--prompt-file",
        str(prompt),
        "--duration-seconds",
        str(args.duration_seconds),
        "--steps",
        str(args.steps),
        "--seed",
        str(args.seed),
        "--guidance-scale",
        str(args.guidance_scale),
        "--weights",
        str(WEIGHTS),
        "--text-encoder",
        str(TEXT_ENCODER),
        "--output",
        str(output),
    ]


def generation_environment(base: dict[str, str] | None = None) -> dict[str, str]:
    environment = dict(os.environ if base is None else base)
    # PYTHONPATH is platform-owned for the image, so only this child gets the creator's package.
    environment.update(PYTHONPATH=str(SOURCE_ROOT))
    return environment


def _kill_group(process: subprocess.Popen[bytes]) -> None:
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    process.wait(timeout=KILL_GRACE_SECONDS)


def run_generation(
    command: list[str], environment: dict[str, str], cwd: Path, deadline: float
) -> None:
    """Run the creator's steps in their own process group, killed as a group at the deadline."""

    budget = deadline - time.monotonic() - PUBLISH_RESERVE_SECONDS
    if budget <= 0:
        raise SystemExit("TangoFlux job budget exhausted before generation")
    process = subprocess.Popen(
        command, env=environment, cwd=cwd, start_new_session=True
    )
    try:
        code = process.wait(timeout=budget)
    except subprocess.TimeoutExpired:
        _kill_group(process)
        raise SystemExit("TangoFlux generation exceeded the job budget") from None
    except BaseException:
        _kill_group(process)
        raise
    if code != 0:
        raise SystemExit(f"TangoFlux generation failed with exit status {code}")


def checked_wav(path: Path, duration: float) -> Path:
    """The clip is stereo 16-bit PCM at 44.1 kHz with exactly the frames the request asks for."""

    if not path.is_file() or path.is_symlink():
        raise SystemExit(f"TangoFlux did not write {path.name}")
    try:
        with wave.open(str(path), "rb") as clip:
            facts = (
                clip.getnchannels(),
                clip.getframerate(),
                clip.getsampwidth(),
                clip.getnframes(),
            )
    except (wave.Error, EOFError) as error:
        raise SystemExit(f"TangoFlux wrote an unreadable WAV: {error}") from error
    expected = (CHANNELS, SAMPLE_RATE, SAMPLE_WIDTH_BYTES, output_frames(duration))
    if facts != expected:
        raise SystemExit(
            f"TangoFlux wrote channels, rate, sample width and frames {facts}, not {expected}"
        )
    return path


def main(argv: list[str] | None = None) -> None:
    args = _arguments(argv)
    deadline = _job_deadline()
    prompt = load_prompt(INPUT_ROOT)
    check_model_files()
    # The platform owns tmp/ and cache/; only our unique workspace must be fresh.
    scratch_root = args.output_dir / "tmp"
    scratch_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="vonk-tangoflux-", dir=scratch_root) as name:
        root = Path(name)
        staged = root / OUTPUT_NAME
        run_generation(
            generation_command(prompt, args, output=staged),
            generation_environment(),
            root,
            deadline,
        )
        os.replace(checked_wav(staged, args.duration_seconds), args.output_dir / OUTPUT_NAME)


if __name__ == "__main__":
    main()
