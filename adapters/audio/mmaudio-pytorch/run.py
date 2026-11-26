"""MMAudio job adapter: a text prompt, optionally with a video, to one mono 44.1 kHz WAV.

The job reads the ``prompt`` slot, the optional ``video`` and ``negative-prompt`` slots, verified
against the job manifest, runs the creator's demo steps in ``worker.py`` in their own process group,
validates the WAV and publishes it with one atomic rename. A video's length caps the duration, as in
the creator's demo.
"""

from __future__ import annotations

import argparse
import math
import os
import re
import signal
import subprocess
import sys
import tempfile
import time
import wave
from argparse import Namespace
from dataclasses import dataclass
from pathlib import Path

from vonk_agent_protocol.job_inputs import RecipeJobInputManifest

ENTRYPOINT = "/opt/vonk/source/run.py"
WORKER = Path("/opt/vonk/source/worker.py")
SOURCE_ROOT = Path("/opt/mmaudio")
INPUT_ROOT = Path("/inputs")
MODEL_FILES = {
    Path("/models/target"): (
        "weights/mmaudio_large_44k_v2.pth",
        "ext_weights/v1-44.pth",
        "ext_weights/synchformer_state_dict.pth",
    ),
    Path("/models/dfn5b-clip"): ("open_clip_config.json", "open_clip_pytorch_model.bin"),
    Path("/models/bigvgan"): ("config.json", "bigvgan_generator.pt"),
}
MAX_MANIFEST_BYTES = 262144
MAX_PROMPT_BYTES = 65536
MAX_NEGATIVE_PROMPT_BYTES = 4096
MAX_VIDEO_BYTES = 536801280
INPUT_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._ -]{0,119}")
VIDEO_TYPES = {
    ".mp4": "video/mp4",
    ".mov": "video/quicktime",
    ".webm": "video/webm",
    ".mkv": "video/x-matroska",
}
MIN_DURATION_SECONDS = 1.0
MAX_DURATION_SECONDS = 30.0
MIN_CFG_STRENGTH = 1.0
MAX_CFG_STRENGTH = 10.0
MAX_SEED = 2**31 - 1
MAX_STEPS = 100
SAMPLING_RATE = 44100
# The creator's 44k config: a latent is 512 * 2 samples, and the length is a whole number of them.
SAMPLES_PER_LATENT = 1024
MAX_FRAMES = math.ceil(MAX_DURATION_SECONDS * SAMPLING_RATE / SAMPLES_PER_LATENT) * SAMPLES_PER_LATENT
OUTPUT_NAME = "audio.wav"
# Time kept after generation for validating and publishing.
PUBLISH_RESERVE_SECONDS = 20
KILL_GRACE_SECONDS = 10


def _arguments(argv: list[str] | None = None) -> Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--entrypoint", required=True)
    parser.add_argument("--output-mime", required=True)
    parser.add_argument("--duration-seconds", type=float, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--cfg-strength", type=float, required=True)
    parser.add_argument("--num-steps", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.entrypoint != ENTRYPOINT:
        raise SystemExit("unexpected signed-bundle entrypoint")
    if args.output_mime != "audio/wav":
        raise SystemExit("MMAudio emits audio/wav only")
    if not MIN_DURATION_SECONDS <= args.duration_seconds <= MAX_DURATION_SECONDS:
        raise SystemExit(
            f"duration must be {MIN_DURATION_SECONDS:g}..{MAX_DURATION_SECONDS:g} seconds"
        )
    if not 0 <= args.seed <= MAX_SEED:
        raise SystemExit(f"seed must be 0..{MAX_SEED}")
    if not MIN_CFG_STRENGTH <= args.cfg_strength <= MAX_CFG_STRENGTH:
        raise SystemExit(f"cfg strength must be {MIN_CFG_STRENGTH:g}..{MAX_CFG_STRENGTH:g}")
    if not 1 <= args.num_steps <= MAX_STEPS:
        raise SystemExit(f"steps must be 1..{MAX_STEPS}")
    return args


@dataclass(frozen=True)
class Inputs:
    prompt: Path
    negative_prompt: Path | None
    video: Path | None


def _checked_text(path: Path, maximum: int, minimum: int) -> None:
    payload = path.read_bytes()
    if not minimum <= len(payload) <= maximum:
        raise SystemExit(f"{path.name} must hold {minimum}..{maximum} bytes")
    try:
        payload.decode("utf-8")
    except UnicodeDecodeError as error:
        raise SystemExit(f"{path.name} is not UTF-8 text") from error


def load_inputs(input_root: Path) -> Inputs:
    """The declared ``prompt``, ``negative-prompt`` and ``video`` files, verified against the job manifest."""

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
        manifest = RecipeJobInputManifest.model_validate_json(manifest_path.read_bytes())
    except (OSError, ValueError) as error:
        raise SystemExit(f"invalid Vonk job input manifest: {error}") from error
    names = [item.name for item in manifest.files]
    if {entry.name for entry in input_root.iterdir()} != {*names, "manifest.json"}:
        raise SystemExit("job input files do not match the manifest")
    found: dict[str, Path] = {}
    for item in manifest.files:
        if item.slot not in ("prompt", "negative-prompt", "video") or item.slot in found:
            raise SystemExit(f"undeclared or repeated input slot {item.slot}")
        path = input_root / item.name
        if INPUT_NAME.fullmatch(item.name) is None or path.is_symlink() or not path.is_file():
            raise SystemExit(f"{item.name} must be a regular file with a plain name")
        suffix = path.suffix.lower()
        if path.stat().st_size != item.size_bytes:
            raise SystemExit(f"{item.name} does not match the job manifest")
        if item.slot == "video":
            if VIDEO_TYPES.get(suffix) != item.media_type or not 1 <= item.size_bytes <= MAX_VIDEO_BYTES:
                raise SystemExit(f"{item.name} must be a .mp4, .mov, .webm or .mkv video of its own media type")
        else:
            if suffix != ".txt" or item.media_type != "text/plain":
                raise SystemExit(f"{item.name} must be a .txt file of media type text/plain")
            if item.slot == "prompt":
                _checked_text(path, MAX_PROMPT_BYTES, 1)
            else:
                _checked_text(path, MAX_NEGATIVE_PROMPT_BYTES, 0)
        found[item.slot] = path
    if "prompt" not in found:
        raise SystemExit("a prompt input is required")
    return Inputs(found["prompt"], found.get("negative-prompt"), found.get("video"))


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


def worker_command(
    inputs: Inputs, args: Namespace, *, output: Path, worker: Path = WORKER
) -> list[str]:
    command = [
        sys.executable,
        str(worker),
        "--prompt-file",
        str(inputs.prompt),
        "--duration-seconds",
        str(args.duration_seconds),
        "--seed",
        str(args.seed),
        "--cfg-strength",
        str(args.cfg_strength),
        "--num-steps",
        str(args.num_steps),
        "--output",
        str(output),
    ]
    if inputs.negative_prompt is not None:
        command += ["--negative-prompt-file", str(inputs.negative_prompt)]
    if inputs.video is not None:
        command += ["--video", str(inputs.video)]
    return command


def worker_environment(base: dict[str, str] | None = None) -> dict[str, str]:
    environment = dict(os.environ if base is None else base)
    # PYTHONPATH is platform-owned for the image, so only this child gets the source tree.
    environment.update(PYTHONPATH=str(SOURCE_ROOT))
    return environment


def _kill_group(process: subprocess.Popen[bytes]) -> None:
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    process.wait(timeout=KILL_GRACE_SECONDS)


def run_worker(
    command: list[str], environment: dict[str, str], cwd: Path, deadline: float
) -> None:
    """Run the creator's steps in their own process group, killed as a group at the deadline."""

    budget = deadline - time.monotonic() - PUBLISH_RESERVE_SECONDS
    if budget <= 0:
        raise SystemExit("MMAudio job budget exhausted before generation")
    process = subprocess.Popen(command, env=environment, cwd=cwd, start_new_session=True)
    try:
        code = process.wait(timeout=budget)
    except subprocess.TimeoutExpired:
        _kill_group(process)
        raise SystemExit("MMAudio generation exceeded the job budget") from None
    except BaseException:
        _kill_group(process)
        raise
    if code != 0:
        raise SystemExit(f"MMAudio generation failed with exit status {code}")


def validated_wav(path: Path) -> None:
    if not path.is_file():
        raise SystemExit(f"MMAudio did not write {path.name}")
    try:
        with wave.open(str(path), "rb") as stream:
            shape = (stream.getnchannels(), stream.getsampwidth(), stream.getframerate())
            frames = stream.getnframes()
    except (wave.Error, EOFError) as error:
        raise SystemExit(f"MMAudio wrote {path.name} that is not PCM WAV: {error}") from error
    if shape != (1, 2, SAMPLING_RATE):
        raise SystemExit(f"MMAudio wrote {shape}; expected mono 16-bit {SAMPLING_RATE} Hz")
    if not SAMPLES_PER_LATENT <= frames <= MAX_FRAMES:
        raise SystemExit(f"MMAudio wrote {frames} frames outside the contract")


def publish(staged: Path, output_dir: Path) -> Path:
    """Validate the WAV, then one atomic rename; a refused file leaves no output."""
    validated_wav(staged)
    final = output_dir / OUTPUT_NAME
    os.replace(staged, final)
    return final


def main(argv: list[str] | None = None) -> None:
    args = _arguments(argv)
    deadline = _job_deadline()
    inputs = load_inputs(INPUT_ROOT)
    check_model_files()
    # The platform owns tmp/ and cache/; only our unique scratch must be fresh.
    scratch_root = args.output_dir / "tmp"
    scratch_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="vonk-mmaudio-", dir=scratch_root) as name:
        scratch = Path(name)
        staged = scratch / OUTPUT_NAME
        run_worker(
            worker_command(inputs, args, output=staged),
            worker_environment(),
            scratch,
            deadline,
        )
        publish(staged, args.output_dir)


if __name__ == "__main__":
    main()
