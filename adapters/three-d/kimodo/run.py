"""NVIDIA Kimodo job adapter: a text prompt to a SOMA motion zip (BVH and NPZ).

The job reads one ``prompt`` text file, runs upstream's own ``kimodo_gen`` entry
point (``python -m kimodo.scripts.generate``) on the pinned Kimodo-SOMA-RP-v1.2
checkpoint, and publishes one ``motion.zip`` holding ``motion.bvh`` (somaskel77
skeleton, centimetres, standard T-pose rest pose) and ``motion.npz`` (Kimodo's own
arrays). The platform's artifact contract has no animation media type, so the two
files travel in one deterministic zip.

Everything the creator's code would fetch from Hugging Face is mounted instead:
the Kimodo checkpoint (``CHECKPOINT_DIR``), the two LLM2Vec adapters
(``TEXT_ENCODERS_DIR``) and the Llama 3 base the adapters sit on. The mounted
adapter config names the base by its Hugging Face repository id, so the job
points a scratch copy of it at the mounted base before the text encoder loads.

This module imports only the standard library at import time; numpy loads inside
the validation that needs it, so the plumbing is testable without a GPU, weights
or the upstream stack.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import signal
import subprocess
import sys
import tempfile
import time
import zipfile
from argparse import Namespace
from pathlib import Path

ENTRYPOINT = "/opt/vonk/source/run.py"
SOURCE_REPOSITORY = "https://github.com/nv-tlabs/kimodo"
SOURCE_COMMIT = "e1c219c99b4b94ba48be1fb6a1cbc64ecee4c954"
INPUT_ROOT = Path("/inputs")
SOURCE_ROOT = Path("/opt/vonk/kimodo")
MODEL_ROOT = Path("/models")
# Upstream resolves a checkpoint as <CHECKPOINT_DIR>/<display name>/config.yaml.
CHECKPOINT_DIR = MODEL_ROOT / "checkpoints"
KIMODO_MODEL = "Kimodo-SOMA-RP-v1.2"
# Upstream joins TEXT_ENCODERS_DIR with the repository ids of its text encoder preset.
TEXT_ENCODER_MOUNT = MODEL_ROOT / "text-encoders"
MNTP = "McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp"
SUPERVISED = "McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp-supervised"
LLAMA_BASE = MODEL_ROOT / "meta-llama/Meta-Llama-3-8B-Instruct"
MODEL_FILES = {
    CHECKPOINT_DIR / KIMODO_MODEL: (
        "config.yaml",
        "model.safetensors",
        "stats/motion/body/mean.npy",
        "stats/motion/body/std.npy",
        "stats/motion/global_root/mean.npy",
        "stats/motion/global_root/std.npy",
        "stats/motion/local_root/mean.npy",
        "stats/motion/local_root/std.npy",
    ),
    TEXT_ENCODER_MOUNT / MNTP: (
        "adapter_config.json",
        "adapter_model.safetensors",
        "config.json",
        "special_tokens_map.json",
        "tokenizer.json",
        "tokenizer_config.json",
    ),
    TEXT_ENCODER_MOUNT / SUPERVISED: (
        "adapter_config.json",
        "adapter_model.safetensors",
    ),
    LLAMA_BASE: (
        "config.json",
        "generation_config.json",
        "model.safetensors.index.json",
        "model-00001-of-00004.safetensors",
        "model-00002-of-00004.safetensors",
        "model-00003-of-00004.safetensors",
        "model-00004-of-00004.safetensors",
    ),
}
MAX_PROMPT_BYTES = 4096
MAX_MANIFEST_BYTES = 262144
# The model card: "Maximum duration is 10 sec (300 frames at 30 frames per second)".
MIN_DURATION_SECONDS = 0.5
MAX_DURATION_SECONDS = 10.0
OUTPUT_NAME = "motion.zip"
MEMBERS = ("motion.bvh", "motion.npz")
# Time kept after generation for validating and writing the zip.
PUBLISH_RESERVE_SECONDS = 10
KILL_GRACE_SECONDS = 10
# Fixed so that identical members always give identical zip bytes.
ZIP_TIMESTAMP = (1980, 1, 1, 0, 0, 0)


def _arguments(argv: list[str] | None = None) -> Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--entrypoint", required=True)
    parser.add_argument("--output-mime", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--duration-seconds", type=float, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.entrypoint != ENTRYPOINT:
        raise SystemExit("unexpected signed-bundle entrypoint")
    if args.output_mime != "application/zip":
        raise SystemExit("Kimodo emits application/zip only")
    if not 0 <= args.seed < 2**32:
        raise SystemExit("seed is outside the numpy/torch seed range")
    if not MIN_DURATION_SECONDS <= args.duration_seconds <= MAX_DURATION_SECONDS:
        raise SystemExit(
            f"duration must be {MIN_DURATION_SECONDS:g}..{MAX_DURATION_SECONDS:g} seconds"
        )
    return args


def _load_prompt(input_root: Path) -> str:
    """The one declared ``prompt`` text file, verified against the job manifest."""

    from vonk_agent_protocol.job_inputs import RecipeJobInputManifest

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
    declared = [item for item in manifest.files if item.slot == "prompt"]
    if len(manifest.files) != 1 or len(declared) != 1:
        raise SystemExit("exactly one declared prompt file is required")
    item = declared[0]
    if item.media_type != "text/plain" or Path(item.name).suffix != ".txt":
        raise SystemExit("the prompt must be a .txt file of media type text/plain")
    if {entry.name for entry in input_root.iterdir()} != {item.name, "manifest.json"}:
        raise SystemExit("job input files do not match the manifest")
    path = input_root / item.name
    if path.is_symlink() or not path.is_file():
        raise SystemExit("the declared prompt must be a regular file")
    if not 1 <= item.size_bytes <= MAX_PROMPT_BYTES:
        raise SystemExit(f"prompt.txt must hold 1..{MAX_PROMPT_BYTES} bytes")
    payload = path.read_bytes()
    if len(payload) != item.size_bytes:
        raise SystemExit("prompt size does not match the job manifest")
    if hashlib.sha256(payload).hexdigest() != item.sha256:
        raise SystemExit("prompt digest does not match the job manifest")
    try:
        prompt = payload.decode("utf-8").strip()
    except UnicodeDecodeError as error:
        raise SystemExit("the prompt must be UTF-8 text") from error
    if not prompt or "\x00" in prompt:
        raise SystemExit("the prompt must be non-empty text")
    return prompt


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
    """Fail now, naming the file, instead of deep inside the upstream loader."""
    for directory, names in files.items():
        for name in names:
            if not (directory / name).is_file():
                raise SystemExit(f"model file is not mounted: {directory / name}")


def build_text_encoder_tree(scratch: Path, *, mount: Path, base: Path) -> Path:
    """``scratch/text-encoders`` laid out the way upstream's TEXT_ENCODERS_DIR expects.

    The mounted MNTP adapter config names the base model by its Hugging Face id,
    which an offline job cannot resolve. The scratch copy of that one file names the
    mounted base instead; every other file is a symlink to the mount, and the
    supervised adapter is linked whole.
    """

    root = scratch / "text-encoders"
    mntp = root / MNTP
    mntp.mkdir(parents=True)
    for source in sorted((mount / MNTP).iterdir()):
        if source.is_file() and source.name != "adapter_config.json":
            (mntp / source.name).symlink_to(source)
    config = json.loads((mount / MNTP / "adapter_config.json").read_text("utf-8"))
    config["base_model_name_or_path"] = str(base)
    (mntp / "adapter_config.json").write_text(json.dumps(config), encoding="utf-8")
    (root / SUPERVISED).symlink_to(mount / SUPERVISED, target_is_directory=True)
    return root


def generation_command(
    prompt: str, *, duration_seconds: float, seed: int, output_stem: Path
) -> list[str]:
    """Upstream's ``kimodo_gen`` with its defaults for steps and guidance.

    Post-processing is off: upstream's MotionCorrection package is x86-only C++
    (SSE4.1/AVX intrinsics) and cannot be built for aarch64. ``--no-postprocess`` is
    upstream's own switch for it; foot-skate cleanup is left to the retargeting.
    """

    return [
        sys.executable,
        "-m",
        "kimodo.scripts.generate",
        "--model",
        KIMODO_MODEL,
        "--duration",
        f"{duration_seconds:g}",
        "--seed",
        str(seed),
        "--bvh",
        "--bvh_standard_tpose",
        "--no-postprocess",
        "--output",
        str(output_stem),
        "--",
        prompt,
    ]


def generation_environment(
    text_encoders: Path, base: dict[str, str] | None = None
) -> dict[str, str]:
    environment = dict(os.environ if base is None else base)
    # PYTHONPATH is platform-owned for the image, so only this child gets the source tree.
    environment.update(
        PYTHONPATH=str(SOURCE_ROOT),
        CHECKPOINT_DIR=str(CHECKPOINT_DIR),
        TEXT_ENCODERS_DIR=str(text_encoders),
        TEXT_ENCODER_MODE="local",
    )
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
    """Run upstream's generator in its own process group, killed as a group at the deadline."""

    budget = deadline - time.monotonic() - PUBLISH_RESERVE_SECONDS
    if budget <= 0:
        raise SystemExit("Kimodo job budget exhausted before generation")
    process = subprocess.Popen(
        command, env=environment, cwd=cwd, start_new_session=True
    )
    try:
        code = process.wait(timeout=budget)
    except subprocess.TimeoutExpired:
        _kill_group(process)
        raise SystemExit("Kimodo generation exceeded the job budget") from None
    except BaseException:
        _kill_group(process)
        raise
    if code != 0:
        raise SystemExit(f"Kimodo generation failed with exit status {code}")


def validate_motion(bvh: Path, npz: Path) -> int:
    """The BVH is a well-formed motion file and the NPZ holds the same number of frames."""

    import numpy

    text = bvh.read_text(encoding="utf-8")
    frames = re.search(r"^Frames:\s*(\d+)\s*$", text, re.MULTILINE)
    if not text.startswith("HIERARCHY") or "\nMOTION" not in text or frames is None:
        raise SystemExit("Kimodo wrote a BVH without hierarchy, motion and frame count")
    count = int(frames.group(1))
    if count < 1:
        raise SystemExit("Kimodo wrote a BVH without frames")
    with numpy.load(npz, allow_pickle=False) as arrays:
        if "posed_joints" not in arrays.files:
            raise SystemExit("Kimodo wrote an NPZ without posed_joints")
        if arrays["posed_joints"].shape[0] != count:
            raise SystemExit("Kimodo BVH and NPZ disagree on the frame count")
    return count


def publish(members: dict[str, Path], scratch: Path, output_dir: Path) -> Path:
    """One zip with fixed member order, timestamps and modes, then one atomic rename."""

    staged = scratch / OUTPUT_NAME
    with zipfile.ZipFile(staged, "w") as archive:
        for name in MEMBERS:
            info = zipfile.ZipInfo(name, ZIP_TIMESTAMP)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.create_system = 3
            info.external_attr = 0o644 << 16
            archive.writestr(info, members[name].read_bytes())
    final = output_dir / OUTPUT_NAME
    os.replace(staged, final)
    return final


def main(argv: list[str] | None = None) -> None:
    args = _arguments(argv)
    deadline = _job_deadline()
    prompt = _load_prompt(INPUT_ROOT)
    check_model_files()
    with tempfile.TemporaryDirectory(prefix="vonk-kimodo-") as name:
        scratch = Path(name)
        text_encoders = build_text_encoder_tree(
            scratch, mount=TEXT_ENCODER_MOUNT, base=LLAMA_BASE
        )
        stem = scratch / "motion"
        run_generation(
            generation_command(
                prompt,
                duration_seconds=args.duration_seconds,
                seed=args.seed,
                output_stem=stem,
            ),
            generation_environment(text_encoders),
            scratch,
            deadline,
        )
        members = {"motion.bvh": stem.with_suffix(".bvh"), "motion.npz": stem.with_suffix(".npz")}
        for path in members.values():
            if not path.is_file():
                raise SystemExit(f"Kimodo did not write {path.name}")
        validate_motion(members["motion.bvh"], members["motion.npz"])
        publish(members, scratch, args.output_dir)


if __name__ == "__main__":
    main()
