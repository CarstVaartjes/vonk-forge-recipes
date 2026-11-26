"""AnyTop job adapter: BVH clips of one rig to new motions of that rig (a BVH zip).

AnyTop has no text prompt. A skeleton it never saw is introduced by a few BVH clips of that
exact rig and the names of four of its joints; the creator's preprocessing turns them into the
rig's conditioning (topology, joint-name embeddings input, offsets, motion statistics) and the
pretrained model then samples motions for it (README: "Preprocessing new skeleton" and
"Generate unseen skeleton outside of Truebones dataset"; there is no fine-tuning).

The job reads the ``clips`` slot (the rig's BVH clips) and an optional ``rest-pose`` BVH, runs
the creator's own steps in ``worker.py`` on the pinned unified checkpoint, and publishes one
``motions.zip`` holding one ``motion-NNN.bvh`` per repetition, at the creator's 20 fps.
The creator's stick-figure mp4 rendering is skipped (see ``worker.py``).

Everything the creator's code would fetch from Hugging Face is mounted instead: the checkpoint
and the T5 encoder that embeds the joint names (the creator's ``t5-base``, read relative to the
working directory, so the job builds a workspace with a symlink to the mount).
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import zipfile
from argparse import Namespace
from dataclasses import dataclass
from pathlib import Path

from vonk_agent_protocol.job_inputs import RecipeJobInputManifest

ENTRYPOINT = "/opt/vonk/source/run.py"
SOURCE_ROOT = Path("/opt/vonk/anytop")
MOTION_ROOT = Path("/opt/vonk/motion")
WORKER = Path("/opt/vonk/source/worker.py")
INPUT_ROOT = Path("/inputs")
CHECKPOINT = Path(
    "/models/anytop/checkpoints/all_model_dataset_truebones_bs_16_latentdim_128"
)
T5 = Path("/models/t5-base")
MODEL_FILES = {
    CHECKPOINT: ("args.json", "model000459999.pt"),
    T5: (
        "config.json",
        "generation_config.json",
        "model.safetensors",
        "spiece.model",
        "tokenizer.json",
    ),
}
MAX_MANIFEST_BYTES = 262144
MAX_CLIPS = 32
MAX_CLIP_BYTES = 64 * 1024 * 1024
# The creator's own file name test is `.bvh`; names become file names in the scratch workspace.
CLIP_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._ -]{0,119}\.bvh")
JOINT_NAME = re.compile(r"[A-Za-z0-9_.:+-]{1,64}")
MAX_SEED = 2**31 - 1
MIN_REPETITIONS = 1
MAX_REPETITIONS = 8
# The README: "--motion_length ... (maximum is 9.8 [sec])"; one second is 20 frames at the creator's fps.
MIN_MOTION_SECONDS = 1.0
MAX_MOTION_SECONDS = 9.8
FPS = 20
OUTPUT_NAME = "motions.zip"
GENERATED = "Rig_rep_{}_#0.bvh"
# Time kept after generation for validating and writing the zip.
PUBLISH_RESERVE_SECONDS = 20
KILL_GRACE_SECONDS = 10
# Fixed so that identical members always give identical zip bytes.
ZIP_TIMESTAMP = (1980, 1, 1, 0, 0, 0)
FACE_JOINT_FLAGS = (
    "face-joint-right-hip",
    "face-joint-left-hip",
    "face-joint-right-shoulder",
    "face-joint-left-shoulder",
)
_FRAMES = re.compile(r"^Frames:\s*(\d+)\s*$", re.MULTILINE)
_FRAME_TIME = re.compile(r"^Frame Time:.*$", re.MULTILINE)


def _arguments(argv: list[str] | None = None) -> Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--entrypoint", required=True)
    parser.add_argument("--output-mime", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--motion-length-seconds", type=float, required=True)
    parser.add_argument("--num-repetitions", type=int, required=True)
    for flag in FACE_JOINT_FLAGS:
        parser.add_argument(f"--{flag}", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.entrypoint != ENTRYPOINT:
        raise SystemExit("unexpected signed-bundle entrypoint")
    if args.output_mime != "application/zip":
        raise SystemExit("AnyTop emits application/zip only")
    if not 0 <= args.seed <= MAX_SEED:
        raise SystemExit(f"seed must be 0..{MAX_SEED}")
    if not MIN_MOTION_SECONDS <= args.motion_length_seconds <= MAX_MOTION_SECONDS:
        raise SystemExit(
            f"motion length must be {MIN_MOTION_SECONDS:g}..{MAX_MOTION_SECONDS:g} seconds"
        )
    if not MIN_REPETITIONS <= args.num_repetitions <= MAX_REPETITIONS:
        raise SystemExit(f"repetitions must be {MIN_REPETITIONS}..{MAX_REPETITIONS}")
    args.face_joints = [getattr(args, flag.replace("-", "_")) for flag in FACE_JOINT_FLAGS]
    if not all(JOINT_NAME.fullmatch(name) for name in args.face_joints):
        raise SystemExit("a face joint name must be 1..64 letters, digits or _.:+-")
    return args


@dataclass(frozen=True)
class Inputs:
    clips: list[Path]
    rest_pose: Path | None


def _checked_bvh(path: Path, size: int) -> None:
    payload = path.read_bytes()
    if len(payload) != size:
        raise SystemExit(f"{path.name} does not match the job manifest")
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as error:
        raise SystemExit(f"{path.name} is not UTF-8 text") from error
    if not text.lstrip().startswith("HIERARCHY") or _FRAMES.search(text) is None:
        raise SystemExit(f"{path.name} is not a BVH motion file")


def load_inputs(input_root: Path) -> Inputs:
    """The declared ``clips`` and ``rest-pose`` files, verified against the job manifest."""

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
    if len({name.lower() for name in names}) != len(names):
        raise SystemExit("input file names must be unique")
    if {entry.name for entry in input_root.iterdir()} != {*names, "manifest.json"}:
        raise SystemExit("job input files do not match the manifest")
    clips: list[Path] = []
    rest: list[Path] = []
    for item in manifest.files:
        if item.slot not in ("clips", "rest-pose"):
            raise SystemExit(f"undeclared input slot {item.slot}")
        if item.media_type != "text/plain" or CLIP_NAME.fullmatch(item.name) is None:
            raise SystemExit(f"{item.name} must be a .bvh file of media type text/plain")
        path = input_root / item.name
        if path.is_symlink() or not path.is_file():
            raise SystemExit(f"{item.name} must be a regular file")
        if not 1 <= item.size_bytes <= MAX_CLIP_BYTES:
            raise SystemExit(f"{item.name} must hold 1..{MAX_CLIP_BYTES} bytes")
        _checked_bvh(path, item.size_bytes)
        (clips if item.slot == "clips" else rest).append(path)
    if not 1 <= len(clips) <= MAX_CLIPS or len(rest) > 1:
        raise SystemExit(f"give 1..{MAX_CLIPS} clips and at most one rest pose")
    return Inputs(clips, rest[0] if rest else None)


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


def build_workspace(root: Path, inputs: Inputs, *, t5: Path = T5) -> Path:
    """The working directory the creator's relative ``t5-base`` resolves in, and the BVH directory."""

    workspace = root / "workspace"
    bvh_dir = root / "bvhs"
    workspace.mkdir()
    bvh_dir.mkdir()
    (workspace / "t5-base").symlink_to(t5, target_is_directory=True)
    for path in [*inputs.clips, *([inputs.rest_pose] if inputs.rest_pose else [])]:
        shutil.copyfile(path, bvh_dir / path.name)
    return workspace


def generation_command(
    inputs: Inputs, args: Namespace, *, root: Path, checkpoint: Path = CHECKPOINT
) -> list[str]:
    rest = root / "bvhs" / inputs.rest_pose.name if inputs.rest_pose else None
    return [
        sys.executable,
        str(WORKER),
        "--bvh-dir",
        str(root / "bvhs"),
        "--rest-pose",
        str(rest or ""),
        "--face-joints",
        *args.face_joints,
        "--checkpoint",
        str(checkpoint / "model000459999.pt"),
        "--seed",
        str(args.seed),
        "--motion-length-seconds",
        str(args.motion_length_seconds),
        "--num-repetitions",
        str(args.num_repetitions),
        "--save-dir",
        str(root / "skeleton"),
        "--output-dir",
        str(root / "generated"),
    ]


def generation_environment(base: dict[str, str] | None = None) -> dict[str, str]:
    environment = dict(os.environ if base is None else base)
    # PYTHONPATH is platform-owned for the image, so only this child gets the source trees.
    environment.update(PYTHONPATH=f"{SOURCE_ROOT}:{MOTION_ROOT}")
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
        raise SystemExit("AnyTop job budget exhausted before generation")
    process = subprocess.Popen(
        command, env=environment, cwd=cwd, start_new_session=True
    )
    try:
        code = process.wait(timeout=budget)
    except subprocess.TimeoutExpired:
        _kill_group(process)
        raise SystemExit("AnyTop generation exceeded the job budget") from None
    except BaseException:
        _kill_group(process)
        raise
    if code != 0:
        raise SystemExit(f"AnyTop generation failed with exit status {code}")


def retimed(text: str) -> str:
    """The BVH with the frame time the motion really has.

    The creator samples 20 fps (``param_utils.FPS``) but saves with ``BVH.save``'s default
    frame time of 1/24 s, so every file would play 20 percent too fast.
    """
    return _FRAME_TIME.sub(f"Frame Time: {1.0 / FPS:.6f}", text, count=1)


def validated_motion(path: Path, frames: int) -> str:
    if not path.is_file():
        raise SystemExit(f"AnyTop did not write {path.name}")
    text = path.read_text(encoding="utf-8")
    found = _FRAMES.search(text)
    if not text.startswith("HIERARCHY") or found is None:
        raise SystemExit(f"AnyTop wrote {path.name} without hierarchy and frame count")
    if int(found.group(1)) != frames:
        raise SystemExit(
            f"AnyTop wrote {found.group(1)} frames in {path.name}, not the {frames} asked for"
        )
    return retimed(text)


def publish(motions: list[str], scratch: Path, output_dir: Path) -> Path:
    """One zip with fixed member names, order, timestamps and modes, then one atomic rename."""

    staged = scratch / OUTPUT_NAME
    with zipfile.ZipFile(staged, "w") as archive:
        for index, text in enumerate(motions):
            info = zipfile.ZipInfo(f"motion-{index:03d}.bvh", ZIP_TIMESTAMP)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.create_system = 3
            info.external_attr = 0o644 << 16
            archive.writestr(info, text.encode("utf-8"))
    final = output_dir / OUTPUT_NAME
    os.replace(staged, final)
    return final


def main(argv: list[str] | None = None) -> None:
    args = _arguments(argv)
    deadline = _job_deadline()
    inputs = load_inputs(INPUT_ROOT)
    check_model_files()
    # The platform owns tmp/ and cache/; only our unique workspace must be fresh.
    scratch_root = args.output_dir / "tmp"
    scratch_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="vonk-anytop-", dir=scratch_root) as name:
        root = Path(name)
        workspace = build_workspace(root, inputs)
        (root / "generated").mkdir()
        run_generation(
            generation_command(inputs, args, root=root),
            generation_environment(),
            workspace,
            deadline,
        )
        frames = int(args.motion_length_seconds * FPS)
        motions = [
            validated_motion(root / "generated" / GENERATED.format(index), frames)
            for index in range(args.num_repetitions)
        ]
        publish(motions, root, args.output_dir)


if __name__ == "__main__":
    main()
