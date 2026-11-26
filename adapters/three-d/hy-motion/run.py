"""Tencent HY-Motion 1.0 job adapter: a text prompt to a wooden-skeleton motion zip (BVH and NPZ).

The job reads one ``prompt`` text file and runs the creator's own ``T2MRuntime`` (``worker.py``)
on the pinned HY-Motion-1.0 checkpoint with the prompt rewriter and duration estimator off,
the creator's guidance scale of 5.0 and its default 50 flow-matching steps. It publishes one
``motion.zip`` holding ``motion.bvh`` (written by ``export_bvh.py``: the creator's FBX export
needs the x86-only Autodesk FBX SDK) and ``motion.npz`` (the creator's own SMPL-H parameters).
The platform's artifact contract has no animation media type, so both travel in one
deterministic zip.

Everything the creator's code would fetch from Hugging Face or read from its Git LFS tree is
mounted instead: the checkpoint, Qwen3-8B, CLIP ViT-L/14 and the skeleton and statistics
assets. The creator reads them relative to its working directory, so the job builds a scratch
workspace of symlinks to the mounts and runs the creator's code there.
"""

from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import tempfile
import time
import zipfile
from argparse import Namespace
from pathlib import Path

import numpy as np
from export_bvh import FPS, JOINTS, bvh_text, load_skeleton
from vonk_agent_protocol.job_inputs import RecipeJobInputManifest

ENTRYPOINT = "/opt/vonk/source/run.py"
SOURCE_ROOT = Path("/opt/vonk/hy-motion")
WORKER = Path("/opt/vonk/source/worker.py")
JOINT_NAMES = Path("/opt/vonk/source/joint_names.json")
INPUT_ROOT = Path("/inputs")
CHECKPOINT = Path("/models/hy-motion/HY-Motion-1.0")
ASSETS = Path("/models/hy-motion-assets")
QWEN = Path("/models/Qwen3-8B")
CLIP = Path("/models/clip-vit-large-patch14")
# The creator's relative paths (T2MRuntime, WoodenMesh, MotionFlowMatching mean_std_dir, HYTextModel).
WOODEN = Path("scripts/gradio/static/assets/dump_wooden")
WOODEN_FILES = (
    "faces.bin",
    "j_template.bin",
    "kintree.bin",
    "skinIndice.bin",
    "skinWeights.bin",
    "v_template.bin",
)
MODEL_FILES = {
    CHECKPOINT: ("config.yml", "latest.ckpt"),
    ASSETS: (
        "stats/Mean.npy",
        "stats/Std.npy",
        *(f"{WOODEN}/{name}" for name in WOODEN_FILES),
    ),
    QWEN: (
        "config.json",
        "generation_config.json",
        "merges.txt",
        "model-00001-of-00005.safetensors",
        "model-00002-of-00005.safetensors",
        "model-00003-of-00005.safetensors",
        "model-00004-of-00005.safetensors",
        "model-00005-of-00005.safetensors",
        "model.safetensors.index.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "vocab.json",
    ),
    CLIP: (
        "config.json",
        "merges.txt",
        "model.safetensors",
        "special_tokens_map.json",
        "tokenizer_config.json",
        "vocab.json",
    ),
}
MAX_PROMPT_BYTES = 4096
MAX_MANIFEST_BYTES = 262144
# The creator's generate() clamps to 20..360 frames at 30 fps (train_frames: 360).
MIN_FRAMES = 20
MAX_FRAMES = 360
MIN_DURATION_SECONDS = 0.7
MAX_DURATION_SECONDS = 12.0
MAX_SEED = 2**31 - 1
# The creator's local_infer.py default (cfg_scale or 5.0), also config.yml's text_guidance_scale.
CFG_SCALE = 5.0
OUTPUT_NAME = "motion.zip"
MEMBERS = ("motion.bvh", "motion.npz")
GENERATED_NPZ = "motion_000.npz"
# Time kept after generation for exporting, validating and writing the zip.
PUBLISH_RESERVE_SECONDS = 20
KILL_GRACE_SECONDS = 10
# Fixed so that identical members always give identical zip bytes.
ZIP_TIMESTAMP = (1980, 1, 1, 0, 0, 0)


def _arguments(argv: list[str] | None = None) -> Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--entrypoint", required=True)
    parser.add_argument("--output-mime", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--duration-seconds", type=float, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.entrypoint != ENTRYPOINT:
        raise SystemExit("unexpected signed-bundle entrypoint")
    if args.output_mime != "application/zip":
        raise SystemExit("HY-Motion emits application/zip only")
    if not 0 <= args.seed <= MAX_SEED:
        raise SystemExit(f"seed must be 0..{MAX_SEED}")
    if not MIN_DURATION_SECONDS <= args.duration_seconds <= MAX_DURATION_SECONDS:
        raise SystemExit(
            f"duration must be {MIN_DURATION_SECONDS:g}..{MAX_DURATION_SECONDS:g} seconds"
        )
    return args


def _load_prompt(input_root: Path) -> str:
    """The one declared ``prompt`` text file, verified against the job manifest."""

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
    """Fail now, naming the file, instead of deep inside the creator's loader."""
    for directory, names in files.items():
        for name in names:
            if not (directory / name).is_file():
                raise SystemExit(f"model file is not mounted: {directory / name}")


def build_workspace(
    root: Path,
    *,
    assets: Path = ASSETS,
    qwen: Path = QWEN,
    clip: Path = CLIP,
    joint_names: Path = JOINT_NAMES,
) -> Path:
    """The working directory the creator's relative paths resolve in, all symlinks to mounts.

    ``ckpts/Qwen3-8B`` and ``ckpts/clip-vit-large-patch14`` are the text encoders
    (``USE_HF_MODELS=0``), ``stats`` the motion statistics (``mean_std_dir: ./stats/``) and
    ``dump_wooden`` the skeleton and mesh templates, with the creator's plain git
    ``joint_names.json`` beside them (it is vendored in the adapter, not an LFS object).
    """

    workspace = root / "workspace"
    (workspace / "ckpts").mkdir(parents=True)
    (workspace / "ckpts/Qwen3-8B").symlink_to(qwen, target_is_directory=True)
    (workspace / "ckpts/clip-vit-large-patch14").symlink_to(
        clip, target_is_directory=True
    )
    (workspace / "stats").symlink_to(assets / "stats", target_is_directory=True)
    wooden = workspace / WOODEN
    wooden.mkdir(parents=True)
    for name in WOODEN_FILES:
        (wooden / name).symlink_to(assets / WOODEN / name)
    (wooden / "joint_names.json").symlink_to(joint_names)
    return workspace


def generation_command(
    prompt_file: Path, *, duration_seconds: float, seed: int, output_dir: Path
) -> list[str]:
    return [
        sys.executable,
        str(WORKER),
        "--config",
        str(CHECKPOINT / "config.yml"),
        "--checkpoint",
        str(CHECKPOINT / "latest.ckpt"),
        "--prompt-file",
        str(prompt_file),
        "--seed",
        str(seed),
        "--duration-seconds",
        str(duration_seconds),
        "--cfg-scale",
        str(CFG_SCALE),
        "--output-dir",
        str(output_dir),
        "--output-name",
        "motion",
    ]


def generation_environment(base: dict[str, str] | None = None) -> dict[str, str]:
    environment = dict(os.environ if base is None else base)
    # PYTHONPATH is platform-owned for the image, so only this child gets the source tree.
    environment.update(PYTHONPATH=str(SOURCE_ROOT), USE_HF_MODELS="0")
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
    """Run the creator's runtime in its own process group, killed as a group at the deadline."""

    budget = deadline - time.monotonic() - PUBLISH_RESERVE_SECONDS
    if budget <= 0:
        raise SystemExit("HY-Motion job budget exhausted before generation")
    process = subprocess.Popen(
        command, env=environment, cwd=cwd, start_new_session=True
    )
    try:
        code = process.wait(timeout=budget)
    except subprocess.TimeoutExpired:
        _kill_group(process)
        raise SystemExit("HY-Motion generation exceeded the job budget") from None
    except BaseException:
        _kill_group(process)
        raise
    if code != 0:
        raise SystemExit(f"HY-Motion generation failed with exit status {code}")


def expected_frames(duration_seconds: float) -> int:
    """The creator's length: round(duration * fps), clamped to its 20..360 frames."""
    return max(MIN_FRAMES, min(MAX_FRAMES, int(round(duration_seconds * FPS))))


def write_bvh(workspace: Path, npz: Path, bvh: Path, *, frames: int) -> None:
    """The BVH of the creator's NPZ on the creator's wooden skeleton."""

    skeleton = load_skeleton(workspace / WOODEN, workspace / WOODEN / "joint_names.json")
    with np.load(npz, allow_pickle=False) as arrays:
        poses, trans = arrays["poses"], arrays["trans"]
    if poses.shape != (frames, JOINTS * 3) or trans.shape != (frames, 3):
        raise SystemExit(
            f"HY-Motion wrote {poses.shape[0]} frames of {poses.shape[1:]} poses, "
            f"not the {frames} frames of {JOINTS * 3} the duration asks for"
        )
    bvh.write_text(bvh_text(skeleton, poses, trans), encoding="utf-8")


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
    # The platform owns tmp/ and cache/; only our unique workspace must be fresh.
    scratch_root = args.output_dir / "tmp"
    scratch_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="vonk-hy-motion-", dir=scratch_root) as name:
        scratch = Path(name)
        workspace = build_workspace(scratch)
        prompt_file = scratch / "prompt.txt"
        prompt_file.write_text(prompt, encoding="utf-8")
        generated = scratch / "generated"
        generated.mkdir()
        run_generation(
            generation_command(
                prompt_file,
                duration_seconds=args.duration_seconds,
                seed=args.seed,
                output_dir=generated,
            ),
            generation_environment(),
            workspace,
            deadline,
        )
        npz = generated / GENERATED_NPZ
        if not npz.is_file():
            raise SystemExit(f"HY-Motion did not write {GENERATED_NPZ}")
        members = {"motion.bvh": scratch / "motion.bvh", "motion.npz": npz}
        write_bvh(
            workspace,
            npz,
            members["motion.bvh"],
            frames=expected_frames(args.duration_seconds),
        )
        publish(members, scratch, args.output_dir)


if __name__ == "__main__":
    main()
