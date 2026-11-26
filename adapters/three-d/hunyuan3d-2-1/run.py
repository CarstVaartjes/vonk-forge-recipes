"""Hunyuan3D 2.1 job adapter: an object image (and optionally a mesh) to one PBR GLB.

Two recipes share this adapter and differ in the literal ``--mode`` argument:

* ``full``: the creator's ``demo.py`` path. The shape pipeline turns the image into a mesh, the paint
  pipeline textures it (albedo, metallic, roughness), and ``mesh.glb`` is written.
* ``paint``: only the paint pipeline, on a mesh the job supplies, with the image as texture reference.

The job reads the ``image`` slot (and, in paint mode, the ``mesh`` slot), verified against the job
manifest, runs the creator's steps in ``worker.py`` in their own process group, validates the GLB
(``glb_validation.py``, profile ``textured-pbr``) and publishes it with one atomic rename.
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
from argparse import Namespace
from dataclasses import dataclass
from pathlib import Path

import numpy
import trimesh
from glb_validation import normalize_glb_json_padding, validate_mesh_glb
from PIL import Image
from vonk_agent_protocol.job_inputs import RecipeJobInputManifest

ENTRYPOINT = "/opt/vonk/source/run.py"
WORKER = Path("/opt/vonk/source/worker.py")
SOURCE_ROOT = Path("/opt/hunyuan3d")
SOURCE_PATHS = (SOURCE_ROOT, SOURCE_ROOT / "hy3dshape", SOURCE_ROOT / "hy3dpaint")
INPUT_ROOT = Path("/inputs")
MODEL_ROOT = Path("/models/target")
DIT = MODEL_ROOT / "hunyuan3d-dit-v2-1"
PAINT = MODEL_ROOT / "hunyuan3d-paintpbr-v2-1"
DINO = Path("/models/dinov2-giant")
UPSCALER = Path("/models/realesrgan")
# Every file the creator's code reads; the full recipe needs the shape files as well.
PAINT_FILES = {
    PAINT: (
        "model_index.json",
        "scheduler/scheduler_config.json",
        "feature_extractor/preprocessor_config.json",
        "image_encoder/config.json",
        "image_encoder/model.safetensors",
        "text_encoder/config.json",
        "text_encoder/pytorch_model.bin",
        "tokenizer/merges.txt",
        "tokenizer/special_tokens_map.json",
        "tokenizer/tokenizer_config.json",
        "tokenizer/vocab.json",
        "unet/config.json",
        "unet/diffusion_pytorch_model.bin",
        "unet/attn_processor.py",
        "unet/model.py",
        "unet/modules.py",
        "vae/config.json",
        "vae/diffusion_pytorch_model.bin",
    ),
    DINO: ("config.json", "preprocessor_config.json", "model.safetensors"),
    UPSCALER: ("RealESRGAN_x4plus.pth",),
}
SHAPE_FILES = {DIT: ("config.yaml", "model.fp16.ckpt")}
MODES = ("full", "paint")
MAX_MANIFEST_BYTES = 262144
MAX_IMAGE_BYTES = 16 * 1024 * 1024
MAX_IMAGE_PIXELS = 64 * 1024 * 1024
MAX_MESH_BYTES = 240 * 1024 * 1024
MAX_SEED = 10_000_000
INPUT_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._ -]{0,119}")
IMAGE_FORMATS = {".png": "PNG", ".jpg": "JPEG", ".jpeg": "JPEG", ".webp": "WEBP"}
IMAGE_TYPES = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
}
MESH_TYPES = {".glb": "model/gltf-binary", ".obj": "model/obj"}
OUTPUT_NAME = "mesh.glb"
# Time kept after the creator's steps for validating and publishing.
PUBLISH_RESERVE_SECONDS = 30
KILL_GRACE_SECONDS = 10


def _arguments(argv: list[str] | None = None) -> Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--entrypoint", required=True)
    parser.add_argument("--output-mime", required=True)
    parser.add_argument("--mode", required=True)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.entrypoint != ENTRYPOINT:
        raise SystemExit("unexpected signed-bundle entrypoint")
    if args.output_mime != "model/gltf-binary":
        raise SystemExit("Hunyuan3D 2.1 emits model/gltf-binary only")
    if args.mode not in MODES:
        raise SystemExit(f"mode must be one of {', '.join(MODES)}")
    if args.mode == "paint" and args.seed is not None:
        # The creator's paint step seeds itself with 0; a seed here would change nothing.
        raise SystemExit("the paint step is seeded by the creator; give no seed")
    if args.mode == "full" and (args.seed is None or not 0 <= args.seed <= MAX_SEED):
        raise SystemExit(f"the shape seed must be 0..{MAX_SEED}")
    return args


@dataclass(frozen=True)
class Inputs:
    image: Path
    mesh: Path | None


def _checked_image(path: Path) -> None:
    suffix = path.suffix.lower()
    try:
        with Image.open(path) as image:
            image.verify()
        with Image.open(path) as image:
            width, height, found = image.width, image.height, image.format
    except (OSError, Image.DecompressionBombError) as error:
        raise SystemExit(f"{path.name} is not a readable image") from error
    if found != IMAGE_FORMATS[suffix]:
        raise SystemExit(f"{path.name} content does not match its extension")
    if width * height > MAX_IMAGE_PIXELS:
        raise SystemExit(f"{path.name} exceeds {MAX_IMAGE_PIXELS} pixels")


def _checked_mesh(path: Path) -> None:
    """A non-empty, finite triangle mesh, before any model is loaded."""
    try:
        if path.suffix.lower() == ".glb":
            validate_mesh_glb(path, profile="geometry")
            return
        mesh = trimesh.load(path, force="mesh", process=False)
        if not len(mesh.faces) or not numpy.isfinite(mesh.vertices).all():
            raise ValueError("the mesh has no faces or a non-finite vertex")
    except (ValueError, OSError) as error:
        raise SystemExit(f"{path.name} is not a usable triangle mesh: {error}") from error


def load_inputs(input_root: Path, mode: str) -> Inputs:
    """The declared ``image`` (and, in paint mode, ``mesh``) files, verified against the job manifest."""

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
    wanted = {"image"} if mode == "full" else {"image", "mesh"}
    found: dict[str, Path] = {}
    for item in manifest.files:
        if item.slot not in wanted or item.slot in found:
            raise SystemExit(f"undeclared or repeated input slot {item.slot}")
        path = input_root / item.name
        if INPUT_NAME.fullmatch(item.name) is None or path.is_symlink() or not path.is_file():
            raise SystemExit(f"{item.name} must be a regular file with a plain name")
        suffix = path.suffix.lower()
        types, limit = (
            (IMAGE_TYPES, MAX_IMAGE_BYTES) if item.slot == "image" else (MESH_TYPES, MAX_MESH_BYTES)
        )
        if types.get(suffix) != item.media_type:
            raise SystemExit(f"{item.name} must be one of {', '.join(types)} of its own media type")
        if not 1 <= item.size_bytes <= limit or path.stat().st_size != item.size_bytes:
            raise SystemExit(f"{item.name} does not match the job manifest or its size bound")
        (_checked_image if item.slot == "image" else _checked_mesh)(path)
        found[item.slot] = path
    if set(found) != wanted:
        raise SystemExit(f"{mode} mode needs exactly the {', '.join(sorted(wanted))} input")
    return Inputs(found["image"], found.get("mesh"))


def _job_deadline() -> float:
    """The platform's job budget as one monotonic deadline."""
    try:
        seconds = int(os.environ["VONK_JOB_TIMEOUT_SECONDS"])
    except (KeyError, ValueError) as error:
        raise SystemExit("job must declare VONK_JOB_TIMEOUT_SECONDS") from error
    if not 1 <= seconds <= 3600:
        raise SystemExit("job timeout must be between 1 and 3600 seconds")
    return time.monotonic() + seconds


def check_model_files(mode: str) -> None:
    """Fail now, naming the file, instead of deep inside the creator's loader."""
    files = PAINT_FILES if mode == "paint" else {**SHAPE_FILES, **PAINT_FILES}
    for directory, names in files.items():
        for name in names:
            if not (directory / name).is_file():
                raise SystemExit(f"model file is not mounted: {directory / name}")


def worker_command(
    inputs: Inputs, args: Namespace, *, scratch: Path, worker: Path = WORKER
) -> list[str]:
    command = [
        sys.executable,
        str(worker),
        "--mode",
        args.mode,
        "--image",
        str(inputs.image),
        "--scratch",
        str(scratch),
        "--output",
        str(scratch / OUTPUT_NAME),
    ]
    if inputs.mesh is not None:
        # The paint step writes beside the mesh it is given, so it gets a scratch copy.
        staged = scratch / f"input{inputs.mesh.suffix.lower()}"
        shutil.copyfile(inputs.mesh, staged)
        command += ["--mesh", str(staged)]
    if args.seed is not None:
        command += ["--seed", str(args.seed)]
    return command


def worker_environment(base: dict[str, str] | None = None) -> dict[str, str]:
    environment = dict(os.environ if base is None else base)
    # PYTHONPATH is platform-owned for the image, so only this child gets the source trees.
    environment.update(PYTHONPATH=os.pathsep.join(str(path) for path in SOURCE_PATHS))
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
        raise SystemExit("Hunyuan3D 2.1 job budget exhausted before generation")
    process = subprocess.Popen(command, env=environment, cwd=cwd, start_new_session=True)
    try:
        code = process.wait(timeout=budget)
    except subprocess.TimeoutExpired:
        _kill_group(process)
        raise SystemExit("Hunyuan3D 2.1 generation exceeded the job budget") from None
    except BaseException:
        _kill_group(process)
        raise
    if code != 0:
        raise SystemExit(f"Hunyuan3D 2.1 generation failed with exit status {code}")


def publish(staged: Path, output_dir: Path) -> Path:
    """Validate the PBR GLB, then one atomic rename; a refused file leaves no output."""
    if not staged.is_file():
        raise SystemExit(f"Hunyuan3D 2.1 did not write {staged.name}")
    normalize_glb_json_padding(staged)
    try:
        validate_mesh_glb(staged, profile="textured-pbr")
    except ValueError as error:
        raise SystemExit(f"Hunyuan3D 2.1 produced an invalid PBR GLB: {error}") from error
    final = output_dir / OUTPUT_NAME
    os.replace(staged, final)
    return final


def main(argv: list[str] | None = None) -> None:
    args = _arguments(argv)
    deadline = _job_deadline()
    inputs = load_inputs(INPUT_ROOT, args.mode)
    check_model_files(args.mode)
    # The platform owns tmp/ and cache/; only our unique scratch must be fresh.
    scratch_root = args.output_dir / "tmp"
    scratch_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="vonk-hunyuan3d-", dir=scratch_root) as name:
        scratch = Path(name)
        run_worker(
            worker_command(inputs, args, scratch=scratch),
            worker_environment(),
            scratch,
            deadline,
        )
        publish(scratch / OUTPUT_NAME, args.output_dir)


if __name__ == "__main__":
    main()
