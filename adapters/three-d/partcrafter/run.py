"""PartCrafter job adapter: one object image to separate part meshes plus the composed scene.

The job reads the one ``image`` file, runs the creator's own steps in ``worker.py`` on the pinned
weights (RMBG-1.4 prepares the image, the pipeline samples ``--num-parts`` parts at the creator's
script defaults), validates every GLB, and publishes ``part_NN.glb`` for each part plus
``object.glb``, the scene with one node per part and the creator's seeded per-part vertex colours.

Everything the creator's script would fetch from Hugging Face is mounted instead: the PartCrafter
pipeline at ``/models/target`` and RMBG-1.4 at ``/models/rmbg``.
"""

from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import tempfile
import time
from argparse import Namespace
from pathlib import Path

from glb_validation import normalize_glb_json_padding, validate_mesh_glb
from PIL import Image
from vonk_agent_protocol.job_inputs import RecipeJobInputManifest

ENTRYPOINT = "/opt/vonk/source/run.py"
SOURCE_ROOT = Path("/opt/partcrafter")
WORKER = Path("/opt/vonk/source/worker.py")
INPUT_ROOT = Path("/inputs")
WEIGHTS = Path("/models/target")
RMBG = Path("/models/rmbg")
MODEL_FILES = {
    WEIGHTS: (
        "model_index.json",
        "scheduler/scheduler_config.json",
        "transformer/config.json",
        "transformer/diffusion_pytorch_model.safetensors",
        "vae/config.json",
        "vae/diffusion_pytorch_model.safetensors",
        "image_encoder_dinov2/config.json",
        "image_encoder_dinov2/model.safetensors",
        "feature_extractor_dinov2/preprocessor_config.json",
    ),
    RMBG: ("config.json", "model.safetensors"),
}
OUTPUT_MIME = "model/gltf-binary"
MAX_MANIFEST_BYTES = 262144
MAX_IMAGE_BYTES = 16 * 1024 * 1024
MAX_IMAGE_PIXELS = 64 * 1024 * 1024
# The media type the manifest declares for each decoded format.
IMAGE_FORMATS = {"PNG": "image/png", "JPEG": "image/jpeg", "WEBP": "image/webp"}
MAX_SEED = 2**31 - 1
# The creator's own bound (MAX_NUM_PARTS in scripts/inference_partcrafter.py).
MIN_PARTS = 1
MAX_PARTS = 16
PART_NAME = "part_{:02}.glb"
SCENE_NAME = "object.glb"
# Time kept after generation for validating and publishing the files.
PUBLISH_RESERVE_SECONDS = 30
KILL_GRACE_SECONDS = 10


def _arguments(argv: list[str] | None = None) -> Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--entrypoint", required=True)
    parser.add_argument("--output-mime", required=True)
    parser.add_argument("--num-parts", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.entrypoint != ENTRYPOINT:
        raise SystemExit("unexpected signed-bundle entrypoint")
    if args.output_mime != OUTPUT_MIME:
        raise SystemExit(f"PartCrafter emits {OUTPUT_MIME} only")
    if not MIN_PARTS <= args.num_parts <= MAX_PARTS:
        raise SystemExit(f"num-parts must be {MIN_PARTS}..{MAX_PARTS}")
    if not 0 <= args.seed <= MAX_SEED:
        raise SystemExit(f"seed must be 0..{MAX_SEED}")
    return args


def load_image(input_root: Path) -> Path:
    """The one declared ``image`` file, verified against the job manifest and decoded."""

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
    if len(manifest.files) != 1 or manifest.files[0].slot != "image":
        raise SystemExit("PartCrafter takes exactly one file in the image slot")
    item = manifest.files[0]
    path = input_root / item.name
    if path.is_symlink() or not path.is_file():
        raise SystemExit(f"{item.name} must be a regular file")
    if not 1 <= item.size_bytes <= MAX_IMAGE_BYTES or (
        path.stat().st_size != item.size_bytes
    ):
        raise SystemExit(f"{item.name} must match the manifest and hold 1..{MAX_IMAGE_BYTES} bytes")
    try:
        with Image.open(path) as image:
            if image.width * image.height > MAX_IMAGE_PIXELS:
                raise SystemExit(f"{item.name} has more than {MAX_IMAGE_PIXELS} pixels")
            decoded = image.format
            image.load()
    except (OSError, ValueError, Image.DecompressionBombError) as error:
        raise SystemExit(f"{item.name} is not a decodable image: {error}") from error
    if IMAGE_FORMATS.get(decoded or "") != item.media_type:
        raise SystemExit(f"{item.name} decodes as {decoded}, not {item.media_type}")
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
    image: Path, args: Namespace, *, generated: Path, worker: Path = WORKER
) -> list[str]:
    return [
        sys.executable,
        str(worker),
        "--image",
        str(image),
        "--num-parts",
        str(args.num_parts),
        "--seed",
        str(args.seed),
        "--weights",
        str(WEIGHTS),
        "--rmbg",
        str(RMBG),
        "--output-dir",
        str(generated),
    ]


def generation_environment(base: dict[str, str] | None = None) -> dict[str, str]:
    environment = dict(os.environ if base is None else base)
    # PYTHONPATH is platform-owned for the image, so only this child gets the creator's source.
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
        raise SystemExit("PartCrafter job budget exhausted before generation")
    process = subprocess.Popen(
        command, env=environment, cwd=cwd, start_new_session=True
    )
    try:
        code = process.wait(timeout=budget)
    except subprocess.TimeoutExpired:
        _kill_group(process)
        raise SystemExit("PartCrafter generation exceeded the job budget") from None
    except BaseException:
        _kill_group(process)
        raise
    if code != 0:
        raise SystemExit(f"PartCrafter generation failed with exit status {code}")


def validated_outputs(generated: Path, num_parts: int) -> list[Path]:
    """Every part and the scene, each a valid mesh GLB; the scene holds one mesh per part."""

    paths = [generated / PART_NAME.format(index) for index in range(num_parts)]
    scene = generated / SCENE_NAME
    for path in [*paths, scene]:
        if not path.is_file() or path.is_symlink():
            raise SystemExit(f"PartCrafter did not write {path.name}")
        normalize_glb_json_padding(path)
        try:
            facts = validate_mesh_glb(path, profile="geometry")
        except ValueError as error:
            raise SystemExit(f"PartCrafter wrote an invalid {path.name}: {error}") from error
        if path == scene and facts["mesh_count"] != num_parts:
            raise SystemExit(
                f"{SCENE_NAME} holds {facts['mesh_count']} meshes, not the {num_parts} parts"
            )
    return [scene, *paths]


def publish(files: list[Path], output_dir: Path) -> None:
    """One atomic rename per file; a failure removes what this job already placed."""

    placed: list[Path] = []
    try:
        for path in files:
            final = output_dir / path.name
            os.replace(path, final)
            placed.append(final)
    except BaseException:
        for final in placed:
            final.unlink(missing_ok=True)
        raise


def main(argv: list[str] | None = None) -> None:
    args = _arguments(argv)
    deadline = _job_deadline()
    image = load_image(INPUT_ROOT)
    check_model_files()
    # The platform owns tmp/ and cache/; only our unique workspace must be fresh.
    scratch_root = args.output_dir / "tmp"
    scratch_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="vonk-partcrafter-", dir=scratch_root) as name:
        generated = Path(name) / "generated"
        generated.mkdir()
        run_generation(
            generation_command(image, args, generated=generated),
            generation_environment(),
            Path(name),
            deadline,
        )
        publish(validated_outputs(generated, args.num_parts), args.output_dir)


if __name__ == "__main__":
    main()
