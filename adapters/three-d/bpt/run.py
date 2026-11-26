"""BPT job adapter: one dense mesh to an artist-style rebuilt mesh at the source's scale.

The job reads the one ``mesh`` file, refuses anything that is not a non-empty finite triangle mesh,
runs the creator's own steps in ``worker.py`` on the pinned checkpoint (a point cloud sampled from
the mesh conditions the model, which writes new vertex codes at ``--temperature``), restores the
source's bounding box with the exact inverse of the creator's normalisation, validates the GLB and
publishes ``retopo.glb``.

Everything the creator's script would load is mounted instead: the checkpoint at ``/models/target``.
The creator's own config, point-cloud encoder config and code come from ``/opt/bpt``.
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
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import trimesh
from glb_validation import normalize_glb_json_padding, validate_mesh_glb
from vonk_agent_protocol.job_inputs import RecipeJobInputManifest

ENTRYPOINT = "/opt/vonk/source/run.py"
SOURCE_ROOT = Path("/opt/bpt")
CONFIG = SOURCE_ROOT / "config/BPT-open-8k-8-16.yaml"
WORKER = Path("/opt/vonk/source/worker.py")
INPUT_ROOT = Path("/inputs")
CHECKPOINT = Path("/models/target/bpt-8-16-500m.pt")
OUTPUT_MIME = "model/gltf-binary"
OUTPUT_NAME = "retopo.glb"
MAX_MANIFEST_BYTES = 262144
MAX_MESH_BYTES = 256 * 1024 * 1024
# The media type the manifest declares for each accepted extension.
MESH_TYPES = {
    ".glb": "model/gltf-binary",
    ".obj": "model/obj",
    ".ply": "model/ply",
}
MAX_SEED = 2**31 - 1
# The creator's CLI default is 0.5 and gives no range; the range is the recipe's.
MIN_TEMPERATURE = 0.1
MAX_TEMPERATURE = 1.0
# apply_normalize in the creator's utils.py: centre the bounding box, scale the longest side to
# 2 * 0.95, so the normalised mesh fills [-0.95, 0.95] and the model quantises over [-1, 1].
NORMALIZED_EXTENT = 2 * 0.95
# Time kept after generation for restoring the scale, validating and publishing.
PUBLISH_RESERVE_SECONDS = 30
KILL_GRACE_SECONDS = 10


def _arguments(argv: list[str] | None = None) -> Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--entrypoint", required=True)
    parser.add_argument("--output-mime", required=True)
    parser.add_argument("--temperature", type=float, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.entrypoint != ENTRYPOINT:
        raise SystemExit("unexpected signed-bundle entrypoint")
    if args.output_mime != OUTPUT_MIME:
        raise SystemExit(f"BPT emits {OUTPUT_MIME} only")
    if not MIN_TEMPERATURE <= args.temperature <= MAX_TEMPERATURE:
        raise SystemExit(f"temperature must be {MIN_TEMPERATURE:g}..{MAX_TEMPERATURE:g}")
    if not 0 <= args.seed <= MAX_SEED:
        raise SystemExit(f"seed must be 0..{MAX_SEED}")
    return args


@dataclass(frozen=True)
class SourceMesh:
    path: Path
    bounds: np.ndarray
    faces: int

    @property
    def center(self) -> np.ndarray:
        return (self.bounds[0] + self.bounds[1]) / 2

    @property
    def longest_side(self) -> float:
        return float((self.bounds[1] - self.bounds[0]).max())


def load_mesh(input_root: Path) -> SourceMesh:
    """The one declared ``mesh`` file, verified against the manifest and loaded as a triangle mesh."""

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
    if len(manifest.files) != 1 or manifest.files[0].slot != "mesh":
        raise SystemExit("BPT takes exactly one file in the mesh slot")
    item = manifest.files[0]
    path = input_root / item.name
    if path.is_symlink() or not path.is_file():
        raise SystemExit(f"{item.name} must be a regular file")
    if not 1 <= item.size_bytes <= MAX_MESH_BYTES or (
        path.stat().st_size != item.size_bytes
    ):
        raise SystemExit(f"{item.name} must match the manifest and hold 1..{MAX_MESH_BYTES} bytes")
    if MESH_TYPES.get(path.suffix.lower()) != item.media_type:
        raise SystemExit(f"{item.name} must be a .glb, .obj or .ply file of its media type")
    try:
        # The creator's own load (utils.py Dataset): force='mesh' joins a scene into one mesh.
        mesh = trimesh.load(str(path), force="mesh")
    except Exception as error:
        raise SystemExit(f"{item.name} could not be loaded as a mesh: {error}") from error
    if not isinstance(mesh, trimesh.Trimesh) or len(mesh.faces) == 0:
        raise SystemExit(f"{item.name} is not a triangle mesh with faces")
    source = SourceMesh(path, np.asarray(mesh.bounds, dtype=np.float64), len(mesh.faces))
    if not np.isfinite(source.bounds).all() or not source.longest_side > 0:
        raise SystemExit(f"{item.name} has no finite, non-degenerate extent")
    return source


def restore_scale(vertices: np.ndarray, source: SourceMesh) -> np.ndarray:
    """The exact inverse of the creator's normalisation with the input's own bounding box.

    Never refits to the output's bounds: the model's mesh keeps its own overhang and gaps.
    """
    return vertices.astype(np.float64) * source.longest_side / NORMALIZED_EXTENT + source.center


def _job_deadline() -> float:
    """The platform's job budget as one monotonic deadline."""
    try:
        seconds = int(os.environ["VONK_JOB_TIMEOUT_SECONDS"])
    except (KeyError, ValueError) as error:
        raise SystemExit("job must declare VONK_JOB_TIMEOUT_SECONDS") from error
    if not 1 <= seconds <= 3600:
        raise SystemExit("job timeout must be between 1 and 3600 seconds")
    return time.monotonic() + seconds


def check_model_files(files: tuple[Path, ...] = (CHECKPOINT, CONFIG)) -> None:
    """Fail now, naming the file, instead of deep inside the creator's loader."""
    for path in files:
        if not path.is_file():
            raise SystemExit(f"required file is missing: {path}")


def build_workspace(root: Path, *, source: Path = SOURCE_ROOT) -> Path:
    """The working directory the creator's relative ``miche/shapevae-256.yaml`` resolves in."""

    workspace = root / "workspace"
    workspace.mkdir()
    (workspace / "miche").symlink_to(source / "miche", target_is_directory=True)
    return workspace


def generation_command(
    mesh: SourceMesh, args: Namespace, *, output: Path, worker: Path = WORKER
) -> list[str]:
    return [
        sys.executable,
        str(worker),
        "--mesh",
        str(mesh.path),
        "--temperature",
        str(args.temperature),
        "--seed",
        str(args.seed),
        "--config",
        str(CONFIG),
        "--checkpoint",
        str(CHECKPOINT),
        "--output",
        str(output),
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
        raise SystemExit("BPT job budget exhausted before generation")
    process = subprocess.Popen(
        command, env=environment, cwd=cwd, start_new_session=True
    )
    try:
        code = process.wait(timeout=budget)
    except subprocess.TimeoutExpired:
        _kill_group(process)
        raise SystemExit("BPT generation exceeded the job budget") from None
    except BaseException:
        _kill_group(process)
        raise
    if code != 0:
        raise SystemExit(f"BPT generation failed with exit status {code}")


def rebuilt_glb(raw: Path, source: SourceMesh, staged: Path) -> Path:
    """The generated mesh at the source's scale, as a validated GLB without colour."""

    if not raw.is_file():
        raise SystemExit("BPT did not write a mesh")
    with np.load(raw) as generated:
        vertices = np.asarray(generated["vertices"])
        faces = np.asarray(generated["faces"])
    if len(vertices) == 0 or len(faces) == 0:
        raise SystemExit("BPT generated an empty mesh")
    restored = restore_scale(vertices, source)
    if not np.isfinite(restored).all():
        raise SystemExit("BPT generated non-finite vertices")
    mesh = trimesh.Trimesh(restored, faces, process=False)
    print(
        f"BPT faces {source.faces} -> {len(mesh.faces)}; "
        f"bounds {source.bounds.round(4).tolist()} -> {mesh.bounds.round(4).tolist()}",
        file=sys.stderr,
    )
    mesh.export(staged, file_type="glb")
    normalize_glb_json_padding(staged)
    try:
        validate_mesh_glb(staged, profile="geometry")
    except ValueError as error:
        raise SystemExit(f"BPT wrote an invalid GLB: {error}") from error
    return staged


def main(argv: list[str] | None = None) -> None:
    args = _arguments(argv)
    deadline = _job_deadline()
    source = load_mesh(INPUT_ROOT)
    check_model_files()
    # The platform owns tmp/ and cache/; only our unique workspace must be fresh.
    scratch_root = args.output_dir / "tmp"
    scratch_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="vonk-bpt-", dir=scratch_root) as name:
        root = Path(name)
        workspace = build_workspace(root)
        raw = root / "generated.npz"
        run_generation(
            generation_command(source, args, output=raw),
            generation_environment(),
            workspace,
            deadline,
        )
        staged = rebuilt_glb(raw, source, root / OUTPUT_NAME)
        os.replace(staged, args.output_dir / OUTPUT_NAME)


if __name__ == "__main__":
    main()
