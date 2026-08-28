"""Offline Hunyuan3D-Omni adapter for image plus optional 3D control."""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
import trimesh
from glb_validation import normalize_glb_json_padding, validate_mesh_glb
from PIL import Image

SOURCE = Path("/opt/hunyuan3d-omni")
INPUTS = Path("/inputs")
TARGET = Path("/models/target")
DINO = Path("/models/dinov2-large")
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}
GEOMETRY_SUFFIXES = {".glb", ".gltf", ".obj", ".ply"}


def safe_named_file(name: object, suffixes: set[str]) -> Path:
    if not isinstance(name, str) or Path(name).name != name:
        raise SystemExit("job manifest paths must be plain filenames")
    path = INPUTS / name
    if not path.is_file() or path.is_symlink() or path.suffix.lower() not in suffixes:
        raise SystemExit(f"invalid job input: {name}")
    return path


def job() -> tuple[str, Path, object]:
    manifest_path = INPUTS / "job.json"
    if manifest_path.is_file() and not manifest_path.is_symlink():
        document = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not isinstance(document, dict) or set(document) - {"control_type", "image", "control"}:
            raise SystemExit("job.json has unsupported fields")
        kind = document.get("control_type")
        image = safe_named_file(document.get("image"), IMAGE_SUFFIXES)
        control = document.get("control")
        if kind in {"point", "voxel"}:
            control = safe_named_file(control, GEOMETRY_SUFFIXES)
        elif kind == "pose":
            if not isinstance(control, list) or not control or not all(
                isinstance(row, list) and len(row) == 3 and all(isinstance(v, (int, float)) for v in row)
                for row in control
            ):
                raise SystemExit("pose control must be a non-empty array of xyz triples")
        elif kind == "bbox":
            if not isinstance(control, list) or len(control) != 6 or not all(
                isinstance(v, (int, float)) for v in control
            ):
                raise SystemExit("bbox control must contain six numbers")
        else:
            raise SystemExit("control_type must be bbox, point, pose, or voxel")
        return kind, image, control

    images = sorted(path for path in INPUTS.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES)
    if len(images) != 1:
        raise SystemExit("provide one image, or job.json plus its referenced inputs")
    return "bbox", images[0], [-0.95, -0.95, -0.95, 0.95, 0.95, 0.95]


def normalized_surface(path: Path, *, sample: bool, seed: int) -> torch.Tensor:
    loaded = trimesh.load(path, force="mesh", process=True)
    if not isinstance(loaded, trimesh.Trimesh) or loaded.is_empty:
        raise SystemExit("control geometry does not contain a mesh")
    extent = float(loaded.extents.max())
    if not np.isfinite(extent) or extent <= 0:
        raise SystemExit("control geometry has invalid bounds")
    loaded.apply_translation(-loaded.bounding_box.centroid)
    loaded.apply_scale(1.9 / extent)
    if sample:
        values, _face_index = trimesh.sample.sample_surface(loaded, 81920, seed=seed)
    else:
        values = loaded.vertices
    return torch.as_tensor(values, dtype=torch.float16, device="cuda").unsqueeze(0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--entrypoint", required=True)
    parser.add_argument("--output-mime", required=True)
    parser.add_argument("--timeout-seconds", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.entrypoint != "/opt/vonk/source/run.py" or args.output_mime != "model/gltf-binary":
        raise SystemExit("unexpected signed adapter contract")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    sys.path.insert(0, str(SOURCE))
    from hy3dshape.pipelines import Hunyuan3DOmniSiTFlowMatchingPipeline
    from transformers import Dinov2Model

    kind, image_path, control = job()
    with Image.open(image_path) as image:
        image.verify()
    if not DINO.is_dir():
        raise SystemExit("missing pinned local DINOv2-large artifact")
    original_dino_loader = Dinov2Model.from_pretrained

    def load_local_dino(identifier: object, *loader_args: object, **loader_kwargs: object) -> object:
        if identifier != "facebook/dinov2-large":
            raise RuntimeError(f"unexpected DINO model identifier: {identifier}")
        loader_kwargs["local_files_only"] = True
        return original_dino_loader(str(DINO), *loader_args, **loader_kwargs)

    with patch.object(Dinov2Model, "from_pretrained", side_effect=load_local_dino):
        pipeline = Hunyuan3DOmniSiTFlowMatchingPipeline.from_pretrained(str(TARGET))
    kwargs: dict[str, object] = {"image": str(image_path)}
    if kind == "bbox":
        kwargs["bbox"] = torch.tensor(control, dtype=torch.float16, device="cuda").reshape(1, 1, 6)
    elif kind == "pose":
        kwargs["pose"] = torch.tensor(control, dtype=torch.float16, device="cuda").unsqueeze(0)
    else:
        assert isinstance(control, Path)
        kwargs[kind] = normalized_surface(control, sample=kind == "voxel", seed=args.seed)
    result = pipeline(
        **kwargs,
        num_inference_steps=50,
        octree_resolution=512,
        mc_level=0,
        guidance_scale=4.5,
        generator=torch.Generator(device="cuda").manual_seed(args.seed),
    )
    mesh = result["shapes"][0][0]
    if not isinstance(mesh, trimesh.Trimesh) or mesh.is_empty:
        raise SystemExit("Hunyuan3D-Omni did not return a mesh")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    temporary = args.output_dir / ".output.tmp.glb"
    output = args.output_dir / "output.glb"
    mesh.export(temporary, file_type="glb")
    normalize_glb_json_padding(temporary)
    try:
        validate_mesh_glb(temporary, profile="geometry")
    except ValueError as exc:
        temporary.unlink(missing_ok=True)
        raise SystemExit(f"Hunyuan3D-Omni produced an invalid GLB artifact: {exc}") from exc
    os.replace(temporary, output)


if __name__ == "__main__":
    main()
