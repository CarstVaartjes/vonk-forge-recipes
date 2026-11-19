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


def slot_files(slot: str) -> list[Path]:
    """The files of ``slot`` from the input manifest, in file name order."""
    manifest = INPUTS / "manifest.json"
    if manifest.is_symlink() or not manifest.is_file():
        raise SystemExit("the input manifest is missing")
    try:
        names = sorted(
            item["name"]
            for item in json.loads(manifest.read_text(encoding="utf-8"))["files"]
            if item["slot"] == slot
        )
    except (UnicodeDecodeError, json.JSONDecodeError, KeyError, TypeError):
        raise SystemExit("the input manifest must list each file's slot") from None
    paths = [INPUTS / name for name in names]
    if any(path.is_symlink() or not path.is_file() for path in paths):
        raise SystemExit(f"{slot} names a missing input file")
    return paths


def job(kind: str, bbox: list[float]) -> tuple[str, Path, object]:
    """The image and the control input; ``kind`` must agree with the control slot."""
    images = slot_files("image")
    if len(images) != 1 or images[0].suffix.lower() not in IMAGE_SUFFIXES:
        raise SystemExit("provide exactly one source image")
    controls = slot_files("control")
    if kind == "bbox":
        if controls:
            raise SystemExit("control-type bbox takes no control file")
        # Hunyuan3D-Omni's bbox is the box's [length, height, width], each in
        # (0, 1]: its encoder takes shape (1, 1, 3) and builds the corners
        # (upstream demos/bbox/data.json, omni_encoder.bbox_to_corners).
        return kind, images[0], bbox
    if not controls:
        raise SystemExit(f"control-type {kind} needs a control file")
    control = controls[0]
    if kind in {"point", "voxel"}:
        if control.suffix.lower() not in GEOMETRY_SUFFIXES:
            raise SystemExit(f"control-type {kind} needs a GLB, glTF, OBJ or PLY file")
        return kind, images[0], control
    if control.suffix.lower() != ".json":
        raise SystemExit("control-type pose needs a JSON file")
    try:
        rows = json.loads(control.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        raise SystemExit("pose control must be UTF-8 JSON") from None
    if (
        not isinstance(rows, list)
        or not rows
        or not all(
            isinstance(row, list)
            and len(row) == 3
            and all(isinstance(v, (int, float)) for v in row)
            for row in rows
        )
    ):
        raise SystemExit("pose control must be a non-empty array of xyz triples")
    return kind, images[0], rows


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
    parser.add_argument("--control-type", required=True)
    parser.add_argument("--bbox-length", type=float, required=True)
    parser.add_argument("--bbox-height", type=float, required=True)
    parser.add_argument("--bbox-width", type=float, required=True)
    parser.add_argument("--timeout-seconds", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if (
        args.entrypoint != "/opt/vonk/source/run.py"
        or args.output_mime != "model/gltf-binary"
    ):
        raise SystemExit("unexpected signed adapter contract")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    sys.path.insert(0, str(SOURCE))
    from hy3dshape.pipelines import Hunyuan3DOmniSiTFlowMatchingPipeline
    from transformers import Dinov2Model

    kind, image_path, control = job(
        args.control_type, [args.bbox_length, args.bbox_height, args.bbox_width]
    )
    with Image.open(image_path) as image:
        image.verify()
    if not DINO.is_dir():
        raise SystemExit("missing pinned local DINOv2-large artifact")
    original_dino_loader = Dinov2Model.from_pretrained

    def load_local_dino(
        identifier: object, *loader_args: object, **loader_kwargs: object
    ) -> object:
        if identifier != "facebook/dinov2-large":
            raise RuntimeError(f"unexpected DINO model identifier: {identifier}")
        loader_kwargs["local_files_only"] = True
        return original_dino_loader(str(DINO), *loader_args, **loader_kwargs)

    with patch.object(Dinov2Model, "from_pretrained", side_effect=load_local_dino):
        pipeline = Hunyuan3DOmniSiTFlowMatchingPipeline.from_pretrained(str(TARGET))
    kwargs: dict[str, object] = {"image": str(image_path)}
    if kind == "bbox":
        kwargs["bbox"] = torch.tensor(
            control, dtype=torch.float16, device="cuda"
        ).reshape(1, 1, 3)
    elif kind == "pose":
        kwargs["pose"] = torch.tensor(
            control, dtype=torch.float16, device="cuda"
        ).unsqueeze(0)
    else:
        assert isinstance(control, Path)
        kwargs[kind] = normalized_surface(
            control, sample=kind == "voxel", seed=args.seed
        )
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
        raise SystemExit(
            f"Hunyuan3D-Omni produced an invalid GLB artifact: {exc}"
        ) from exc
    os.replace(temporary, output)


if __name__ == "__main__":
    main()
