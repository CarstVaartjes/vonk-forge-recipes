"""The creator's Hunyuan3D 2.1 ``demo.py`` steps, run once for one job.

``run.py`` starts this module in its own process group with the scratch directory as the working
directory and ``PYTHONPATH`` set to the creator's source trees. It calls the creator's own steps in
the creator's order:

1. ``full`` mode only: ``Hunyuan3DDiTFlowMatchingPipeline`` turns the image (opened as RGBA, with no
   background removal, exactly as ``demo.py`` does) into a mesh. The generator is a CPU generator,
   as in ``gradio_app.py``; every other call default stays.
2. ``Hunyuan3DPaintPipeline`` remeshes to at most 40,000 faces, unwraps UVs, paints six views with
   the PBR multiview model, upscales them and bakes albedo, metallic and roughness into
   ``textured.obj``. The paint step seeds itself with 0.
3. ``export_pbr_glb.write_pbr_glb`` writes the textures into one GLB (the creator's GLB writers need
   bpy, which has no aarch64 build).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import trimesh
from export_pbr_glb import write_pbr_glb
from hy3dshape.pipelines import Hunyuan3DDiTFlowMatchingPipeline
from PIL import Image
from textureGenPipeline import Hunyuan3DPaintConfig, Hunyuan3DPaintPipeline
from torchvision_fix import apply_fix

SOURCE = Path("/opt/hunyuan3d")
DIT = Path("/models/target/hunyuan3d-dit-v2-1")
MULTIVIEW_ROOT = "/models/target"
DINO = "/models/dinov2-giant"
UPSCALER = "/models/realesrgan/RealESRGAN_x4plus.pth"
# demo.py: max_num_view = 6 (6 to 9) and resolution = 512 (512 or 768).
MAX_NUM_VIEW = 6
RESOLUTION = 512


def _describe(label: str, path: Path) -> None:
    """Faces and bounding box on stderr: the axis frame of the paint step is unverified on GB10."""
    mesh = trimesh.load(path, force="mesh", process=False)
    print(
        f"{label}: faces={len(mesh.faces)} bounds={mesh.bounds.round(4).tolist()}",
        file=sys.stderr,
        flush=True,
    )


def shape(image_path: Path, seed: int, destination: Path) -> Hunyuan3DDiTFlowMatchingPipeline:
    pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_single_file(
        ckpt_path=str(DIT / "model.fp16.ckpt"),
        config_path=str(DIT / "config.yaml"),
        dtype=torch.float16,
    )
    with Image.open(image_path) as opened:
        image = opened.convert("RGBA")
    mesh = pipeline(image=image, generator=torch.Generator().manual_seed(seed))[0]
    mesh.export(destination)
    return pipeline


def paint(mesh_path: Path, image_path: Path, scratch: Path) -> Path:
    config = Hunyuan3DPaintConfig(MAX_NUM_VIEW, RESOLUTION)
    config.multiview_pretrained_path = MULTIVIEW_ROOT
    config.dino_ckpt_path = DINO
    config.realesrgan_ckpt_path = UPSCALER
    config.multiview_cfg_path = str(SOURCE / "hy3dpaint/cfgs/hunyuan-paint-pbr.yaml")
    textured = scratch / "textured.obj"
    Hunyuan3DPaintPipeline(config)(
        mesh_path=str(mesh_path),
        image_path=str(image_path),
        output_mesh_path=str(textured),
        save_glb=False,
    )
    return textured


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("full", "paint"), required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--mesh", type=Path)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--scratch", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    # basicsr imports the removed torchvision.transforms.functional_tensor; it is first imported
    # when the paint pipeline loads its upscaler, after this.
    apply_fix()
    if args.mode == "full":
        mesh_path = args.scratch / "shape.glb"
        # demo.py keeps the shape pipeline alive while it paints; so does this.
        resident = shape(args.image, args.seed, mesh_path)
    else:
        mesh_path = args.mesh
    _describe("input mesh", mesh_path)
    textured = paint(mesh_path, args.image, args.scratch)
    write_pbr_glb(
        textured,
        textured.with_suffix(".jpg"),
        textured.with_name("textured_metallic.jpg"),
        textured.with_name("textured_roughness.jpg"),
        args.output,
    )
    _describe("output mesh", args.output)


if __name__ == "__main__":
    main()
