"""Offline single-image Pixal3D adapter that emits one GLB artifact."""

from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path
from types import MethodType


SUPPORTED_SUFFIXES = {".jpeg", ".jpg", ".png", ".webp"}


def one_input(input_dir: Path) -> Path:
    candidates = sorted(
        path for path in input_dir.iterdir() if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES
    )
    if len(candidates) != 1:
        raise SystemExit(f"expected exactly one JPEG, PNG, or WebP input; found {len(candidates)}")
    return candidates[0]


def camera_params(field_of_view: float, mesh_scale: float, image_resolution: int = 512) -> dict[str, float]:
    focal_pixels = (16.0 / math.tan(field_of_view / 2.0)) * image_resolution / 32.0
    x_world = -0.5 / mesh_scale
    y_world = 0.0
    x_ndc = -image_resolution / 2.0
    distance = focal_pixels * x_world / x_ndc - y_world
    return {"camera_angle_x": field_of_view, "distance": distance, "mesh_scale": mesh_scale}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--entrypoint", default="/opt/vonk/source/pixal3d_job.py")
    parser.add_argument("--output-mime", required=True)
    parser.add_argument("--input-dir", type=Path, default=Path("/inputs"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pipeline-resolution", choices=("1024_cascade", "1536_cascade"), default="1024_cascade")
    parser.add_argument("--field-of-view-radians", type=float, default=0.2)
    parser.add_argument("--mesh-scale", type=float, default=1.0)
    parser.add_argument("--max-num-tokens", type=int, default=49152)
    parser.add_argument("--decimation-target", type=int, default=1_000_000)
    parser.add_argument("--texture-size", type=int, choices=(1024, 2048, 4096), default=2048)
    parser.add_argument("--timeout-seconds", type=int, default=3600)
    args = parser.parse_args()
    if args.entrypoint != "/opt/vonk/source/pixal3d_job.py":
        raise SystemExit("unexpected adapter entrypoint")
    if args.output_mime != "model/gltf-binary":
        raise SystemExit("Pixal3D adapter only emits model/gltf-binary")
    if not 0.05 <= args.field_of_view_radians <= 1.5:
        raise SystemExit("field of view must be between 0.05 and 1.5 radians")
    if not 0.25 <= args.mesh_scale <= 4.0:
        raise SystemExit("mesh scale must be between 0.25 and 4")
    if not 1 <= args.max_num_tokens <= 65536:
        raise SystemExit("max token count must be between 1 and 65,536")
    if not 1 <= args.decimation_target <= 2_000_000:
        raise SystemExit("decimation target must be between 1 and 2,000,000")

    os.environ.setdefault("ATTN_BACKEND", "sdpa")
    os.environ.setdefault("SPARSE_ATTN_BACKEND", "sdpa")
    os.environ.setdefault("SPARSE_CONV_BACKEND", "flex_gemm")

    import numpy as np
    import torch
    from PIL import Image, ImageOps
    import o_voxel
    from pixal3d.pipelines import Pixal3DImageTo3DPipeline
    from pixal3d.pipelines import rembg
    from pixal3d.trainers.flow_matching.mixins.image_conditioned_proj import (
        DinoV3ProjFeatureExtractor,
    )

    class NoBackgroundModel:
        def __init__(self, **_: object) -> None:
            pass

        def to(self, _: object) -> "NoBackgroundModel":
            return self

        cuda = to
        cpu = to

    rembg.BiRefNet = NoBackgroundModel

    def load_local_naf(self: object) -> None:
        if self.naf_model is None:
            device = next(self.model.parameters()).device
            sys.path.insert(0, "/opt/vonk/naf-source")
            from src.model.naf import NAF

            model = NAF().to(device)
            state = torch.load("/models/naf/naf_release.pth", map_location=device, weights_only=True)
            model.load_state_dict(state, strict=True)
            model.eval()
            model.requires_grad_(False)
            self.naf_model = model

    pipeline = Pixal3DImageTo3DPipeline.from_pretrained("/models/target")
    configs = {
        "ss": {"image_size": 512, "grid_resolution": 16},
        "shape_512": {
            "image_size": 512,
            "grid_resolution": 32,
            "use_naf_upsample": True,
            "naf_target_size": 512,
        },
        "shape_1024": {
            "image_size": 1024,
            "grid_resolution": 64,
            "use_naf_upsample": True,
            "naf_target_size": 512,
        },
        "tex_1024": {
            "image_size": 1024,
            "grid_resolution": 64,
            "use_naf_upsample": True,
            "naf_target_size": 1024,
        },
    }
    modules = {}
    for name, config in configs.items():
        module = DinoV3ProjFeatureExtractor(model_name="/models/dino", **config)
        module._load_naf = MethodType(load_local_naf, module)
        modules[name] = module
    pipeline.image_cond_model_ss = modules["ss"]
    pipeline.image_cond_model_shape_512 = modules["shape_512"]
    pipeline.image_cond_model_shape_1024 = modules["shape_1024"]
    pipeline.image_cond_model_tex_1024 = modules["tex_1024"]
    pipeline._device = torch.device("cuda")
    pipeline.low_vram = True
    for module in modules.values():
        if module.use_naf_upsample:
            module._load_naf()

    with Image.open(one_input(args.input_dir)) as opened:
        image = ImageOps.exif_transpose(opened).convert("RGB")
    meshes, (_, _, resolution) = pipeline.run(
        image,
        camera_params=camera_params(args.field_of_view_radians, args.mesh_scale),
        seed=args.seed,
        preprocess_image=False,
        return_latent=True,
        pipeline_type=args.pipeline_resolution,
        max_num_tokens=args.max_num_tokens,
    )
    mesh = meshes[0]
    glb = o_voxel.postprocess.to_glb(
        vertices=mesh.vertices,
        faces=mesh.faces,
        attr_volume=mesh.attrs,
        coords=mesh.coords,
        attr_layout=pipeline.pbr_attr_layout,
        grid_size=resolution,
        aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
        decimation_target=args.decimation_target,
        texture_size=args.texture_size,
        remesh=True,
        remesh_band=1,
        remesh_project=0,
        use_tqdm=True,
    )
    rotation = np.array(
        [[-1, 0, 0, 0], [0, 0, -1, 0], [0, -1, 0, 0], [0, 0, 0, 1]],
        dtype=np.float64,
    )
    glb.apply_transform(rotation)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    temporary = args.output_dir / ".output.tmp.glb"
    glb.export(temporary, extension_webp=True)
    os.replace(temporary, args.output_dir / "output.glb")


if __name__ == "__main__":
    main()
