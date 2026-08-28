"""Offline Step1X-3D image-to-mesh and mesh-texturing adapter.

The upstream project exposes Python functions and a Gradio application, but no
bounded filesystem job interface.  This adapter deliberately accepts exactly
one image for geometry jobs and exactly one image plus one GLB for texture
jobs.  All model authorities are mounted by the recipe at immutable paths.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from glb_validation import normalize_glb_json_padding, validate_mesh_glb

_IMAGE_SUFFIXES = frozenset({".jpeg", ".jpg", ".png", ".webp"})


def _publish_glb(temporary: Path, output: Path, *, profile: str) -> None:
    normalize_glb_json_padding(temporary)
    try:
        validate_mesh_glb(temporary, profile=profile)
    except ValueError as exc:
        temporary.unlink(missing_ok=True)
        raise SystemExit(f"Step1X produced an invalid {profile} GLB artifact: {exc}") from exc
    os.replace(temporary, output)


def _files(suffixes: frozenset[str]) -> list[Path]:
    root = Path("/inputs")
    return sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in suffixes
    )


def _one_input(label: str, suffixes: frozenset[str]) -> Path:
    matches = _files(suffixes)
    if len(matches) != 1:
        raise SystemExit(f"expected exactly one {label} input, found {len(matches)}")
    return matches[0]


def _validated_input_glb(path: Path) -> Path:
    """Reject malformed or semantically empty meshes before loading upstream."""

    try:
        validate_mesh_glb(path, profile="geometry")
    except ValueError as exc:
        raise SystemExit(f"Step1X texture input is not a valid GLB mesh: {exc}") from exc
    return path


def _offline_geometry_image(source: Path, destination: Path) -> Path:
    """Give upstream an alpha-bearing input so it never downloads BRIA RMBG.

    Step1X skips its implicit background-removal authority when an alpha channel
    contains transparency.  Preserve an existing alpha channel; for an opaque
    image, make only the bottom-right pixel transparent.  This is deterministic,
    keeps the useful image content intact, and makes offline behavior explicit.
    """

    from PIL import Image

    with Image.open(source) as opened:
        image = opened.convert("RGBA")
    alpha_minimum, alpha_maximum = image.getchannel("A").getextrema()
    if alpha_minimum == alpha_maximum == 255:
        image.putpixel((image.width - 1, image.height - 1), (255, 255, 255, 0))
    image.save(destination, format="PNG")
    return destination


def _geometry(*, label_control: bool, seed: int, output_dir: Path) -> None:
    import torch
    from step1x3d_geometry.models.pipelines.pipeline import (
        Step1X3DGeometryPipeline,
    )

    source = _one_input("image", _IMAGE_SUFFIXES)
    prepared = _offline_geometry_image(source, output_dir / "prepared-input.png")
    subfolder = (
        "Step1X-3D-Geometry-Label-1300m"
        if label_control
        else "Step1X-3D-Geometry-1300m"
    )
    pipeline = Step1X3DGeometryPipeline.from_pretrained(
        "/models/target",
        subfolder=subfolder,
        local_files_only=True,
    ).to("cuda")
    generator = torch.Generator(device=pipeline.device).manual_seed(seed)
    arguments: dict[str, object] = {
        "guidance_scale": 7.5,
        "num_inference_steps": 50,
        "generator": generator,
        # These optional post-processors import pymeshlab, which has no official
        # Linux ARM64 wheel.  Trimesh can export the native generated mesh.
        "do_remove_floater": False,
        "do_remove_degenerate_face": False,
        "do_reduce_face": False,
    }
    if label_control:
        arguments.update(
            label={"symmetry": "x", "geometry_type": "sharp"},
            octree_resolution=384,
            max_facenum=400000,
        )
    result = pipeline(str(prepared), **arguments)
    temporary = output_dir / ".step1x-geometry.tmp.glb"
    result.mesh[0].export(temporary)
    _publish_glb(temporary, output_dir / "step1x-geometry.glb", profile="geometry")


def _texture(*, seed: int, output_dir: Path) -> None:
    image_path = _one_input("image", _IMAGE_SUFFIXES)
    mesh_path = _validated_input_glb(
        _one_input("GLB mesh", frozenset({".glb"}))
    )

    import trimesh
    from step1x3d_texture.pipelines.step1x_3d_texture_synthesis_pipeline import (
        Step1X3DTextureConfig,
        Step1X3DTexturePipeline,
    )

    config = Step1X3DTextureConfig()
    config.base_model = "/models/sdxl"
    config.vae_model = "/models/sdxl-vae"
    config.adapter_path = "/models/target/Step1X-3D-Texture"
    pipeline = Step1X3DTexturePipeline(config)
    mesh = trimesh.load(mesh_path)
    # BiRefNet is optional upstream preprocessing.  Disabling it prevents the
    # hidden remote-code download and retains the caller-provided image exactly.
    textured_mesh = pipeline(image_path, mesh, remove_bg=False, seed=seed)
    temporary = output_dir / ".step1x-textured.tmp.glb"
    textured_mesh.export(temporary)
    _publish_glb(temporary, output_dir / "step1x-textured.glb", profile="textured")


def main(mode: str) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--entrypoint", required=True)
    parser.add_argument("--output-mime", required=True)
    parser.add_argument("--timeout-seconds", type=int, default=3600)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    expected_entrypoint = f"/opt/vonk/source/{mode}.py"
    if args.entrypoint != expected_entrypoint:
        raise SystemExit("unexpected pipeline entrypoint")
    if args.output_mime != "model/gltf-binary":
        raise SystemExit("Step1X jobs emit model/gltf-binary only")
    if not 1 <= args.timeout_seconds <= 3600:
        raise SystemExit("timeout must be between 1 and 3600 seconds")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if mode == "geometry":
        _geometry(label_control=False, seed=args.seed, output_dir=args.output_dir)
    elif mode == "label-geometry":
        _geometry(label_control=True, seed=args.seed, output_dir=args.output_dir)
    elif mode == "texture":
        _texture(seed=args.seed, output_dir=args.output_dir)
    else:
        raise SystemExit(f"unsupported VONK_STEP1X_MODE: {mode}")


if __name__ == "__main__":
    raise SystemExit("invoke a signed Step1X mode entrypoint")
