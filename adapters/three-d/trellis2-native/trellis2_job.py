"""Offline single-image TRELLIS.2 adapter that emits one GLB artifact."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
from pathlib import Path


SUPPORTED_SUFFIXES = {".jpeg", ".jpg", ".png", ".webp"}


def one_input(input_dir: Path) -> Path:
    candidates = sorted(
        path for path in input_dir.iterdir() if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES
    )
    if len(candidates) != 1:
        raise SystemExit(f"expected exactly one JPEG, PNG, or WebP input; found {len(candidates)}")
    return candidates[0]


def local_model_root(target: Path, decoder_config: Path, decoder_weights: Path) -> Path:
    root = Path(tempfile.mkdtemp(prefix="vonk-trellis-model-"))
    for item in target.iterdir():
        if item.name != "pipeline.json":
            (root / item.name).symlink_to(item, target_is_directory=item.is_dir())
    decoder = root / "companion" / "ss_dec_conv3d_16l8_fp16"
    decoder.parent.mkdir(parents=True)
    decoder.with_suffix(".json").symlink_to(decoder_config / "ss_dec_conv3d_16l8_fp16.json")
    decoder.with_suffix(".safetensors").symlink_to(
        decoder_weights / "ss_dec_conv3d_16l8_fp16.safetensors"
    )
    config = json.loads((target / "pipeline.json").read_text())
    config["args"]["models"]["sparse_structure_decoder"] = str(
        decoder
    )
    config["args"]["image_cond_model"]["args"]["model_name"] = "/models/dino"
    (root / "pipeline.json").write_text(json.dumps(config))
    return root


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--entrypoint", default="/opt/vonk/source/trellis2_job.py")
    parser.add_argument("--output-mime", required=True)
    parser.add_argument("--input-dir", type=Path, default=Path("/inputs"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--pipeline-resolution", choices=("512", "1024", "1024_cascade"), default="1024_cascade")
    parser.add_argument("--decimation-target", type=int, default=1_000_000)
    parser.add_argument("--texture-size", type=int, choices=(1024, 2048, 4096), default=2048)
    parser.add_argument("--timeout-seconds", type=int, default=3600)
    args = parser.parse_args()
    if args.entrypoint != "/opt/vonk/source/trellis2_job.py":
        raise SystemExit("unexpected adapter entrypoint")
    if args.output_mime != "model/gltf-binary":
        raise SystemExit("TRELLIS.2 adapter only emits model/gltf-binary")
    if not 1 <= args.decimation_target <= 2_000_000:
        raise SystemExit("decimation target must be between 1 and 2,000,000")

    os.environ.setdefault("ATTN_BACKEND", "sdpa")
    os.environ.setdefault("SPARSE_ATTN_BACKEND", "sdpa")
    os.environ.setdefault("SPARSE_CONV_BACKEND", "flex_gemm")

    from PIL import Image, ImageOps
    import o_voxel
    from trellis2.pipelines import Trellis2ImageTo3DPipeline
    from trellis2.pipelines import rembg

    class NoBackgroundModel:
        def __init__(self, **_: object) -> None:
            pass

        def to(self, _: object) -> "NoBackgroundModel":
            return self

        cuda = to
        cpu = to

    rembg.BiRefNet = NoBackgroundModel
    model_root = local_model_root(
        Path("/models/target"),
        Path("/models/sparse-decoder-config"),
        Path("/models/sparse-decoder-weights"),
    )
    try:
        pipeline = Trellis2ImageTo3DPipeline.from_pretrained(str(model_root))
        pipeline.cuda()
        with Image.open(one_input(args.input_dir)) as opened:
            image = ImageOps.exif_transpose(opened).convert("RGB")
        mesh = pipeline.run(
            image,
            seed=args.seed,
            preprocess_image=False,
            pipeline_type=args.pipeline_resolution,
        )[0]
        mesh.simplify(16_777_216)
        glb = o_voxel.postprocess.to_glb(
            vertices=mesh.vertices,
            faces=mesh.faces,
            attr_volume=mesh.attrs,
            coords=mesh.coords,
            attr_layout=mesh.layout,
            voxel_size=mesh.voxel_size,
            aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
            decimation_target=args.decimation_target,
            texture_size=args.texture_size,
            remesh=True,
            remesh_band=1,
            remesh_project=0,
            verbose=True,
        )
        args.output_dir.mkdir(parents=True, exist_ok=True)
        temporary = args.output_dir / ".output.tmp.glb"
        glb.export(temporary, extension_webp=True)
        os.replace(temporary, args.output_dir / "output.glb")
    finally:
        shutil.rmtree(model_root, ignore_errors=True)


if __name__ == "__main__":
    main()
