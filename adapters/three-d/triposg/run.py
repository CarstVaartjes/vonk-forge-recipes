"""Offline, one-image TripoSG adapter with a deterministic GLB result."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import trimesh
from PIL import Image

SOURCE = Path("/opt/triposg")
INPUTS = Path("/inputs")
TARGET = Path("/models/target")
RMBG = Path("/models/rmbg")


def one_input_image() -> Path:
    candidates = sorted(
        path
        for path in INPUTS.iterdir()
        if path.is_file() and not path.is_symlink() and path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
    )
    if len(candidates) != 1:
        raise SystemExit("TripoSG requires exactly one JPEG, PNG, or WebP input")
    with Image.open(candidates[0]) as image:
        image.verify()
    return candidates[0]


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
    if not 1 <= args.timeout_seconds <= 3600:
        raise SystemExit("timeout is outside the harness contract")

    sys.path.insert(0, str(SOURCE))
    sys.path.insert(0, str(SOURCE / "scripts"))
    from briarmbg import BriaRMBG
    from image_process import prepare_image
    from triposg.pipelines.pipeline_triposg import TripoSGPipeline

    image_path = one_input_image()
    rmbg = BriaRMBG.from_pretrained(str(RMBG), local_files_only=True).to("cuda")
    rmbg.eval()
    pipe = TripoSGPipeline.from_pretrained(str(TARGET), local_files_only=True).to(
        "cuda", torch.float16
    )
    image = prepare_image(
        str(image_path), bg_color=np.array([1.0, 1.0, 1.0]), rmbg_net=rmbg
    )
    samples = pipe(
        image=image,
        generator=torch.Generator(device="cuda").manual_seed(args.seed),
        num_inference_steps=50,
        guidance_scale=7.0,
    ).samples[0]
    mesh = trimesh.Trimesh(
        samples[0].astype(np.float32), np.ascontiguousarray(samples[1]), process=True
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / "output.glb"
    mesh.export(output, file_type="glb")
    if not output.is_file() or output.stat().st_size < 20 or output.read_bytes()[:4] != b"glTF":
        raise SystemExit("TripoSG did not produce a valid GLB artifact")


if __name__ == "__main__":
    main()
