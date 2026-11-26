"""The creator's PartCrafter steps (``scripts/inference_partcrafter.py``) on the mounted weights.

run.py starts this in its own process group with the creator's source on ``PYTHONPATH``. It does
what the script's ``--rmbg`` path does: RMBG-1.4 prepares the image (a valid alpha channel is used as
it is, any other image has its background removed, composited on white, cropped and padded), the
pipeline samples every part with the script's defaults, and each part is written as ``part_NN.glb``
beside ``object.glb``, the composed scene with one node per part and seeded per-part vertex colours.
The creator's Hugging Face downloads, renders, manifest and Gemini helpers are not used.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from accelerate.utils import set_seed

from src.models.briarmbg import BriaRMBG
from src.pipelines.pipeline_partcrafter import PartCrafterPipeline
from src.utils.data_utils import get_colored_mesh_composition
from src.utils.image_utils import prepare_image

# The script's own argparse defaults (the pipeline's call defaults differ).
NUM_TOKENS = 1024
NUM_INFERENCE_STEPS = 50
GUIDANCE_SCALE = 7.0
MAX_NUM_EXPANDED_COORDS = 1e9
DEVICE = "cuda"


@torch.no_grad()
def generate(pipe, image, num_parts, seed):
    return pipe(
        image=[image] * num_parts,
        attention_kwargs={"num_parts": num_parts},
        num_tokens=NUM_TOKENS,
        generator=torch.Generator(device=pipe.device).manual_seed(seed),
        num_inference_steps=NUM_INFERENCE_STEPS,
        guidance_scale=GUIDANCE_SCALE,
        max_num_expanded_coords=MAX_NUM_EXPANDED_COORDS,
        use_flash_decoder=False,
    ).meshes


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--num-parts", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--rmbg", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    rmbg_net = BriaRMBG.from_pretrained(str(args.rmbg), local_files_only=True)
    rmbg_net = rmbg_net.to(DEVICE).eval()
    pipe = PartCrafterPipeline.from_pretrained(
        str(args.weights), local_files_only=True
    ).to(DEVICE, torch.float16)
    set_seed(args.seed)
    image = prepare_image(
        str(args.image), bg_color=np.array([1.0, 1.0, 1.0]), rmbg_net=rmbg_net
    )
    meshes = generate(pipe, image, args.num_parts, args.seed)
    # The creator swaps a part that failed to decode for a one-vertex dummy mesh. That is no
    # artifact, so the job fails and names the part.
    missing = [index for index, mesh in enumerate(meshes) if mesh is None]
    if missing:
        raise SystemExit(f"PartCrafter could not decode part(s) {missing}")
    for index, mesh in enumerate(meshes):
        mesh.export(args.output_dir / f"part_{index:02}.glb")
    get_colored_mesh_composition(meshes).export(args.output_dir / "object.glb")


if __name__ == "__main__":
    main()
