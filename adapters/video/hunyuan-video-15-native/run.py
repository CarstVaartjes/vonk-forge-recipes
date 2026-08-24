"""Offline Diffusers adapter for complete HunyuanVideo 1.5 snapshots."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch
from diffusers import HunyuanVideo15ImageToVideoPipeline, HunyuanVideo15Pipeline
from diffusers.utils import export_to_video
from PIL import Image

_IMAGE_SUFFIXES = frozenset({".jpeg", ".jpg", ".png", ".webp"})
_MODEL_INDEX = Path("/models/model_index.json")


def _one_image(path: Path) -> Image.Image:
    candidates = sorted(
        item
        for item in path.iterdir()
        if item.is_file() and item.suffix.lower() in _IMAGE_SUFFIXES
    )
    if len(candidates) != 1:
        raise SystemExit(
            "image-to-video requires exactly one supported file in /inputs"
        )
    with Image.open(candidates[0]) as image:
        return image.convert("RGB")


def _variant(pipeline: str, resolution: int, steps: int, guidance: float) -> str:
    """Map the signed recipe tuple to one exact self-contained model snapshot."""
    if (
        pipeline == "text-to-video"
        and resolution == 720
        and steps == 50
        and guidance == 6.0
    ):
        return "HunyuanVideo15Pipeline"
    if (
        pipeline == "text-to-video"
        and resolution == 480
        and steps == 50
        and guidance == 1.0
    ):
        return "HunyuanVideo15Pipeline"
    if (
        pipeline == "image-to-image"
        and resolution == 480
        and steps in {8, 12}
        and guidance == 1.0
    ):
        return "HunyuanVideo15ImageToVideoPipeline"
    raise SystemExit("unsupported HunyuanVideo 1.5 recipe variant")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pipeline", required=True, choices=("text-to-video", "image-to-image")
    )
    parser.add_argument("--output-mime", required=True)
    parser.add_argument("--num-inference-steps", required=True, type=int)
    parser.add_argument("--guidance-scale", required=True, type=float)
    parser.add_argument("--resolution", required=True, type=int, choices=(480, 720))
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    if args.output_mime != "video/mp4":
        raise SystemExit("HunyuanVideo 1.5 only emits video/mp4")
    class_name = _variant(
        args.pipeline, args.resolution, args.num_inference_steps, args.guidance_scale
    )
    if not _MODEL_INDEX.is_file():
        raise SystemExit(
            "complete pinned Diffusers snapshot is missing model_index.json"
        )

    pipeline_class = (
        HunyuanVideo15ImageToVideoPipeline
        if class_name == "HunyuanVideo15ImageToVideoPipeline"
        else HunyuanVideo15Pipeline
    )
    pipe = pipeline_class.from_pretrained(
        "/models",
        torch_dtype=torch.bfloat16,
        local_files_only=True,
    )
    if pipe.__class__.__name__ != class_name:
        raise SystemExit(
            f"snapshot resolved unexpected pipeline class {pipe.__class__.__name__}"
        )
    if float(pipe.guider.guidance_scale) != args.guidance_scale:
        raise SystemExit("snapshot guider does not match the signed recipe")
    pipe.enable_model_cpu_offload()
    pipe.vae.enable_tiling()

    prompt = os.environ.get(
        "VONK_PROMPT", "A red fox walking through a quiet alpine meadow"
    )
    call = {
        "prompt": prompt,
        "negative_prompt": os.environ.get("VONK_NEGATIVE_PROMPT", ""),
        "generator": torch.Generator(device="cuda:0").manual_seed(args.seed),
        "num_frames": 121,
        "num_inference_steps": args.num_inference_steps,
    }
    if args.pipeline == "image-to-image":
        call["image"] = _one_image(Path("/inputs"))

    frames = pipe(**call).frames[0]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / "output.mp4"
    export_to_video(frames, output, fps=24)
    if not output.is_file() or output.stat().st_size == 0:
        raise SystemExit("Diffusers did not produce output.mp4")


if __name__ == "__main__":
    main()
