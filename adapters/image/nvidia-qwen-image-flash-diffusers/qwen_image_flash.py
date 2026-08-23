from __future__ import annotations

import argparse
import os
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pipeline", required=True)
    parser.add_argument("--output-mime", required=True)
    parser.add_argument("--num-inference-steps", type=int, default=4)
    parser.add_argument("--true-cfg-scale", type=float, default=1.0)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    if args.pipeline != "text-to-image":
        raise SystemExit("NVIDIA Qwen Image Flash supports text-to-image only")
    if args.output_mime != "image/png":
        raise SystemExit("NVIDIA Qwen Image Flash emits image/png only")
    if args.num_inference_steps != 4:
        raise SystemExit("NVIDIA Qwen Image Flash requires its packaged four-step schedule")
    if args.width % 16 or args.height % 16:
        raise SystemExit("width and height must be divisible by 16")

    import torch
    from diffusers import QwenImagePipeline

    prompt = os.environ.get(
        "VONK_PROMPT",
        "A red fox in a snowy pine forest at golden hour, photorealistic, sharp focus, soft bokeh",
    )
    pipe = QwenImagePipeline.from_pretrained(
        "/models",
        dtype=torch.bfloat16,
        local_files_only=True,
    ).to("cuda")
    image = pipe(
        prompt=prompt,
        width=args.width,
        height=args.height,
        num_inference_steps=args.num_inference_steps,
        true_cfg_scale=args.true_cfg_scale,
        guidance_scale=None,
        negative_prompt=None,
        generator=torch.Generator(device="cuda").manual_seed(args.seed),
    ).images[0]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    image.save(args.output_dir / "qwen-image-flash.png", format="PNG")


if __name__ == "__main__":
    main()
