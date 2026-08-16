from __future__ import annotations

import argparse
import os
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pipeline", required=True)
    parser.add_argument("--output-mime", required=True)
    parser.add_argument("--num-inference-steps", type=int, default=50)
    parser.add_argument("--guidance-scale", type=float, default=4.0)
    parser.add_argument("--width", type=int, default=1328)
    parser.add_argument("--height", type=int, default=1328)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    if args.pipeline != "text-to-image":
        raise SystemExit("this candidate currently supports text-to-image only")
    if args.output_mime != "image/png":
        raise SystemExit("this candidate currently emits image/png only")

    import torch
    from diffusers import QwenImagePipeline

    prompt = os.environ.get(
        "VONK_PROMPT",
        "A studio photograph of a small red fox reading a book beside a window",
    )
    generator = torch.Generator(device="cuda").manual_seed(args.seed)
    pipe = QwenImagePipeline.from_pretrained(
        "/models",
        torch_dtype=torch.bfloat16,
        local_files_only=True,
    ).to("cuda")
    image = pipe(
        prompt=prompt,
        width=args.width,
        height=args.height,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        generator=generator,
    ).images[0]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    image.save(args.output_dir / "qwen-image.png", format="PNG")


if __name__ == "__main__":
    main()
