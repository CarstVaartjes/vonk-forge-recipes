from __future__ import annotations

import argparse
import os
from pathlib import Path

_IMAGE_SUFFIXES = frozenset({".jpg", ".jpeg", ".png", ".webp"})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pipeline", required=True)
    parser.add_argument("--output-mime", required=True)
    parser.add_argument("--num-inference-steps", type=int, default=40)
    parser.add_argument("--true-cfg-scale", type=float, default=4.0)
    parser.add_argument("--guidance-scale", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    if args.pipeline != "image-to-image":
        raise SystemExit("this candidate currently supports image-to-image only")
    if args.output_mime != "image/png":
        raise SystemExit("this candidate currently emits image/png only")

    input_files = sorted(
        path for path in Path("/inputs").iterdir() if path.suffix.lower() in _IMAGE_SUFFIXES
    )
    if not input_files:
        raise SystemExit("at least one image input is required")
    if len(input_files) > 2:
        raise SystemExit("this candidate accepts at most two image inputs")

    import torch
    from diffusers import QwenImageEditPlusPipeline
    from PIL import Image

    prompt = os.environ.get(
        "VONK_PROMPT",
        "Preserve the subject and composition while improving the lighting and color grading",
    )
    images = [Image.open(path).convert("RGB") for path in input_files]
    pipe = QwenImageEditPlusPipeline.from_pretrained(
        "/models",
        dtype=torch.bfloat16,
        local_files_only=True,
    ).to("cuda")
    output = pipe(
        image=images,
        prompt=prompt,
        generator=torch.Generator(device="cuda").manual_seed(args.seed),
        true_cfg_scale=args.true_cfg_scale,
        negative_prompt=" ",
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        num_images_per_prompt=1,
    ).images[0]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output.save(args.output_dir / "qwen-image-edit.png", format="PNG")


if __name__ == "__main__":
    main()
