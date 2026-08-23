"""Small offline-first adapter for standard Diffusers-compatible pipelines.

Model-specific upstream pipelines can replace this context without changing
the recipe contract. The generic path is intentionally conservative: it loads
the frozen local snapshot, emits one bounded artifact, and never downloads at
runtime.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--entrypoint", default="/opt/vonk/source/run.py")
    parser.add_argument("--output-mime", required=True)
    parser.add_argument("--pipeline", default="text-to-image")
    parser.add_argument("--num-inference-steps", type=int, default=4)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--frames", type=int, default=17)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.entrypoint != "/opt/vonk/source/run.py":
        raise SystemExit("unexpected pipeline entrypoint")
    if args.output_mime not in {"image/png", "video/mp4", "audio/wav", "model/gltf-binary"}:
        raise SystemExit("unsupported output MIME")
    if args.num_inference_steps < 1 or args.num_inference_steps > 100:
        raise SystemExit("inference step bound is invalid")

    import torch
    from diffusers import DiffusionPipeline

    pipe = DiffusionPipeline.from_pretrained(
        "/models",
        dtype=torch.bfloat16,
        local_files_only=True,
        trust_remote_code=False,
    )
    pipe = pipe.to("cuda")
    prompt = os.environ.get("VONK_PROMPT", "A small red fox in a quiet alpine landscape")
    generator = torch.Generator(device="cuda").manual_seed(args.seed)
    kwargs = {
        "prompt": prompt,
        "num_inference_steps": args.num_inference_steps,
        "generator": generator,
    }
    if args.pipeline == "text-to-image":
        kwargs.update(width=args.width, height=args.height)
    output = pipe(**kwargs)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.output_mime == "image/png":
        output.images[0].save(args.output_dir / "output.png", format="PNG")
    else:
        raise SystemExit(
            "this generic adapter requires a model-specific upstream pipeline for non-image output"
        )


if __name__ == "__main__":
    main()
