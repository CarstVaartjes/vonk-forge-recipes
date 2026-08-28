from __future__ import annotations

import argparse
from pathlib import Path

_INPUT_DIR = Path("/inputs")
_PROMPT_SUFFIXES = frozenset({".txt", ".text"})
_MAX_PROMPT_BYTES = 16 * 1024


def _prompt() -> str:
    if not _INPUT_DIR.is_dir():
        raise SystemExit("the read-only /inputs job mount is required")
    prompt_files = sorted(
        path
        for path in _INPUT_DIR.iterdir()
        if path.is_file() and path.suffix.lower() in _PROMPT_SUFFIXES
    )
    if len(prompt_files) != 1:
        raise SystemExit(
            "exactly one UTF-8 text prompt input (.txt or .text) is required"
        )
    raw = prompt_files[0].read_bytes()
    if not raw or len(raw) > _MAX_PROMPT_BYTES:
        raise SystemExit("the text prompt must contain 1..16384 UTF-8 bytes")
    try:
        prompt = raw.decode("utf-8").strip()
    except UnicodeDecodeError as error:
        raise SystemExit("the text prompt must be valid UTF-8") from error
    if not prompt:
        raise SystemExit("the text prompt must not be blank")
    return prompt


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
        raise SystemExit(
            "NVIDIA Qwen Image Flash requires its packaged four-step schedule"
        )
    if args.width % 16 or args.height % 16:
        raise SystemExit("width and height must be divisible by 16")

    import torch
    from diffusers import QwenImagePipeline

    prompt = _prompt()
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
