from __future__ import annotations

import argparse
from pathlib import Path

_INPUT_DIR = Path("/inputs")
_IMAGE_SUFFIXES = frozenset({".jpg", ".jpeg", ".png", ".webp"})
_PROMPT_SUFFIXES = frozenset({".txt", ".text"})
_MAX_PROMPT_BYTES = 16 * 1024


def _input_files(suffixes: frozenset[str]) -> list[Path]:
    if not _INPUT_DIR.is_dir():
        raise SystemExit("the read-only /inputs job mount is required")
    return sorted(
        path
        for path in _INPUT_DIR.iterdir()
        if path.is_file() and path.suffix.lower() in suffixes
    )


def _prompt() -> str:
    prompt_files = _input_files(_PROMPT_SUFFIXES)
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


def _image_inputs() -> list[Path]:
    input_files = _input_files(_IMAGE_SUFFIXES)
    if not 1 <= len(input_files) <= 2:
        raise SystemExit("image editing requires one or two image inputs")
    return input_files


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

    prompt = _prompt()
    input_files = _image_inputs()

    import torch
    from diffusers import QwenImageEditPlusPipeline
    from PIL import Image

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
