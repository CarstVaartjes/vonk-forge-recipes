from __future__ import annotations

import argparse
from pathlib import Path

_IMAGE_SUFFIXES = frozenset({".jpg", ".jpeg", ".png", ".webp"})


def _image_input(input_dir: Path = Path("/inputs")) -> Path:
    if not input_dir.is_dir():
        raise SystemExit("the read-only /inputs job mount is required")
    input_files = sorted(
        path
        for path in input_dir.iterdir()
        if path.is_file() and path.suffix.lower() in _IMAGE_SUFFIXES
    )
    if len(input_files) != 1:
        raise SystemExit("exactly one image input is required")
    return input_files[0]


def _save_layers(
    layers: list[object], output_dir: Path, expected_count: int
) -> list[Path]:
    values = list(layers)
    if len(values) != expected_count:
        raise SystemExit(
            f"layer decomposition returned {len(values)} layers; expected {expected_count}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []
    for index, layer in enumerate(values):
        destination = output_dir / f"layer-{index:02d}.png"
        layer.save(destination, format="PNG")
        outputs.append(destination)
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pipeline", required=True)
    parser.add_argument("--output-mime", required=True)
    parser.add_argument("--num-inference-steps", type=int, default=50)
    parser.add_argument("--true-cfg-scale", type=float, default=4.0)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--resolution", type=int, default=640)
    parser.add_argument("--cfg-normalize", choices=("true", "false"), default="true")
    parser.add_argument("--seed", type=int, default=777)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    if args.pipeline != "image-to-layers":
        raise SystemExit("this candidate currently supports image-to-layers only")
    if args.output_mime != "image/png":
        raise SystemExit("this candidate currently emits image/png artifacts only")
    if not 1 <= args.layers <= 8:
        raise SystemExit("layers must be between 1 and 8")

    input_file = _image_input()

    import torch
    from diffusers import QwenImageLayeredPipeline
    from PIL import Image

    image = Image.open(input_file).convert("RGBA")
    pipe = QwenImageLayeredPipeline.from_pretrained(
        "/models",
        dtype=torch.bfloat16,
        local_files_only=True,
    ).to("cuda")
    output = pipe(
        image=image,
        generator=torch.Generator(device="cuda").manual_seed(args.seed),
        true_cfg_scale=args.true_cfg_scale,
        negative_prompt=" ",
        num_inference_steps=args.num_inference_steps,
        num_images_per_prompt=1,
        layers=args.layers,
        resolution=args.resolution,
        cfg_normalize=args.cfg_normalize == "true",
        use_en_prompt=True,
    ).images[0]
    _save_layers(output, args.output_dir, args.layers)


if __name__ == "__main__":
    main()
