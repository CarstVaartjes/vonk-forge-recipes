"""Run the creator's CHORD model on one image, as its test.py does, and save the four maps."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from omegaconf import OmegaConf
from torchvision.transforms import v2

from chord import ChordModel
from chord.io import load_torch_file, read_image, save_maps

_CONFIG = "/opt/chord/config/chord.yaml"
_CHECKPOINT = "/models/chord/chord_v1.safetensors"
# Only the configs and tokenizer of Stable Diffusion 2.1 Base come from here; the
# checkpoint holds every weight. A hf_key makes the creator's code load locally.
_STABLE_DIFFUSION = "/models/stable-diffusion-2-1-base"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CHORD needs a CUDA device")
    device = torch.device("cuda")

    config = OmegaConf.load(_CONFIG)
    config.model.stable_diffusion.hf_key = _STABLE_DIFFUSION
    model = ChordModel(config)
    model.load_state_dict(load_torch_file(_CHECKPOINT))
    model.eval()
    model.to(device)

    image = read_image(str(args.image)).to(device)
    original_height, original_width = image.shape[-2:]
    x = v2.Resize(size=(1024, 1024), antialias=True)(image).unsqueeze(0)
    with torch.no_grad(), torch.autocast(device_type="cuda"):
        output = model(x)
    for key in output:
        output[key] = v2.Resize(size=(original_height, original_width), antialias=True)(
            output[key]
        )
    save_maps(str(args.output_dir), output)


if __name__ == "__main__":
    main()
