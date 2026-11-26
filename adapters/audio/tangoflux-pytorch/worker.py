"""The creator's TangoFlux steps (``tangoflux/__init__.py``) on the mounted weights.

run.py starts this in its own process group with the creator's package on ``PYTHONPATH``. It builds
what ``TangoFluxInference.__init__`` builds, but reads the VAE, the model and the FLAN-T5 text encoder
from the mounted directories instead of ``snapshot_download``, and requires the creator's expected
key mismatch exactly (``generation.check_load_result``). It then runs the creator's generation at
the requested seed and writes the stereo 44.1 kHz clip as a 16-bit PCM WAV.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import soundfile
import torch
from diffusers import AutoencoderOobleck
from safetensors.torch import load_file

from generation import SAMPLE_RATE, check_load_result, generate_wave
from tangoflux.model import TangoFlux

DEVICE = "cuda"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt-file", type=Path, required=True)
    parser.add_argument("--duration-seconds", type=float, required=True)
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--guidance-scale", type=float, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--text-encoder", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    vae = AutoencoderOobleck()
    vae.load_state_dict(load_file(str(args.weights / "vae.safetensors")))
    weights = load_file(str(args.weights / "tangoflux.safetensors"))
    config = json.loads((args.weights / "config.json").read_text(encoding="utf-8"))
    model = TangoFlux(config, text_encoder_dir=str(args.text_encoder))
    check_load_result(model.load_state_dict(weights, strict=False))
    vae.to(DEVICE)
    model.to(DEVICE)

    prompt = args.prompt_file.read_text(encoding="utf-8")
    with torch.no_grad():
        wave = generate_wave(
            model,
            vae,
            prompt,
            duration=args.duration_seconds,
            steps=args.steps,
            seed=args.seed,
            guidance_scale=args.guidance_scale,
        )
    if int(vae.config.sampling_rate) != SAMPLE_RATE:
        raise SystemExit(f"unexpected VAE sampling rate {vae.config.sampling_rate}")
    samples = np.clip(wave.float().numpy().T, -1.0, 1.0)
    soundfile.write(str(args.output), samples, SAMPLE_RATE, subtype="PCM_16")


if __name__ == "__main__":
    main()
