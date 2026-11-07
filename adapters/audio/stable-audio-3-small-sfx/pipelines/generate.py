"""Generate one sound effect with the creator's StableAudioModel and write PCM16 WAV."""

from __future__ import annotations

import argparse
from pathlib import Path

import soundfile
from stable_audio_3 import StableAudioModel


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt-file", required=True, type=Path)
    parser.add_argument("--duration-seconds", required=True, type=float)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    prompt = args.prompt_file.read_text(encoding="utf-8").strip()
    # The creator's defaults for small-sfx: float16 on CUDA, steps=8, cfg_scale=1.0,
    # and the pingpong sampler its rf_denoiser objective selects (stable_audio_3/model.py,
    # inference/sampling.py); only the prompt, duration and the config's sample_size
    # (as the creator's CLI passes it) are ours.
    model = StableAudioModel.from_pretrained("small-sfx")
    audio = model.generate(
        prompt=prompt,
        duration=args.duration_seconds,
        sample_size=model.model_config["sample_size"],
    )
    # generate() returns float32 in [-1, 1] shaped (batch, channels, samples).
    soundfile.write(
        str(args.output),
        audio[0].T.cpu().numpy(),
        model.model.sample_rate,
        subtype="PCM_16",
        format="WAV",
    )


if __name__ == "__main__":
    main()
