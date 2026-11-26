"""The creator's MMAudio ``demo.py`` steps for ``large_44k_v2``, run once for one job.

``run.py`` starts this module in its own process group with the scratch directory as the working
directory and ``PYTHONPATH`` set to the creator's source tree. It follows ``demo.py`` without its
``download_if_needed`` (every checkpoint is a recipe mount): bf16 weights, 25 Euler steps through the
creator's flow matching, the creator's ``load_video`` and ``generate``. The audio is mono at the
creator's 44.1 kHz and is not trimmed, as the creator does; it is written as 16-bit PCM WAV, the
job's audio format, and the creator's video mux is left out.
"""

from __future__ import annotations

import argparse
import wave
from pathlib import Path

import numpy as np
import torch
from mmaudio.eval_utils import generate, load_video
from mmaudio.model.flow_matching import FlowMatching
from mmaudio.model.networks import get_my_mmaudio
from mmaudio.model.sequence_config import CONFIG_44K
from mmaudio.model.utils.features_utils import FeaturesUtils

VARIANT = "large_44k_v2"
NETWORK = "/models/target/weights/mmaudio_large_44k_v2.pth"
VAE = "/models/target/ext_weights/v1-44.pth"
SYNCHFORMER = "/models/target/ext_weights/synchformer_state_dict.pth"
DEVICE = "cuda"
SAMPLE_WIDTH_BYTES = 2


def write_wav(path: Path, samples: np.ndarray, sampling_rate: int) -> None:
    """Mono 16-bit PCM; samples outside [-1, 1] are clipped, not wrapped."""
    if not np.isfinite(samples).all():
        raise SystemExit("MMAudio produced a non-finite sample")
    pcm = np.round(np.clip(samples, -1.0, 1.0) * 32767.0).astype("<i2")
    with wave.open(str(path), "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(SAMPLE_WIDTH_BYTES)
        stream.setframerate(sampling_rate)
        stream.writeframes(pcm.tobytes())


def synthesize(
    *,
    prompt: str,
    negative_prompt: str,
    video: Path | None,
    duration: float,
    seed: int,
    cfg_strength: float,
    num_steps: int,
) -> np.ndarray:
    # demo.py lines 14-15; bf16 is its default without --full_precision.
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    dtype = torch.bfloat16
    with torch.inference_mode():
        net = get_my_mmaudio(VARIANT).to(DEVICE, dtype).eval()
        net.load_weights(torch.load(NETWORK, map_location=DEVICE, weights_only=True))
        rng = torch.Generator(device=DEVICE)
        rng.manual_seed(seed)
        fm = FlowMatching(min_sigma=0, inference_mode="euler", num_steps=num_steps)
        feature_utils = FeaturesUtils(
            tod_vae_ckpt=VAE,
            synchformer_ckpt=SYNCHFORMER,
            enable_conditions=True,
            mode="44k",
            bigvgan_vocoder_ckpt=None,
            need_vae_encoder=False,
        )
        feature_utils = feature_utils.to(DEVICE, dtype).eval()

        if video is not None:
            # Only the audio is produced, so the frames the creator keeps for its video mux are not read.
            video_info = load_video(video, duration, load_all_frames=False)
            clip_frames = video_info.clip_frames.unsqueeze(0)
            sync_frames = video_info.sync_frames.unsqueeze(0)
            duration = video_info.duration_sec
        else:
            clip_frames = sync_frames = None

        CONFIG_44K.duration = duration
        net.update_seq_lengths(
            CONFIG_44K.latent_seq_len, CONFIG_44K.clip_seq_len, CONFIG_44K.sync_seq_len
        )
        audios = generate(
            clip_frames,
            sync_frames,
            [prompt],
            negative_text=[negative_prompt],
            feature_utils=feature_utils,
            net=net,
            fm=fm,
            rng=rng,
            cfg_strength=cfg_strength,
        )
        return audios.float().cpu()[0].numpy()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt-file", type=Path, required=True)
    parser.add_argument("--negative-prompt-file", type=Path)
    parser.add_argument("--video", type=Path)
    parser.add_argument("--duration-seconds", type=float, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--cfg-strength", type=float, required=True)
    parser.add_argument("--num-steps", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    negative = args.negative_prompt_file
    samples = synthesize(
        prompt=args.prompt_file.read_text(encoding="utf-8"),
        negative_prompt=negative.read_text(encoding="utf-8") if negative else "",
        video=args.video,
        duration=args.duration_seconds,
        seed=args.seed,
        cfg_strength=args.cfg_strength,
        num_steps=args.num_steps,
    )
    write_wav(args.output, samples, CONFIG_44K.sampling_rate)


if __name__ == "__main__":
    main()
