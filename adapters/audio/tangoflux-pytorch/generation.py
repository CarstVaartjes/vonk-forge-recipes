"""TangoFlux generation steps that need no torch import: the creator's ``generate()`` and its
load-time key check, over the model and VAE objects the worker passes in."""

from __future__ import annotations

# The creator's TangoFluxInference comment: "_IncompatibleKeys(missing_keys=
# ['text_encoder.encoder.embed_tokens.weight'], unexpected_keys=[]) this behaviour is expected".
# load_state_dict(strict=False) would otherwise hide key drift across diffusers versions.
EXPECTED_MISSING_KEYS = ["text_encoder.encoder.embed_tokens.weight"]
SAMPLE_RATE = 44100
# audio_seq_len 645 latents of 2,048 samples each: the longest clip the model can write.
MAX_FRAMES = 645 * 2048


def check_load_result(result: object) -> None:
    """Refuse a checkpoint whose keys differ from the creator's expectation."""
    missing = list(getattr(result, "missing_keys"))
    unexpected = list(getattr(result, "unexpected_keys"))
    if missing != EXPECTED_MISSING_KEYS or unexpected:
        raise SystemExit(
            f"TangoFlux checkpoint keys differ from the creator's: missing {missing}, "
            f"unexpected {unexpected}"
        )


def output_frames(duration: float, sample_rate: int = SAMPLE_RATE) -> int:
    """Samples of a ``duration`` clip: the creator's trim, capped at what the model writes."""
    return min(int(duration * sample_rate), MAX_FRAMES)


def generate_wave(model, vae, prompt, *, duration, steps, seed, guidance_scale):
    """The creator's ``generate()`` (run under ``torch.no_grad``), with the seed it hard-wires to 0."""
    latents = model.inference_flow(
        prompt,
        duration=duration,
        num_inference_steps=steps,
        guidance_scale=guidance_scale,
        seed=seed,
    )
    wave = vae.decode(latents.transpose(2, 1)).sample.cpu()[0]
    return wave[:, : int(duration * vae.config.sampling_rate)]
