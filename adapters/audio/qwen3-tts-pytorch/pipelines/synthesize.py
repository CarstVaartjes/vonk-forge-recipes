"""Run one Qwen3-TTS generation with the official qwen-tts API and write a PCM16 WAV."""

from __future__ import annotations

import argparse
from pathlib import Path

import soundfile
from speech_mode import Mode


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", required=True, choices=[mode.value for mode in Mode])
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--language", required=True)
    parser.add_argument("--text", required=True)
    parser.add_argument("--voice")
    parser.add_argument("--reference-audio", type=Path)
    parser.add_argument("--reference-text")
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = _arguments()
    mode = Mode(args.mode)
    if mode is not Mode.DESIGN:
        # An unreadable reference clip fails here, before the model loads.
        soundfile.info(args.reference_audio)

    import torch
    from qwen_tts import Qwen3TTSModel

    model = Qwen3TTSModel.from_pretrained(
        args.model_dir,
        device_map="cuda:0",
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
    )
    if mode is Mode.DESIGN:
        waves, sample_rate = model.generate_voice_design(
            text=args.text, language=args.language, instruct=args.voice
        )
    else:
        # Full text in the prefill (non-streaming input) keeps one voice throughout.
        waves, sample_rate = model.generate_voice_clone(
            text=args.text,
            language=args.language,
            ref_audio=str(args.reference_audio),
            ref_text=args.reference_text,
            x_vector_only_mode=mode is Mode.CLONE_SPEAKER_ONLY,
            non_streaming_mode=True,
        )
    if len(waves) != 1 or waves[0].size == 0:
        raise SystemExit("Qwen3-TTS returned no audio")
    soundfile.write(str(args.output), waves[0], sample_rate, subtype="PCM_16")


if __name__ == "__main__":
    main()
