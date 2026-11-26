"""Run one MOSS-TTSD dialogue the way the creator's app does and write a PCM16 WAV.

The steps follow ``run_inference`` in MOSS-TTS ``clis/moss_ttsd_app.py`` at
934d6826b084c46a0d033402174d5f8ac4ed2519 with its defaults: text normalization on,
sample-rate normalization off, audio temperature 1.1, top-p 0.9, top-k 50, repetition
penalty 1.1, bf16 weights and SDPA attention (its CUDA fallback without FlashAttention).
Reference clips are read with soundfile instead of ``torchaudio.load``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import soundfile
from dialogue import (
    merge_consecutive_speaker_tags,
    normalize_prompt_text,
    normalize_text,
    validate_dialogue_text,
)

TEMPERATURE = 1.1
TOP_P = 0.9
TOP_K = 50
REPETITION_PENALTY = 1.1


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--codec-dir", required=True, type=Path)
    parser.add_argument("--request", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = _arguments()
    request = json.loads(args.request.read_text(encoding="utf-8"))
    speaker_count = int(request["speakers"])
    references = [Path(path) for path in request["reference_audio"]]
    # An unreadable reference clip fails here, before the model loads.
    clips = [soundfile.read(str(path), dtype="float32", always_2d=True) for path in references]

    import torch
    from transformers import AutoModel, AutoProcessor

    torch.backends.cuda.enable_cudnn_sdp(False)
    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_mem_efficient_sdp(True)
    torch.backends.cuda.enable_math_sdp(True)

    def resample(wav: torch.Tensor, orig_sr: int, target_sr: int) -> torch.Tensor:
        # The creator's _resample_wav.
        if int(orig_sr) == int(target_sr):
            return wav
        new_num_samples = int(round(wav.shape[-1] * float(target_sr) / float(orig_sr)))
        if new_num_samples <= 0:
            raise ValueError(f"Invalid resample length from {orig_sr}Hz to {target_sr}Hz.")
        return torch.nn.functional.interpolate(
            wav.unsqueeze(0), size=new_num_samples, mode="linear", align_corners=False
        ).squeeze(0)

    device = torch.device("cuda:0")
    processor = AutoProcessor.from_pretrained(
        str(args.model_dir), trust_remote_code=True, codec_path=str(args.codec_dir)
    )
    processor.audio_tokenizer = processor.audio_tokenizer.to(device)
    processor.audio_tokenizer.eval()
    model = AutoModel.from_pretrained(
        str(args.model_dir),
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
    ).to(device)
    model.eval()
    sample_rate = int(getattr(processor.model_config, "sampling_rate", 24000))

    dialogue = validate_dialogue_text(normalize_text(request["script"]), speaker_count)
    conversation_text = dialogue
    prompt_audio = None
    reference_codes: list[torch.Tensor | None] = []
    if clips:
        prompts = [
            normalize_prompt_text(text, speaker_id)
            for speaker_id, text in enumerate(request["reference_text"], start=1)
        ]
        conversation_text = validate_dialogue_text(
            normalize_text(merge_consecutive_speaker_tags("".join(prompts) + dialogue)),
            speaker_count,
        )
        wavs = []
        for samples, rate in clips:
            wav = torch.from_numpy(samples.T.copy()).mean(dim=0, keepdim=True)
            if wav.numel() == 0:
                raise SystemExit("a reference clip is empty")
            wavs.append(resample(wav, int(rate), sample_rate))
        encoded = processor.encode_audios_from_wav(wavs, sampling_rate=sample_rate)
        reference_codes = [None] * speaker_count
        for speaker_index, codes in enumerate(encoded):
            reference_codes[speaker_index] = codes
        prompt_audio = processor.encode_audios_from_wav(
            [torch.cat(wavs, dim=-1)], sampling_rate=sample_rate
        )[0]

    if prompt_audio is None:
        conversations = [[processor.build_user_message(text=conversation_text)]]
        mode = "generation"
    else:
        conversations = [
            [
                processor.build_user_message(text=conversation_text, reference=reference_codes),
                processor.build_assistant_message(audio_codes_list=[prompt_audio]),
            ]
        ]
        mode = "continuation"

    batch = processor(conversations, mode=mode)
    with torch.no_grad():
        outputs = model.generate(
            input_ids=batch["input_ids"].to(device),
            attention_mask=batch["attention_mask"].to(device),
            max_new_tokens=int(request["max_new_tokens"]),
            audio_temperature=TEMPERATURE,
            audio_top_p=TOP_P,
            audio_top_k=TOP_K,
            audio_repetition_penalty=REPETITION_PENALTY,
        )
    messages = processor.decode(outputs)
    if not messages or messages[0] is None:
        raise SystemExit("MOSS-TTSD did not return a decodable audio result")
    audio = messages[0].audio_codes_list[0].detach().float().cpu().numpy().reshape(-1)
    if audio.size == 0:
        raise SystemExit("MOSS-TTSD returned no audio")
    soundfile.write(str(args.output), audio, sample_rate, subtype="PCM_16")


if __name__ == "__main__":
    main()
