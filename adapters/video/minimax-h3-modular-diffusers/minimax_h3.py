from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from pathlib import Path

MODEL_ROOT = Path("/models")
INPUT_ROOT = Path("/inputs")
MAX_PER_KIND = {"image": 9, "video": 3, "audio": 3}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pipeline", required=True)
    parser.add_argument("--output-mime", required=True)
    parser.add_argument("--num-frames", type=int, required=True)
    parser.add_argument("--num-inference-steps", type=int, required=True)
    parser.add_argument("--width", type=int, required=True)
    parser.add_argument("--height", type=int, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def _slot_files(slot: str) -> list[tuple[str, Path]]:
    """(media type, path) of each file in ``slot``, in file name order."""
    path = INPUT_ROOT / "manifest.json"
    try:
        files = json.loads(path.read_text(encoding="utf-8"))["files"]
        found = sorted(
            (item["name"], item["media_type"]) for item in files if item["slot"] == slot
        )
    except (UnicodeDecodeError, json.JSONDecodeError, KeyError, TypeError) as error:
        raise ValueError("the input manifest must list each file's slot") from error
    return [(media_type, INPUT_ROOT / name) for name, media_type in found]


def _keyframe(slot: str) -> Path | None:
    files = _slot_files(slot)
    return files[0][1] if files else None


def _load_prompt() -> str:
    if not INPUT_ROOT.is_dir() or INPUT_ROOT.is_symlink():
        raise ValueError("/inputs must contain one UTF-8 .txt prompt")
    prompt_files = [
        path for path in INPUT_ROOT.iterdir() if path.suffix.lower() == ".txt"
    ]
    if len(prompt_files) != 1:
        raise ValueError("exactly one UTF-8 .txt prompt file is required")
    path = prompt_files[0]
    if (
        path.is_symlink()
        or not path.is_file()
        or not 1 <= path.stat().st_size <= 16 * 1024
    ):
        raise ValueError("prompt file must contain 1..16384 UTF-8 bytes")
    try:
        prompt = path.read_text(encoding="utf-8").strip()
    except UnicodeDecodeError as error:
        raise ValueError("prompt file must contain valid UTF-8") from error
    if not 1 <= len(prompt) <= 4096 or "\x00" in prompt:
        raise ValueError("prompt must contain 1..4096 non-NUL characters")
    return prompt


def _load_references() -> list[object] | None:
    """The ``references`` slot in file name order, read in that order by the model."""
    files = _slot_files("references")
    if not files:
        return None
    from diffusers.modular_pipelines.minimax_h3 import (
        MiniMaxH3AudioReference,
        MiniMaxH3ImageReference,
        MiniMaxH3VideoReference,
    )

    constructors = {
        "image": MiniMaxH3ImageReference.from_file,
        "video": MiniMaxH3VideoReference.from_file,
        "audio": MiniMaxH3AudioReference.from_file,
    }
    clean = [(media_type.split("/", 1)[0], path) for media_type, path in files]
    counts = Counter(kind for kind, _ in clean)
    if any(counts[kind] > limit for kind, limit in MAX_PER_KIND.items()):
        raise ValueError("references exceed MiniMax H3 per-modality limits")
    if counts["audio"] == len(clean):
        raise ValueError(
            "audio references require at least one image or video reference"
        )
    return [constructors[kind](str(path)) for kind, path in clean]


def _load_pipeline(workflow: str):
    import torch
    from diffusers import MiniMaxH3Transformer3DModel, ModularPipeline, TorchAoConfig
    from diffusers.hooks import apply_group_offloading
    from torchao.quantization import Int8WeightOnlyConfig
    from transformers import Qwen3VLForConditionalGeneration
    from transformers import TorchAoConfig as TransformersTorchAoConfig

    pipe = ModularPipeline.from_pretrained(
        str(MODEL_ROOT),
        workflow=workflow,
        local_files_only=True,
    )
    transformer_name = "transformer_ref" if workflow == "ref2va" else "transformer"
    transformer = MiniMaxH3Transformer3DModel.from_pretrained(
        str(MODEL_ROOT),
        subfolder=transformer_name,
        dtype=torch.bfloat16,
        local_files_only=True,
        quantization_config=TorchAoConfig(
            Int8WeightOnlyConfig(version=2),
            modules_to_not_convert=[
                "proj_in",
                "audio_proj_in",
                "context_embedder",
                "time_embedder",
                "time_proj",
                "token_refiner",
                "norm_out",
                "proj_out",
                "audio_proj_out",
            ],
        ),
    )
    pipe.update_components(
        **{transformer_name: transformer},
        text_encoder=Qwen3VLForConditionalGeneration.from_pretrained(
            str(MODEL_ROOT),
            subfolder="text_encoder",
            dtype=torch.bfloat16,
            local_files_only=True,
            quantization_config=TransformersTorchAoConfig(
                Int8WeightOnlyConfig(version=2),
                modules_to_not_convert=[
                    "model.visual",
                    "model.language_model.embed_tokens",
                    "model.language_model.norm",
                    "lm_head",
                ],
            ),
        ),
    )
    pipe.load_components(
        pretrained_model_name_or_path=str(MODEL_ROOT),
        dtype=torch.bfloat16,
        local_files_only=True,
    )

    transformer = getattr(pipe, transformer_name)
    transformer.requires_grad_(False)
    pipe.text_encoder.requires_grad_(False)
    streamed = {
        "onload_device": torch.device("cuda"),
        "offload_device": torch.device("cpu"),
        "use_stream": True,
    }
    transformer.enable_group_offload(
        offload_type="block_level",
        num_blocks_per_group=1,
        **streamed,
    )
    apply_group_offloading(
        pipe.text_encoder.model,
        offload_type="leaf_level",
        **streamed,
    )
    pipe.vae.to("cuda")
    pipe.audio_vae.to("cuda")
    return pipe


def _verify_joint_av(path: Path, *, width: int, height: int, frame_count: int) -> None:
    import av

    with av.open(str(path), mode="r") as container:
        video_streams = list(container.streams.video)
        audio_streams = list(container.streams.audio)
        if len(video_streams) != 1 or len(audio_streams) != 1:
            raise RuntimeError(
                "MiniMax H3 output must contain one video and one audio stream"
            )
        video = video_streams[0]
        audio = audio_streams[0].codec_context
        if video.codec_context.name != "h264" or audio.name != "aac":
            raise RuntimeError(
                "MiniMax H3 output must contain H.264 video and AAC audio"
            )
        if video.width != width or video.height != height:
            raise RuntimeError("MiniMax H3 output dimensions do not match the job")
        if (
            audio.sample_rate != 32000
            or audio.layout is None
            or audio.layout.name != "stereo"
        ):
            raise RuntimeError("MiniMax H3 output must contain 32 kHz stereo audio")
        if video.average_rate is None or float(video.average_rate) != 24.0:
            raise RuntimeError("MiniMax H3 output must contain 24 fps video")

    with av.open(str(path), mode="r") as container:
        decoded_frames = sum(1 for _ in container.decode(video=0))
    if decoded_frames != frame_count:
        raise RuntimeError("MiniMax H3 output frame count changed")

    audio_samples = 0
    with av.open(str(path), mode="r") as container:
        for frame in container.decode(audio=0):
            audio_samples += frame.samples
    if audio_samples == 0:
        raise RuntimeError("MiniMax H3 output audio is empty")
    expected_duration = frame_count / 24
    audio_duration = audio_samples / 32_000
    if abs(expected_duration - audio_duration) > max(1 / 24, 1024 / 32_000):
        raise RuntimeError(
            "MiniMax H3 output audio and video durations are not synchronized"
        )


def main() -> None:
    args = _parse_args()
    if args.pipeline != "text-to-video":
        raise SystemExit("this runtime supports MiniMax H3 audio-video generation only")
    if args.output_mime != "video/mp4":
        raise SystemExit("this runtime emits a joint video/audio MP4 only")
    if not 0 <= args.seed < 2**63:
        raise SystemExit("seed is outside the supported range")

    references = _load_references()
    first_image = _keyframe("first-image")
    last_image = _keyframe("last-image")
    if references is not None and (first_image is not None or last_image is not None):
        raise ValueError("references cannot be combined with first/last keyframes")

    workflow = (
        "ref2va"
        if references is not None
        else ("fl2va" if first_image or last_image else "t2va")
    )
    pipe = _load_pipeline(workflow)

    import torch
    from diffusers.utils import load_image
    from diffusers.utils.export_utils import encode_video

    call: dict[str, object] = {
        "prompt": _load_prompt(),
        "num_frames": args.num_frames,
        "height": args.height,
        "width": args.width,
        "num_inference_steps": args.num_inference_steps,
        "generator": torch.Generator().manual_seed(args.seed),
        "output": ["videos", "audio", "sampling_rate"],
    }
    if references is not None:
        call["references"] = references
    if first_image is not None:
        call["image"] = load_image(str(first_image))
    if last_image is not None:
        call["last_image"] = load_image(str(last_image))

    results = pipe(**call)
    if results["sampling_rate"] != 32000:
        raise RuntimeError("MiniMax H3 returned an unexpected audio sample rate")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    temporary = args.output_dir / ".minimax-h3.partial.mp4"
    destination = args.output_dir / "minimax-h3.mp4"
    encode_video(
        results["videos"][0],
        fps=24,
        output_path=str(temporary),
        audio=results["audio"][0],
        audio_sample_rate=results["sampling_rate"],
    )
    _verify_joint_av(temporary, width=args.width, height=args.height, frame_count=args.num_frames)
    os.replace(temporary, destination)


if __name__ == "__main__":
    main()
