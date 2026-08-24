from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any


MODEL_ROOT = Path("/models")
INPUT_ROOT = Path("/inputs")
DEFAULT_PROMPT = (
    "A small red fox trots through a snowy pine forest, snow crunching underfoot"
)
ALLOWED_REQUEST_KEYS = {
    "prompt",
    "num_frames",
    "width",
    "height",
    "first_image",
    "last_image",
    "references",
}
ALLOWED_REFERENCE_KEYS = {"type", "path"}
ALLOWED_REFERENCE_TYPES = {"image", "video", "audio"}
MAX_REQUEST_BYTES = 64 * 1024


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pipeline", required=True)
    parser.add_argument("--output-mime", required=True)
    parser.add_argument("--num-inference-steps", type=int, default=4)
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=544)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def _safe_input_path(value: object, field: str) -> Path:
    if not isinstance(value, str) or not value or len(value) > 256:
        raise ValueError(f"{field} must be a non-empty relative input path")
    relative = Path(value)
    if relative.is_absolute() or ".." in relative.parts or relative == Path("."):
        raise ValueError(f"{field} must stay inside /inputs")
    candidate = INPUT_ROOT / relative
    if candidate.is_symlink() or not candidate.is_file():
        raise ValueError(f"{field} does not name a regular input file")
    try:
        candidate.resolve(strict=True).relative_to(INPUT_ROOT.resolve(strict=True))
    except ValueError as error:
        raise ValueError(f"{field} escapes /inputs") from error
    return candidate


def _load_request() -> dict[str, Any]:
    request_path = INPUT_ROOT / "request.json"
    if not request_path.exists():
        return {}
    if request_path.is_symlink() or not request_path.is_file():
        raise ValueError("request.json must be a regular file")
    if request_path.stat().st_size > MAX_REQUEST_BYTES:
        raise ValueError("request.json exceeds 64 KiB")
    try:
        value = json.loads(request_path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("request.json must contain UTF-8 JSON") from error
    if not isinstance(value, dict) or not set(value).issubset(ALLOWED_REQUEST_KEYS):
        raise ValueError("request.json contains unsupported fields")
    return value


def _positive_multiple(value: object, field: str, default: int) -> int:
    if value is None:
        return default
    if type(value) is not int or not 64 <= value <= 2048 or value % 32:
        raise ValueError(f"{field} must be a multiple of 32 between 64 and 2048")
    return value


def _num_frames(value: object) -> int:
    if value is None:
        return 124
    if type(value) is not int or not 120 <= value <= 345:
        raise ValueError("num_frames must request 120..345 frames at 24 fps")
    return value


def _prompt(value: object) -> str:
    if value is None:
        return DEFAULT_PROMPT
    if not isinstance(value, str) or not value.strip() or len(value) > 4096:
        raise ValueError("prompt must contain 1..4096 characters")
    return value.strip()


def _load_references(values: object) -> list[object] | None:
    if values is None:
        return None
    if not isinstance(values, list) or not values or len(values) > 12:
        raise ValueError("references must contain 1..12 ordered entries")

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
    clean: list[tuple[str, Path]] = []
    for index, item in enumerate(values):
        if not isinstance(item, dict) or set(item) != ALLOWED_REFERENCE_KEYS:
            raise ValueError(f"references[{index}] must contain exactly type and path")
        kind = item.get("type")
        if kind not in ALLOWED_REFERENCE_TYPES:
            raise ValueError(f"references[{index}].type is unsupported")
        clean.append(
            (kind, _safe_input_path(item.get("path"), f"references[{index}].path"))
        )

    counts = Counter(kind for kind, _ in clean)
    if counts["image"] > 9 or counts["video"] > 3 or counts["audio"] > 3:
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
        low_cpu_mem_usage=False,
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
        workflow=workflow,
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


def _verify_joint_av(path: Path) -> None:
    import av

    with av.open(str(path), mode="r") as container:
        video_streams = list(container.streams.video)
        audio_streams = list(container.streams.audio)
        if len(video_streams) != 1 or len(audio_streams) != 1:
            raise RuntimeError(
                "MiniMax H3 output must contain one video and one audio stream"
            )
        audio = audio_streams[0].codec_context
        if (
            audio.sample_rate != 32000
            or audio.layout is None
            or audio.layout.name != "stereo"
        ):
            raise RuntimeError("MiniMax H3 output must contain 32 kHz stereo audio")
        if (
            video_streams[0].average_rate is None
            or float(video_streams[0].average_rate) != 24.0
        ):
            raise RuntimeError("MiniMax H3 output must contain 24 fps video")


def main() -> None:
    args = _parse_args()
    if args.pipeline != "text-to-video":
        raise SystemExit("this runtime supports MiniMax H3 audio-video generation only")
    if args.output_mime != "video/mp4":
        raise SystemExit("this runtime emits a joint video/audio MP4 only")
    if not 2 <= args.num_inference_steps <= 1000:
        raise SystemExit("num-inference-steps must be between 2 and 1000")
    if not 0 <= args.seed < 2**63:
        raise SystemExit("seed is outside the supported range")

    request = _load_request()
    references = _load_references(request.get("references"))
    first_image = (
        _safe_input_path(request["first_image"], "first_image")
        if request.get("first_image") is not None
        else None
    )
    last_image = (
        _safe_input_path(request["last_image"], "last_image")
        if request.get("last_image") is not None
        else None
    )
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
        "prompt": _prompt(request.get("prompt")),
        "num_frames": _num_frames(request.get("num_frames")),
        "height": _positive_multiple(request.get("height"), "height", args.height),
        "width": _positive_multiple(request.get("width"), "width", args.width),
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
    _verify_joint_av(temporary)
    os.replace(temporary, destination)


if __name__ == "__main__":
    main()
