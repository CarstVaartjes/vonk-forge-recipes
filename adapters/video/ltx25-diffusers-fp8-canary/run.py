"""Offline LTX 2.5 FP8-cast single-Spark canary adapter."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

MODEL_ROOT = Path("/models/target")
INPUT_ROOT = Path("/inputs")
MAX_REQUEST_BYTES = 64 * 1024
MODEL_REVISION = "426936f8b22dc28e4def61e515478b0b7e4a53cc"
DIFFUSERS_REVISION = "d035dcd7cc7c88e0a154609b62887d50bba9fdc2"
PROFILES = frozenset({"fp8-cast-sequential-offload"})
DEFAULT_PROFILE = "fp8-cast-sequential-offload"
ALLOWED_REQUEST_KEYS = frozenset({"seed"})
TRANSFORMER_BF16_SHARD_BYTES = {
    "transformer/diffusion_pytorch_model-00001-of-00004.safetensors": 9_932_336_016,
    "transformer/diffusion_pytorch_model-00002-of-00004.safetensors": 9_919_463_480,
    "transformer/diffusion_pytorch_model-00003-of-00004.safetensors": 9_986_144_288,
    "transformer/diffusion_pytorch_model-00004-of-00004.safetensors": 8_138_277_304,
}
TRANSFORMER_BF16_BYTES = sum(TRANSFORMER_BF16_SHARD_BYTES.values())
DECLARED_MINIMUM_FP8_SAVINGS_BYTES = 8_000_000_000
EXPECTED_SHARDS = {
    "connectors/diffusion_pytorch_model.safetensors.index.json": {
        f"diffusion_pytorch_model-{index:05d}-of-00002.safetensors"
        for index in range(1, 3)
    },
    "text_encoder/model.safetensors.index.json": {
        f"model-{index:05d}-of-00005.safetensors" for index in range(1, 6)
    },
    "transformer/diffusion_pytorch_model.safetensors.index.json": {
        f"diffusion_pytorch_model-{index:05d}-of-00004.safetensors"
        for index in range(1, 5)
    },
}
REQUIRED_FILES = frozenset(
    {
        "audio_vae/config.json",
        "audio_vae/diffusion_pytorch_model.safetensors",
        "connectors/config.json",
        "scheduler/scheduler_config.json",
        "text_encoder/config.json",
        "text_encoder/generation_config.json",
        "tokenizer/chat_template.jinja",
        "tokenizer/tokenizer.json",
        "tokenizer/tokenizer_config.json",
        "transformer/config.json",
        "vae/config.json",
        "vae/diffusion_pytorch_model.safetensors",
        "vocoder/config.json",
        "vocoder/diffusion_pytorch_model.safetensors",
        *EXPECTED_SHARDS,
        *(
            f"connectors/diffusion_pytorch_model-{index:05d}-of-00002.safetensors"
            for index in range(1, 3)
        ),
        *(
            f"text_encoder/model-{index:05d}-of-00005.safetensors"
            for index in range(1, 6)
        ),
        *(
            f"transformer/diffusion_pytorch_model-{index:05d}-of-00004.safetensors"
            for index in range(1, 5)
        ),
    }
)
FORBIDDEN_PATHS = (
    "diffusion_decoder",
    "duration_head",
    "latent_upsampler",
    "processor",
    "prompt_enhancer",
    "temporal_latent_upsampler",
    "transformer_full",
    "ltx-2.5-22b-distilled-lora-450-bf16.safetensors",
    "model_index.json",
    "modular_model_index.json",
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pipeline", required=True)
    parser.add_argument("--output-mime", required=True)
    parser.add_argument("--num-inference-steps", required=True, type=int)
    parser.add_argument("--width", required=True, type=int)
    parser.add_argument("--height", required=True, type=int)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args()


def _load_request() -> dict[str, Any]:
    path = INPUT_ROOT / "request.json"
    if not path.exists():
        return {}
    if path.is_symlink() or not path.is_file():
        raise ValueError("request.json must be a regular file")
    if path.stat().st_size > MAX_REQUEST_BYTES:
        raise ValueError("request.json exceeds 64 KiB")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("request.json must contain UTF-8 JSON") from error
    if not isinstance(value, dict) or not set(value).issubset(ALLOWED_REQUEST_KEYS):
        raise ValueError("request.json contains unsupported fields")
    return value


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


def _profile(value: object) -> str:
    if value is None:
        return DEFAULT_PROFILE
    if not isinstance(value, str) or value not in PROFILES:
        raise ValueError(f"profile must be one of: {', '.join(sorted(PROFILES))}")
    return value


def _seed(value: object, default: int) -> int:
    if value is None:
        value = default
    if type(value) is not int or not 0 <= value < 2**63:
        raise ValueError("seed must be an integer between 0 and 2^63-1")
    return value


def _validate_model_closure() -> None:
    missing = sorted(path for path in REQUIRED_FILES if not (MODEL_ROOT / path).is_file())
    if missing:
        raise SystemExit(f"LTX 2.5 filtered snapshot is incomplete: {missing[0]}")
    forbidden = sorted(path for path in FORBIDDEN_PATHS if (MODEL_ROOT / path).exists())
    if forbidden:
        raise SystemExit(f"LTX 2.5 snapshot contains excluded component: {forbidden[0]}")
    for relative, expected in EXPECTED_SHARDS.items():
        document = json.loads((MODEL_ROOT / relative).read_text(encoding="utf-8"))
        weight_map = document.get("weight_map")
        actual = set(weight_map.values()) if isinstance(weight_map, dict) else set()
        if actual != expected:
            raise SystemExit(f"LTX 2.5 shard index changed: {relative}")


def _load_pipeline(profile: str):
    import torch
    from diffusers import (
        AutoencoderKLLTX2Audio,
        AutoencoderKLLTX2Video,
        FlowMatchEulerDiscreteScheduler,
        LTX2VideoTransformer3DModel,
    )
    from diffusers.pipelines.ltx2 import (
        LTX2Pipeline,
        LTX2TextConnectors,
        LTX2VocoderWithBWE,
    )
    from transformers import AutoModelForImageTextToText, AutoTokenizer

    local = {"local_files_only": True}
    transformer = LTX2VideoTransformer3DModel.from_pretrained(
        MODEL_ROOT / "transformer", dtype=torch.bfloat16, **local
    )
    if profile.startswith("fp8-cast"):
        transformer.enable_layerwise_casting(
            storage_dtype=torch.float8_e4m3fn,
            compute_dtype=torch.bfloat16,
        )
    pipe = LTX2Pipeline(
        scheduler=FlowMatchEulerDiscreteScheduler.from_pretrained(
            MODEL_ROOT / "scheduler", **local
        ),
        vae=AutoencoderKLLTX2Video.from_pretrained(
            MODEL_ROOT / "vae", dtype=torch.bfloat16, **local
        ),
        audio_vae=AutoencoderKLLTX2Audio.from_pretrained(
            MODEL_ROOT / "audio_vae", dtype=torch.bfloat16, **local
        ),
        text_encoder=AutoModelForImageTextToText.from_pretrained(
            MODEL_ROOT / "text_encoder", dtype=torch.bfloat16, **local
        ),
        tokenizer=AutoTokenizer.from_pretrained(MODEL_ROOT / "tokenizer", **local),
        connectors=LTX2TextConnectors.from_pretrained(
            MODEL_ROOT / "connectors", dtype=torch.bfloat16, **local
        ),
        transformer=transformer,
        vocoder=LTX2VocoderWithBWE.from_pretrained(
            MODEL_ROOT / "vocoder", dtype=torch.bfloat16, **local
        ),
    )
    pipe.vae.enable_tiling()
    if profile != DEFAULT_PROFILE:
        raise RuntimeError("the canary must use FP8 storage and sequential offload")
    # Sequential offload controls CUDA residency. The FP8 layerwise storage above
    # is the part that reduces physical unified-memory use on DGX Spark.
    pipe.enable_sequential_cpu_offload(device="cuda")
    return pipe


def _verify_joint_av(
    path: Path, *, width: int, height: int, frame_count: int, sample_rate: int
) -> dict[str, int | float]:
    import av
    import numpy as np

    with av.open(str(path), mode="r") as container:
        if len(container.streams.video) != 1 or len(container.streams.audio) != 1:
            raise RuntimeError("LTX 2.5 output must contain one video and one audio stream")
        video_stream = container.streams.video[0]
        audio_stream = container.streams.audio[0]
        if video_stream.codec_context.name != "h264":
            raise RuntimeError("LTX 2.5 output video codec must be H.264")
        if audio_stream.codec_context.name != "aac":
            raise RuntimeError("LTX 2.5 output audio codec must be AAC")
        if video_stream.width != width or video_stream.height != height:
            raise RuntimeError("LTX 2.5 output dimensions do not match the job")
        if video_stream.average_rate is None or float(video_stream.average_rate) != 24.0:
            raise RuntimeError("LTX 2.5 output must be exactly 24 fps")
        if audio_stream.codec_context.sample_rate != sample_rate:
            raise RuntimeError("LTX 2.5 output audio sample rate changed")
        if audio_stream.codec_context.layout.name != "stereo":
            raise RuntimeError("LTX 2.5 output audio must be stereo")

    with av.open(str(path), mode="r") as container:
        decoded_frames = sum(1 for _ in container.decode(video=0))
    if decoded_frames != frame_count:
        raise RuntimeError("LTX 2.5 output frame count changed")

    audio_samples = 0
    audio_peak = 0.0
    with av.open(str(path), mode="r") as container:
        for frame in container.decode(audio=0):
            audio_samples += frame.samples
            samples = frame.to_ndarray()
            if samples.size:
                audio_peak = max(audio_peak, float(np.max(np.abs(samples))))
    if audio_samples == 0 or audio_peak == 0.0:
        raise RuntimeError("LTX 2.5 output audio is empty or silent")
    video_seconds = frame_count / 24.0
    audio_seconds = audio_samples / sample_rate
    if abs(video_seconds - audio_seconds) > 1 / 24:
        raise RuntimeError("LTX 2.5 output audio and video durations are not synchronized")
    return {
        "audio_channels": 2,
        "audio_codec": "aac",
        "audio_samples": audio_samples,
        "audio_sample_rate": sample_rate,
        "duration_seconds": video_seconds,
        "fps": 24,
        "frames": decoded_frames,
        "height": height,
        "video_codec": "h264",
        "width": width,
    }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _array_receipt(value: Any) -> dict[str, Any]:
    import numpy as np

    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    array = np.ascontiguousarray(value)
    return {
        "dtype": str(array.dtype),
        "shape": list(array.shape),
        "sha256": hashlib.sha256(array.tobytes(order="C")).hexdigest(),
    }


def main() -> None:
    args = _parse_args()
    if args.pipeline != "text-to-video" or args.output_mime != "video/mp4":
        raise SystemExit("this LTX 2.5 adapter supports text-to-video MP4 jobs only")
    if args.num_inference_steps != 8:
        raise SystemExit("LTX 2.5 distilled requires its fixed eight-sigma schedule")
    if (args.width, args.height) != (768, 512):
        raise SystemExit("this baseline is qualified only at 768x512")

    _validate_model_closure()
    request = _load_request()
    prompt = _load_prompt()
    profile = _profile(request.get("profile"))
    seed = _seed(request.get("seed"), args.seed)

    import torch
    from diffusers.pipelines.ltx2.utils import (
        DEFAULT_NEGATIVE_PROMPT,
        DISTILLED_SIGMA_VALUES,
    )
    from diffusers.utils import encode_video

    if len(DISTILLED_SIGMA_VALUES) != 8:
        raise RuntimeError("pinned Diffusers distilled sigma schedule changed")
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    pipe = _load_pipeline(profile)
    generator = torch.Generator("cuda").manual_seed(seed)
    video, audio = pipe(
        prompt=prompt,
        negative_prompt=DEFAULT_NEGATIVE_PROMPT,
        width=args.width,
        height=args.height,
        num_frames=65,
        frame_rate=24.0,
        sigmas=DISTILLED_SIGMA_VALUES,
        guidance_scale=1.0,
        audio_guidance_scale=1.0,
        stg_scale=0.0,
        audio_stg_scale=0.0,
        modality_scale=1.0,
        audio_modality_scale=1.0,
        guidance_rescale=0.0,
        audio_guidance_rescale=0.0,
        noise_scale=0.0,
        use_cross_timestep=True,
        enable_prompt_enhancement=False,
        max_sequence_length=1024,
        generator=generator,
        output_type="np",
        return_dict=False,
    )
    sample_rate = int(pipe.vocoder.config.output_sampling_rate)
    if sample_rate != 48000:
        raise RuntimeError("official LTX 2.5 BWE vocoder must emit 48 kHz audio")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    temporary = args.output_dir / ".ltx-2.5.partial.mp4"
    destination = args.output_dir / "ltx-2.5.mp4"
    raw_video = video[0]
    raw_audio = audio[0].float().cpu()
    tensor_receipt = {
        "audio": _array_receipt(raw_audio),
        "video": _array_receipt(raw_video),
    }
    encode_video(
        raw_video,
        fps=24,
        output_path=str(temporary),
        audio=raw_audio,
        audio_sample_rate=sample_rate,
    )
    media = _verify_joint_av(
        temporary,
        width=args.width,
        height=args.height,
        frame_count=65,
        sample_rate=sample_rate,
    )
    os.replace(temporary, destination)
    receipt = {
        "media": media,
        "output_sha256": _sha256_file(destination),
        "profile": profile,
        "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        "runtime": {
            "cuda": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(),
            "diffusers_revision": DIFFUSERS_REVISION,
            "model_revision": MODEL_REVISION,
            "torch": torch.__version__,
        },
        "seed": seed,
        "tensors": tensor_receipt,
    }
    (args.output_dir / "ltx-2.5-receipt.json").write_text(
        json.dumps(receipt, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
