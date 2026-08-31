"""One-stage Wan-Dancer generation with pinned DiffSynth disk offload."""

from __future__ import annotations

import argparse
import os
from pathlib import Path


MODEL_ROOT = Path("/models")
FRAME_COUNT = 149
VRAM_LIMIT_GIB = 64.0
NEGATIVE_PROMPT = (
    "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，"
    "整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，"
    "画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，"
    "手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"
)
STYLE_PROMPTS = {
    "chinese-classical": (
        "一个人正在跳舞，舞蹈种类是古典舞。",
        "一个人正在跳舞，舞蹈种类是古典舞,图像清晰程度高,人物动作平均幅度中等,"
        "人物动作最大幅度中等。",
    ),
    "k-pop": (
        "一个人正在跳舞，舞蹈种类是韩舞。",
        "一个人正在跳舞，舞蹈种类是韩舞,图像清晰程度高,人物动作平均幅度中等,"
        "人物动作最大幅度中等。",
    ),
    "street": (
        "一个人正在跳舞，舞蹈种类是街舞。",
        "一个人正在跳舞，舞蹈种类是街舞,图像清晰程度高,人物动作平均幅度中等,"
        "人物动作最大幅度中等。",
    ),
    "tap": (
        "一个人正在跳舞，舞蹈种类是踢踏舞。",
        "一个人正在跳舞，舞蹈种类是踢踏舞,图像清晰程度高,人物动作平均幅度高,"
        "人物动作最大幅度高。",
    ),
    "latin": (
        "一个人正在跳舞，舞蹈种类是拉丁舞。",
        "一个人正在跳舞，舞蹈种类是拉丁舞,图像清晰程度高,人物动作平均幅度高,"
        "人物动作最大幅度中等。",
    ),
}
REQUIRED_MODEL_FILES = (
    "global_model.safetensors",
    "local_model.safetensors",
    "models_t5_umt5-xxl-enc-bf16.pth",
    "models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth",
    "Wan2.1_VAE.pth",
    "google/umt5-xxl/tokenizer.json",
    "google/umt5-xxl/tokenizer_config.json",
    "google/umt5-xxl/spiece.model",
)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", required=True, choices=("global", "local"))
    parser.add_argument("--reference-image", required=True, type=Path)
    parser.add_argument("--music", required=True, type=Path)
    parser.add_argument("--prompt-file", required=True, type=Path)
    parser.add_argument("--style", required=True, choices=tuple(STYLE_PROMPTS))
    parser.add_argument("--height", required=True, type=int)
    parser.add_argument("--width", required=True, type=int)
    parser.add_argument("--steps", required=True, type=int)
    parser.add_argument("--cfg-scale", required=True, type=float)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--global-video", type=Path)
    parser.add_argument("--output-frames", type=int)
    return parser.parse_args()


def _bounded_path(path: Path, label: str) -> Path:
    value = path.resolve()
    if path.is_symlink() or not value.is_file():
        raise SystemExit(f"{label} must be a regular file")
    return value


def _fit_reference(path: Path, height: int, width: int):
    from PIL import Image

    with Image.open(path) as source:
        source = source.convert("RGB")
        source.thumbnail((width, height), Image.Resampling.LANCZOS)
        canvas = Image.new("RGB", (width, height), (127, 127, 127))
        canvas.paste(source, ((width - source.width) // 2, (height - source.height) // 2))
    return canvas


def _pipeline(stage: str):
    if os.environ.get("DIFFSYNTH_SKIP_DOWNLOAD", "").lower() != "true":
        raise SystemExit("offline DiffSynth model loading is not enforced")

    import torch
    from diffsynth.pipelines.wan_video import ModelConfig, WanVideoPipeline

    disk = {
        "offload_dtype": "disk",
        "offload_device": "disk",
        "onload_dtype": torch.bfloat16,
        "onload_device": "cpu",
        "preparing_dtype": torch.bfloat16,
        "preparing_device": "cuda",
        "computation_dtype": torch.bfloat16,
        "computation_device": "cuda",
    }
    expert = "global_model.safetensors" if stage == "global" else "local_model.safetensors"
    model_configs = [
        ModelConfig(path=str(MODEL_ROOT / expert), **disk),
        ModelConfig(path=str(MODEL_ROOT / "models_t5_umt5-xxl-enc-bf16.pth"), **disk),
        ModelConfig(path=str(MODEL_ROOT / "Wan2.1_VAE.pth"), **disk),
        ModelConfig(
            path=str(MODEL_ROOT / "models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth"),
            **disk,
        ),
    ]
    return WanVideoPipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device="cuda",
        model_configs=model_configs,
        tokenizer_config=ModelConfig(path=str(MODEL_ROOT / "google/umt5-xxl")),
        redirect_common_files=False,
        use_usp=False,
        vram_limit=VRAM_LIMIT_GIB,
    )


def _global_keyframes(reference, height: int, width: int):
    from PIL import Image

    black = Image.new("RGB", (width, height), (0, 0, 0))
    return [reference] + [black] * (FRAME_COUNT - 1), [1] + [0] * (FRAME_COUNT - 1)


def _local_keyframes(path: Path, output_frames: int, height: int, width: int):
    from PIL import Image
    from diffsynth.utils.data import VideoData

    if not 1 <= output_frames <= FRAME_COUNT:
        raise SystemExit("output frame count is outside the one-segment canary bound")
    source = VideoData(video_file=str(path), height=height, width=width)
    if len(source) != FRAME_COUNT:
        raise SystemExit(f"global stage emitted {len(source)} frames, expected {FRAME_COUNT}")

    black = Image.new("RGB", (width, height), (0, 0, 0))
    keyframes = [black] * FRAME_COUNT
    mask = [0] * FRAME_COUNT
    # Preserve the official native adapter's one-segment mapping: for a music
    # clip shorter than 149/30 seconds, consume the leading global keyframes at
    # the corresponding local-frame positions and leave the tail unconditioned.
    for destination in range(output_frames):
        keyframes[destination] = source[destination]
        mask[destination] = 1
    return keyframes, mask


def main() -> None:
    args = _arguments()
    missing = [path for path in REQUIRED_MODEL_FILES if not (MODEL_ROOT / path).is_file()]
    if missing:
        raise SystemExit(f"incomplete Wan-Dancer snapshot; missing: {missing}")
    if args.height * args.width > 921_600:
        raise SystemExit("generation canvas exceeds 921,600 pixels")
    if args.height % 16 or args.width % 16:
        raise SystemExit("generation dimensions must be multiples of 16")

    reference_path = _bounded_path(args.reference_image, "reference image")
    music = _bounded_path(args.music, "music")
    prompt_path = _bounded_path(args.prompt_file, "prompt")
    prompt = prompt_path.read_text(encoding="utf-8").strip()
    if not prompt:
        raise SystemExit("prompt is empty")
    reference = _fit_reference(reference_path, args.height, args.width)
    style = STYLE_PROMPTS[args.style][0 if args.stage == "global" else 1]
    prompt = f"{prompt}, {style}"

    from diffsynth.utils.data import save_video

    pipe = _pipeline(args.stage)
    if args.stage == "global":
        keyframes, mask = _global_keyframes(reference, args.height, args.width)
        video = pipe(
            prompt=prompt + " 帧率是7.5000",
            negative_prompt=NEGATIVE_PROMPT,
            seed=args.seed,
            tiled=False,
            height=args.height,
            width=args.width,
            num_frames=FRAME_COUNT,
            num_inference_steps=args.steps,
            cfg_scale=args.cfg_scale,
            wantodance_music_path=str(music),
            wantodance_reference_image=reference,
            wantodance_fps=7.5,
            wantodance_keyframes=keyframes,
            wantodance_keyframes_mask=mask,
            framewise_decoding=True,
        )
        save_video(video, str(args.output), fps=7.5, quality=5)
        return

    if args.global_video is None or args.output_frames is None:
        raise SystemExit("local stage requires global-video and output-frames")
    global_video = _bounded_path(args.global_video, "global video")
    keyframes, mask = _local_keyframes(
        global_video, args.output_frames, args.height, args.width
    )
    video = pipe(
        prompt=prompt + " 帧率是30fps。",
        negative_prompt=NEGATIVE_PROMPT,
        seed=args.seed,
        tiled=True,
        height=args.height,
        width=args.width,
        num_frames=FRAME_COUNT,
        num_inference_steps=args.steps,
        cfg_scale=args.cfg_scale,
        wantodance_music_path=str(music),
        wantodance_reference_image=reference,
        wantodance_fps=30.0,
        wantodance_keyframes=keyframes,
        wantodance_keyframes_mask=mask,
    )
    save_video(video, str(args.output), fps=30, quality=5)


if __name__ == "__main__":
    main()
