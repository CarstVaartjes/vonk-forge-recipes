from __future__ import annotations

import argparse
import math
from pathlib import Path

_BASE_DIR = Path("/models/base")
_LORA_DIR = Path("/models/target")
_INPUT_DIR = Path("/inputs")
_IMAGE_SUFFIXES = frozenset({".jpg", ".jpeg", ".png", ".webp"})
_PROMPT_SUFFIXES = frozenset({".txt", ".text"})
_MAX_PROMPT_BYTES = 16 * 1024
_TEXT_TO_IMAGE_LORA = "Qwen-Image-2512-Lightning-4steps-V1.0-bf16.safetensors"
_IMAGE_EDIT_LORA = "Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors"
_NEGATIVE_PROMPT_2512 = (
    "低分辨率，低画质，肢体畸形，手指畸形，画面过饱和，蜡像感，人脸无细节，"
    "过度光滑，画面具有AI感。构图混乱。文字模糊，扭曲。"
)


def _input_files(suffixes: frozenset[str]) -> list[Path]:
    if not _INPUT_DIR.is_dir():
        raise SystemExit("the read-only /inputs job mount is required")
    return sorted(
        path
        for path in _INPUT_DIR.iterdir()
        if path.is_file() and path.suffix.lower() in suffixes
    )


def _prompt() -> str:
    prompt_files = _input_files(_PROMPT_SUFFIXES)
    if len(prompt_files) != 1:
        raise SystemExit("exactly one UTF-8 text prompt input (.txt or .text) is required")
    raw = prompt_files[0].read_bytes()
    if not raw or len(raw) > _MAX_PROMPT_BYTES:
        raise SystemExit("the text prompt must contain 1..16384 UTF-8 bytes")
    try:
        prompt = raw.decode("utf-8").strip()
    except UnicodeDecodeError as error:
        raise SystemExit("the text prompt must be valid UTF-8") from error
    if not prompt:
        raise SystemExit("the text prompt must not be blank")
    return prompt


def _scheduler_config() -> dict[str, object]:
    # Exact shift=3 distillation schedule published by ModelTC.
    return {
        "base_image_seq_len": 256,
        "base_shift": math.log(3),
        "invert_sigmas": False,
        "max_image_seq_len": 8192,
        "max_shift": math.log(3),
        "num_train_timesteps": 1000,
        "shift": 1.0,
        "shift_terminal": None,
        "stochastic_sampling": False,
        "time_shift_type": "exponential",
        "use_beta_sigmas": False,
        "use_dynamic_shifting": True,
        "use_exponential_sigmas": False,
        "use_karras_sigmas": False,
    }


def _pipeline(pipeline_name: str):
    import torch
    from diffusers import (
        FlowMatchEulerDiscreteScheduler,
        QwenImageEditPlusPipeline,
        QwenImagePipeline,
    )
    from diffusers.models import QwenImageTransformer2DModel

    pipeline_class = (
        QwenImagePipeline
        if pipeline_name == "text-to-image"
        else QwenImageEditPlusPipeline
    )
    transformer = QwenImageTransformer2DModel.from_pretrained(
        _BASE_DIR,
        subfolder="transformer",
        torch_dtype=torch.bfloat16,
        local_files_only=True,
    )
    scheduler = FlowMatchEulerDiscreteScheduler.from_config(_scheduler_config())
    pipe = pipeline_class.from_pretrained(
        _BASE_DIR,
        transformer=transformer,
        scheduler=scheduler,
        torch_dtype=torch.bfloat16,
        local_files_only=True,
    )
    lora_name = (
        _TEXT_TO_IMAGE_LORA
        if pipeline_name == "text-to-image"
        else _IMAGE_EDIT_LORA
    )
    lora_path = _LORA_DIR / lora_name
    if not lora_path.is_file():
        raise SystemExit(f"the exact Lightning LoRA is missing: {lora_path}")
    pipe.load_lora_weights(str(lora_path))
    return pipe.to("cuda")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pipeline", choices=("text-to-image", "image-to-image"), required=True)
    parser.add_argument("--output-mime", required=True)
    parser.add_argument("--num-inference-steps", type=int, default=4)
    parser.add_argument("--true-cfg-scale", type=float, default=1.0)
    parser.add_argument("--width", type=int, default=1328)
    parser.add_argument("--height", type=int, default=1328)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    if args.output_mime != "image/png":
        raise SystemExit("Qwen Image Lightning emits image/png only")
    if args.num_inference_steps != 4:
        raise SystemExit("the selected Lightning LoRA requires exactly four inference steps")
    if args.true_cfg_scale != 1.0:
        raise SystemExit("the selected Lightning LoRA requires true CFG 1.0")
    if args.width % 16 or args.height % 16:
        raise SystemExit("width and height must be divisible by 16")

    import torch
    from PIL import Image

    prompt = _prompt()
    pipe = _pipeline(args.pipeline)
    call: dict[str, object] = {
        "prompt": prompt,
        "generator": torch.Generator(device="cuda").manual_seed(args.seed),
        "true_cfg_scale": args.true_cfg_scale,
        "num_inference_steps": args.num_inference_steps,
        "num_images_per_prompt": 1,
    }
    if args.pipeline == "text-to-image":
        if _input_files(_IMAGE_SUFFIXES):
            raise SystemExit("text-to-image accepts a text prompt only")
        call.update(
            {
                "prompt": f"{prompt}, Ultra HD, 4K, cinematic composition.",
                "negative_prompt": _NEGATIVE_PROMPT_2512,
                "width": args.width,
                "height": args.height,
            }
        )
        output_name = "qwen-image-2512-lightning.png"
    else:
        image_files = _input_files(_IMAGE_SUFFIXES)
        if not 1 <= len(image_files) <= 2:
            raise SystemExit("image editing requires one or two image inputs")
        call.update(
            {
                "image": [Image.open(path).convert("RGB") for path in image_files],
                "negative_prompt": " ",
            }
        )
        output_name = "qwen-image-edit-2511-lightning.png"

    image = pipe(**call).images[0]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    image.save(args.output_dir / output_name, format="PNG")


if __name__ == "__main__":
    main()
