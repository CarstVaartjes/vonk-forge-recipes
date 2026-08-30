"""Offline Diffusers adapter for complete HunyuanVideo 1.5 snapshots."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path

import torch
from diffusers import HunyuanVideo15ImageToVideoPipeline, HunyuanVideo15Pipeline
from diffusers.utils import export_to_video
from PIL import Image

_IMAGE_SUFFIXES = frozenset({".jpeg", ".jpg", ".png", ".webp"})
_MODEL_INDEX = Path("/models/model_index.json")
_MAX_IMAGE_BYTES = 16 * 1024 * 1024


def _slot_files(path: Path, slot: str) -> list[Path]:
    manifest = path / "manifest.json"
    try:
        document = json.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise SystemExit("missing or invalid signed input manifest") from error
    files = document.get("files") if isinstance(document, dict) else None
    if not isinstance(files, list):
        raise SystemExit("signed input manifest files are invalid")
    selected: list[Path] = []
    for item in files:
        if not isinstance(item, dict) or item.get("slot") != slot:
            continue
        name = item.get("name")
        if not isinstance(name, str) or Path(name).name != name:
            raise SystemExit("signed input manifest contains an unsafe name")
        candidate = path / name
        if candidate.is_symlink() or not candidate.is_file():
            raise SystemExit("signed input file is unavailable")
        selected.append(candidate)
    return selected


def _one_prompt(path: Path) -> str:
    candidates = _slot_files(path, "prompt")
    if len(candidates) != 1 or candidates[0].suffix.lower() != ".txt":
        raise SystemExit("text/video inference requires one prompt text file")
    try:
        prompt = candidates[0].read_text(encoding="utf-8").strip()
    except UnicodeDecodeError as error:
        raise SystemExit("prompt text must be UTF-8") from error
    if not 1 <= len(prompt.encode("utf-8")) <= 65536:
        raise SystemExit("prompt text must contain 1..65536 UTF-8 bytes")
    return prompt


def _one_image(path: Path) -> Image.Image:
    candidates = sorted(_slot_files(path, "image"))
    if (
        len(candidates) != 1
        or candidates[0].suffix.lower() not in _IMAGE_SUFFIXES
        or candidates[0].stat().st_size > _MAX_IMAGE_BYTES
    ):
        raise SystemExit(
            "image-to-video requires exactly one supported file in /inputs"
        )
    with Image.open(candidates[0]) as image:
        return image.convert("RGB")


def _validate_video(path: Path, resolution: int) -> None:
    command = [
        "ffprobe",
        "-v",
        "error",
        "-count_frames",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=codec_type,width,height,nb_read_frames:format=format_name,duration",
        "-of",
        "json",
        str(path),
    ]
    try:
        completed = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            timeout=120,
        )
        document = json.loads(completed.stdout)
        streams = document.get("streams")
        stream = streams[0] if isinstance(streams, list) and len(streams) == 1 else None
        container = document.get("format")
        if not isinstance(stream, dict) or not isinstance(container, dict):
            raise TypeError("missing one video stream")
        width = int(stream["width"])
        height = int(stream["height"])
        frames = int(stream["nb_read_frames"])
        duration = float(container["duration"])
        formats = str(container["format_name"]).split(",")
        if (
            stream.get("codec_type") != "video"
            or "mp4" not in formats
            or min(width, height) != resolution
            or frames != 121
            or duration <= 0
        ):
            raise ValueError("unexpected MP4 stream contract")
    except (
        FileNotFoundError,
        KeyError,
        TypeError,
        ValueError,
        json.JSONDecodeError,
        subprocess.SubprocessError,
    ) as error:
        raise SystemExit("Diffusers produced an invalid 121-frame MP4 artifact") from error


def _variant(pipeline: str, resolution: int, steps: int, guidance: float) -> str:
    """Map the signed recipe tuple to one exact self-contained model snapshot."""
    if (
        pipeline == "text-to-video"
        and resolution == 720
        and steps == 50
        and guidance == 6.0
    ):
        return "HunyuanVideo15Pipeline"
    if (
        pipeline == "text-to-video"
        and resolution == 480
        and steps == 50
        and guidance == 1.0
    ):
        return "HunyuanVideo15Pipeline"
    if (
        pipeline == "image-to-image"
        and resolution == 480
        and steps in {8, 12}
        and guidance == 1.0
    ):
        return "HunyuanVideo15ImageToVideoPipeline"
    raise SystemExit("unsupported HunyuanVideo 1.5 recipe variant")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pipeline", required=True, choices=("text-to-video", "image-to-image")
    )
    parser.add_argument("--output-mime", required=True)
    parser.add_argument("--num-inference-steps", required=True, type=int)
    parser.add_argument("--guidance-scale", required=True, type=float)
    parser.add_argument("--resolution", required=True, type=int, choices=(480, 720))
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    if args.output_mime != "video/mp4":
        raise SystemExit("HunyuanVideo 1.5 only emits video/mp4")
    class_name = _variant(
        args.pipeline, args.resolution, args.num_inference_steps, args.guidance_scale
    )
    if not _MODEL_INDEX.is_file():
        raise SystemExit(
            "complete pinned Diffusers snapshot is missing model_index.json"
        )

    input_root = Path("/inputs")
    prompt = _one_prompt(input_root)
    image = _one_image(input_root) if args.pipeline == "image-to-image" else None

    pipeline_class = (
        HunyuanVideo15ImageToVideoPipeline
        if class_name == "HunyuanVideo15ImageToVideoPipeline"
        else HunyuanVideo15Pipeline
    )
    pipe = pipeline_class.from_pretrained(
        "/models",
        torch_dtype=torch.bfloat16,
        local_files_only=True,
    )
    if pipe.__class__.__name__ != class_name:
        raise SystemExit(
            f"snapshot resolved unexpected pipeline class {pipe.__class__.__name__}"
        )
    if float(pipe.guider.guidance_scale) != args.guidance_scale:
        raise SystemExit("snapshot guider does not match the signed recipe")
    pipe.enable_model_cpu_offload()
    pipe.vae.enable_tiling()

    call = {
        "prompt": prompt,
        "negative_prompt": "",
        "generator": torch.Generator(device="cuda:0").manual_seed(args.seed),
        "num_frames": 121,
        "num_inference_steps": args.num_inference_steps,
    }
    if args.pipeline == "image-to-image":
        call["image"] = image

    frames = pipe(**call).frames[0]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    temporary = args.output_dir / ".output.tmp.mp4"
    output = args.output_dir / "output.mp4"
    export_to_video(frames, temporary, fps=24)
    try:
        _validate_video(temporary, args.resolution)
    except SystemExit:
        temporary.unlink(missing_ok=True)
        raise
    os.replace(temporary, output)


if __name__ == "__main__":
    main()
