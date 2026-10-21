"""Offline Diffusers adapter for complete HunyuanVideo 1.5 snapshots."""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
from pathlib import Path
from typing import NamedTuple

import numpy as np

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


class FrameMetrics(NamedTuple):
    frames: int
    shape: tuple[int, ...]
    mean_luminance: float | None
    nan_count: int
    inf_count: int
    black_frames: int


def _frame_metrics(frames: np.ndarray) -> FrameMetrics:
    # Inspect floats before export_to_video casts them to uint8 and erases NaNs.
    nan_count = inf_count = black_frames = 0
    luminance = 0.0
    rgb_video = frames.ndim == 4 and frames.shape[-1] == 3
    for frame in frames:
        nan_count += int(np.count_nonzero(np.isnan(frame)))
        inf_count += int(np.count_nonzero(np.isinf(frame)))
        # Match the exporter's float-to-uint8 quantization.
        black_frames += int(np.all(frame < 1 / 255))
        if rgb_video and np.isfinite(frame).all():
            luminance += sum(
                float(frame[..., channel].mean(dtype=np.float64)) * weight
                for channel, weight in enumerate((0.2126, 0.7152, 0.0722))
            )
    return FrameMetrics(
        len(frames),
        tuple(frames.shape),
        luminance / len(frames)
        if rgb_video and len(frames) and not (nan_count or inf_count)
        else None,
        nan_count,
        inf_count,
        black_frames,
    )


def _video_error(
    path: Path,
    expected_size: tuple[int, int],
    metrics: FrameMetrics,
    reason: str,
    probe: str = "unavailable",
) -> SystemExit:
    reason = reason.replace("\n", "\\n")
    return SystemExit(
        f"Diffusers produced an invalid {metrics.frames}-frame MP4 artifact: "
        f"{reason}; expected_size={expected_size}, fps=24, "
        f"bytes={path.stat().st_size if path.is_file() else 0}; "
        f"source={json.dumps(metrics._asdict(), allow_nan=False)}; probe={probe}"
    )


def _expected_size(pipe, image: Image.Image | None) -> tuple[int, int]:
    # '480p'/'720p' identify area buckets, not a mandatory short edge. Use the
    # pinned pipeline's own aspect-ratio selection for I2V and T2V alike.
    width, height = image.size if image is not None else pipe.default_aspect_ratio
    height, width = pipe.video_processor.calculate_default_height_width(
        height=height, width=width, target_size=pipe.target_size
    )
    return width, height


def _validate_video(
    path: Path,
    expected_size: tuple[int, int],
    metrics: FrameMetrics,
    expected_frames: int = 121,
) -> None:
    command = [
        "ffprobe",
        "-v",
        "error",
        "-count_frames",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=codec_type,codec_name,pix_fmt,width,height,nb_read_frames,avg_frame_rate:format=format_name,duration",
        "-of",
        "json",
        str(path),
    ]
    probe = "unavailable"
    try:
        completed = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            timeout=120,
        )
        probe = completed.stdout[:4000].replace("\n", "")
        document = json.loads(completed.stdout)
        if not isinstance(document, dict):
            raise TypeError("ffprobe inventory is not an object")
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
            or stream.get("codec_name") != "h264"
            or stream.get("pix_fmt") != "yuv420p"
            or "mp4" not in formats
            or (width, height) != expected_size
            or frames != expected_frames
            or stream.get("avg_frame_rate") != "24/1"
            or not math.isfinite(duration)
            or abs(duration - expected_frames / 24) > 1 / 24
            or path.stat().st_size == 0
        ):
            raise ValueError("unexpected MP4 stream contract")
        if completed.stderr:
            raise ValueError(f"ffprobe decode errors: {completed.stderr[:2000]}")
    except (
        OSError,
        KeyError,
        TypeError,
        ValueError,
        subprocess.SubprocessError,
    ) as error:
        stderr = getattr(error, "stderr", None)
        raise _video_error(
            path,
            expected_size,
            metrics,
            f"{type(error).__name__}: {error}; stderr={str(stderr)[:2000]}",
            probe,
        ) from error


def _write_video(
    frames: np.ndarray, directory: Path, expected_size: tuple[int, int], expected_frames: int = 121
) -> None:
    metrics = _frame_metrics(frames)
    directory.mkdir(parents=True, exist_ok=True)
    temporary = directory / ".output.tmp.mp4"
    try:
        if (
            metrics.shape != (expected_frames, expected_size[1], expected_size[0], 3)
            or metrics.nan_count
            or metrics.inf_count
            or metrics.black_frames == metrics.frames
        ):
            raise _video_error(
                temporary, expected_size, metrics, "invalid decoded RGB frames"
            )
        # Native buckets are multiples of 16; avoid imageio silently resizing.
        export_to_video(frames, temporary, fps=24, macro_block_size=1)
        _validate_video(temporary, expected_size, metrics, expected_frames)
        os.replace(temporary, directory / "output.mp4")
    except SystemExit:
        temporary.unlink(missing_ok=True)
        raise
    except Exception as error:
        diagnostic = _video_error(
            temporary, expected_size, metrics, f"{type(error).__name__}: {error}"
        )
        temporary.unlink(missing_ok=True)
        raise diagnostic from error


def _variant(pipeline: str, resolution: int, steps: int, guidance: float) -> str:
    """Map the signed recipe tuple to one exact self-contained model snapshot."""
    if (
        pipeline == "text-to-video"
        and resolution == 720
        and 1 <= steps <= 50
        and guidance == 6.0
    ):
        return "HunyuanVideo15Pipeline"
    if (
        pipeline == "text-to-video"
        and resolution == 480
        and 1 <= steps <= 50
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
    parser.add_argument("--num-frames", type=int, default=121)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    if not 1 <= args.num_frames <= 121 or args.num_frames % 4 != 1:
        raise SystemExit("num-frames must be 4k+1 between 1 and 121")
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
        dtype=torch.bfloat16,
        local_files_only=True,
    )
    if pipe.__class__.__name__ != class_name:
        raise SystemExit(
            f"snapshot resolved unexpected pipeline class {pipe.__class__.__name__}"
        )
    if float(pipe.guider.guidance_scale) != args.guidance_scale:
        raise SystemExit("snapshot guider does not match the signed recipe")
    pipe.to("cuda")
    pipe.vae.enable_tiling()

    call = {
        "prompt": prompt,
        "negative_prompt": "",
        "generator": torch.Generator(device="cuda:0").manual_seed(args.seed),
        "num_frames": args.num_frames,
        "num_inference_steps": args.num_inference_steps,
    }
    if args.pipeline == "image-to-image":
        call["image"] = image

    expected_size = _expected_size(pipe, image)
    frames = pipe(**call, output_type="np").frames[0]
    _write_video(frames, args.output_dir, expected_size, args.num_frames)


if __name__ == "__main__":
    main()
