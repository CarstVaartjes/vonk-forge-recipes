"""Offline 4x texture upscaler: 4xNomos8kSC (RRDBNet) loaded with Spandrel, tiled."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

_ENTRYPOINT = "/opt/vonk/source/run.py"
_MODEL = Path("/models/target/4xNomos8kSC.safetensors")
_INPUT_ROOT = Path("/inputs")
_SUFFIXES = frozenset({".jpeg", ".jpg", ".png"})
_FORMATS = frozenset({"JPEG", "PNG"})
_UNSUPPORTED_MODES = frozenset({"F", "I", "I;16", "I;16B", "I;16L"})
_MAX_MANIFEST_BYTES = 256 * 1024
_MAX_INPUT_BYTES = 32 * 1024 * 1024
_MAX_SIDE = 2048
_SCALE = 4
# A tile is upscaled with this many pixels of real neighbours around it, which
# are dropped from the result, so a tile edge is not a model-visible border.
_TILE = 512
_CONTEXT = 32

TileModel = Callable[[np.ndarray], np.ndarray]


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--entrypoint", required=True)
    parser.add_argument("--output-mime", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.entrypoint != _ENTRYPOINT:
        raise SystemExit("unexpected signed-bundle entrypoint")
    if args.output_mime != "image/png":
        raise SystemExit("the texture upscaler emits image/png only")
    return args


def _job_deadline() -> float:
    try:
        seconds = int(os.environ["VONK_JOB_TIMEOUT_SECONDS"])
    except (KeyError, ValueError) as error:
        raise SystemExit("job must declare VONK_JOB_TIMEOUT_SECONDS") from error
    if not 1 <= seconds <= 3600:
        raise SystemExit("job timeout must be between 1 and 3600 seconds")
    return time.monotonic() + seconds


def _check_deadline(deadline: float) -> None:
    if time.monotonic() >= deadline:
        raise SystemExit("upscaling exceeded the declared job budget")


def _input_path() -> Path:
    manifest = _INPUT_ROOT / "manifest.json"
    try:
        if manifest.is_symlink() or manifest.stat().st_size > _MAX_MANIFEST_BYTES:
            raise ValueError("input manifest must be a small regular file")
        files = json.loads(manifest.read_text(encoding="utf-8"))["files"]
        names = [item["name"] for item in files if item["slot"] == "image"]
    except (OSError, ValueError, KeyError, TypeError) as error:
        raise SystemExit("missing or invalid signed input manifest") from error
    if len(names) != 1:
        raise SystemExit("upscaling requires exactly one image input")
    name = names[0]
    path = _INPUT_ROOT / name
    if (
        not isinstance(name, str)
        or Path(name).name != name
        or path.suffix.lower() not in _SUFFIXES
        or path.is_symlink()
        or not path.is_file()
    ):
        raise SystemExit("the image input must be a regular .png, .jpg or .jpeg file")
    if path.stat().st_size > _MAX_INPUT_BYTES:
        raise SystemExit("the image input exceeds 32 MiB")
    return path


def _read_image(path: Path) -> tuple[np.ndarray, Image.Image | None]:
    """Return the RGB pixels and, when the input has transparency, its alpha."""
    try:
        with Image.open(path) as source:
            if source.format not in _FORMATS:
                raise SystemExit("the image input must be PNG or JPEG")
            if source.mode in _UNSUPPORTED_MODES:
                raise SystemExit(
                    f"{source.mode} images are not supported; use an 8-bit image"
                )
            if max(source.size) > _MAX_SIDE:
                raise SystemExit(
                    f"the image is {source.width}x{source.height}; the limit is "
                    f"{_MAX_SIDE}x{_MAX_SIDE}. Crop or downscale it first; it is "
                    "never resized silently"
                )
            transparent = "A" in source.getbands() or "transparency" in source.info
            upright = ImageOps.exif_transpose(source)
            pixels = upright.convert("RGBA" if transparent else "RGB")
    except (OSError, ValueError, Image.DecompressionBombError) as error:
        raise SystemExit(f"the image input cannot be decoded: {error}") from error
    if not transparent:
        return np.asarray(pixels), None
    alpha = pixels.getchannel("A")
    return np.asarray(pixels.convert("RGB")), alpha


def upscale(pixels: np.ndarray, model: TileModel, deadline: float) -> np.ndarray:
    """Upscale HxWx3 uint8 pixels by _SCALE with ``model`` one tile at a time."""
    height, width = pixels.shape[:2]
    padded = np.pad(pixels, ((_CONTEXT, _CONTEXT), (_CONTEXT, _CONTEXT), (0, 0)), "edge")
    result = np.empty((height * _SCALE, width * _SCALE, 3), dtype=np.uint8)
    for top in range(0, height, _TILE):
        for left in range(0, width, _TILE):
            _check_deadline(deadline)
            bottom = min(top + _TILE, height)
            right = min(left + _TILE, width)
            tile = model(padded[top : bottom + 2 * _CONTEXT, left : right + 2 * _CONTEXT])
            inner = tile[
                _CONTEXT * _SCALE : (bottom - top + _CONTEXT) * _SCALE,
                _CONTEXT * _SCALE : (right - left + _CONTEXT) * _SCALE,
            ]
            result[top * _SCALE : bottom * _SCALE, left * _SCALE : right * _SCALE] = inner
    return result


def _load_model() -> TileModel:
    import torch
    from spandrel import ImageModelDescriptor, ModelLoader

    descriptor = ModelLoader().load_from_file(_MODEL)
    if (
        not isinstance(descriptor, ImageModelDescriptor)
        or descriptor.scale != _SCALE
        or descriptor.input_channels != 3
        or descriptor.output_channels != 3
    ):
        raise SystemExit("the pinned weights are not a 4x RGB image model")
    # The card gives no inference settings; fp32 keeps the fp16-stored weights'
    # full range and costs little for a 17M-parameter network.
    descriptor.to("cuda").eval()

    def run(tile: np.ndarray) -> np.ndarray:
        with torch.inference_mode():
            batch = torch.from_numpy(np.ascontiguousarray(tile)).to("cuda")
            batch = batch.permute(2, 0, 1).unsqueeze(0).float().div_(255)
            output = descriptor(batch).squeeze(0).permute(1, 2, 0)
            return output.clamp_(0, 1).mul_(255).round_().to(torch.uint8).cpu().numpy()

    return run


def main() -> None:
    deadline = _job_deadline()
    args = _arguments()
    if not _MODEL.is_file():
        raise SystemExit("the pinned 4xNomos8kSC weights are missing")
    pixels, alpha = _read_image(_input_path())
    upscaled = Image.fromarray(upscale(pixels, _load_model(), deadline))
    if alpha is not None:
        upscaled.putalpha(alpha.resize(upscaled.size, Image.Resampling.BICUBIC))
    # The platform owns tmp/ and cache/; only our unique workspace must be fresh.
    scratch = args.output_dir / "tmp"
    scratch.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="vonk-upscale-", dir=scratch) as staging:
        staged = Path(staging) / "upscaled.png"
        upscaled.save(staged, format="PNG")
        _check_deadline(deadline)
        os.replace(staged, args.output_dir / "upscaled.png")


if __name__ == "__main__":
    main()
