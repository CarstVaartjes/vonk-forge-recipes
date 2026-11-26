"""Offline BiRefNet job adapter: one image in, an RGBA cutout and its mask out.

The inference is the creator's model-card example (BiRefNet_HR README, "Use the loaded
BiRefNet for inference"): fp16, a 2048x2048 resize, ImageNet normalisation, the last
prediction through a sigmoid, the mask resized back to the input. The weights are mounted
as a Model whose remote code (``birefnet.py``) the card's AutoModel call loads.
"""

from __future__ import annotations

import argparse
import os
import struct
import tempfile
import time
from collections.abc import Callable
from pathlib import Path

from PIL import Image, ImageOps
from vonk_agent_protocol.job_inputs import RecipeJobInputManifest

_ENTRYPOINT = "/opt/vonk/source/run.py"
_MODEL = Path("/models/target")
_INPUT_ROOT = Path("/inputs")
_FORMATS = {
    ".png": ("PNG", "image/png"),
    ".jpg": ("JPEG", "image/jpeg"),
    ".jpeg": ("JPEG", "image/jpeg"),
}
_UNSUPPORTED_MODES = frozenset({"F", "I", "I;16", "I;16B", "I;16L"})
_MAX_MANIFEST_BYTES = 256 * 1024
_MAX_INPUT_BYTES = 32 * 1024 * 1024
_MAX_SIDE = 4096
_MODEL_SIDE = 2048
_MEAN = [0.485, 0.456, 0.406]
_STD = [0.229, 0.224, 0.225]
# Each output PNG with its colour type: 6 is RGBA, 0 is 8-bit grayscale.
_OUTPUTS = {"cutout.png": 6, "mask.png": 0}

Segmenter = Callable[[Image.Image], Image.Image]


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--entrypoint", required=True)
    parser.add_argument("--output-mime", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.entrypoint != _ENTRYPOINT:
        raise SystemExit("unexpected signed-bundle entrypoint")
    if args.output_mime != "image/png":
        raise SystemExit("BiRefNet emits image/png only")
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
        raise SystemExit("segmentation exceeded the declared job budget")


def _input_path() -> Path:
    """The one declared ``image`` file, verified against the job manifest."""
    manifest_path = _INPUT_ROOT / "manifest.json"
    if (
        not _INPUT_ROOT.is_dir()
        or _INPUT_ROOT.is_symlink()
        or manifest_path.is_symlink()
        or not manifest_path.is_file()
        or manifest_path.stat().st_size > _MAX_MANIFEST_BYTES
    ):
        raise SystemExit("a regular, bounded Vonk job input manifest is required")
    try:
        manifest = RecipeJobInputManifest.model_validate_json(
            manifest_path.read_bytes()
        )
    except (OSError, ValueError) as error:
        raise SystemExit(f"invalid Vonk job input manifest: {error}") from error
    if len(manifest.files) != 1 or manifest.files[0].slot != "image":
        raise SystemExit("segmentation requires exactly one image input")
    item = manifest.files[0]
    suffix = Path(item.name).suffix.lower()
    if suffix not in _FORMATS or item.media_type != _FORMATS[suffix][1]:
        raise SystemExit("the image input must be a .png, .jpg or .jpeg file")
    if {entry.name for entry in _INPUT_ROOT.iterdir()} != {item.name, "manifest.json"}:
        raise SystemExit("job input files do not match the manifest")
    path = _INPUT_ROOT / item.name
    if path.is_symlink() or not path.is_file():
        raise SystemExit("the image input must be a regular file")
    if not 1 <= item.size_bytes <= _MAX_INPUT_BYTES:
        raise SystemExit("the image input must hold 1 byte to 32 MiB")
    if path.stat().st_size != item.size_bytes:
        raise SystemExit("image size does not match the job manifest")
    return path


def _read_image(path: Path) -> Image.Image:
    """The upright RGB image; transparency is dropped, never an oversize resize."""
    expected = _FORMATS[path.suffix.lower()][0]
    try:
        with Image.open(path) as source:
            if source.format != expected:
                raise SystemExit("the image content does not match its file extension")
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
            return ImageOps.exif_transpose(source).convert("RGB")
    except (OSError, ValueError, Image.DecompressionBombError) as error:
        raise SystemExit(f"the image input cannot be decoded: {error}") from error


def _load_model() -> Segmenter:
    import torch
    from torchvision import transforms
    from transformers import AutoModelForImageSegmentation

    # The creator's card path: remote code from the mounted Model, fp16, then half on CUDA.
    torch.set_float32_matmul_precision("high")
    model = AutoModelForImageSegmentation.from_pretrained(
        _MODEL,
        trust_remote_code=True,
        local_files_only=True,
        torch_dtype=torch.float16,
    )
    model.to("cuda").eval()
    prepare = transforms.Compose(
        [
            transforms.Resize((_MODEL_SIDE, _MODEL_SIDE)),
            transforms.ToTensor(),
            transforms.Normalize(_MEAN, _STD),
        ]
    )

    def segment(image: Image.Image) -> Image.Image:
        batch = prepare(image).unsqueeze(0).to("cuda").half()
        with torch.no_grad():
            prediction = model(batch)[-1].sigmoid().cpu()
        mask = transforms.ToPILImage()(prediction[0].squeeze())
        return mask.resize(image.size)

    return segment


def _validate(path: Path, size: tuple[int, int], color_type: int) -> None:
    with path.open("rb") as stream:
        header = stream.read(26)
    if len(header) < 26 or header[:8] != b"\x89PNG\r\n\x1a\n":
        raise SystemExit(f"{path.name} is not a PNG")
    width, height, bit_depth, kind = struct.unpack(">IIBB", header[16:26])
    if (width, height, bit_depth, kind) != (*size, 8, color_type):
        raise SystemExit(f"{path.name} is outside the output contract")


def main() -> None:
    deadline = _job_deadline()
    args = _arguments()
    if not _MODEL.is_dir():
        raise SystemExit("the pinned BiRefNet model is missing")
    image = _read_image(_input_path())
    segment = _load_model()
    _check_deadline(deadline)
    mask = segment(image)
    if mask.mode != "L" or mask.size != image.size:
        raise SystemExit("the model returned a mask outside the output contract")
    _check_deadline(deadline)
    cutout = image.copy()
    cutout.putalpha(mask)
    # The platform owns tmp/ and cache/; only our unique workspace must be fresh.
    scratch = args.output_dir / "tmp"
    scratch.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="vonk-birefnet-", dir=scratch) as staging:
        staged = Path(staging)
        cutout.save(staged / "cutout.png", format="PNG")
        mask.save(staged / "mask.png", format="PNG")
        for name, color_type in _OUTPUTS.items():
            _validate(staged / name, image.size, color_type)
        _check_deadline(deadline)
        for name in _OUTPUTS:
            os.replace(staged / name, args.output_dir / name)


if __name__ == "__main__":
    main()
