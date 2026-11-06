"""Offline job adapter for Ubisoft La Forge's CHORD image-to-PBR-maps model."""

from __future__ import annotations

import argparse
import json
import os
import signal
import struct
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from PIL import Image

_WORKER = "/opt/vonk/source/pipelines/infer.py"
_INPUT_ROOT = Path("/inputs")
_MAX_MANIFEST_BYTES = 256 * 1024
_MAX_IMAGE_BYTES = 16 * 1024 * 1024
_MAX_PIXELS = 16 * 1024 * 1024
_FORMATS = {".png": "PNG", ".jpg": "JPEG", ".jpeg": "JPEG"}
# The four maps the creator's save_maps writes, with each PNG colour type:
# 2 is RGB, 0 is 8-bit grayscale (roughness and metalness are single channel).
_MAPS = {"basecolor.png": 2, "normal.png": 2, "roughness.png": 0, "metalness.png": 0}


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-mime", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_mime != "image/png":
        raise SystemExit("CHORD emits image/png only")
    return args


def _image_file() -> tuple[Path, tuple[int, int]]:
    manifest = _INPUT_ROOT / "manifest.json"
    if (
        manifest.is_symlink()
        or not manifest.is_file()
        or manifest.stat().st_size > _MAX_MANIFEST_BYTES
    ):
        raise SystemExit("missing or invalid signed input manifest")
    try:
        document = json.loads(manifest.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise SystemExit("signed input manifest must be UTF-8 JSON") from error
    files = document.get("files") if isinstance(document, dict) else None
    if not isinstance(files, list):
        raise SystemExit("signed input manifest files are invalid")
    names: set[str] = set()
    images: list[Path] = []
    for item in files:
        name = item.get("name") if isinstance(item, dict) else None
        if not isinstance(name, str) or Path(name).name != name:
            raise SystemExit("signed input manifest contains an unsafe name")
        names.add(name)
        if item.get("slot") == "image":
            images.append(_INPUT_ROOT / name)
    if {path.name for path in _INPUT_ROOT.iterdir()} != names | {"manifest.json"}:
        raise SystemExit("undeclared job input")
    if len(images) != 1 or images[0].suffix.lower() not in _FORMATS:
        raise SystemExit("PBR estimation requires one PNG or JPEG image")
    image = images[0]
    if (
        image.is_symlink()
        or not image.is_file()
        or not 1 <= image.stat().st_size <= _MAX_IMAGE_BYTES
    ):
        raise SystemExit("the image must be a regular file of 1..16 MiB")
    try:
        with Image.open(image) as header:
            size = header.size
            kind = header.format
    except (OSError, Image.DecompressionBombError) as error:
        raise SystemExit("the image is not a readable PNG or JPEG") from error
    if kind != _FORMATS[image.suffix.lower()]:
        raise SystemExit("the image content does not match its file extension")
    if size[0] * size[1] > _MAX_PIXELS:
        raise SystemExit("the image exceeds 16 megapixels")
    return image, size


def _job_deadline() -> float:
    try:
        seconds = int(os.environ["VONK_JOB_TIMEOUT_SECONDS"])
    except (KeyError, ValueError) as error:
        raise SystemExit("job must declare VONK_JOB_TIMEOUT_SECONDS") from error
    if not 1 <= seconds <= 3600:
        raise SystemExit("job timeout must be between 1 and 3600 seconds")
    return time.monotonic() + seconds


def _remaining(deadline: float) -> float:
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise SystemExit("PBR estimation exceeded the declared job budget")
    return remaining


def _estimate(command: list[str], scratch: Path, deadline: float) -> None:
    # Own session: expiry must stop the worker and anything it started.
    with subprocess.Popen(
        command, cwd=scratch, env=dict(os.environ), start_new_session=True
    ) as process:
        try:
            code = process.wait(timeout=_remaining(deadline))
        except (subprocess.TimeoutExpired, SystemExit) as error:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
            raise SystemExit(
                "PBR estimation exceeded the declared job budget"
            ) from error
        if code:
            raise SystemExit(f"PBR estimation failed with exit code {code}")


def _validate_maps(directory: Path, size: tuple[int, int]) -> None:
    if {path.name for path in directory.iterdir()} != _MAPS.keys():
        raise SystemExit("PBR estimation did not produce exactly the four maps")
    for name, color_type in _MAPS.items():
        with (directory / name).open("rb") as stream:
            header = stream.read(26)
        if len(header) < 26 or header[:8] != b"\x89PNG\r\n\x1a\n":
            raise SystemExit(f"{name} is not a PNG")
        width, height, bit_depth, kind = struct.unpack(">IIBB", header[16:26])
        if (width, height, bit_depth, kind) != (*size, 8, color_type):
            raise SystemExit(f"{name} is outside the output contract")


def main() -> None:
    deadline = _job_deadline()
    args = _arguments()
    image, size = _image_file()
    # The platform owns tmp/ and cache/ under /outputs; only our workspace is fresh.
    scratch_root = args.output_dir / "tmp"
    scratch_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="vonk-chord-", dir=scratch_root) as scratch:
        staged = Path(scratch) / "maps"
        _estimate(
            [sys.executable, _WORKER, "--image", str(image), "--output-dir", str(staged)],
            Path(scratch),
            deadline,
        )
        _validate_maps(staged, size)
        _remaining(deadline)
        for name in _MAPS:
            os.replace(staged / name, args.output_dir / name)


if __name__ == "__main__":
    main()
