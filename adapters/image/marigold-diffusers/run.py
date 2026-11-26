"""Offline Marigold job adapter: one image in, the maps of one Marigold task out.

Every task is mainline diffusers with the creator's defaults (the model's own steps,
processing resolution and output resampling; only the seed is a setting). The tasks:

- ``intrinsics-appearance``: albedo, roughness and metallic (Marigold IID Appearance).
- ``intrinsics-lighting``: albedo, shading and residual (Marigold IID Lighting).
- ``normals-depth``: a camera-space normal map and a 16-bit depth map (Marigold Normals
  and Marigold Depth, two Models run one after the other).
"""

from __future__ import annotations

import argparse
import os
import struct
import tempfile
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageOps
from vonk_agent_protocol.job_inputs import RecipeJobInputManifest

_ENTRYPOINT = "/opt/vonk/source/run.py"
_INPUT_ROOT = Path("/inputs")
_TARGET = Path("/models/target")
_NORMALS = Path("/models/normals")
_DEPTH = Path("/models/depth")
_FORMATS = {
    ".png": ("PNG", "image/png"),
    ".jpg": ("JPEG", "image/jpeg"),
    ".jpeg": ("JPEG", "image/jpeg"),
}
_UNSUPPORTED_MODES = frozenset({"F", "I", "I;16", "I;16B", "I;16L"})
_MAX_MANIFEST_BYTES = 256 * 1024
_MAX_INPUT_BYTES = 32 * 1024 * 1024
_MAX_SIDE = 4096
_MAX_SEED = 2**31 - 1
_RGB, _RGBA, _GRAY = 2, 6, 0

Render = Callable[[Any, Any], dict[str, Image.Image]]


@dataclass(frozen=True)
class Stage:
    """One pipeline run: the diffusers pipeline kind, its mounted Model and how its maps render.

    ``render`` returns the maps by file stem (``albedo`` for ``albedo.png``).
    """

    kind: str
    mount: Path
    render: Render


def _check_targets(pipeline: Any, expected: dict[str, Any]) -> dict[str, Any]:
    """The pipeline's own target properties, which must be exactly the ones we read."""
    properties = pipeline.target_properties
    found = {
        "target_names": list(properties["target_names"]),
        "sub_target_names": properties.get("material", {}).get("sub_target_names"),
    }
    if found != expected:
        raise SystemExit(
            f"the pinned Model's target properties are {found}, not the expected {expected}"
        )
    return properties


def _gray(channel: np.ndarray) -> Image.Image:
    return Image.fromarray((np.clip(channel, 0.0, 1.0) * 255).astype(np.uint8))


def _render_appearance(pipeline: Any, output: Any) -> dict[str, Image.Image]:
    properties = _check_targets(
        pipeline,
        {
            "target_names": ["albedo", "material"],
            "sub_target_names": ["roughness", "metallicity", None],
        },
    )
    shown = pipeline.image_processor.visualize_intrinsics(
        output.prediction, properties
    )[0]
    # The creator's visualisation colours the single-channel maps with a white-to-black
    # "binary" map (so rough would be dark) and applies a display gamma; a PBR roughness or
    # metallic map is the predicted value itself, so those two come from the prediction.
    material = output.prediction[1]  # [H, W, 3]: roughness, metallicity, unused
    return {
        "albedo": shown["albedo"],
        "roughness": _gray(material[..., 0]),
        "metallic": _gray(material[..., 1]),
    }


def _render_lighting(pipeline: Any, output: Any) -> dict[str, Image.Image]:
    properties = _check_targets(
        pipeline,
        {"target_names": ["albedo", "shading", "residual"], "sub_target_names": None},
    )
    shown = pipeline.image_processor.visualize_intrinsics(
        output.prediction, properties
    )[0]
    return {name: shown[name] for name in properties["target_names"]}


def _render_normals(pipeline: Any, output: Any) -> dict[str, Image.Image]:
    return {"normal": pipeline.image_processor.visualize_normals(output.prediction)[0]}


def _render_depth(pipeline: Any, output: Any) -> dict[str, Image.Image]:
    return {
        "depth": pipeline.image_processor.export_depth_to_16bit_png(output.prediction)[
            0
        ]
    }


_TASKS: dict[str, tuple[Stage, ...]] = {
    "intrinsics-appearance": (Stage("intrinsics", _TARGET, _render_appearance),),
    "intrinsics-lighting": (Stage("intrinsics", _TARGET, _render_lighting),),
    "normals-depth": (
        Stage("normals", _NORMALS, _render_normals),
        Stage("depth", _DEPTH, _render_depth),
    ),
}
# The files each task writes: (PNG bit depth, colour type); 2 is RGB, 0 is grayscale.
# Marigold's albedo takes the input's alpha, which makes it RGBA (6).
_OUTPUTS: dict[str, dict[str, tuple[int, int]]] = {
    "intrinsics-appearance": {
        "albedo.png": (8, _RGB),
        "roughness.png": (8, _GRAY),
        "metallic.png": (8, _GRAY),
    },
    "intrinsics-lighting": {
        "albedo.png": (8, _RGB),
        "shading.png": (8, _RGB),
        "residual.png": (8, _RGB),
    },
    "normals-depth": {"normal.png": (8, _RGB), "depth.png": (16, _GRAY)},
}


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--entrypoint", required=True)
    parser.add_argument("--output-mime", required=True)
    parser.add_argument("--task", required=True, choices=sorted(_TASKS))
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.entrypoint != _ENTRYPOINT:
        raise SystemExit("unexpected signed-bundle entrypoint")
    if args.output_mime != "image/png":
        raise SystemExit("Marigold emits image/png only")
    if not 0 <= args.seed <= _MAX_SEED:
        raise SystemExit(f"seed must be 0..{_MAX_SEED}")
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
        raise SystemExit("estimation exceeded the declared job budget")


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
        raise SystemExit("estimation requires exactly one image input")
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


def _read_image(path: Path) -> tuple[Image.Image, Image.Image | None]:
    """The upright RGB image the model sees and, if the input had one, its alpha.

    Transparent pixels are composited over black for the model; the alpha itself goes onto
    the albedo unchanged. A larger image is refused, never resized.
    """
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
            transparent = "A" in source.getbands() or "transparency" in source.info
            upright = ImageOps.exif_transpose(source)
            if not transparent:
                return upright.convert("RGB"), None
            rgba = upright.convert("RGBA")
    except (OSError, ValueError, Image.DecompressionBombError) as error:
        raise SystemExit(f"the image input cannot be decoded: {error}") from error
    over_black = Image.alpha_composite(Image.new("RGBA", rgba.size, (0, 0, 0, 255)), rgba)
    return over_black.convert("RGB"), rgba.getchannel("A")


def _generator(seed: int) -> Any:
    import torch

    return torch.Generator("cuda").manual_seed(seed)


def _load_pipeline(kind: str, mount: Path) -> Any:
    import torch
    from diffusers import (
        MarigoldDepthPipeline,
        MarigoldIntrinsicsPipeline,
        MarigoldNormalsPipeline,
    )

    pipelines = {
        "intrinsics": MarigoldIntrinsicsPipeline,
        "normals": MarigoldNormalsPipeline,
        "depth": MarigoldDepthPipeline,
    }
    # The creator's card path: the fp16 variant, fp16 weights, then onto the GPU.
    return pipelines[kind].from_pretrained(
        mount, variant="fp16", torch_dtype=torch.float16, local_files_only=True
    ).to("cuda")


def _run_stage(stage: Stage, image: Image.Image, seed: int) -> dict[str, Image.Image]:
    """Load, run with the model's own defaults and render one stage; the pipeline is freed."""
    pipeline = _load_pipeline(stage.kind, stage.mount)
    output = pipeline(image, generator=_generator(seed))
    return stage.render(pipeline, output)


def _validate(path: Path, size: tuple[int, int], depth_and_type: tuple[int, int]) -> None:
    with path.open("rb") as stream:
        header = stream.read(26)
    if len(header) < 26 or header[:8] != b"\x89PNG\r\n\x1a\n":
        raise SystemExit(f"{path.name} is not a PNG")
    width, height, bit_depth, kind = struct.unpack(">IIBB", header[16:26])
    if (width, height, bit_depth, kind) != (*size, *depth_and_type):
        raise SystemExit(f"{path.name} is outside the output contract")


def main() -> None:
    deadline = _job_deadline()
    args = _arguments()
    image, alpha = _read_image(_input_path())
    maps: dict[str, Image.Image] = {}
    for stage in _TASKS[args.task]:
        if not stage.mount.is_dir():
            raise SystemExit(f"the pinned Model is missing at {stage.mount}")
        _check_deadline(deadline)
        maps |= _run_stage(stage, image, args.seed)
    contract = dict(_OUTPUTS[args.task])
    maps = {f"{stem}.png": picture for stem, picture in maps.items()}
    if alpha is not None and args.task.startswith("intrinsics"):
        maps["albedo.png"] = maps["albedo.png"].convert("RGB")
        maps["albedo.png"].putalpha(alpha)
        contract["albedo.png"] = (8, _RGBA)
    if maps.keys() != contract.keys():
        raise SystemExit(f"the task produced {sorted(maps)}, not {sorted(contract)}")
    _check_deadline(deadline)
    # The platform owns tmp/ and cache/; only our unique workspace must be fresh.
    scratch = args.output_dir / "tmp"
    scratch.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="vonk-marigold-", dir=scratch) as staging:
        staged = Path(staging)
        for name, picture in maps.items():
            picture.save(staged / name, format="PNG")
            _validate(staged / name, image.size, contract[name])
        _check_deadline(deadline)
        for name in contract:
            os.replace(staged / name, args.output_dir / name)


if __name__ == "__main__":
    main()
