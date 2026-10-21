"""Resolve producer-only bucket inputs into ordinary metadata assertions."""

from __future__ import annotations

import base64
import hashlib
import struct
from enum import StrEnum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from qualification.native_buckets import (
    calculate_dimensions,
    generate_crop_size_list,
    get_closest_ratio,
)


class NativeBucketKind(StrEnum):
    HUNYUAN_VIDEO_15 = "hunyuan-video-15"
    QWEN_IMAGE = "qwen-image"


class NativeDimensions(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: NativeBucketKind
    target_size: int = Field(gt=0)
    aspect_ratio: tuple[int, int] | None = None


def resolve_native_dimensions(
    case: dict[str, Any], fixtures: dict[str, Any], root: Path
) -> None:
    for assertion in case.get("assertions", []):
        if "native_dimensions" not in assertion:
            continue
        if "width" in assertion or "height" in assertion:
            raise ValueError("native dimensions cannot also hard-code width/height")
        native = NativeDimensions.model_validate(assertion.pop("native_dimensions"))
        if native.aspect_ratio is not None:
            width, height = native.aspect_ratio
        else:
            # Qwen adapters sort reference files by name before selecting the
            # first image; Hunyuan accepts exactly one image.
            image_ids = [i["fixture"] for i in case["inputs"] if i["slot"] == "image"]
            fixture_id = min(image_ids, key=lambda key: fixtures[key]["name"])
            fixture = fixtures[fixture_id]
            raw = (root / fixture["path"]).read_bytes()
            content = base64.b64decode(raw) if fixture["encoding"] == "base64" else raw
            if hashlib.sha256(content).hexdigest() != fixture["sha256"]:
                raise ValueError(
                    f"fixture {fixture_id} does not match its recorded digest"
                )
            # Qualification image fixtures are PNGs; their IHDR dimensions do
            # not require an image decoder in the catalog build environment.
            if (
                len(content) < 33
                or content[:16] != b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR"
            ):
                raise ValueError(f"fixture {fixture_id} must have a PNG IHDR header")
            width, height = struct.unpack(">II", content[16:24])
            if not width or not height:
                raise ValueError(f"fixture {fixture_id} must have positive dimensions")
        if native.kind == NativeBucketKind.HUNYUAN_VIDEO_15:
            buckets = generate_crop_size_list(
                base_size=native.target_size, patch_size=16
            )
            ratios = [round(float(h) / float(w), 5) for h, w in buckets]
            height, width = get_closest_ratio(height, width, ratios, buckets)[0]
        else:
            width, height = calculate_dimensions(native.target_size**2, width / height)
        assertion.update(width=width, height=height)
