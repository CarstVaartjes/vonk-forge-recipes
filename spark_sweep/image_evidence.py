"""Untrusted CI image hints cannot prevent a source-built signed publication."""

import json
from collections.abc import Callable
from pathlib import Path
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, JsonValue, TypeAdapter

from .tolerant import InputDiagnostic, read_optional

Digest = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]


class PrebuiltImage(BaseModel):
    model_config = ConfigDict(strict=True)

    reference: Annotated[
        str, Field(pattern=r"^[a-z0-9][a-z0-9./:_-]*@sha256:[0-9a-f]{64}$")
    ]
    build_key: Digest


IMAGE_MAP = TypeAdapter(dict[str, JsonValue])
DIGEST = TypeAdapter(Digest)


def read_images(
    value: object, diagnose: Callable[[InputDiagnostic], None]
) -> dict[str, dict[str, str]]:
    rows = read_optional(
        "prebuilt-images", lambda: IMAGE_MAP.validate_python(value), diagnose
    )
    images: dict[str, dict[str, str]] = {}
    for digest, row in (rows or {}).items():
        key = read_optional(
            f"prebuilt-images/{digest}",
            lambda digest=digest: DIGEST.validate_python(digest),
            diagnose,
        )
        image = read_optional(
            f"prebuilt-images/{digest}",
            lambda row=row: PrebuiltImage.model_validate(row),
            diagnose,
        )
        if image is not None and key is not None:
            images[key] = image.model_dump()
    return images


def load_images(
    path: Path, diagnose: Callable[[InputDiagnostic], None]
) -> dict[str, dict[str, str]]:
    return (
        read_optional(
            str(path),
            lambda: read_images(json.loads(path.read_text(encoding="utf-8")), diagnose),
            diagnose,
        )
        or {}
    )
