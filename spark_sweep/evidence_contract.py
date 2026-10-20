"""Independently validated optional result evidence; source bytes stay in state."""

from collections.abc import Mapping
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter

from .removal_contract import RecipeVerdict
from .tolerant import InputDiagnostic, read_optional


class ResultCore(BaseModel):
    status: RecipeVerdict = RecipeVerdict.PENDING
    content_sha256: str | None = None


class Metrics(BaseModel):
    model_config = ConfigDict(strict=True, allow_inf_nan=False)

    download_s: float | None = Field(default=None, ge=0)
    load_s: float | None = Field(default=None, ge=0)
    smoke_s: float | None = Field(default=None, ge=0)
    total_s: float | None = Field(default=None, ge=0)
    ttft_ms: float | None = Field(default=None, ge=0)
    tokens_per_second: float | None = Field(default=None, ge=0)


class QualityNote(BaseModel):
    model_config = ConfigDict(strict=True)

    case: str
    expected: str
    got: str
    ok: bool | None = None


QUALITY = TypeAdapter(list[QualityNote])


class ResultEvidence(BaseModel):
    core: ResultCore
    timings: Metrics = Field(default_factory=Metrics)
    perf: Metrics = Field(default_factory=Metrics)
    quality: list[QualityNote] = Field(default_factory=list)
    diagnostics: list[InputDiagnostic] = Field(default_factory=list)


def read_result(entry: Mapping[str, Any], source: str) -> ResultEvidence:
    result = ResultEvidence(core=ResultCore.model_validate(entry))
    for name in ("timings", "perf"):
        if entry.get(name) is not None:
            value = read_optional(
                f"{source}/{name}",
                lambda name=name: Metrics.model_validate(entry[name]),
                result.diagnostics.append,
            )
            if value is not None:
                setattr(result, name, value)
    if entry.get("quality") is not None:
        notes = read_optional(
            f"{source}/quality",
            lambda: QUALITY.validate_python(entry["quality"], strict=True),
            result.diagnostics.append,
        )
        if notes is not None:
            result.quality = notes
    return result
