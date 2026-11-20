"""Build-context files whose bytes live in the mirror, not in Git.

``adapter-sources.json`` at the repository root declares each such file once by
its content: the SHA-256 is the key, the mirror serves it at ``mirror + sha256``
and ``origin`` optionally names where upstream published the same bytes. One
blob may be placed at several repository paths (a vendor overlay several
adapters share). The package build places every declared path before it reads
build contexts, so packages, source bundles and recipe identities are exactly
what they would be with the bytes committed. Repository tooling reads this
file; no package, Controller or Spark reads it.
"""

from __future__ import annotations

from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, StrictInt, StrictStr, model_validator

from .recipe import RelativePath, Sha256

HttpsUrl = Annotated[StrictStr, Field(max_length=2048, pattern=r"^https://\S+$")]


class _AdapterSourcesContract(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)


class AdapterSource(_AdapterSourcesContract):
    size_bytes: StrictInt = Field(gt=0)
    origin: HttpsUrl | None = None
    paths: tuple[RelativePath, ...] = Field(min_length=1)


class AdapterSources(_AdapterSourcesContract):
    mirror: HttpsUrl
    sources: dict[Sha256, AdapterSource]

    @model_validator(mode="after")
    def one_content_per_path(self) -> AdapterSources:
        if not self.mirror.endswith("/"):
            raise ValueError("mirror must end with '/'")
        paths = [path for source in self.sources.values() for path in source.paths]
        if len(paths) != len(set(paths)):
            raise ValueError("a path is declared more than once")
        if any(".git" in path.split("/") for path in paths):
            raise ValueError("a declared path is inside .git")
        return self

    def urls(self, sha256: str) -> tuple[str, ...]:
        """Where the blob is fetched from, mirror first."""
        origin = self.sources[sha256].origin
        mirror = f"{self.mirror}{sha256}"
        return (mirror,) if origin is None else (mirror, origin)
