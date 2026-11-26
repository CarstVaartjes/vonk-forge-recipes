"""The contract of an adapter's ``vendored-upstream.json``.

One typed shape for the manifest ``recipe_tooling.vendored_upstream`` checks: unknown
keys are refused, so a misspelt ``ours`` or ``renamed`` cannot silently classify nothing.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, StrictBool, StrictStr


class _Manifest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)


class ManifestSource(_Manifest):
    repo: StrictStr
    commit: StrictStr
    line: StrictStr | None = None  # the release line the refresh follows
    unverified: StrictStr | None = None  # the commit is not served by GitHub


class ManifestAlso(_Manifest):
    """A second upstream and the path prefixes taken from it."""

    repo: StrictStr = Field(min_length=1)
    commit: StrictStr = Field(pattern=r"^[0-9a-f]{40}$")
    files: list[StrictStr] = Field(min_length=1)


class ManifestDependency(_Manifest):
    """Where the kit declares one dependency (``tools/kit-pins``)."""

    datasource: StrictStr
    file: StrictStr
    pattern: StrictStr
    repo: StrictStr | None = None
    repo_pattern: StrictStr | None = None
    binds: StrictStr | None = None
    override: dict[StrictStr, StrictStr] | None = None


class ManifestKit(_Manifest):
    dependencies: dict[StrictStr, ManifestDependency]
    whole: StrictBool | None = None


class VendoredManifest(_Manifest):
    source: ManifestSource | None = None
    kit: ManifestKit | None = None
    ours: list[StrictStr] = []
    renamed: dict[StrictStr, StrictStr] = {}
    renamed_dirs: dict[StrictStr, StrictStr] = {}
    also: list[ManifestAlso] = []
    reason: dict[StrictStr, StrictStr] = {}
