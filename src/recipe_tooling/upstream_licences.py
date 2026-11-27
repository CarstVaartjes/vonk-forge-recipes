"""Require a licence entry for every upstream an adapter vendors files from.

``upstream-licences.json`` at the repository root records, per upstream repository,
its SPDX licence and the path of its licence text, or ``unlicensed-upstream`` with
the issue that asks the creator for a licence. It lives outside every build context,
so recording or changing a licence never changes a recipe's identity or image key.

An adapter vendors from its upstream when it has a ``vendored-upstream.json``, a
source (the manifest's ``source``, its kit's locked base, or the Dockerfile labels)
and a file that is not ours (the same classification ``check-vendored-upstream``
uses); files under an ``also`` prefix vendor from that second upstream. Licences are
informative: the entry records the status, it never gates a recipe.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, StrictStr, field_validator

from recipe_tooling.adapter_sources import materialize_or_exit
from recipe_tooling.paths import REPO_ROOT
from recipe_tooling.vendored_manifest import VendoredManifest
from recipe_tooling.vendored_upstream import (
    MANIFEST,
    _local_files,
    _source,
    adapter_dirs,
)

LICENCES = "upstream-licences.json"
UNLICENSED = "unlicensed-upstream"


class _Entry(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)


class LicensedUpstream(_Entry):
    license: StrictStr = Field(pattern=r"^\S+$")
    texts: list[StrictStr] = Field(min_length=1)

    @field_validator("license")
    @classmethod
    def _not_unlicensed(cls, value: str) -> str:
        if value == UNLICENSED:
            raise ValueError(f"'{UNLICENSED}' takes a request issue, not texts")
        return value


class UnlicensedUpstream(_Entry):
    license: Literal["unlicensed-upstream"]
    request: StrictStr = Field(
        pattern=r"^https://github\.com/[^/\s]+/[^/\s]+/issues/\d+$"
    )


class UpstreamLicences(_Entry):
    upstreams: dict[StrictStr, UnlicensedUpstream | LicensedUpstream]


def vendoring_upstreams(root: Path) -> dict[str, list[str]]:
    """Upstream repository -> the adapters that vendor files from it."""
    found: dict[str, list[str]] = {}
    for adapter in adapter_dirs(root):
        path = adapter / MANIFEST
        if not path.is_file():
            continue
        manifest = VendoredManifest.model_validate_json(
            path.read_text(encoding="utf-8")
        )
        ours = manifest.ours
        rest = [
            rel
            for rel in _local_files(adapter)
            if rel not in ours
            and not any(o.endswith("/") and rel.startswith(o) for o in ours)
        ]
        name = adapter.relative_to(root / "adapters").as_posix()
        for also in manifest.also:
            taken = [rel for rel in rest if rel.startswith(tuple(also.files))]
            if taken:
                found.setdefault(also.repo, []).append(name)
            rest = [rel for rel in rest if rel not in taken]
        source = _source(adapter, manifest)
        if source is not None and rest:
            found.setdefault(source[0], []).append(name)
    return found


def problems(root: Path) -> list[str]:
    licences = UpstreamLicences.model_validate_json(
        (root / LICENCES).read_text(encoding="utf-8")
    ).upstreams
    found: list[str] = []
    for repo, adapters in sorted(vendoring_upstreams(root).items()):
        entry = licences.get(repo)
        if entry is None:
            found.append(
                f"{repo}: vendored by {', '.join(adapters)} but has no entry in {LICENCES}; "
                f"add its SPDX licence and licence text (licenses/{repo}/), or "
                f"'{UNLICENSED}' with the licence-request issue"
            )
        elif isinstance(entry, LicensedUpstream):
            found.extend(
                f"{repo}: licence text {text} is missing"
                for text in entry.texts
                if not (root / text).is_file()
            )
    return found


def main() -> int:
    materialize_or_exit(REPO_ROOT)
    found = problems(REPO_ROOT)
    for problem in found:
        print(problem, file=sys.stderr)
    if found:
        return 1
    print(
        json.dumps(
            {"upstreams_with_licence_entry": len(vendoring_upstreams(REPO_ROOT))}
        )
    )
    return 0
