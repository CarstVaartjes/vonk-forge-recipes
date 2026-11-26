"""Keep a held base image where it is held.

``policy/base-image-holds.json`` names recipes whose base image must not move with a refresh
(a newer engine release breaks them), with the reason and an upstream reference::

    {"<recipe id>": {"base_image": {"repository": "...", "digest": "<sha256 hex>"},
                     "version": "v0.27.1", "reason": "why", "upstream": "https://...",
                     "source": "<owner/repo>"}}

``source`` is optional: the GitHub repository whose newer releases the held image stands
for, so refresh declines those too.

The recipe's ``execution.build.base_image`` and every ``FROM`` of that repository in its
Dockerfile must carry the held digest. Moving a held recipe therefore means updating or
removing its hold in the same change. ``tools/refresh-upstream`` never moves a held image.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, RootModel, ValidationError

from recipe_tooling.paths import REPO_ROOT
from recipe_tooling.qualification import documents
from recipe_tooling.refresh_upstream import catalog_view
from vonk_forge_contracts import RecipeDefinition

ROOT = REPO_ROOT
HOLDS = "policy/base-image-holds.json"
FROM = re.compile(
    r"^[ \t]*FROM[ \t]+(?:--\S+[ \t]+)*(?P<image>\S+)", re.MULTILINE | re.IGNORECASE
)


class HeldBaseImage(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    repository: str = Field(min_length=1)
    digest: str = Field(pattern=r"^[0-9a-f]{64}$")


class BaseImageHold(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    base_image: HeldBaseImage
    version: str | None = None
    reason: str = Field(min_length=1)
    upstream: str = Field(pattern=r"^https://\S+$")
    source: str | None = Field(
        default=None, pattern=r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$"
    )


class BaseImageHolds(RootModel[dict[str, BaseImageHold]]):
    pass


def holds(root: Path) -> dict[str, BaseImageHold]:
    """The declared holds by recipe id; none when the file is absent."""
    path = root / HOLDS
    if not path.is_file():
        return {}
    return BaseImageHolds.model_validate_json(path.read_bytes()).root


def moves_held_image(hold: BaseImageHold | None, repository: str) -> bool:
    """Whether moving ``repository`` would move this hold's base image."""
    return hold is not None and hold.base_image.repository.lower() == repository.lower()


def follows_held_source(hold: BaseImageHold | None, repo: str) -> bool:
    """Whether ``repo`` is the GitHub repository whose newer releases this hold declines."""
    return (
        hold is not None
        and hold.source is not None
        and hold.source.lower() == repo.lower()
    )


def describe(hold: BaseImageHold) -> str:
    base = hold.base_image
    return (
        f"{base.repository}@sha256:{base.digest} ({hold.version or '?'}) "
        f"is held: {hold.reason} ({hold.upstream})"
    )


def followed_repositories(root: Path, recipe: RecipeDefinition) -> set[str]:
    """The GitHub repositories (lower case) the recipe follows: its source and its adapter's."""
    followed: set[str] = set()
    document = recipe.model_dump(mode="json")
    source = catalog_view.source_pin(document)
    if source is not None and source[0] == "github":
        followed.add(source[1].lower())
    pin = catalog_view.adapter_pin(root, document)
    if pin is not None:
        followed.add(pin.repo.lower())
    return followed


def problems(root: Path) -> list[str]:
    try:
        declared = holds(root)
    except ValidationError as error:
        return [f"{HOLDS}: {error}"]
    found: list[str] = []
    for recipe_id, hold in declared.items():
        where = f"{HOLDS} {recipe_id}"
        repository, digest = hold.base_image.repository, hold.base_image.digest
        path = root / "recipes" / f"{recipe_id}.json"
        if not path.is_file():
            found.append(f"{where}: no recipe {path.name}; remove the hold")
            continue
        recipe = documents.recipe(recipe_id, root)
        if hold.source is not None and hold.source.lower() not in followed_repositories(
            root, recipe
        ):
            found.append(
                f"{where}: the recipe follows no GitHub repository {hold.source}, so "
                "the hold's source declines nothing; fix or remove it"
            )
        build = recipe.execution.build
        recorded = build.base_image
        if (
            recorded.repository.lower() != repository.lower()
            or recorded.digest != digest
        ):
            found.append(
                f"{where}: the recipe's base_image is {recorded.repository}"
                f"@sha256:{recorded.digest}, but {describe(hold)}. Keep the held "
                "digest, or update/remove the hold in the same change."
            )
        dockerfile = root / build.dockerfile
        if dockerfile.is_file():
            for match in FROM.finditer(dockerfile.read_text(encoding="utf-8")):
                name, _, pinned = match["image"].partition("@")
                if ":" in name.rsplit("/", 1)[-1]:
                    name = name.rsplit(":", 1)[0]  # a tag beside the digest
                if name.lower() == repository.lower() and pinned != f"sha256:{digest}":
                    found.append(
                        f"{where}: {dockerfile.relative_to(root)} builds FROM "
                        f"{match['image']}, but {describe(hold)}"
                    )
    return found


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    arguments = parser.parse_args(argv)
    found = problems(arguments.root)
    for problem in found:
        print(problem, file=sys.stderr)
    if found:
        return 1
    print("Every held base image is where its hold says.")
    return 0
