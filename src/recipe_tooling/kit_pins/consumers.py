"""What follows a locked kit: base archive, model and image."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

from recipe_tooling.kit_pins.model import PUBLISHED_IMAGES, Declaration, Pin

# ---------------------------------------------------------------------------
# Consumers: what follows a pin
# ---------------------------------------------------------------------------

ARCHIVE_NAME = re.compile(r"^(?P<stem>.+)-(?P<commit>[0-9a-f]{40})\.tar\.gz$")
ARCHIVE_DIGEST = re.compile(
    r'echo "(?P<digest>[0-9a-f]{64})  [^"]+"\s*\|\s*sha256sum -c'
)
FROM_LINE = re.compile(
    r"^(?P<lead>[ \t]*FROM[ \t]+(?:--\S+[ \t]+)*)(?P<image>\S+)",
    re.MULTILINE | re.IGNORECASE,
)


def base_consumers(adapter: Path, pin: Pin) -> list[str]:
    """What the files that consume the base pin say; empty when they follow it."""
    problems: list[str] = []
    dockerfile = adapter / "Dockerfile"
    text = dockerfile.read_text(encoding="utf-8") if dockerfile.is_file() else ""
    archives = (
        sorted((adapter / "vendor").glob("*.tar.gz"))
        if (adapter / "vendor").is_dir()
        else []
    )
    if archives:
        named = [
            a
            for a in archives
            if (m := ARCHIVE_NAME.match(a.name)) and m["commit"] == pin.immutable
        ]
        if not named:
            problems.append(
                f"vendor/ holds {', '.join(a.name for a in archives)}, not an archive of the "
                f"locked {pin.repo} commit {pin.immutable}"
            )
        elif len(archives) > 1:
            problems.append(
                "vendor/ holds more than one archive; keep only the locked one"
            )
        else:
            digest = hashlib.sha256(named[0].read_bytes()).hexdigest()
            found = ARCHIVE_DIGEST.search(text)
            if found is None or found["digest"] != digest:
                problems.append(
                    f"Dockerfile does not verify {named[0].name} with its sha256 {digest}"
                )
            if f"vendor/{named[0].name}" not in text:
                problems.append(f"Dockerfile does not COPY vendor/{named[0].name}")
    revision = re.search(r'org\.opencontainers\.image\.revision="([^"]+)"', text)
    if revision is not None and revision.group(1) != pin.immutable:
        problems.append(
            f"Dockerfile's image revision label is {revision.group(1)}, not the locked "
            f"commit {pin.immutable}"
        )
    elif pin.immutable not in text:
        problems.append(f"Dockerfile does not name the locked commit {pin.immutable}")
    return problems


def _document(path: Path) -> dict[str, Any] | None:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def bound_model(
    root: Path, recipe: dict[str, Any], model_id: str
) -> tuple[dict[str, Any], dict[str, Any]] | None:
    """``(the recipe's model reference, its catalogued Model document)`` for a ``models[].id``."""
    for reference in recipe.get("models", []):
        if reference.get("id") == model_id:
            document = _document(root / "models" / f"{reference['model']['slug']}.json")
            return None if document is None else (reference, document)
    return None


def model_matches(document: dict[str, Any], repo: str, revision: str) -> bool:
    source = document.get("source") or {}
    return (
        str(source.get("repository", "")).lower()
        == f"https://huggingface.co/{repo}".lower()
        and source.get("revision") == revision
    )


def model_consumers(
    root: Path, recipe: dict[str, Any], declaration: Declaration, pin: Pin
) -> list[str]:
    """Why the bound Model is not the repository and revision the kit locks."""
    assert declaration.binds is not None
    found = bound_model(root, recipe, declaration.binds)
    if found is None:
        return [
            (
                f"the recipe selects no Model '{declaration.binds}' (or it is not "
                f"catalogued) for the kit's {declaration.name} "
                f"({pin.repo}@{pin.immutable[:12]})"
            )
        ]
    reference, document = found
    if model_matches(document, pin.repo, pin.immutable):
        return []
    source = document.get("source") or {}
    was = str(source.get("repository", "?")).removeprefix("https://huggingface.co/")
    return [
        (
            f"the recipe's Model '{declaration.binds}' ({reference['model']['slug']}) is "
            f"{was}@{str(source.get('revision', '?'))[:12]}, but the kit locks "
            f"{pin.repo}@{pin.immutable[:12]}; catalogue that revision "
            f"(tools/catalog-hf-model --repository {pin.repo} --revision {pin.immutable}) "
            "and run tools/kit-pins lock"
        )
    ]


def split_image(image: str) -> tuple[str, str | None, str | None]:
    """``(repository, tag, digest)`` of an image reference."""
    name, _, digest = image.partition("@")
    colon, slash = name.rfind(":"), name.rfind("/")
    if colon > slash:
        return name[:colon], name[colon + 1 :], digest or None
    return name, None, digest or None


def image_reference(pin: Pin, written: str) -> str:
    """The image reference of a pin, in the form ``written`` has it.

    A reference that names a tag keeps naming it (the locked tag); a digest-only
    reference stays digest-only.
    """
    repo, tag, _ = split_image(written)
    if tag is not None and pin.kind == "tag":
        return f"{repo}:{pin.ref}@{pin.immutable}"
    return f"{repo}@{pin.immutable}"


def followed_image(
    root: Path, adapter: Path, source: str
) -> tuple[str, str, str] | None:
    """``(image repository, tag, pinned digest)`` when the adapter runs ``source``'s image.

    ``policy/published-images.json`` names the images a GitHub repository publishes and the tag
    it publishes as current. An adapter whose Dockerfile builds ``FROM`` such an image,
    for a recipe whose source is that repository, runs what the image is: the image is
    the pin, and the source commit is the one the image records (``Registry.provenance``).
    """
    path = root / PUBLISHED_IMAGES
    dockerfile = adapter / "Dockerfile"
    if not path.is_file() or not dockerfile.is_file():
        return None
    images = json.loads(path.read_text(encoding="utf-8"))["images"]
    for match in FROM_LINE.finditer(dockerfile.read_text(encoding="utf-8")):
        repo, _, digest = split_image(match["image"])
        entry = images.get(repo.lower())
        if digest and entry and entry["source"].lower() == source.lower():
            return repo, entry["tag"], digest
    return None


def image_consumers(adapter: Path, recipe: dict[str, Any], pin: Pin) -> list[str]:
    """Why the Dockerfile and the recipe's base image are not the locked image."""
    problems: list[str] = []
    dockerfile = adapter / "Dockerfile"
    text = dockerfile.read_text(encoding="utf-8") if dockerfile.is_file() else ""
    named = [
        m["image"]
        for m in FROM_LINE.finditer(text)
        if "$" not in m["image"]
        and split_image(m["image"])[0].lower() == pin.repo.lower()
    ]
    if not named:
        problems.append(f"Dockerfile has no FROM of the kit's image {pin.repo}")
    problems.extend(
        f"Dockerfile's FROM {image} is not the kit's locked image "
        f"{image_reference(pin, image)}"
        for image in named
        if image != image_reference(pin, image)
    )
    base = recipe.get("execution", {}).get("build", {}).get("base_image") or {}
    if str(base.get("repository", "")).lower() == pin.repo.lower():
        locked = pin.immutable.removeprefix("sha256:")
        if base.get("digest") != locked:
            problems.append(
                f"the recipe's base_image digest is {base.get('digest')}, not the locked {locked}"
            )
    return problems


def consumers(
    root: Path,
    adapter: Path,
    recipe: dict[str, Any],
    declaration: Declaration,
    pin: Pin,
) -> list[str]:
    """What consumes a locked pin and does not follow it, for each datasource."""
    if pin.datasource == "hf-revision":
        return model_consumers(root, recipe, declaration, pin)
    if pin.datasource == "oci-image":
        return image_consumers(adapter, recipe, pin)
    return base_consumers(adapter, pin) if pin.name == "base" else []
