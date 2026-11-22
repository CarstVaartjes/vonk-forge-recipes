"""Scanning archives and Dockerfiles for image references."""

from __future__ import annotations

import io
import json
import re
import tarfile
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Iterable
from pathlib import Path
from typing import Any, cast

from recipe_tooling.kit_pins.archive import archive_url
from recipe_tooling.kit_pins.consumers import FROM_LINE, split_image
from recipe_tooling.kit_pins.lock import adapter_of
from recipe_tooling.kit_pins.model import FETCH_ERRORS, MANIFEST, KitError
from recipe_tooling.kit_pins.upstream import Upstream
from recipe_tooling.upstream_http import fetch as fetch_url

# ---------------------------------------------------------------------------
# Prebuilt images a kit references (the recipe runs them, or records why not)
# ---------------------------------------------------------------------------

EXCEPTIONS = "policy/kit-image-exceptions.json"
MAX_SCAN_FILE = 400_000
SCANNED_NAME = re.compile(
    r"(?i)(^|/)(dockerfile[^/]*|docker-compose[^/]*|compose[^/]*\.ya?ml|makefile|"
    r"[^/]*\.(sh|bash|ya?ml|env|example|conf|toml|md))$"
)
UNSCANNED_DIRS = {
    ".github",
    "docs",
    "doc",
    "test",
    "tests",
    "evidence",
    "bench",
    "benchmarks",
    "results",
    "node_modules",
}
# Engines and bases every recipe may build on (the engines' own projects, NVIDIA's NGC
# containers, base operating systems): not a creator's prebuilt image.
GENERIC_IMAGE = re.compile(
    r"^(docker\.io/(library|nvidia|lmsysorg|pytorch|rocm|vllm|grafana|prom|otel)/|nvcr\.io/|"
    r"docker\.io/(ubuntu|debian|python|alpine|local)/?)"
)
_IMAGE = re.compile(
    r"^(?P<name>(?:(?:ghcr\.io|docker\.io|quay\.io|public\.ecr\.aws)/)?"
    r"[a-z0-9][a-z0-9._-]*(?:/[a-z0-9][a-z0-9._-]*)+)"
    r"(?::(?P<tag>[A-Za-z0-9_][\w.-]*))?(?:@(?P<digest>sha256:[0-9a-f]{64}))?$",
    re.IGNORECASE,
)
_REGISTRY_PREFIX = re.compile(r"(?i)^(ghcr\.io|docker\.io|quay\.io|public\.ecr\.aws)/")
_FROM = re.compile(r"(?i)^\s*FROM\s+(?:--\S+\s+)*(?P<t>\S+)")
_ARG = re.compile(
    r"(?i)^\s*ARG\s+\w*(?:IMAGE|IMG|BASE)\w*\s*=\s*[\"']?(?P<t>[^\"'\s]+)"
)
_ASSIGN = re.compile(
    r"(?i)(?:^|[\s;])(?:export\s+)?\w*(?:IMAGE|IMG)(?:_(?:NAME|SOURCE|REF|REPO|URL|TAG|DEFAULT))?"
    r"\s*[:=]\s*[\"']?"
    r"(?:\$\{\w+:?-)?(?P<t>[^\"'\s}]+)"
)
_DOCKER = re.compile(r"(?i)\bdocker(?:\s+[\w-]+)*?\s+(?:run|pull|create)\b(?P<rest>.*)")
_NOT_AN_IMAGE_SUFFIX = re.compile(
    r"(?i)\.(sh|py|md|json|ya?ml|txt|toml|conf|jinja|patch|diff|tar|gz|safetensors|gguf)$"
)


def image_name(name: str) -> str:
    """The canonical repository of an image name: a registry host, lower case."""
    lowered = name.lower()
    first = lowered.split("/", 1)[0]
    return lowered if "." in first or ":" in first else f"docker.io/{lowered}"


def _candidates(line: str, markdown: bool) -> Iterable[str]:
    """The tokens of a line that sit where an image reference does."""
    if not markdown:
        for lead in (_FROM, _ARG):
            if (match := lead.match(line)) is not None:
                yield match["t"]
        for match in _ASSIGN.finditer(line):
            yield match["t"]
    docker = _DOCKER.search(line)
    if docker is not None:
        previous = ""
        for token in docker["rest"].split():
            value = token.strip("\"'")
            if not token.startswith("-") and (
                not previous.startswith("-") or _REGISTRY_PREFIX.match(value)
            ):
                yield value
            previous = token


def scan_text(path: str, text: str, scope: str = "") -> list[dict[str, Any]]:
    """The images one kit file runs or pulls (``repo``, ``reference``, ``file``, ``line``)."""
    parts = path.split("/")
    if scope and not path.startswith(scope):
        return []
    if any(part.lower() in UNSCANNED_DIRS for part in parts[:-1]):
        return []
    if not SCANNED_NAME.search(path):
        return []
    markdown = path.lower().endswith(".md")
    found: list[dict[str, Any]] = []
    for number, line in enumerate(text.splitlines(), 1):
        if not markdown and line.lstrip().startswith("#"):
            continue
        for token in _candidates(line, markdown):
            match = _IMAGE.match(token)
            if match is None or "$" in token:
                continue
            name = match["name"]
            if _NOT_AN_IMAGE_SUFFIX.search(name) or name.lower().startswith("local/"):
                continue
            if not _REGISTRY_PREFIX.match(name) and not (
                match["tag"] or match["digest"]
            ):
                continue  # a bare owner/name is more likely a path or a model id
            found.append(
                {
                    "repo": image_name(name),
                    "reference": token,
                    "file": path,
                    "line": number,
                }
            )
    return found


def scan_archive(data: bytes, scope: str = "") -> list[dict[str, Any]]:
    """Every creator image a kit archive references, once per repository."""
    return scan_stream(io.BytesIO(data), scope)


def scan_stream(stream: Any, scope: str = "") -> list[dict[str, Any]]:
    """``scan_archive`` over a gzip tar stream, read once and never held whole."""
    first: dict[str, dict[str, Any]] = {}
    with tarfile.open(fileobj=stream, mode="r|gz") as archive:
        for member in archive:
            cast(
                Any, archive
            ).members.clear()  # typeshed omits it; stream mode keeps every member
            if not member.isfile() or member.size > MAX_SCAN_FILE:
                continue
            path = member.name.split("/", 1)[1] if "/" in member.name else member.name
            if not SCANNED_NAME.search(path) or (scope and not path.startswith(scope)):
                continue
            handle = archive.extractfile(member)
            if handle is None:
                continue
            for found in scan_text(
                path, handle.read().decode("utf-8", "replace"), scope
            ):
                if not GENERIC_IMAGE.match(found["repo"]):
                    first.setdefault(found["repo"], found)
    return [first[repo] for repo in sorted(first)]


def scan_download(repo: str, commit: str, scope: str = "") -> list[dict[str, Any]]:
    """Scan a commit's archive as it downloads: a large monorepo needs no size cap."""
    url = archive_url(repo, commit)
    request = urllib.request.Request(url, headers={"User-Agent": "vonk-forge-recipes"})
    try:
        return fetch_url(
            request, lambda response: scan_stream(response, scope), timeout=120
        )
    except (*FETCH_ERRORS, tarfile.TarError) as error:
        raise KitError(f"cannot scan {url}: {error}") from error


def kit_images(
    source: Upstream, repo: str, commit: str, scope: str = ""
) -> list[dict[str, Any]]:
    """The creator images the kit references at the commit (under ``scope``, if any)."""
    if source.images is None:
        raise KitError("no kit image scan is configured")
    return source.images(repo, commit, scope)


def kit_scope(recipe: dict[str, Any]) -> str:
    """The directory of a monorepo kit: where the recipe's citation of a file points."""
    reference = str(recipe.get("provenance", {}).get("source_reference", ""))
    match = re.match(
        r"^https://github\.com/[^/]+/[^/]+/blob/[0-9a-f]{40}/(?P<dir>.+)/[^/]+$",
        reference,
    )
    return f"{match['dir']}/" if match else ""


def dockerfile_images(adapter: Path, recipe: dict[str, Any]) -> set[str]:
    """The image repositories the recipe's Dockerfile builds FROM (and its recorded base)."""
    dockerfile = adapter / "Dockerfile"
    text = dockerfile.read_text(encoding="utf-8") if dockerfile.is_file() else ""
    used = {
        image_name(split_image(m["image"])[0])
        for m in FROM_LINE.finditer(text)
        if "$" not in m["image"]
    }
    base = recipe.get("execution", {}).get("build", {}).get("base_image") or {}
    if base.get("repository"):
        used.add(image_name(str(base["repository"])))
    return used


def read_exceptions(root: Path) -> dict[str, str]:
    path = root / EXCEPTIONS
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def image_problems(
    root: Path,
    recipe_id: str,
    recipe: dict[str, Any],
    referenced: list[dict[str, Any]],
    exceptions: dict[str, str],
) -> list[str]:
    """Why a recipe neither runs a prebuilt image its kit references nor records why not."""
    adapter = adapter_of(root, recipe)
    used = dockerfile_images(adapter, recipe) if adapter is not None else set()
    reason = exceptions.get(recipe_id)
    if reason is not None and not str(reason).strip():
        return [f"{recipe_id}: the {EXCEPTIONS} entry has no reason"]
    if not referenced or any(found["repo"] in used for found in referenced) or reason:
        return []
    listed = "; ".join(
        f"{found['reference']} ({found['file']}:{found['line']})"
        for found in referenced[:6]
    )
    more = " and more" if len(referenced) > 6 else ""
    return [
        (
            f"{recipe_id}: the kit provides prebuilt image(s) {listed}{more}, but the "
            f"recipe does not build FROM one: run it by digest (declare it as an "
            f"oci-image dependency in {MANIFEST}, then tools/kit-pins lock {recipe_id}), "
            f"or record why not in {EXCEPTIONS}"
        )
    ]


def unneeded_exception(
    recipe_id: str,
    recipe: dict[str, Any],
    root: Path,
    referenced: list[dict[str, Any]],
    exceptions: dict[str, str],
) -> str | None:
    """A recorded exception the recipe no longer needs (it runs a kit image, or none exists)."""
    if recipe_id not in exceptions:
        return None
    adapter = adapter_of(root, recipe)
    used = dockerfile_images(adapter, recipe) if adapter is not None else set()
    if not referenced:
        return f"{recipe_id}: its kit references no prebuilt image"
    runs = [found["repo"] for found in referenced if found["repo"] in used]
    return f"{recipe_id}: it runs the kit's image {runs[0]}" if runs else None
