"""Author and heal the adapter-source mirror: a release whose assets are named by SHA-256.

``add`` is the author-side ingress: it hashes each local file once, uploads it as
asset ``<sha256>`` when the mirror lacks it, and only then writes the manifest
entry, so no entry ever points at a missing asset. Assets are never replaced or
deleted. ``reconcile`` (run on main) re-uploads any declared blob the mirror lost
from its ``origin``, verified at ingress like every download; it only reports
what it cannot heal, so it never blocks a publication.
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import subprocess
import time
from collections.abc import Callable
from pathlib import Path

from tools.adapter_sources import (
    CHUNK_BYTES,
    MANIFEST,
    ROOT,
    AdapterSource,
    AdapterSourceError,
    AdapterSources,
    Opener,
    default_cache,
    fetch,
    load,
    open_url,
)

Gh = Callable[[list[str]], str]
UPLOAD_ROUNDS = 3
_MIRROR = re.compile(
    r"^https://github\.com/(?P<repo>[^/]+/[^/]+)/releases/download/(?P<tag>[^/]+)/$"
)


def run_gh(arguments: list[str]) -> str:
    return subprocess.run(
        ["gh", *arguments], check=True, capture_output=True, text=True
    ).stdout


def _release(sources: AdapterSources) -> tuple[str, str]:
    found = _MIRROR.match(sources.mirror)
    if found is None:
        raise AdapterSourceError(
            f"mirror is not a GitHub release URL: {sources.mirror}"
        )
    return found["repo"], found["tag"]


def _assets(sources: AdapterSources, gh: Gh) -> dict[str, str | None]:
    """Asset name to its GitHub-reported digest (``sha256:...``) when it has one."""
    repo, tag = _release(sources)
    listing = gh(["release", "view", tag, "--repo", repo, "--json", "assets"])
    return {
        asset["name"]: asset.get("digest") for asset in json.loads(listing)["assets"]
    }


def _tampered(assets: dict[str, str | None]) -> list[str]:
    """Assets whose reported digest is not their name: refused, never overwritten."""
    return [
        f"mirror asset {name} reports {digest}"
        for name, digest in assets.items()
        if digest is not None and digest != f"sha256:{name}"
    ]


def _upload(sources: AdapterSources, gh: Gh, sha256: str, blob: Path) -> str:
    """Upload the verified blob (a file named by its digest); return "" or the evidence."""
    repo, tag = _release(sources)
    last = ""
    for _ in range(UPLOAD_ROUNDS):
        try:
            gh(["release", "upload", tag, str(blob), "--repo", repo])
            return ""
        except subprocess.CalledProcessError as error:
            last = (error.stderr or str(error)).strip()
            if sha256 in _assets(sources, gh):  # a concurrent run uploaded it
                return ""
    return f"upload of {sha256} failed: {last}"


def _hash(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        while chunk := handle.read(CHUNK_BYTES):
            digest.update(chunk)
            size += len(chunk)
    return digest.hexdigest(), size


def _write(root: Path, sources: AdapterSources) -> None:
    document = sources.model_dump(mode="json", exclude_none=True)
    document["sources"] = dict(sorted(document["sources"].items()))
    (root / MANIFEST).write_text(json.dumps(document, indent=2) + "\n")


def add(
    paths: list[str],
    *,
    root: Path = ROOT,
    origin: str | None = None,
    cache: Path | None = None,
    gh: Gh = run_gh,
) -> list[str]:
    """Mirror each file, then declare it; return the digests that were uploaded."""
    sources = load(root)
    cache = default_cache() if cache is None else cache
    found: dict[str, tuple[int, list[str]]] = {}
    for path in paths:
        resolved = (root / path).resolve()
        relative = resolved.relative_to(root.resolve()).as_posix()
        sha256, size = _hash(resolved)
        found.setdefault(sha256, (size, []))[1].append(relative)
        cache.mkdir(parents=True, exist_ok=True)
        if not (cache / sha256).is_file():
            shutil.copyfile(resolved, cache / sha256)
    if origin is not None and len(found) != 1:
        raise AdapterSourceError("--origin names one blob, but the files differ")
    assets = _assets(sources, gh)
    if bad := _tampered(assets):
        raise AdapterSourceError("; ".join(bad))
    uploaded = []
    for sha256 in found:
        if sha256 not in assets:
            if failure := _upload(sources, gh, sha256, cache / sha256):
                raise AdapterSourceError(failure)
            uploaded.append(sha256)
    # Every asset exists now, so the entries may be written. A path moves to the
    # newest digest declared for it.
    declared = {key: value.model_dump() for key, value in sources.sources.items()}
    for sha256, (size, held) in found.items():
        for entry in declared.values():
            entry["paths"] = tuple(p for p in entry["paths"] if p not in held)
        entry = declared.setdefault(
            sha256, {"size_bytes": size, "origin": None, "paths": []}
        )
        entry["origin"] = origin or entry["origin"]
        entry["paths"] = tuple(sorted({*entry["paths"], *held}))
    merged = {
        key: AdapterSource.model_validate(entry)
        for key, entry in declared.items()
        if entry["paths"]
    }
    _write(root, AdapterSources(mirror=sources.mirror, sources=merged))
    return uploaded


def reconcile(
    root: Path = ROOT,
    *,
    cache: Path | None = None,
    opener: Opener = open_url,
    sleep: Callable[[float], None] = time.sleep,
    gh: Gh = run_gh,
) -> list[str]:
    """Re-upload declared blobs the mirror lacks; return what could not be healed."""
    sources = load(root)
    cache = default_cache() if cache is None else cache
    try:
        assets = _assets(sources, gh)
    except (subprocess.CalledProcessError, ValueError, KeyError) as error:
        return [f"mirror listing unavailable: {error}"]
    problems = _tampered(assets)
    cache.mkdir(parents=True, exist_ok=True)
    for sha256, source in sources.sources.items():
        if sha256 in assets:
            continue
        if source.origin is None:
            problems.append(f"sha256:{sha256} is not in the mirror and has no origin")
            continue
        blob = cache / sha256
        if not (blob.is_file() and blob.stat().st_size == source.size_bytes):
            failure = fetch(
                sources, sha256, cache, opener, sleep, urls=(source.origin,)
            )
            if failure:
                problems.append(failure)
                continue
        if failure := _upload(sources, gh, sha256, blob):
            problems.append(failure)
    return problems
