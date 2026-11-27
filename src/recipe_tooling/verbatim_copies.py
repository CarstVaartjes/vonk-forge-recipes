"""Which adapter files are verbatim creator copies, for the added-line guards.

Standard library only: the guards run under plain ``python3`` in CI and the pre-commit
hook. A file is a verbatim creator copy, which ``recipe_tooling.vendored_upstream``
compares byte for byte with its upstream, when it sits under an ``also`` prefix of the
nearest ``vendored-upstream.json``, or when it is not ours (``ours``, the Dockerfile,
the manifest, ``kit-lock.json``) and the adapter has a pinned upstream: a ``source``
without ``unverified``, a locked kit base, or GitHub source and 40-hex revision
Dockerfile labels (``recipe_tooling.image_labels``). A guard about our own code skips such a
file, so a refresh that brings it is never blocked; every other file is judged.
"""

from __future__ import annotations

import json
from pathlib import Path

from recipe_tooling.image_labels import label_source

MANIFEST = "vendored-upstream.json"


OWN = {MANIFEST, "Dockerfile", "kit-lock.json"}


def _pinned_upstream(adapter: Path, manifest: dict[str, object]) -> bool:
    """The same pin ``vendored_upstream`` compares against, without an ``unverified`` mark."""
    source = manifest.get("source")
    if isinstance(source, dict):
        return bool(source.get("commit")) and not source.get("unverified")
    lock = adapter / "kit-lock.json"
    if lock.is_file():
        dependencies = json.loads(lock.read_text(encoding="utf-8")).get(
            "dependencies", {}
        )
        if dependencies.get("base") is not None:
            return True
    dockerfile = adapter / "Dockerfile"
    return (
        dockerfile.is_file()
        and label_source(dockerfile.read_text(encoding="utf-8", errors="replace"))
        is not None
    )


def verbatim_copy(root: Path, name: str) -> bool:
    """Whether ``name`` (relative to ``root``) is a verbatim creator copy."""
    path = Path(name)
    for adapter in path.parents:
        if adapter == Path("adapters") or adapter == Path():
            return False
        manifest_path = root / adapter / MANIFEST
        if not manifest_path.is_file():
            continue
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        relative = path.relative_to(adapter).as_posix()
        if any(
            relative.startswith(tuple(also.get("files", [])))
            for also in manifest.get("also", [])
        ):
            return True
        ours = manifest.get("ours", [])
        if relative in OWN or any(
            relative == entry or (entry.endswith("/") and relative.startswith(entry))
            for entry in ours
        ):
            return False
        return _pinned_upstream(root / adapter, manifest)
    return False
