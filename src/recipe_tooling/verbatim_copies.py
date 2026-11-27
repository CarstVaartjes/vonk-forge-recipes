"""Which adapter files are verbatim creator copies, for the added-line guards.

Standard library only: the guards run under plain ``python3`` in CI and the pre-commit
hook. A file is a verbatim creator copy, which ``recipe_tooling.vendored_upstream``
compares byte for byte with its upstream, when it sits under an ``also`` prefix of the
nearest ``vendored-upstream.json``, or when it is not ours (``ours``, the Dockerfile,
the manifest, ``kit-lock.json``) and the adapter has a pinned upstream: a ``source``
without ``unverified`` or a locked kit base. A guard about our own code skips such a
file, so a refresh that brings it is never blocked; every other file is judged.
"""

from __future__ import annotations

import json
from pathlib import Path

MANIFEST = "vendored-upstream.json"


OWN = {MANIFEST, "Dockerfile", "kit-lock.json"}


def _pinned_upstream(adapter: Path, manifest: dict[str, object]) -> bool:
    source = manifest.get("source")
    if (
        isinstance(source, dict)
        and source.get("commit")
        and not source.get("unverified")
    ):
        return True
    lock = adapter / "kit-lock.json"
    if not lock.is_file():
        return False
    dependencies = json.loads(lock.read_text(encoding="utf-8")).get("dependencies", {})
    return dependencies.get("base") is not None


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
