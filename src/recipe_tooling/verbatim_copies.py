"""Which adapter files are verbatim creator copies, for the added-line guards.

Standard library only: the guards run under plain ``python3`` in CI and the pre-commit
hook. A file the nearest ``vendored-upstream.json`` does not list under ``ours`` is a
byte-identical creator copy that ``recipe_tooling.vendored_upstream`` compares with its
upstream, so a guard about our own code skips it and a refresh that brings it is never
blocked.
"""

from __future__ import annotations

import json
from pathlib import Path

MANIFEST = "vendored-upstream.json"


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
        if not set(manifest) - {"kit"}:
            return False  # a kit-only manifest claims no vendored files
        relative = path.relative_to(adapter).as_posix()
        ours = manifest.get("ours", [])
        return not any(
            relative == entry or (entry.endswith("/") and relative.startswith(entry))
            for entry in ours
        )
    return False
