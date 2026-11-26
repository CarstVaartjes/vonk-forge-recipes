"""Keep every Hugging Face revision a Dockerfile fetches under a declaration.

A Dockerfile that downloads ``https://huggingface.co/<repo>/resolve/<revision>/...`` at
build time pins that revision by hand; nothing else tells the next refresh the pin
exists. The adapter must declare the same ``repo`` and ``revision`` in one of

* ``vendored-upstream.json`` ``"huggingface": [{"repo": "<owner>/<name>", "revision": "<40 hex>"}]``
  (a build-time fetch of a file the recipe does not serve as a Model), or
* ``kit-lock.json`` (a kit dependency with an ``hf-revision`` datasource, which the kit
  check keeps current).

A revision given as ``${NAME}`` resolves through the Dockerfile's ``ARG NAME=<default>``;
one with no default is read from the kit lock at build time and is already watched there.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

from recipe_tooling.paths import REPO_ROOT
from recipe_tooling.vendored_upstream import KIT_LOCK, MANIFEST, adapter_dirs

ROOT = REPO_ROOT
FETCH = re.compile(
    r"huggingface\.co/(?P<repo>[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+)/resolve/"
    r"(?P<revision>\$\{?\w+\}?|[A-Za-z0-9_.-]+)"
)
ARG = re.compile(r"^\s*ARG\s+(\w+)=(\S+)", re.MULTILINE | re.IGNORECASE)


def declared(adapter: Path) -> set[tuple[str, str]]:
    """The ``(repo, revision)`` pairs (lower case) the adapter declares."""
    pairs: set[tuple[str, str]] = set()
    manifest = adapter / MANIFEST
    if manifest.is_file():
        for entry in json.loads(manifest.read_text(encoding="utf-8")).get(
            "huggingface", []
        ):
            pairs.add((entry["repo"].lower(), entry["revision"].lower()))
    lock = adapter / KIT_LOCK
    if lock.is_file():
        dependencies = json.loads(lock.read_text(encoding="utf-8")).get(
            "dependencies", {}
        )
        for entry in dependencies.values():
            if entry.get("datasource") == "hf-revision":
                pairs.add((entry["repo"].lower(), entry["revision"].lower()))
    return pairs


def problems(root: Path) -> list[str]:
    found: list[str] = []
    for adapter in adapter_dirs(root):
        known = declared(adapter)
        for dockerfile in sorted(adapter.rglob("Dockerfile")):
            text = dockerfile.read_text(encoding="utf-8", errors="replace")
            defaults = dict(ARG.findall(text))
            for match in FETCH.finditer(text):
                revision = match["revision"]
                if revision.startswith("$"):
                    revision = defaults.get(revision.strip("${}"), "")
                    if not revision:
                        continue  # read from the kit lock at build time
                if (match["repo"].lower(), revision.lower()) not in known:
                    found.append(
                        f"{dockerfile.relative_to(root)}: fetches huggingface.co/"
                        f"{match['repo']} at {revision}, which no {MANIFEST} "
                        f'"huggingface" entry or {KIT_LOCK} hf-revision declares'
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
    print("Every Hugging Face revision a Dockerfile fetches is declared.")
    return 0
