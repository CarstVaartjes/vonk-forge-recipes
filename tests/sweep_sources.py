"""The source of ``Sweep``: spark_sweep/run.py and the mixin modules holding its methods."""

from __future__ import annotations

import ast
from pathlib import Path

from repo_root import ROOT

SWEEP = ROOT / "spark_sweep"


def run_paths() -> list[Path]:
    """run.py and the module of every class ``Sweep`` takes as a base (derived, never listed)."""
    run = SWEEP / "run.py"
    tree = ast.parse(run.read_text(encoding="utf-8"))
    sweep = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "Sweep"
    )
    bases = {base.id for base in sweep.bases if isinstance(base, ast.Name)}
    modules = {
        node.module
        for node in tree.body
        if isinstance(node, ast.ImportFrom)
        and node.level == 1
        and node.module
        and bases & {alias.name for alias in node.names}
    }
    return [run, *(SWEEP / f"{module}.py" for module in sorted(modules))]


def run_source() -> str:
    """The Sweep modules as one parseable text (one ``from __future__`` line is enough)."""
    return "\n".join(
        path.read_text(encoding="utf-8").replace(
            "from __future__ import annotations\n", ""
        )
        for path in run_paths()
    )


RUN_FILES = tuple(path.name for path in run_paths())
