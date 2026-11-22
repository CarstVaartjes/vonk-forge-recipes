"""Every repository path a guard names exists.

A path-scoped guard goes blind when the code it scopes moves and its path literal does
not: it checks nothing and passes. So each path literal in the ``tools/check-*`` scripts,
``tools/_added_lines.py``, the two ``single-source-values.json`` files and the
``LAYERS`` directories of the import-direction test must match at least one path in the
tree (a directory prefix, a file, or a glob). No allowlist: when a path moves, move its
literal.
"""

from __future__ import annotations

import ast
import fnmatch
import json
import os
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

#: First segment that makes a string a repository path. A moved directory stays listed.
TOP_LEVEL = frozenset(
    {
        ".github",
        ".githooks",
        "adapters",
        "api",
        "contracts",
        "display",
        "docs",
        "inventory",
        "models",
        "policy",
        "qualification",
        "recipes",
        "scripts",
        "spark_sweep",
        "tests",
        "tools",
    }
)
PATHLIKE = re.compile(r"^[\w.\-*/]+$")
SKIP_DIRS = {".git", "__pycache__", "node_modules", ".venv", ".pytest_cache"}
JSON_FILES = (
    "qualification/single-source-values.json",
    "tools/protocol-wheel/single-source-values.json",
)


def path_literals(strings) -> set[str]:
    """The strings that name a repository path (a plain literal with a known first segment)."""
    return {
        text
        for text in strings
        if "/" in text
        and PATHLIKE.fullmatch(text)
        and text.split("/", 1)[0] in TOP_LEVEL
    }


def python_strings(source: str) -> list[str]:
    return [
        node.value
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    ]


def shell_strings(source: str) -> list[str]:
    return re.split(r"[\s\"'=:;()]+", source)


def source_strings(source: str) -> list[str]:
    first = source.splitlines()[0] if source else ""
    return python_strings(source) if "python" in first else shell_strings(source)


def json_strings(value) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        return [text for item in value for text in json_strings(item)]
    if isinstance(value, dict):
        return [text for item in value.values() for text in json_strings(item)]
    return []


def tree_paths(root: Path) -> list[str]:
    found = []
    for directory, names, files in os.walk(root):
        names[:] = [name for name in names if name not in SKIP_DIRS]
        base = Path(directory).relative_to(root).as_posix()
        for name in [*names, *files]:
            found.append(name if base == "." else f"{base}/{name}")
    return found


def dangling(literals, paths) -> list[str]:
    """Literals matching no path: a glob matches by pattern, anything else by prefix."""
    missing = []
    for literal in sorted(literals):
        if "*" in literal:
            hit = any(fnmatch.fnmatch(path, literal) for path in paths)
        else:
            stem = literal.rstrip("/")
            hit = any(path == stem or path.startswith(stem + "/") for path in paths)
        if not hit:
            missing.append(literal)
    return missing


def guard_sources() -> dict[str, str]:
    files = sorted((ROOT / "tools").glob("check-*")) + [ROOT / "tools/_added_lines.py"]
    return {path.relative_to(ROOT).as_posix(): path.read_text() for path in files}


def layer_directories() -> list[str]:
    source = (ROOT / "tests/test_package_import_direction.py").read_text()
    for node in ast.parse(source).body:
        if isinstance(node, ast.Assign) and any(
            getattr(t, "id", None) == "LAYERS" for t in node.targets
        ):
            return [directory for directory, _ in ast.literal_eval(node.value).values()]
    return []


def test_every_path_a_guard_names_exists():
    paths = tree_paths(ROOT)
    named: dict[str, set[str]] = {}
    for name, source in guard_sources().items():
        named[name] = path_literals(source_strings(source))
    for name in JSON_FILES:
        named[name] = path_literals(json_strings(json.loads((ROOT / name).read_text())))
    named["tests/test_package_import_direction.py LAYERS"] = set(layer_directories())

    missing = {
        name: bad
        for name, literals in named.items()
        if (bad := dangling(literals, paths))
    }
    assert not missing, (
        f"guard paths that match nothing (move the literal with the code): {missing}"
    )


def test_the_scan_sees_the_guards_it_protects():
    # A scan that finds nothing would pass forever; pin that it reads the known owners.
    sources = guard_sources()
    assert "tools/_upstream_http.py" in path_literals(
        python_strings(sources["tools/check-raw-urlopen"])
    )
    assert layer_directories()
    assert path_literals(json_strings(json.loads((ROOT / JSON_FILES[0]).read_text())))


def test_a_moved_path_literal_is_reported():
    paths = ["tools/_upstream_http.py", "tests/test_a.py", "adapters/x/Dockerfile.base"]
    literals = path_literals(
        python_strings(
            'A = "tools/_upstream_http.py"\nB = "tools/gone/"\nC = "adapters/*Dockerfile*"\n'
            'D = "adapters/*.nope"\nE = "not a path/x"\nF = "tests/"\n'
        )
    )
    assert dangling(literals, paths) == ["adapters/*.nope", "tools/gone/"]
