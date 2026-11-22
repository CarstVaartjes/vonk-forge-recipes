"""Every repository path a guard names exists.

A path-scoped guard goes blind when the code it scopes moves and its path literal does
not: it checks nothing and passes. So each path scope must match at least one path in the
tree (a directory prefix, a file, or a glob). Sources: slash literals, ``ROOT / "a" / "b"``
chains and bare top-level names in the ``tools/check-*`` scripts and ``tools/_added_lines.py``;
the composed paths and bare names of four test-side scans; the two
``single-source-values.json`` files; and the ``LAYERS`` directories of the import-direction
test, each of which must also own a file that test's walker reads (``.py`` or a shebang
launcher). A prefix scope on a directory that survives a move is not detected. No allowlist:
when a path moves, move its literal.
"""

from __future__ import annotations

import ast
import fnmatch
import json
import os
import re
from pathlib import Path

from repo_root import ROOT
from test_package_import_direction import LAYERS, tree_sources

#: First segment that makes a string a repository path. A moved directory stays listed.
RETIRED_AND_LISTED = frozenset(
    {
        ".github",
        ".githooks",
        "adapters",
        "api",
        "contracts",
        "display",
        "docs",
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
SKIP_DIRS = {".git", "__pycache__", "node_modules", ".venv", ".pytest_cache"}


def top_level(root: Path = ROOT) -> frozenset[str]:
    """The listed names (kept for retired directories) plus today's top-level directories."""
    current = {
        path.name
        for path in root.iterdir()
        if path.is_dir() and path.name not in SKIP_DIRS
    }
    return RETIRED_AND_LISTED | current


PATHLIKE = re.compile(r"^[\w.\-*/]+$")
#: Tests that scope a scan to a directory of the tree (a moved directory empties them).
TEST_SIDE_SCOPES = (
    "tests/test_admissible_memory_declarations.py",
    "tests/test_admission_maximum_cap.py",
    "tests/test_own_pulls.py",
    "tests/test_qualification_evidence_publisher.py",
)
JSON_FILES = (
    "qualification/single-source-values.json",
    "tools/protocol-wheel/single-source-values.json",
)


def path_literals(strings, names: frozenset[str] | None = None) -> set[str]:
    """Strings naming a repository path: a plain literal whose first segment is a top-level
    name, or a bare top-level name (``RECIPES = "recipes"``)."""
    names = top_level() if names is None else names
    return {
        text
        for text in strings
        if PATHLIKE.fullmatch(text)
        and text.split("/", 1)[0] in names
        and ("/" in text or text in names)
    }


def _string_run(node: ast.expr) -> list[str]:
    """The trailing string constants of a ``/`` chain: ``ROOT / "a" / "b"`` -> ["a", "b"]."""
    run: list[str] = []
    while isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
        if not (
            isinstance(node.right, ast.Constant) and isinstance(node.right.value, str)
        ):
            return []
        run.insert(0, node.right.value)
        node = node.left
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        run.insert(0, node.value)
    return run


def _base(node: ast.expr) -> ast.expr:
    while isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
        node = node.left
    return node


def composed_paths(source: str, bases: frozenset[str] | None = None) -> list[str]:
    """``ROOT / "display" / "models"`` -> ``display/models`` (only chains rooted at one of
    ``bases`` when given: a test's ``tmp_path / "x"`` is a sample, not a scope)."""
    found = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
            base = _base(node)
            if bases is not None and not (
                isinstance(base, ast.Name) and base.id in bases
            ):
                continue
            run = _string_run(node)
            if run:
                found.append("/".join(part.strip("/") for part in run))
    return found


def python_strings(source: str) -> list[str]:
    constants = [
        node.value
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    ]
    return constants + composed_paths(source)


def scope_strings(source: str) -> list[str]:
    """What scopes a scan inside a test: composed paths and bare names, never the sample
    paths a test feeds its subject."""
    scopes = [
        path
        for path in composed_paths(source, frozenset({"ROOT", "root"}))
        if "." not in path.rsplit("/", 1)[-1]  # a directory, not a sample file
    ]
    return scopes + [text for text in python_strings(source) if "/" not in text]


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


def layers_without_sources(directories, walked) -> list[str]:
    """Layer directories that own no file the import-direction walker reads."""
    return [
        directory
        for directory in directories
        if not any(name.startswith(directory.rstrip("/") + "/") for name in walked)
    ]


def test_every_path_a_guard_names_exists():
    paths = tree_paths(ROOT)
    named: dict[str, set[str]] = {}
    for name, source in guard_sources().items():
        named[name] = path_literals(source_strings(source))
    for name in TEST_SIDE_SCOPES:
        named[name] = path_literals(scope_strings((ROOT / name).read_text()))
    for name in JSON_FILES:
        named[name] = path_literals(json_strings(json.loads((ROOT / name).read_text())))
    layers = [directory for directory, _ in LAYERS.values()]
    named["tests/test_package_import_direction.py LAYERS"] = set(layers)
    empty = layers_without_sources(layers, tree_sources())
    assert not empty, f"LAYERS directories the walker reads nothing from: {empty}"

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
    assert LAYERS
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


def test_a_composed_path_is_seen():
    source = 'DIR = ROOT / "display" / "models"\nOTHER = ROOT / "x" / name\n'
    assert "display/models" in path_literals(python_strings(source))
    assert dangling({"display/models"}, ["display/families/a"]) == ["display/models"]


def test_a_bare_top_level_constant_is_seen():
    literals = path_literals(python_strings('RECIPES = "recipes"\nword = "tools-x"\n'))
    assert literals == {"recipes"}
    assert dangling(literals, ["adapters/a"]) == ["recipes"]


def test_a_new_top_level_directory_is_checked(tmp_path):
    (tmp_path / "src").mkdir()
    (tmp_path / ".git").mkdir()
    names = top_level(tmp_path)
    assert "src" in names and ".git" not in names
    assert path_literals(["src/pkg/mod.py", "other/x"], names) == {"src/pkg/mod.py"}


def test_a_layer_directory_holding_only_data_is_reported():
    walked = {"tools/launcher": "#!/usr/bin/env python3", "qualification/a.py": ""}
    assert layers_without_sources(["tools", "qualification", "docs"], walked) == [
        "docs"
    ]
