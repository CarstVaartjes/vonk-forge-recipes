"""Python layers: contracts -> qualification -> spark_sweep -> tools, and no import cycles.

Whole-tree, no counts and no exceptions. A function-level import counts, a
``TYPE_CHECKING`` import does not. Tools may import the sweep only through its
three launchers.
"""

from __future__ import annotations

import ast

from repo_root import ROOT

#: Layer name -> (directory, importable top-level package).
LAYERS = {
    "contracts": ("contracts/src/vonk_forge_contracts", "vonk_forge_contracts"),
    "qualification": (
        "src/recipe_tooling/qualification",
        "recipe_tooling.qualification",
    ),
    "spark_sweep": ("spark_sweep", "spark_sweep"),
    "tooling": ("src/recipe_tooling", "recipe_tooling"),
    "tools": ("tools", "tools"),
}
ALLOWED = {
    "contracts": {"contracts"},
    "qualification": {"contracts", "qualification"},
    "spark_sweep": {"contracts", "qualification", "spark_sweep"},
    "tooling": {"contracts", "qualification", "tooling"},
    "tools": {"contracts", "qualification", "tooling", "tools"},
}
LAUNCHERS = {
    "sweep-recipes",
    "sync-qualification-evidence",
    "backfill-qualification-evidence",
}

Node = tuple[str, str]  # (layer, dotted module; the file name for a tool script)
Edge = tuple[Node, str, str, int]  # (source, target layer, dotted target, line)


def _is_type_checking(test: ast.expr) -> bool:
    return (isinstance(test, ast.Name) and test.id == "TYPE_CHECKING") or (
        isinstance(test, ast.Attribute) and test.attr == "TYPE_CHECKING"
    )


def _runtime_imports(tree: ast.AST) -> list[ast.Import | ast.ImportFrom]:
    found: list[ast.Import | ast.ImportFrom] = []

    def visit(node: ast.AST) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.If) and _is_type_checking(child.test):
                for other in child.orelse:
                    visit_one(other)
                continue
            visit_one(child)

    def visit_one(child: ast.AST) -> None:
        if isinstance(child, (ast.Import, ast.ImportFrom)):
            found.append(child)
        visit(child)

    visit(tree)
    return found


def _dotted_targets(node: ast.Import | ast.ImportFrom, source: Node) -> list[str]:
    """Dotted names an import may bind: the module, then each imported name under it."""
    if isinstance(node, ast.Import):
        return [alias.name for alias in node.names]
    if node.level == 0:
        base = node.module or ""
    else:
        layer, module = source
        parts = module.split(".")[:-1]
        parts = parts[: len(parts) - (node.level - 1)]
        base = ".".join(
            [LAYERS[layer][1], *parts, *([node.module] if node.module else [])]
        )
    return [base, *(f"{base}.{alias.name}" for alias in node.names)]


def scan(sources: dict[str, str]) -> tuple[set[Node], list[Edge]]:
    """Nodes and runtime edges of ``{repository path: source}``."""
    nodes: dict[str, Node] = {}
    for path in sources:
        owners = [
            (len(directory), layer, directory)
            for layer, (directory, _) in LAYERS.items()
            if path.startswith(f"{directory}/")
        ]
        if owners:  # the deepest directory owns the file
            _, layer, directory = max(owners)
            name = path[len(directory) + 1 :].removesuffix(".py").replace("/", ".")
            nodes[path] = (layer, name)
    known = set(nodes.values())
    edges: list[Edge] = []
    for path, node in nodes.items():
        for statement in _runtime_imports(ast.parse(sources[path])):
            dotted = _dotted_targets(statement, node)
            resolved = [_resolve(name, known) for name in dotted]
            # `from . import b` binds the module b, not the package's __init__.
            if isinstance(statement, ast.ImportFrom) and any(
                target and target[1] != "__init__" for target in resolved[1:]
            ):
                resolved = resolved[1:]
            edges.extend(
                (node, target[0], target[1], statement.lineno)
                for target in resolved
                if target and target != node
            )
    return set(nodes.values()), edges


def _resolve(dotted: str, known: set[Node]) -> Node | None:
    owners = [
        (len(package), layer)
        for layer, (_, package) in LAYERS.items()
        if dotted == package or dotted.startswith(f"{package}.")
    ]
    if not owners:
        return None
    length, layer = max(owners)  # the deepest package owns the name
    rest = dotted[length + 1 :]
    parts = rest.split(".") if rest else []
    while parts:
        if (layer, ".".join(parts)) in known:
            return layer, ".".join(parts)
        parts.pop()
    return layer, "__init__"


def violations(sources: dict[str, str]) -> list[str]:
    _, edges = scan(sources)
    found = []
    for (layer, module), target_layer, target_module, line in edges:
        if target_layer in ALLOWED[layer]:
            continue
        if (target_layer, target_module) == ("tooling", "paths"):
            continue  # the repository root owner imports nothing, so every layer may read it
        if target_layer == "spark_sweep" and module in LAUNCHERS:
            continue
        found.append(f"{layer}/{module}:{line} imports {target_layer}")
    return sorted(set(found))


def cycles(sources: dict[str, str]) -> list[list[str]]:
    nodes, edges = scan(sources)
    graph: dict[Node, set[Node]] = {node: set() for node in nodes}
    for source, layer, module, _ in edges:
        if (layer, module) in graph:
            graph[source].add((layer, module))
    index: dict[Node, int] = {}
    low: dict[Node, int] = {}
    stack: list[Node] = []
    on_stack: set[Node] = set()
    found: list[list[str]] = []

    def strong(node: Node) -> None:
        index[node] = low[node] = len(index)
        stack.append(node)
        on_stack.add(node)
        for other in graph[node]:
            if other not in index:
                strong(other)
                low[node] = min(low[node], low[other])
            elif other in on_stack:
                low[node] = min(low[node], index[other])
        if low[node] == index[node]:
            members = []
            while True:
                other = stack.pop()
                on_stack.discard(other)
                members.append(other)
                if other == node:
                    break
            if len(members) > 1:
                found.append(sorted(f"{layer}/{module}" for layer, module in members))

    for node in sorted(graph):
        if node not in index:
            strong(node)
    return sorted(found)


def tree_sources() -> dict[str, str]:
    sources = {}
    for directory, _ in LAYERS.values():
        for path in sorted((ROOT / directory).rglob("*")):
            relative = path.relative_to(ROOT).as_posix()
            if (
                not path.is_file()
                or "__pycache__" in path.parts
                or "/fixtures/" in relative
            ):
                continue
            if path.suffix == ".py":
                sources[relative] = path.read_text()
            elif not path.suffix and path.read_bytes().startswith(b"#!"):
                text = path.read_text()
                if "python" in text.splitlines()[0]:
                    sources[relative] = text
    return sources


def test_every_import_points_down_the_layers_and_none_is_a_cycle() -> None:
    sources = tree_sources()
    assert violations(sources) == []
    assert cycles(sources) == []


def test_a_function_level_import_counts() -> None:
    sources = {
        "src/recipe_tooling/qualification/a.py": "def f():\n    import spark_sweep.b\n",
        "spark_sweep/b.py": "",
    }
    assert violations(sources) == ["qualification/a:2 imports spark_sweep"]


def test_a_type_checking_import_is_free() -> None:
    sources = {
        "src/recipe_tooling/qualification/a.py": (
            "from typing import TYPE_CHECKING\n"
            "if TYPE_CHECKING:\n    from spark_sweep import b\n"
        ),
        "spark_sweep/b.py": "from recipe_tooling.qualification import a\n",
    }
    assert violations(sources) == []
    assert cycles(sources) == []


def test_a_two_file_cycle_is_reported_through_relative_imports() -> None:
    sources = {
        "spark_sweep/a.py": "from . import b\n",
        "spark_sweep/b.py": "def f():\n    from .a import x\n",
    }
    assert cycles(sources) == [["spark_sweep/a", "spark_sweep/b"]]


def test_only_the_launchers_may_import_the_sweep() -> None:
    sources = {
        "spark_sweep/b.py": "",
        "tools/sweep-recipes": "#!/usr/bin/env python3\nfrom spark_sweep import b\n",
        "tools/other": "#!/usr/bin/env python3\nfrom spark_sweep import b\n",
    }
    assert violations(sources) == ["tools/other:2 imports spark_sweep"]
