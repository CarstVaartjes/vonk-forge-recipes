"""Strict owner registry checks for the repaired value classes.

Registries contain selectors, never copied values. Generated consumers are
reproduced, not trusted because they carry a generated marker.
"""

from __future__ import annotations

import ast
import json
import operator
import re
import subprocess
import tomllib
from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel, ConfigDict


class ValueKind(StrEnum):
    CONTRACT_VERSION = "contract-version"
    MEMORY_BYTES = "memory-bytes"
    PROTOCOL_WHEEL = "protocol-wheel"


class AdversarialFixture(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    path: str
    function: str
    reason: str


class Registration(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    semantic_id: str
    owner: str
    selector: str
    kind: ValueKind
    adversarial_fixtures: tuple[AdversarialFixture, ...] = ()


class Violation(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    semantic_id: str
    path: str
    line: int
    observed: str
    owner: str
    derivation: str

    def __str__(self) -> str:
        return (
            f"{self.semantic_id}: {self.path}:{self.line}: {self.observed!r}; "
            f"owner {self.owner}; {self.derivation}"
        )


def tracked_files(root: Path) -> list[str]:
    return (
        subprocess.run(
            ["git", "ls-files", "-z"],
            cwd=root,
            check=True,
            capture_output=True,
            timeout=30,
        )
        .stdout.decode()
        .strip("\0")
        .split("\0")
    )


def literal(node: ast.AST) -> str | int | float | None:
    """Evaluate only literal arithmetic; never execute repository code."""
    if isinstance(node, ast.Constant) and type(node.value) in (str, int, float):
        return node.value
    if isinstance(node, ast.BinOp):
        left, right = literal(node.left), literal(node.right)
        operations = {
            ast.Add: operator.add,
            ast.Sub: operator.sub,
            ast.Mult: operator.mul,
            ast.Pow: operator.pow,
            ast.Div: operator.truediv,
        }
        if isinstance(left, (int, float)) and isinstance(right, (int, float)):
            if type(node.op) in operations and abs(right) <= 64:
                return operations[type(node.op)](left, right)
            if isinstance(node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div)):
                return operations[type(node.op)](left, right)
    return None


def assignment(tree: ast.Module, selector: str) -> ast.expr:
    values = [
        node.value
        for node in tree.body
        if isinstance(node, (ast.Assign, ast.AnnAssign))
        and any(
            isinstance(target, ast.Name) and target.id == selector
            for target in (
                node.targets if isinstance(node, ast.Assign) else [node.target]
            )
        )
        and node.value is not None
    ]
    if len(values) != 1:
        raise ValueError(f"owner selector {selector}: expected exactly one definition")
    return values[0]


def owner_value(root: Path, registration: Registration) -> str | int | float:
    path = root / registration.owner
    text = path.read_text()
    if path.suffix in {".toml", ".json"}:
        value = tomllib.loads(text) if path.suffix == ".toml" else json.loads(text)
        for key in registration.selector.split("."):
            value = value[key]
    else:
        value = literal(assignment(ast.parse(text), registration.selector))
    if type(value) not in (str, int, float):
        raise ValueError(
            f"{registration.owner}:{registration.selector}: missing literal owner"
        )
    return value


_WHEEL = re.compile(r"vonk[_-]agent[_-]protocol-[A-Za-z0-9_.+!\-]+\.whl")
_MEMORY_CONTEXT = re.compile(
    r"memory|spark|physical|guard|os_floor|os_allowance|capacity|\bCAP\b", re.IGNORECASE
)
_VERSION = re.compile(
    r"\b(?:CONTRACT_VERSION|CONTRACT_MAJOR|__version__)\s*(?::[^=\n]+)?=\s*"
    r"(?:[\"']\d+(?:\.\d+)*[\"']|\d+(?:\.\d+)*)"
)
_UNITS = re.compile(r"(\d+(?:\.\d+)?)\s*(GiB|GB|MiB|MB|bytes)", re.IGNORECASE)


def _byte_value(value: str | float | None) -> int | float | None:
    if isinstance(value, (int, float)):
        return value
    if isinstance(value, str):
        match = _UNITS.fullmatch(value)
        if match:
            factors = {
                "gib": 1024**3,
                "gb": 1000**3,
                "mib": 1024**2,
                "mb": 1000**2,
                "bytes": 1,
            }
            return float(match[1]) * factors[match[2].lower()]
    return None


def scan_source(
    registration: Registration,
    path: str,
    text: str,
    value: str | float,
) -> list[Violation]:
    """Scan parsed Python, executable documentation and line-oriented build inputs."""
    if path.endswith(".md"):
        issues = []
        for match in re.finditer(r"```[^\n]*\n(.*?)```", text, re.DOTALL):
            padding = "\n" * text[: match.start(1)].count("\n")
            issues.extend(
                scan_source(registration, path + ":fence", padding + match[1], value)
            )
        return issues
    tree = None
    if (
        path.endswith(".py")
        or text.startswith("#!/")
        and "python" in text.splitlines()[0]
    ):
        tree = ast.parse(text)
    exempt_lines: set[int] = set()
    for fixture in registration.adversarial_fixtures:
        if fixture.path != path:
            continue
        if tree is None:
            raise ValueError(f"adversarial fixture must be a Python test: {path}")
        functions = [
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == fixture.function
        ]
        if (
            len(functions) != 1
            or not fixture.function.startswith("test_")
            or not fixture.reason
        ):
            raise ValueError(f"missing named adversarial fixture: {fixture}")
        node = functions[0]
        exempt_lines.update(range(node.lineno, (node.end_lineno or node.lineno) + 1))
    hits: list[tuple[int, str]] = []
    if registration.kind == ValueKind.PROTOCOL_WHEEL:
        hits = [
            (text[: m.start()].count("\n") + 1, m[0]) for m in _WHEEL.finditer(text)
        ]
    elif registration.kind == ValueKind.CONTRACT_VERSION:
        if "contracts/" not in path and "vonk_forge_contracts" not in text:
            text = re.sub(r"__version__", "external_version", text)
        hits = [
            (text[: m.start()].count("\n") + 1, m[0]) for m in _VERSION.finditer(text)
        ]
    elif tree is not None:
        for node in ast.walk(tree):
            if not isinstance(
                node,
                (
                    ast.Assign,
                    ast.AnnAssign,
                    ast.Assert,
                    ast.Return,
                    ast.Expr,
                    ast.If,
                    ast.Compare,
                    ast.Call,
                ),
            ):
                continue
            source = ast.get_source_segment(text, node) or ""
            if not _MEMORY_CONTEXT.search(path + " " + source):
                continue
            for expression in ast.walk(node):
                if _byte_value(literal(expression)) == value:
                    hits.append(
                        (
                            expression.lineno,
                            ast.get_source_segment(text, expression) or "",
                        )
                    )
    else:
        for line, source in enumerate(text.splitlines(), 1):
            if not _MEMORY_CONTEXT.search(path + " " + source):
                continue
            for token in re.findall(
                r"\d[\d_]*(?:\.\d+)?(?:\s*(?:GiB|GB|MiB|MB|bytes))?", source
            ):
                compact = token.replace("_", "")
                if _byte_value(compact) == value or compact == str(value):
                    hits.append((line, token))
    if path == registration.owner:
        if tree is not None:
            node = assignment(tree, registration.selector)
            exempt_lines.update(
                range(node.lineno, (node.end_lineno or node.lineno) + 1)
            )
        else:
            # Metadata values are permitted only on the selected key's line.
            key = registration.selector.split(".")[-1]
            exempt_lines.update(
                i
                for i, line in enumerate(text.splitlines(), 1)
                if re.match(rf'\s*["\']?{re.escape(key)}["\']?\s*[:=]', line)
            )
    return [
        Violation(
            semantic_id=registration.semantic_id,
            path=path,
            line=line,
            observed=observed,
            owner=f"{registration.owner}:{registration.selector}",
            derivation="import the owner or reproduce its exact generated projection",
        )
        for line, observed in sorted(set(hits))
        if line not in exempt_lines
    ]


def check(root: Path, files: list[str] | None = None) -> list[Violation]:
    files = tracked_files(root) if files is None else files
    registrations = [
        Registration.model_validate(item)
        for path in files
        if path.endswith("/single-source-values.json")
        for item in json.loads((root / path).read_text())
    ]
    ids: set[str] = set()
    owners: set[tuple[str, str]] = set()
    families: set[tuple[ValueKind, str | int | float]] = set()
    issues = []
    for registration in registrations:
        value = owner_value(root, registration)
        family = (
            registration.kind,
            value if registration.kind == ValueKind.MEMORY_BYTES else "*",
        )
        owner = (registration.owner, registration.selector)
        if registration.semantic_id in ids or owner in owners or family in families:
            raise ValueError(f"duplicate/overlapping owner: {registration.semantic_id}")
        ids.add(registration.semantic_id)
        owners.add(owner)
        families.add(family)
        generated: dict[str, bytes] = {}
        if registration.kind == ValueKind.PROTOCOL_WHEEL:
            from tools.protocol_wheel import projections

            generated = projections(root)
        for path in files:
            if (
                registration.kind == ValueKind.PROTOCOL_WHEEL
                and _WHEEL.search(path)
                and path not in generated
            ):
                issues.append(
                    Violation(
                        semantic_id=registration.semantic_id,
                        path=path,
                        line=1,
                        observed="unregistered protocol wheel artifact",
                        owner=registration.owner,
                        derivation="use only the selected artifact projections",
                    )
                )
            payload = (root / path).read_bytes()
            if path in generated:
                if payload != generated[path]:
                    issues.append(
                        Violation(
                            semantic_id=registration.semantic_id,
                            path=path,
                            line=1,
                            observed="stale generated projection",
                            owner=registration.owner,
                            derivation="run tools/materialize-protocol-wheel",
                        )
                    )
                continue
            try:
                source = payload.decode("utf-8")
            except UnicodeDecodeError:
                continue
            issues.extend(scan_source(registration, path, source, value))
        for path in generated.keys() - set(files):
            issues.append(
                Violation(
                    semantic_id=registration.semantic_id,
                    path=path,
                    line=1,
                    observed="missing generated projection",
                    owner=registration.owner,
                    derivation="run tools/materialize-protocol-wheel",
                )
            )
    return issues
