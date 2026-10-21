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

from pydantic import BaseModel, ConfigDict, Field


class ValueKind(StrEnum):
    CONTRACT_VERSION = "contract-version"
    MEMORY_BYTES = "memory-bytes"
    PROTOCOL_WHEEL = "protocol-wheel"


class AdversarialFixture(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    path: str = Field(pattern=r"^tests/(?:[A-Za-z0-9_-]+/)*test_[A-Za-z0-9_]+\.py$")
    function: str = Field(pattern=r"^test_[A-Za-z0-9_]+$")
    reason: str = Field(min_length=1)


class Registration(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    semantic_id: str
    owner: str
    selector: str
    kind: ValueKind
    consumer_symbols: tuple[str, ...] = ()
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
    if (
        isinstance(node, ast.Constant)
        and isinstance(node.value, (str, int, float))
        and not isinstance(node.value, bool)
    ):
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
        if isinstance(node.op, ast.Div) and right == 0:
            return None
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
    if not isinstance(value, (str, int, float)) or isinstance(value, bool):
        raise TypeError(
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
        if re.fullmatch(r"[0-9][0-9_]*", value):
            return int(value.replace("_", ""))
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
    if (
        registration.kind == ValueKind.PROTOCOL_WHEEL
        and not _WHEEL.search(text)
        and not any(
            fixture.path == path for fixture in registration.adversarial_fixtures
        )
    ):
        return []
    if registration.kind == ValueKind.CONTRACT_VERSION and not _VERSION.search(text):
        return []
    if (
        registration.kind == ValueKind.MEMORY_BYTES
        and not _MEMORY_CONTEXT.search(text)
        and not any(symbol in text for symbol in registration.consumer_symbols)
    ):
        return []
    tree = None
    if (
        path.endswith(".py")
        or text.startswith("#!/")
        and "python" in text.splitlines()[0]
    ):
        tree = ast.parse(text)
    owned_expression = (
        assignment(tree, registration.selector)
        if tree is not None and path == registration.owner
        else None
    )
    owned_nodes = (
        set(ast.walk(owned_expression)) if owned_expression is not None else set()
    )
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
        if len(functions) != 1:
            raise ValueError(f"missing named adversarial fixture: {fixture}")
        function = functions[0]
        exempt_lines.update(
            range(function.lineno, (function.end_lineno or function.lineno) + 1)
        )
    hits: list[tuple[int, str]] = []
    if registration.kind == ValueKind.PROTOCOL_WHEEL:
        owner_span = None
        if path == registration.owner:
            matches = list(
                re.finditer(
                    rf'"{re.escape(registration.selector)}"\s*:\s*("[^"\n]*")', text
                )
            )
            if len(matches) != 1:
                raise ValueError(
                    f"owner selector {registration.selector}: expected exactly one definition"
                )
            owner_span = matches[0].span(1)
        hits = [
            (text[: match.start()].count("\n") + 1, match[0])
            for match in _WHEEL.finditer(text)
            if owner_span is None
            or not (owner_span[0] <= match.start() < owner_span[1])
        ]
    elif registration.kind == ValueKind.CONTRACT_VERSION:
        if tree is not None:
            names = {"CONTRACT_VERSION", "CONTRACT_MAJOR"}
            if "contracts/" in path or "vonk_forge_contracts" in text:
                names.add("__version__")
            for node in ast.walk(tree):
                if (
                    not isinstance(node, (ast.Assign, ast.AnnAssign))
                    or node.value is None
                    or node.value is owned_expression
                ):
                    continue
                targets = (
                    node.targets if isinstance(node, ast.Assign) else [node.target]
                )
                if (
                    any(
                        isinstance(target, ast.Name) and target.id in names
                        for target in targets
                    )
                    and re.fullmatch(r"\d+(?:\.\d+)*", str(literal(node.value)))
                    is not None
                ):
                    hits.append((node.lineno, ast.get_source_segment(text, node) or ""))
        else:
            hits = [
                (text[: match.start()].count("\n") + 1, match[0])
                for match in _VERSION.finditer(text)
            ]
    elif tree is not None:
        parents = {
            child: parent
            for parent in ast.walk(tree)
            for child in ast.iter_child_nodes(parent)
        }
        for node in ast.walk(tree):
            if (
                not isinstance(node, (ast.Assign, ast.AnnAssign))
                or node.value is None
                or node.value is owned_expression
            ):
                continue
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            names = {target.id for target in targets if isinstance(target, ast.Name)}
            if (
                names.intersection(registration.consumer_symbols)
                and _byte_value(literal(node.value)) is not None
            ):
                hits.append(
                    (node.value.lineno, ast.get_source_segment(text, node.value) or "")
                )
        for expression in ast.walk(tree):
            if (
                not isinstance(expression, (ast.Constant, ast.BinOp))
                or expression in owned_nodes
            ):
                continue
            if _byte_value(literal(expression)) != value:
                continue
            statement = expression
            while statement in parents and not isinstance(statement, ast.stmt):
                statement = parents[statement]
            source = ast.get_source_segment(text, statement) or ""
            if _MEMORY_CONTEXT.search(source):
                hits.append(
                    (expression.lineno, ast.get_source_segment(text, expression) or "")
                )
    else:
        # Recipe declarations and historical observations are independent data
        # selections. Policy/default keys in configuration still belong here.
        for line, source in enumerate(text.splitlines(), 1):
            if source.lstrip().startswith("#"):
                continue
            if path.endswith((".json", ".jsonl", ".toml")) and not re.search(
                r"os_(?:floor|allowance)|default_spark_memory|guard_bytes|physical_memory",
                source,
                re.IGNORECASE,
            ):
                continue
            if not _MEMORY_CONTEXT.search(source):
                continue
            tokens = re.findall(
                r"\d[\d_]*(?:\.\d+)?(?:\s*(?:GiB|GB|MiB|MB|bytes))", source
            )
            expressions = re.findall(
                r"\d[\d_]*(?:\.\d+)?(?:\s*(?:\*\*|[+*/-])\s*\d[\d_]*(?:\.\d+)?)*",
                source,
            )
            for token in tokens + expressions:
                try:
                    number = _byte_value(literal(ast.parse(token, mode="eval").body))
                except SyntaxError:
                    number = _byte_value(token)
                if number == value:
                    hits.append((line, token))
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
        if registration.owner not in files:
            raise ValueError(f"owner is not tracked: {registration.owner}")
        for fixture in registration.adversarial_fixtures:
            if fixture.path not in files:
                raise ValueError(f"adversarial fixture is not tracked: {fixture.path}")
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
