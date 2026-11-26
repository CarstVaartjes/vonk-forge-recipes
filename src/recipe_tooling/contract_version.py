"""The contract version, computed from the change class since the last release.

A pull request never sets the version (G3). ``_version.py`` holds the hand-set
``CONTRACT_MAJOR`` and the placeholder ``__version__ = "<major>.0.0"``. Publication
computes the version from what changed since the last ``v*`` tag and stamps it into
``_version.py`` in its own workspace only, before it builds the index and the wheel.

The change class is the strongest finding over the inputs:

* **breaking**: a property, definition, alternative or file removed; ``required``
  grown, including a new required property on an existing model (the dead-field
  rule: a new field is born optional); a type, const or ``$ref`` changed
  incompatibly; an enum shrunk; a bound or pattern narrowed; an existing vector or
  example no longer holds what it held (a value changed, a key or element removed).
* **minor**: anything else that changes a schema or a vector, such as a property,
  definition, alternative or enum member added, a requirement or bound relaxed,
  a new vector file, or a vector that only gained keys or appended elements (a
  born-optional field shows up in a full dump as a new key).
* **patch**: ``contracts/src`` changed while the schemas and vectors are identical.
* **none**: ``contracts/src`` unchanged; publication updates the current release.

A breaking class needs ``CONTRACT_MAJOR`` one above the released major. The tool
never runs git: the workflow exports the released tree and passes directories.
Standard library only, so it runs on the runner's bare ``python3``.
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import sys
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from enum import IntEnum
from pathlib import Path

_VERSION = re.compile(r"([0-9]+)\.([0-9]+)\.([0-9]+)")
_ABSENT = object()

_ANNOTATIONS = frozenset({"title", "description", "examples", "$comment"})
_LOWER_BOUNDS = frozenset(
    {"minimum", "exclusiveMinimum", "minLength", "minItems", "minProperties"}
)
_UPPER_BOUNDS = frozenset(
    {"maximum", "exclusiveMaximum", "maxLength", "maxItems", "maxProperties"}
)
_NAMED_SCHEMAS = frozenset({"properties", "patternProperties", "$defs", "definitions"})
_SUBSCHEMAS = frozenset(
    {"items", "propertyNames", "contains", "not", "if", "then", "else"}
)
_ALTERNATIVES = ("anyOf", "oneOf")


class ChangeClass(IntEnum):
    """Ordered so the strongest finding wins."""

    NONE = 0
    PATCH = 1
    MINOR = 2
    BREAKING = 3

    @property
    def word(self) -> str:
        return self.name.lower()


class ContractVersionError(ValueError):
    """The change cannot be released as committed."""


@dataclass(frozen=True)
class Finding:
    change: ChangeClass
    where: str
    what: str

    def __str__(self) -> str:
        return f"{self.change.word}: {self.where}: {self.what}"


@dataclass(frozen=True)
class Surface:
    """One released tree's published surface: every input the class is computed over.

    Each field is one input, compared by its own function in ``findings``.
    ``shared_functions`` is the package's ``SHARED_FUNCTIONS`` names, or None for a
    tree that has no such list (it is then not compared).
    """

    schemas: Mapping[str, object] = field(default_factory=dict)
    vectors: Mapping[str, bytes] = field(default_factory=dict)
    shared_functions: frozenset[str] | None = None


def shared_functions(package_init: str) -> frozenset[str] | None:
    """The names in ``SHARED_FUNCTIONS = (...)`` of an ``__init__.py``, read, never imported."""

    for node in ast.parse(package_init).body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target, value = node.targets[0], node.value
        elif isinstance(node, ast.AnnAssign) and node.value is not None:
            target, value = node.target, node.value
        else:
            continue
        if isinstance(target, ast.Name) and target.id == "SHARED_FUNCTIONS":
            if not isinstance(value, (ast.Tuple, ast.List)):
                raise ContractVersionError("SHARED_FUNCTIONS must be a literal tuple")
            return frozenset(
                item.value
                for item in value.elts
                if isinstance(item, ast.Constant) and isinstance(item.value, str)
            )
    return None


def read_surface(
    schemas: Path, vectors: Path, package_init: Path | None = None
) -> Surface:
    """Read a schema directory (``*.json``), a vector tree (every file) and,
    when given, the package ``__init__.py`` holding ``SHARED_FUNCTIONS``."""

    return Surface(
        shared_functions=(
            None
            if package_init is None or not package_init.is_file()
            else shared_functions(package_init.read_text("utf-8"))
        ),
        schemas={
            path.name: json.loads(path.read_text("utf-8"))
            for path in sorted(schemas.glob("*.json"))
        },
        vectors={
            path.relative_to(vectors).as_posix(): path.read_bytes()
            for path in sorted(vectors.rglob("*"))
            if path.is_file()
        },
    )


def findings(base: Surface, head: Surface) -> list[Finding]:
    """Every difference between two surfaces, each with its change class."""

    return [
        *_files(base.schemas, head.schemas, "schema", _schema_file),
        *_files(base.vectors, head.vectors, "vector", _vector_file),
        *_shared(base.shared_functions, head.shared_functions),
    ]


def _shared(
    base: frozenset[str] | None, head: frozenset[str] | None
) -> Iterator[Finding]:
    if base is None or head is None:
        return
    for name in sorted(base - head):
        yield Finding(ChangeClass.BREAKING, "SHARED_FUNCTIONS", f"{name} removed")
    for name in sorted(head - base):
        yield Finding(ChangeClass.MINOR, "SHARED_FUNCTIONS", f"{name} added")


def change_class(found: Sequence[Finding], *, src_changed: bool) -> ChangeClass:
    if not src_changed:
        return ChangeClass.NONE
    return max((item.change for item in found), default=ChangeClass.PATCH)


def parse_version(text: str) -> tuple[int, int, int]:
    match = _VERSION.fullmatch(text)
    if match is None:
        raise ContractVersionError(f"not a MAJOR.MINOR.PATCH version: {text!r}")
    major, minor, patch = (int(part) for part in match.groups())
    return major, minor, patch


def next_version(base: str, major: int, change: ChangeClass) -> str | None:
    """The version to release after ``base``, or None when nothing is cut."""

    base_major, base_minor, base_patch = parse_version(base)
    if major == base_major + 1:
        return f"{major}.0.0"
    if major != base_major:
        raise ContractVersionError(
            f"CONTRACT_MAJOR {major} must be the released major {base_major} "
            f"or one above it"
        )
    if change is ChangeClass.BREAKING:
        raise ContractVersionError(
            "breaking change needs CONTRACT_MAJOR bump "
            f"(to {base_major + 1} in contracts/src/vonk_forge_contracts/_version.py)"
        )
    if change is ChangeClass.MINOR:
        return f"{base_major}.{base_minor + 1}.0"
    if change is ChangeClass.PATCH:
        return f"{base_major}.{base_minor}.{base_patch + 1}"
    return None


def _assignments(text: str) -> dict[str, object]:
    values: dict[str, object] = {}
    for node in ast.parse(text).body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            if isinstance(target, ast.Name):
                try:
                    values[target.id] = ast.literal_eval(node.value)
                except ValueError as error:
                    raise ContractVersionError(
                        f"_version.py: {target.id} must be a literal"
                    ) from error
    return values


def committed_major(text: str, base: str) -> int:
    """G3: the committed ``_version.py`` holds only the major and its placeholder.

    ``CONTRACT_MAJOR`` is the released major or one above it, and ``__version__``
    is ``"<CONTRACT_MAJOR>.0.0"``: the minor and patch are publication's to compute.
    """

    values = _assignments(text)
    major = values.get("CONTRACT_MAJOR")
    if type(major) is not int:
        raise ContractVersionError("_version.py must set CONTRACT_MAJOR to an integer")
    if set(values) != {"CONTRACT_MAJOR", "__version__"}:
        raise ContractVersionError(
            "_version.py holds only CONTRACT_MAJOR and the __version__ placeholder"
        )
    if values["__version__"] != f"{major}.0.0":
        raise ContractVersionError(
            f'__version__ is computed at publication; keep the placeholder "{major}.0.0"'
        )
    base_major = parse_version(base)[0]
    if major not in (base_major, base_major + 1):
        raise ContractVersionError(
            f"CONTRACT_MAJOR {major} must be the released major {base_major} "
            f"or one above it"
        )
    return major


def stamp(text: str, version: str) -> str:
    """``_version.py`` with the computed ``__version__`` in place of the placeholder."""

    values = _assignments(text)
    if parse_version(version)[0] != values.get("CONTRACT_MAJOR"):
        raise ContractVersionError(
            f"version {version} is not in CONTRACT_MAJOR {values.get('CONTRACT_MAJOR')}"
        )
    stamped, count = re.subn(
        r'^__version__ = "[^"\n]*"$',
        f'__version__ = "{version}"',
        text,
        flags=re.MULTILINE,
    )
    if count != 1:
        raise ContractVersionError("_version.py must assign __version__ exactly once")
    return stamped


# --- files -------------------------------------------------------------------


def _files(
    base: Mapping[str, object],
    head: Mapping[str, object],
    kind: str,
    compare: Callable[[str, object, object], Iterator[Finding]],
) -> Iterator[Finding]:
    for name in sorted(base.keys() | head.keys()):
        if name not in head:
            yield Finding(ChangeClass.BREAKING, name, f"{kind} file removed")
        elif name not in base:
            yield Finding(ChangeClass.MINOR, name, f"{kind} file added")
        elif base[name] != head[name]:
            yield from compare(name, base[name], head[name])


def _vector_file(name: str, base: object, head: object) -> Iterator[Finding]:
    try:
        old, new = json.loads(_bytes(base)), json.loads(_bytes(head))
    except ValueError:
        yield Finding(ChangeClass.BREAKING, name, "vector bytes changed")
        return
    if _extends(old, new):
        yield Finding(ChangeClass.MINOR, name, "vector gained keys or elements")
    else:
        yield Finding(
            ChangeClass.BREAKING, name, "vector no longer holds a value it held"
        )


def _bytes(value: object) -> bytes:
    if not isinstance(value, bytes):
        raise TypeError("vector files are bytes")
    return value


def _extends(old: object, new: object) -> bool:
    """Whether ``new`` keeps every value of ``old``, adding only keys or appended items."""

    if isinstance(old, dict) and isinstance(new, dict):
        return all(key in new and _extends(old[key], new[key]) for key in old)
    if isinstance(old, list) and isinstance(new, list):
        return len(new) >= len(old) and all(map(_extends, old, new))
    return type(old) is type(new) and old == new


def _schema_file(name: str, base: object, head: object) -> Iterator[Finding]:
    walk = _SchemaWalk(base, head)
    walk.schema(base, head, name)
    if not walk.found:
        # Different bytes with the same meaning (an inlined ``$ref``) still ship.
        walk.add(ChangeClass.MINOR, name, "schema bytes changed")
    return iter(walk.found)


# --- schemas -----------------------------------------------------------------


class _SchemaWalk:
    """Compare a released schema with the head's, as a reader of documents sees it."""

    def __init__(self, base_root: object, head_root: object) -> None:
        self.roots = (base_root, head_root)
        self.seen: set[tuple[str, str]] = set()
        self.found: list[Finding] = []

    def add(self, change: ChangeClass, where: str, what: str) -> None:
        self.found.append(Finding(change, where, what))

    def schema(self, base: object, head: object, where: str) -> None:
        if base == head:
            return
        if not isinstance(base, dict) or not isinstance(head, dict):
            self.add(ChangeClass.BREAKING, where, "schema changed")
            return
        base, head = _const_as_enum(base), _const_as_enum(head)
        # Wrap first: a ``$ref`` that became one alternative of a union stays a ref.
        base, head = self._as_alternatives(base, head)
        base, head = self._dereferenced(base, head)
        for key in sorted(base.keys() | head.keys()):
            old, new = base.get(key, _ABSENT), head.get(key, _ABSENT)
            if old != new:
                self.keyword(key, old, new, f"{where}/{key}")

    def keyword(self, key: str, old: object, new: object, where: str) -> None:
        if key in _ANNOTATIONS:
            self.add(ChangeClass.MINOR, where, "annotation changed")
        elif key in _NAMED_SCHEMAS:
            self.named(old, new, where)
        elif key == "components":
            self.components(old, new, where)
        elif key == "required":
            self.required(old, new, where)
        elif key == "enum":
            self.enum(old, new, where)
        elif key == "type":
            self.types(old, new, where)
        elif key == "$ref":
            self.ref(old, new, where)
        elif key in _ALTERNATIVES:
            self.alternatives(old, new, where)
        elif key in ("allOf", "prefixItems"):
            self.positional(old, new, where)
        elif key == "additionalProperties":
            self.additional(old, new, where)
        elif key in _SUBSCHEMAS:
            self.subschema(old, new, where)
        elif key in _LOWER_BOUNDS or key in _UPPER_BOUNDS:
            self.bound(old, new, where, lower=key in _LOWER_BOUNDS)
        elif key == "discriminator":
            self.discriminator(old, new, where)
        elif key == "uniqueItems":
            narrowed = new is True
            self.relaxed_unless(where, "uniqueItems", narrowed=narrowed)
        elif key in ("pattern", "format", "default"):
            self.only_removal_relaxes(key, old, new, where)
        else:
            self.add(ChangeClass.BREAKING, where, f"{key} changed")

    def _resolve(self, side: int, ref: str) -> object:
        node = self.roots[side]
        if not ref.startswith("#"):
            return _ABSENT
        for escaped in filter(None, ref[1:].split("/")):
            part = escaped.replace("~1", "/").replace("~0", "~")
            if not isinstance(node, dict) or part not in node:
                return _ABSENT
            node = node[part]
        return node

    def _dereferenced(
        self, base: dict[str, object], head: dict[str, object]
    ) -> tuple[dict[str, object], dict[str, object]]:
        """Inline a ``$ref`` that only one side uses, so its target is compared."""

        if ("$ref" in base) == ("$ref" in head):
            return base, head
        sides = []
        for side, node in enumerate((base, head)):
            ref = node.get("$ref")
            target = self._resolve(side, ref) if isinstance(ref, str) else None
            rest = {k: v for k, v in node.items() if k != "$ref"}
            sides.append({**target, **rest} if isinstance(target, dict) else node)
        return sides[0], sides[1]

    def _as_alternatives(
        self, base: dict[str, object], head: dict[str, object]
    ) -> tuple[dict[str, object], dict[str, object]]:
        """Treat a plain schema as a one-alternative ``anyOf`` when the other side has one.

        So ``X`` becoming ``X or null`` reads as an added alternative.
        """

        for key in _ALTERNATIVES:
            if (key in base) == (key in head):
                continue
            plain = head if key in base else base
            moved = {
                k: v
                for k, v in plain.items()
                if k not in _ANNOTATIONS and k != "default"
            }
            wrapped = {k: v for k, v in plain.items() if k not in moved}
            wrapped[key] = [moved]
            return (wrapped, head) if plain is base else (base, wrapped)
        return base, head

    def named(self, old: object, new: object, where: str) -> None:
        old_map = old if isinstance(old, dict) else {}
        new_map = new if isinstance(new, dict) else {}
        for name in sorted(old_map.keys() | new_map.keys()):
            if name not in new_map:
                self.add(ChangeClass.BREAKING, f"{where}/{name}", "removed")
            elif name not in old_map:
                self.add(ChangeClass.MINOR, f"{where}/{name}", "added")
            else:
                self.schema(old_map[name], new_map[name], f"{where}/{name}")

    def components(self, old: object, new: object, where: str) -> None:
        old_map = old if isinstance(old, dict) else {}
        new_map = new if isinstance(new, dict) else {}
        for key in sorted(old_map.keys() | new_map.keys()):
            before, after = old_map.get(key, _ABSENT), new_map.get(key, _ABSENT)
            if before == after:
                continue
            if key == "schemas":
                self.named(before, after, f"{where}/schemas")
            else:
                self.add(ChangeClass.BREAKING, f"{where}/{key}", "changed")

    def required(self, old: object, new: object, where: str) -> None:
        before = set(old) if isinstance(old, list) else set()
        after = set(new) if isinstance(new, list) else set()
        if after - before:
            self.add(
                ChangeClass.BREAKING,
                where,
                f"now required {sorted(after - before)} (a new field is born optional)",
            )
        if before - after:
            self.add(
                ChangeClass.MINOR, where, f"no longer required {sorted(before - after)}"
            )

    def enum(self, old: object, new: object, where: str) -> None:
        if old is _ABSENT or new is _ABSENT:
            self.relaxed_unless(where, "enum", narrowed=old is _ABSENT)
            return
        before = {json.dumps(item, sort_keys=True) for item in _list(old)}
        after = {json.dumps(item, sort_keys=True) for item in _list(new)}
        if before - after:
            self.add(ChangeClass.BREAKING, where, f"enum lost {sorted(before - after)}")
        if after - before:
            self.add(ChangeClass.MINOR, where, f"enum gained {sorted(after - before)}")

    def types(self, old: object, new: object, where: str) -> None:
        if old is _ABSENT or new is _ABSENT:
            self.relaxed_unless(where, "type", narrowed=old is _ABSENT)
            return
        before = set(_list(old))
        after = set(_list(new))
        self.relaxed_unless(
            where, f"type {old!r} became {new!r}", narrowed=not before <= after
        )

    def ref(self, old: object, new: object, where: str) -> None:
        if not isinstance(old, str) or not isinstance(new, str):
            self.add(ChangeClass.BREAKING, where, f"$ref {old!r} became {new!r}")
            return
        if (old, new) in self.seen:
            return  # already being compared: a recursive definition
        self.seen.add((old, new))
        before, after = self._resolve(0, old), self._resolve(1, new)
        if before is _ABSENT or after is _ABSENT:
            self.add(ChangeClass.BREAKING, where, f"$ref {old!r} became {new!r}")
            return
        self.schema(before, after, f"{where}->{new}")

    def _members(self, side: int, items: object, refs: frozenset[str]) -> list[object]:
        """A union's alternatives, with a nested or referenced union expanded."""

        members: list[object] = []
        for item in _list(items):
            node, seen = item, refs
            if isinstance(item, dict) and set(item) - _ANNOTATIONS == {"$ref"}:
                ref = item["$ref"]
                if isinstance(ref, str) and ref not in refs:
                    node, seen = self._resolve(side, ref), refs | {ref}
            if isinstance(node, dict):
                nested = [key for key in _ALTERNATIVES if key in node]
                if len(nested) == 1 and set(node) - _ANNOTATIONS == set(nested):
                    members.extend(self._members(side, node[nested[0]], seen))
                    continue
            members.append(item)
        return members

    def _trial(self, before: object, after: object, where: str) -> list[Finding]:
        """The findings of one comparison, without recording them."""

        trial = _SchemaWalk(*self.roots)
        trial.seen = set(self.seen)
        trial.schema(before, after, where)
        return trial.found

    def alternatives(self, old: object, new: object, where: str) -> None:
        if old is _ABSENT or new is _ABSENT:
            self.relaxed_unless(where, "alternatives", narrowed=old is _ABSENT)
            return
        unmatched_new = self._members(1, new, frozenset())
        unmatched_old: list[object] = []
        for item in self._members(0, old, frozenset()):
            if item in unmatched_new:
                unmatched_new.remove(item)
            else:
                unmatched_old.append(item)
        # An old alternative is kept when some new one accepts at least as much.
        for item in list(unmatched_old):
            for index, candidate in enumerate(unmatched_new):
                found = self._trial(item, candidate, f"{where}/{index}")
                if all(entry.change is not ChangeClass.BREAKING for entry in found):
                    self.found.extend(found)
                    unmatched_old.remove(item)
                    del unmatched_new[index]
                    break
        if unmatched_old and len(unmatched_old) == len(unmatched_new):
            for index, (before, after) in enumerate(
                zip(unmatched_old, unmatched_new, strict=True)
            ):
                self.schema(before, after, f"{where}/{index}")
            return
        for item in unmatched_old:
            self.add(ChangeClass.BREAKING, where, f"alternative removed {item!r}")
        for item in unmatched_new:
            self.add(ChangeClass.MINOR, where, f"alternative added {item!r}")

    def positional(self, old: object, new: object, where: str) -> None:
        before, after = _list(old), _list(new)
        if len(before) != len(after):
            self.add(ChangeClass.BREAKING, where, "item count changed")
            return
        for index, (left, right) in enumerate(zip(before, after, strict=True)):
            self.schema(left, right, f"{where}/{index}")

    def additional(self, old: object, new: object, where: str) -> None:
        before = True if old is _ABSENT else old
        after = True if new is _ABSENT else new
        if before == after:
            return
        if isinstance(before, dict) and isinstance(after, dict):
            self.schema(before, after, where)
        else:
            widened = after is True or before is False
            self.relaxed_unless(where, "additionalProperties", narrowed=not widened)

    def subschema(self, old: object, new: object, where: str) -> None:
        if old is _ABSENT or new is _ABSENT:
            self.relaxed_unless(where, "subschema", narrowed=old is _ABSENT)
        else:
            self.schema(old, new, where)

    def bound(self, old: object, new: object, where: str, *, lower: bool) -> None:
        if old is _ABSENT or new is _ABSENT:
            self.relaxed_unless(where, "bound", narrowed=old is _ABSENT)
            return
        if not isinstance(old, (int, float)) or not isinstance(new, (int, float)):
            self.add(ChangeClass.BREAKING, where, "bound changed")
            return
        narrowed = new > old if lower else new < old
        self.relaxed_unless(where, f"bound {old!r} became {new!r}", narrowed=narrowed)

    def discriminator(self, old: object, new: object, where: str) -> None:
        if not isinstance(old, dict) or not isinstance(new, dict):
            # A routing hint only: the alternatives and their consts carry any break.
            self.add(ChangeClass.MINOR, where, "discriminator added or removed")
            return
        before, after = old, new
        if before.get("propertyName") != after.get("propertyName"):
            self.add(ChangeClass.BREAKING, where, "discriminator property changed")
        old_map = before.get("mapping") or {}
        new_map = after.get("mapping") or {}
        for name in sorted(old_map.keys() | new_map.keys()):
            if name not in new_map or (
                name in old_map and old_map[name] != new_map[name]
            ):
                self.add(
                    ChangeClass.BREAKING,
                    f"{where}/{name}",
                    "mapping removed or changed",
                )
            elif name not in old_map:
                self.add(ChangeClass.MINOR, f"{where}/{name}", "mapping added")

    def only_removal_relaxes(
        self, key: str, old: object, new: object, where: str
    ) -> None:
        if key == "default":
            # A new default comes with an optional field; changing one changes absent values.
            self.relaxed_unless(where, "default", narrowed=old is not _ABSENT)
        else:
            self.relaxed_unless(where, key, narrowed=new is not _ABSENT)

    def relaxed_unless(self, where: str, what: str, *, narrowed: bool) -> None:
        if narrowed:
            self.add(ChangeClass.BREAKING, where, f"{what} narrowed or changed")
        else:
            self.add(ChangeClass.MINOR, where, f"{what} relaxed or added")


def _const_as_enum(node: dict[str, object]) -> dict[str, object]:
    """``const: x`` is ``enum: [x]``, so a const that grew into an enum is a widening."""

    if "const" not in node:
        return node
    rest = {key: value for key, value in node.items() if key != "const"}
    return rest if "enum" in node else {**rest, "enum": [node["const"]]}


def _list(value: object) -> list[object]:
    return list(value) if isinstance(value, list) else [value]


# --- command line --------------------------------------------------------------


def _next(arguments: argparse.Namespace) -> int:
    found = findings(
        read_surface(
            arguments.base_schemas, arguments.base_vectors, arguments.base_init
        ),
        read_surface(
            arguments.head_schemas, arguments.head_vectors, arguments.head_init
        ),
    )
    change = change_class(found, src_changed=arguments.src_changed == "yes")
    for item in found:
        print(item, file=sys.stderr)
    print(f"change class: {change.word}", file=sys.stderr)
    try:
        version = next_version(arguments.base_version, arguments.major, change)
    except ContractVersionError as error:
        print(f"::error::{error}", file=sys.stderr)
        return 1
    print(version or "none")
    return 0


def _committed(arguments: argparse.Namespace) -> int:
    try:
        major = committed_major(
            arguments.file.read_text("utf-8"), arguments.base_version
        )
    except ContractVersionError as error:
        print(f"::error::{error}", file=sys.stderr)
        return 1
    print(major)
    return 0


def _stamp(arguments: argparse.Namespace) -> int:
    try:
        text = stamp(arguments.file.read_text("utf-8"), arguments.version)
    except ContractVersionError as error:
        print(f"::error::{error}", file=sys.stderr)
        return 1
    arguments.file.write_text(text, "utf-8")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Compute the contract version from the change class."
    )
    commands = parser.add_subparsers(dest="command", required=True)
    compute = commands.add_parser("next", help="print the next version, or none")
    for name in ("base-schemas", "base-vectors", "head-schemas", "head-vectors"):
        compute.add_argument(f"--{name}", type=Path, required=True)
    for name in ("base-init", "head-init"):
        compute.add_argument(
            f"--{name}", type=Path, help="the package __init__.py with SHARED_FUNCTIONS"
        )
    compute.add_argument("--base-version", required=True)
    compute.add_argument("--major", type=int, required=True)
    compute.add_argument("--src-changed", choices=("yes", "no"), required=True)
    compute.set_defaults(run=_next)
    committed = commands.add_parser(
        "committed-major", help="check the committed _version.py (G3), print its major"
    )
    committed.add_argument("file", type=Path)
    committed.add_argument("--base-version", required=True)
    committed.set_defaults(run=_committed)
    stamping = commands.add_parser(
        "stamp", help="write the computed version into a workspace _version.py"
    )
    stamping.add_argument("file", type=Path)
    stamping.add_argument("--version", required=True)
    stamping.set_defaults(run=_stamp)
    arguments = parser.parse_args(argv)
    return arguments.run(arguments)
