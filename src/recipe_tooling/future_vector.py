"""The future vector (G4): the real index, as a newer minor release could publish it.

``build_future_vector`` takes the built index and its generated schema and returns
a self-consistent index that a consumer must still read:

* an unknown word (``zz-future-word``) at every site a consumer reader keeps,
* an unknown field (``zz_future_field``) on every object the schema closes,
* one extra recipe entry whose serving kind is an unknown executed discriminator
  (``zz.future-kind``), which a reader withholds and the rest of the index survives.

The sites are found by walking the schema over the real index, so a new site is
covered with no hand list. A site is *tolerant* when the contract reader keeps an
entry that carries the word there; a site where it does not (a discriminator, an
execution field) is left alone, never guessed. The vector is a reader test, not a
signed catalog: digests are recomputed so it stays self-consistent, and it keeps only
the entries that cover every schema shape (plus the Models they pin), because a full
copy with an unknown field on every object would pass the index size limit.
"""

from __future__ import annotations

import copy
import json
import re
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from typing import Any, Final

from vonk_forge_contracts import document_sha256
from vonk_forge_contracts.catalog_reader import (
    CatalogSection,
    EntryOutcome,
    FaultReason,
    SkippedEntry,
    UnrecognisedWordReport,
    read_catalog,
    read_catalog_model_entry,
    read_catalog_recipe_entry,
)

WORD: Final = "zz-future-word"
FIELD: Final = "zz_future_field"
KIND: Final = "zz.future-kind"
SLUG: Final = "zz-future-recipe"
_OPEN: Final = frozenset({WORD, KIND})
_DIGEST: Final = re.compile(r"[0-9a-f]{64}")
_ENTRY_SCHEMAS: Final = {
    "catalog_entities": "CatalogModelEntry",
    "recipes": "CatalogRecipeEntry",
}
_READERS: Final = {
    "catalog_entities": read_catalog_model_entry,
    "recipes": read_catalog_recipe_entry,
}

type JsonPath = tuple[str | int, ...]


@dataclass(frozen=True, slots=True)
class FutureVector:
    """The vector and what it proves: every site kept, every site left alone."""

    document: dict[str, Any]
    tolerant: tuple[str, ...]
    intolerant: tuple[str, ...]
    withheld: str


class Schema:
    """The generated OpenAPI schema, walked over a document."""

    def __init__(
        self, document: Mapping[str, Any], *, open_words: bool = False
    ) -> None:
        self.schemas: Mapping[str, Any] = document["components"]["schemas"]
        self.open = _OPEN if open_words else frozenset()

    def fits(self, value: object, node: Mapping[str, Any]) -> bool:
        """Whether ``value`` has the shape ``node`` describes (unknown fields allowed)."""
        if "$ref" in node:
            return self.fits(value, self.schemas[node["$ref"].rsplit("/", 1)[-1]])
        for key in ("anyOf", "oneOf"):
            if key in node:
                return any(self.fits(value, alternative) for alternative in node[key])
        if "const" in node:
            return value == node["const"] or value in self.open
        if "enum" in node:
            return value in node["enum"] or (
                isinstance(value, str) and value in self.open
            )
        match node.get("type"):
            case None:
                return True
            case "null":
                return value is None
            case "string":
                return isinstance(value, str)
            case "boolean":
                return isinstance(value, bool)
            case "integer":
                return isinstance(value, int) and not isinstance(value, bool)
            case "number":
                return isinstance(value, int | float) and not isinstance(value, bool)
            case "array":
                items = node.get("items", {})
                return isinstance(value, list) and all(
                    self.fits(item, items) for item in value
                )
            case "object":
                if not isinstance(value, dict):
                    return False
                properties = node.get("properties", {})
                extra = node.get("additionalProperties")
                return set(node.get("required", ())) <= value.keys() and all(
                    self.fits(item, properties[key])
                    if key in properties
                    else self.fits(item, extra)
                    if isinstance(extra, dict)
                    else True
                    for key, item in value.items()
                )
            case _:
                return False

    def pick(
        self, value: object, node: Mapping[str, Any], where: JsonPath
    ) -> tuple[str | None, Mapping[str, Any]]:
        """The concrete schema ``value`` follows: through ``$ref`` and the union member."""
        name: str | None = None
        while True:
            if "$ref" in node:
                reference: str = node["$ref"].rsplit("/", 1)[-1]
                name = reference
                node = self.schemas[reference]
                continue
            alternatives = node.get("anyOf") or node.get("oneOf")
            if alternatives:
                match = next((a for a in alternatives if self.fits(value, a)), None)
                if match is None:
                    raise ValueError(
                        f"no schema alternative fits {'.'.join(map(str, where))}"
                    )
                node = match
                continue
            return name, node

    def walk(
        self,
        value: object,
        node: Mapping[str, Any],
        path: JsonPath = (),
        trail: tuple[str, ...] = (),
    ) -> Iterator[tuple[str, JsonPath, Any, Mapping[str, Any], tuple[str, ...]]]:
        """Yield ``("object" | "word", path, value, schema node, trail)`` for a document.

        ``trail`` names the site by schema (property names and object schema names), so
        two union members that share a property name stay two sites.
        """
        name, concrete = self.pick(value, node, path)
        if isinstance(value, dict):
            here = (*trail, f"<{name}>") if name else trail
            yield "object", path, value, concrete, here
            properties = concrete.get("properties", {})
            extra = concrete.get("additionalProperties")
            for key, item in list(value.items()):
                if key in properties:
                    yield from self.walk(
                        item, properties[key], (*path, key), (*here, key)
                    )
                elif isinstance(extra, dict):
                    yield from self.walk(item, extra, (*path, key), (*here, "{}"))
        elif isinstance(value, list):
            for position, item in enumerate(value):
                yield from self.walk(
                    item, concrete.get("items", {}), (*path, position), (*trail, "[]")
                )
        elif isinstance(value, str) and len(concrete.get("enum", ())) > 1:
            yield "word", path, value, concrete, trail


APPEND: Final = -1


def _at(root: Any, path: JsonPath) -> Any:
    for step in path:
        root = root[step]
    return root


def _insert_word(entry: dict[str, Any], target: JsonPath) -> None:
    """Put the unknown word at a site: appended to its list (``APPEND``), else replacing the value."""
    *parents, last = target
    container = _at(entry, tuple(parents))
    if last == APPEND:
        container.append(WORD)
    else:
        container[last] = WORD


def _target(path: JsonPath) -> JsonPath:
    """The insertion point of a word: a list element is addressed by its list."""
    return (*path[:-1], APPEND) if isinstance(path[-1], int) else path


def _entry_shape(
    schema: Schema, section: str, entry: dict[str, Any]
) -> tuple[set[str], dict[str, set[JsonPath]]]:
    """The shapes an entry exercises, and each word site's insertion points (outside ``document``)."""
    keys: set[str] = set()
    sites: dict[str, set[JsonPath]] = {}
    for kind, path, _value, _node, trail in schema.walk(
        entry, schema.schemas[_ENTRY_SCHEMAS[section]]
    ):
        if kind == "object":
            keys.add("/".join(trail))
        elif path[0] != "document":
            site = f"{section}/{'/'.join(trail)}"
            sites.setdefault(site, set()).add(_target(path))
            keys.add(site)
    return keys, sites


def _covering(shapes: Mapping[str, set[str]]) -> set[str]:
    """Entry names that together cover every shape, preferring the most new shapes."""
    remaining = dict(shapes)
    wanted = set().union(*remaining.values())
    chosen: set[str] = set()
    covered: set[str] = set()
    while covered != wanted:
        name = max(sorted(remaining), key=lambda item: len(remaining[item] - covered))
        chosen.add(name)
        covered |= remaining.pop(name)
    return chosen


def _digests(value: object) -> Iterator[str]:
    if isinstance(value, str):
        if _DIGEST.fullmatch(value):
            yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from _digests(item)
    elif isinstance(value, list):
        for item in value:
            yield from _digests(item)


def _select(schema: Schema, index: dict[str, Any]) -> dict[str, Any]:
    """The index cut to the entries covering every shape, plus the Models they pin."""
    entries = {
        f"{section}:{position}": entry
        for section in _ENTRY_SCHEMAS
        for position, entry in enumerate(index[section])
    }
    chosen = _covering(
        {
            name: _entry_shape(schema, name.split(":")[0], entry)[0]
            for name, entry in entries.items()
        }
    )
    models = {
        entry["content_sha256"]: name
        for name, entry in entries.items()
        if name.startswith("catalog_entities")
    }
    queue = sorted(chosen)
    while queue:
        for digest in _digests(entries[queue.pop()]["document"]):
            name = models.get(digest)
            if name is not None and name not in chosen:
                chosen.add(name)
                queue.append(name)
    selected = {
        key: copy.deepcopy(value)
        for key, value in index.items()
        if key not in _ENTRY_SCHEMAS
    }
    for section in _ENTRY_SCHEMAS:
        selected[section] = [
            copy.deepcopy(entry)
            for name, entry in entries.items()
            if name.startswith(f"{section}:") and name in chosen
        ]
    return selected


def _tolerant_sites(
    schema: Schema, index: dict[str, Any]
) -> tuple[dict[str, list[tuple[str, int, JsonPath]]], list[str]]:
    """Each word site's insertion points; the sites the reader keeps are split from the rest."""
    occurrences: dict[str, list[tuple[str, int, JsonPath]]] = {}
    for section in _ENTRY_SCHEMAS:
        for position, entry in enumerate(index[section]):
            for site, targets in _entry_shape(schema, section, entry)[1].items():
                occurrences.setdefault(site, []).extend(
                    (section, position, target) for target in sorted(targets, key=str)
                )
    kept: dict[str, list[tuple[str, int, JsonPath]]] = {}
    refused: list[str] = []
    for site, where in sorted(occurrences.items()):
        section, position, target = where[0]
        probe = copy.deepcopy(index[section][position])
        _insert_word(probe, target)
        read = _READERS[section](probe)
        if read.outcome is EntryOutcome.KEEP_UNRECOGNISED and any(
            isinstance(problem, UnrecognisedWordReport) and problem.word == WORD
            for problem in read.problems
        ):
            kept[site] = where
        else:
            refused.append(site)
    return kept, refused


def _rebind(index: dict[str, Any]) -> None:
    """Make every ``content_sha256`` the digest of its (changed) document again."""
    mapping: dict[str, str] = {}
    while True:
        moved = False
        for section in _ENTRY_SCHEMAS:
            for entry in index[section]:
                digest = document_sha256(entry["document"])
                if entry["content_sha256"] != digest:
                    mapping[entry["content_sha256"]] = digest
                    moved = True
        if not moved:
            return
        _replace_strings(index, mapping)


def _replace_strings(value: Any, mapping: Mapping[str, str]) -> None:
    items = value.items() if isinstance(value, dict) else enumerate(value)
    for key, item in list(items):
        if isinstance(item, str):
            if item in mapping:
                value[key] = mapping[item]
        elif isinstance(item, dict | list):
            _replace_strings(item, mapping)
    digests = value.get("model_digests") if isinstance(value, dict) else None
    if isinstance(digests, list):
        # An evidence identity keeps its Model digests sorted and unique.
        digests[:] = sorted(set(digests))


def _withhold(entry: dict[str, Any]) -> dict[str, Any]:
    """A copy of a recipe entry whose serving kind a reader cannot execute."""
    extra = copy.deepcopy(entry)
    document = extra["document"]
    document["identity"]["slug"] = SLUG
    document["validation"]["serving"]["checks"][0]["kind"] = KIND
    for block in ("display", "api"):
        if isinstance(extra.get(block), dict):
            extra[block]["identity"]["slug"] = SLUG
    digest = document_sha256(document)
    extra["content_sha256"] = digest
    extra["package"]["recipe_content_sha256"] = digest
    extra["source_path"] = f"recipes/{SLUG}.json"
    extra["package"]["path"] = f"packages/{SLUG}.tar.gz"
    return extra


def words_in(schema_document: Mapping[str, Any], vector: dict[str, Any]) -> set[str]:
    """The word sites of ``vector`` that hold the unknown word."""
    schema = Schema(schema_document, open_words=True)
    return {
        f"{section}/{'/'.join(trail)}"
        for section in _ENTRY_SCHEMAS
        for entry in vector[section]
        for kind, path, value, _node, trail in schema.walk(
            entry, schema.schemas[_ENTRY_SCHEMAS[section]]
        )
        if kind == "word" and path[0] != "document" and value == WORD
    }


def missing_fields(
    schema_document: Mapping[str, Any], vector: dict[str, Any]
) -> list[str]:
    """Paths of closed objects of ``vector`` that lack the unknown field."""
    schema = Schema(schema_document, open_words=True)
    return [
        ".".join(map(str, path))
        for kind, path, value, node, _trail in schema.walk(
            vector, schema.schemas["CatalogIndex"]
        )
        if kind == "object"
        and node.get("additionalProperties") is False
        and FIELD not in value
    ]


def build_future_vector(
    index: dict[str, Any], schema_document: Mapping[str, Any]
) -> FutureVector:
    """The future vector for ``index``; raises ``ValueError`` if its reader would not survive it."""
    vector = _select(Schema(schema_document), index)
    sites, intolerant = _tolerant_sites(Schema(schema_document), vector)
    for where in sites.values():
        for section, position, target in where:
            _insert_word(vector[section][position], target)
    vector["vocabulary"]["capabilities"].append(
        {
            "name": WORD,
            "label": "Future word",
            "description": "A word a newer release adds.",
            "category": vector["vocabulary"]["categories"][0]["id"],
        }
    )
    opened = Schema(schema_document, open_words=True)
    for kind, _path, value, node, _trail in list(
        opened.walk(vector, opened.schemas["CatalogIndex"])
    ):
        if kind == "object" and node.get("additionalProperties") is False:
            value[FIELD] = 1
    _rebind(vector)
    vector["recipes"].append(_withhold(vector["recipes"][-1]))
    _check(vector, sites)
    return FutureVector(
        document=vector,
        tolerant=tuple(sorted(sites)),
        intolerant=tuple(sorted(intolerant)),
        withheld=f"vonk-forge/{SLUG}",
    )


def _check(vector: dict[str, Any], sites: Mapping[str, Any]) -> None:
    """The reader must keep everything but the withheld entry, and every word."""
    problems = read_catalog(vector).problems
    skipped = [problem for problem in problems if isinstance(problem, SkippedEntry)]
    others = [
        problem
        for problem in problems
        if not isinstance(problem, UnrecognisedWordReport | SkippedEntry)
    ]
    if (
        len(skipped) != 1
        or skipped[0].section is not CatalogSection.RECIPES
        or skipped[0].reason is not FaultReason.UNKNOWN_WORD
        or others
    ):
        detail = json.dumps([problem.model_dump(mode="json") for problem in problems])
        raise ValueError(f"the vector is not read as designed: {detail[:2000]}")
    reported = {
        problem.field
        for problem in problems
        if isinstance(problem, UnrecognisedWordReport)
    }
    fields = {
        ".".join(str(step) for step in target if isinstance(step, str))
        for where in sites.values()
        for _section, _position, target in where
    }
    if reported != fields:
        raise ValueError(
            f"words not kept: {sorted(fields - reported)}; unexpected: {sorted(reported - fields)}"
        )
