"""Typed views of what ``vonkctl`` reports: recipes, models, Sparks.

Only fields the sweep needs are read, each through a defensive accessor, so a
library row with an unavailable assessment (``library.assessment_unavailable``)
or a missing optional field never stops a sweep.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

from .vonkctl import Vonkctl

PAGE_LIMIT = "512"


def dig(value: Any, *path: str | int, default: Any = None) -> Any:
    """Read a nested JSON value, returning ``default`` when any step is absent."""
    current = value
    for step in path:
        if isinstance(step, int):
            if not isinstance(current, list) or step >= len(current):
                return default
            current = current[step]
        else:
            if not isinstance(current, dict) or step not in current:
                return default
            current = current[step]
    return default if current is None else current


def as_int(value: Any, default: int = 0) -> int:
    return value if isinstance(value, int) and not isinstance(value, bool) else default


@dataclass(frozen=True)
class Model:
    selector: str
    digest: str
    bytes: int
    local: str


@dataclass(frozen=True)
class Recipe:
    key: str
    title: str
    node_count: int
    usage: tuple[str, ...]
    engine: str
    creator: str
    model_digests: tuple[str, ...]
    model_selectors: tuple[str, ...]
    memory_bytes: int
    image_bytes: int
    artifact_bytes: int
    ports: tuple[int, ...]
    adapter: str
    aliases: tuple[str, ...]
    content_sha256: str
    revision_id: str
    local: str
    updated_at: str
    base_image: str
    checks: tuple[dict[str, Any], ...] = field(default=(), compare=False)

    @property
    def model_set(self) -> frozenset[str]:
        return frozenset(self.model_digests)

    @property
    def is_openai(self) -> bool:
        return self.adapter == "openai"

    @property
    def slug(self) -> str:
        return self.key.partition("/")[2]


def parse_recipe(row: dict[str, Any]) -> Recipe | None:
    key = dig(row, "selector")
    document = dig(row, "document", default={})
    if not isinstance(key, str) or not isinstance(document, dict):
        return None
    interfaces = [
        item
        for item in dig(document, "interfaces", default=[])
        if isinstance(item, dict)
    ]
    roles = [
        item
        for item in dig(document, "topology", "roles", default=[])
        if isinstance(item, dict)
    ]
    models = [
        item["model"]
        for item in dig(document, "models", default=[])
        if isinstance(item, dict) and isinstance(item.get("model"), dict)
    ]
    aliases: list[str] = []
    for item in interfaces:
        aliases += [
            a for a in dig(item, "model_aliases", default=[]) if isinstance(a, str)
        ]
    return Recipe(
        key=key,
        title=str(dig(row, "identity", "title", default=key)),
        node_count=as_int(
            dig(row, "node_count"), as_int(dig(document, "topology", "node_count"), 1)
        ),
        usage=tuple(str(u) for u in dig(row, "usage", default=[])),
        engine=str(
            dig(row, "engine", default=dig(document, "runtime", "engine", default=""))
        ),
        creator=str(dig(row, "creator", default="")),
        model_digests=tuple(
            sorted(
                str(m.get("content_sha256")) for m in models if m.get("content_sha256")
            )
        ),
        model_selectors=tuple(
            sorted(str(s) for s in dig(row, "model_selectors", default=[]))
        ),
        memory_bytes=as_int(dig(row, "resources", "memory_bytes")),
        image_bytes=as_int(dig(row, "resources", "image_bytes")),
        artifact_bytes=max(
            (
                as_int(dig(role, "resources", "disk", "artifact_bytes"))
                for role in roles
            ),
            default=0,
        ),
        ports=tuple(
            sorted({as_int(i.get("port")) for i in interfaces if i.get("port")})
        ),
        adapter=str(interfaces[0].get("adapter", "")) if interfaces else "",
        aliases=tuple(aliases),
        content_sha256=str(dig(row, "identity", "content_sha256", default="")),
        revision_id=str(dig(row, "identity", "recipe_revision_id", default="")),
        local=str(dig(row, "local", "controller", default="unknown")),
        updated_at=str(dig(row, "updated_at", default="")),
        base_image=str(
            dig(document, "execution", "build", "base_image", "digest", default="")
        ),
        checks=tuple(
            c
            for c in dig(document, "validation", "serving", "checks", default=[])
            if isinstance(c, dict)
        ),
    )


def parse_model(row: dict[str, Any]) -> Model | None:
    digest = dig(row, "identity", "content_sha256")
    selector = dig(row, "selector")
    if not isinstance(digest, str) or not isinstance(selector, str):
        return None
    return Model(
        selector=selector,
        digest=digest,
        bytes=as_int(dig(row, "resources", "disk_bytes")),
        local=str(dig(row, "local", "controller", default="unknown")),
    )


def fetch_pages(
    vk: Vonkctl, noun: str, collection: str
) -> tuple[list[dict[str, Any]], str]:
    """Every row of ``vonkctl <noun> library`` (following ``next_cursor``) and the library commit."""
    rows: list[dict[str, Any]] = []
    cursor: str | None = None
    seen: set[str] = set()
    commit = ""
    while True:
        args = [noun, "library", "--limit", PAGE_LIMIT]
        if cursor:
            args += ["--cursor", cursor]
        page = vk.call(*args, timeout=300)
        rows += [
            row for row in dig(page, collection, default=[]) if isinstance(row, dict)
        ]
        commit = commit or str(dig(page, "library", "commit", default=""))
        cursor = dig(page, "next_cursor")
        if not isinstance(cursor, str) or cursor in seen:
            return rows, commit
        seen.add(cursor)


@dataclass(frozen=True)
class Spark:
    id: str
    name: str
    memory_total: int
    disk_free: int
    online: bool


@dataclass(frozen=True)
class Presence:
    """One rank of a loaded run, as ``vonkctl fleet`` reports it."""

    node_id: str
    alias: str
    run_id: str
    healthy: bool
    route_state: str
    run_state: str


@dataclass(frozen=True)
class Fleet:
    sparks: tuple[Spark, ...]
    presences: tuple[Presence, ...]

    def by_ref(self, ref: str) -> Spark | None:
        return next((s for s in self.sparks if ref in (s.id, s.name)), None)


def parse_fleet(document: Any, default_memory: int) -> Fleet:
    sparks: list[Spark] = []
    presences: list[Presence] = []
    for node in dig(document, "nodes", default=[]):
        if not isinstance(node, dict):
            continue
        node_id = str(node.get("id", ""))
        sparks.append(
            Spark(
                id=node_id,
                name=str(node.get("display_name") or node.get("hostname") or node_id),
                memory_total=as_int(
                    dig(node, "inventory", "host_memory_total_bytes"), default_memory
                ),
                disk_free=as_int(dig(node, "inventory", "disk_free_bytes")),
                online=dig(node, "connection", "online_state") == "online",
            )
        )
        for item in dig(node, "loaded", default=[]):
            if isinstance(item, dict):
                presences.append(
                    Presence(
                        node_id=node_id,
                        alias=str(item.get("alias", "")),
                        run_id=str(item.get("run_id", "")),
                        healthy=item.get("healthy") is True,
                        route_state=str(item.get("route_state", "")),
                        run_state=str(item.get("run_state", "")),
                    )
                )
    return Fleet(tuple(sparks), tuple(presences))


def serving_run(
    fleet: Fleet, alias: str, node_ids: Iterable[str], node_count: int
) -> str | None:
    """The run id once every rank is healthy and published, else None."""
    wanted = set(node_ids)
    found = [p for p in fleet.presences if p.alias == alias and p.node_id in wanted]
    run_ids = {p.run_id for p in found}
    if (
        len(found) != node_count
        or len(run_ids) != 1
        or any(
            not (
                p.healthy and p.route_state == "published" and p.run_state == "running"
            )
            for p in found
        )
    ):
        return None
    return run_ids.pop() or None


def fetch_recipes(vk: Vonkctl) -> tuple[list[Recipe], str]:
    rows, commit = fetch_pages(vk, "recipe", "recipes")
    parsed = (parse_recipe(row) for row in rows)
    return [recipe for recipe in parsed if recipe is not None], commit


def fetch_models(vk: Vonkctl) -> dict[str, Model]:
    parsed = (parse_model(row) for row in fetch_pages(vk, "model", "models")[0])
    return {model.digest: model for model in parsed if model is not None}


def fetch_fleet(vk: Vonkctl, default_memory: int) -> Fleet:
    return parse_fleet(vk.call("fleet"), default_memory)
