"""Typed views of what ``vonkctl`` reports: recipes, models, Sparks.

Only fields the sweep needs are read, each through a defensive accessor, so a
library row with an unavailable assessment (``library.assessment_unavailable``)
or a missing optional field never stops a sweep.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import Any

from .vonkctl import Vonkctl

PAGE_LIMIT = "512"
MAX_PAGE_FAILURES = 3


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
    cache_ready: bool = False  # the library's own assessment: exact NAS assets ready
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
        cache_ready=dig(row, "assessment", "cache", "state") == "ready",
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


class Listing[T]:
    """A library listing read one page at a time and kept between passes.

    The Controller bounds a library read (a slow page times out) and invalidates
    a cursor when the selection changes under it. So a failed page is retried
    on the next call, an invalid cursor restarts the pass from the first page,
    and rows already read stay usable: scheduling never waits for a complete
    pass. Rows absent from a complete, uninterrupted pass are dropped.
    """

    def __init__(
        self,
        vk: Vonkctl,
        noun: str,
        collection: str,
        parse: Callable[[dict[str, Any]], T | None],
        key: Callable[[T], str],
    ) -> None:
        self.vk = vk
        self.noun = noun
        self.collection = collection
        self.parse = parse
        self.key = key
        self.rows: dict[str, T] = {}
        self.cursor: str | None = None
        self.seen: set[str] = set()
        self.in_pass = False
        self.passes = 0
        self.pass_started_at = 0.0
        self.pass_done_at = -1e18
        self.complete_started_at = 0.0  # when the newest *complete* pass began
        self.commit = ""
        self.last_error = ""
        self.failures = 0

    def begin(self, now: float) -> None:
        self.in_pass = True
        self.pass_started_at = now
        self._restart()

    def _restart(self) -> None:
        self.cursor = None
        self.seen = set()
        self.failures = 0

    def step(self, now: float = 0.0) -> bool:
        """Read one page. True when this call completed a pass."""
        args = [self.noun, "library", "--limit", PAGE_LIMIT]
        if self.cursor:
            args += ["--cursor", self.cursor]
        reply = self.vk.run(*args, timeout=300)
        if not reply.ok or not isinstance(reply.document, dict):
            self.last_error = reply.error_text
            self.failures += 1
            if "cursor" in self.last_error or self.failures >= MAX_PAGE_FAILURES:
                self._restart()  # the selection changed under the cursor: start again
            return False
        self.last_error = ""
        self.failures = 0
        for row in dig(reply.document, self.collection, default=[]):
            item = self.parse(row) if isinstance(row, dict) else None
            if item is not None:
                self.rows[self.key(item)] = item
                self.seen.add(self.key(item))
        self.commit = self.commit or str(
            dig(reply.document, "library", "commit", default="")
        )
        following = dig(reply.document, "next_cursor")
        if isinstance(following, str) and following != self.cursor:
            self.cursor = following
            return False
        for gone in set(self.rows) - self.seen:
            del self.rows[gone]
        self.in_pass = False
        self.passes += 1
        self.pass_done_at = now
        self.complete_started_at = self.pass_started_at
        return True


def recipe_listing(vk: Vonkctl) -> Listing[Recipe]:
    return Listing(vk, "recipe", "recipes", parse_recipe, lambda r: r.key)


def model_listing(vk: Vonkctl) -> Listing[Model]:
    return Listing(vk, "model", "models", parse_model, lambda m: m.digest)


def read_fully[T](listing: Listing[T], max_steps: int = 200) -> None:
    listing.begin(0.0)
    for _ in range(max_steps):
        if listing.step():
            return
    raise RuntimeError(
        f"the {listing.noun} library could not be read: {listing.last_error}"
    )


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
    listing = recipe_listing(vk)
    read_fully(listing)
    return list(listing.rows.values()), listing.commit


def fetch_models(vk: Vonkctl) -> dict[str, Model]:
    listing = model_listing(vk)
    read_fully(listing)
    return dict(listing.rows)


def fetch_fleet(vk: Vonkctl, default_memory: int) -> Fleet:
    return parse_fleet(vk.call("fleet"), default_memory)
