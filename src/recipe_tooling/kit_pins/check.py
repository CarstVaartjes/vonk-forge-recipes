"""Checking a recipe against its lock."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from recipe_tooling.kit_pins.consumers import consumers
from recipe_tooling.kit_pins.declarations import declarations, read_declared
from recipe_tooling.kit_pins.image_scan import (
    EXCEPTIONS,
    image_problems,
    kit_images,
    kit_scope,
    read_exceptions,
    unneeded_exception,
)
from recipe_tooling.kit_pins.lock import (
    adapter_of,
    kit_of,
    lock_document,
    manifest_of,
    read_lock,
)
from recipe_tooling.kit_pins.model import (
    IMMUTABLE_KEYS,
    LOCK,
    MANIFEST,
    Declaration,
    KitError,
    Pin,
)
from recipe_tooling.kit_pins.sources import withdrawn
from recipe_tooling.kit_pins.upstream import Upstream


def check_images(
    root: Path,
    recipes: dict[str, dict[str, Any]],
    source: Upstream,
    names: Iterable[str] | None = None,
    notice: Callable[[str], None] = lambda message: None,
) -> list[str]:
    """The prebuilt-image rule over recipes built from a kit (one scan per kit commit).

    An exception the recipe no longer needs is reported to ``notice`` (to be removed), not
    failed: a recipe that switched to the kit's image in another change must not break the
    check on the exceptions file that change did not touch.
    """
    exceptions = read_exceptions(root)
    problems = [
        f"{EXCEPTIONS}: '{recipe_id}' is not a recipe"
        for recipe_id in sorted(set(exceptions) - set(recipes))
    ]
    wanted = {
        recipe_id: (*kit, kit_scope(recipes[recipe_id]))
        for recipe_id in sorted(names if names is not None else recipes)
        if (kit := kit_of(recipes[recipe_id])) is not None
    }
    for recipe_id in [r for r in wanted if withdrawn(root, recipes[r])]:
        notice(f"{recipe_id}: its kit is withdrawn (source.unverified); not rescanned")
        del wanted[recipe_id]

    def scan(key: tuple[str, str, str]) -> list[dict[str, Any]] | KitError:
        try:
            return kit_images(source, *key)
        except (KitError, RuntimeError) as error:  # incl. UpstreamBusy
            return KitError(str(error))

    keys = sorted(set(wanted.values()))
    with ThreadPoolExecutor(max_workers=8) as pool:
        scans = dict(zip(keys, pool.map(scan, keys), strict=True))
    for recipe_id, key in wanted.items():
        found = scans[key]
        if isinstance(found, KitError):
            problems.append(f"{recipe_id}: {found}")
            continue
        problems.extend(
            image_problems(root, recipe_id, recipes[recipe_id], found, exceptions)
        )
        unneeded = unneeded_exception(
            recipe_id, recipes[recipe_id], root, found, exceptions
        )
        if unneeded:
            notice(f"remove the {EXCEPTIONS} entry of {unneeded}")
    return problems


# ---------------------------------------------------------------------------
# Check ("is the lock up to date")
# ---------------------------------------------------------------------------


def declares_base(wanted: Iterable[Declaration]) -> bool:
    """Whether the kit declares the base the adapter's vendored files come from."""
    return any(d.name == "base" and d.datasource == "git-ref" for d in wanted)


def lock_difference(lock: dict[str, Any], expected: dict[str, Any]) -> str:
    """What differs between a committed lock and the one its kit declaration yields."""
    parts: list[str] = []
    if lock.get("kit") != expected["kit"]:
        parts.append(
            f"the lock is for kit {lock.get('kit', {}).get('commit', '?')[:12]}"
        )
    old, new = lock.get("dependencies", {}), expected["dependencies"]
    for name in sorted(set(old) | set(new)):
        if old.get(name) == new.get(name):
            continue
        fields = sorted(
            key
            for key in set(old.get(name) or {}) | set(new.get(name) or {})
            if (old.get(name) or {}).get(key) != (new.get(name) or {}).get(key)
        )
        parts.append(f"'{name}' differs in {', '.join(fields) or 'its presence'}")
    return "which declares otherwise: " + "; ".join(parts)


def check_recipe(
    root: Path, recipe_id: str, recipe: dict[str, Any], source: Upstream
) -> list[str]:
    """Reasons the recipe's committed lock is not what its kit declares ([] when current)."""
    adapter = adapter_of(root, recipe)
    if adapter is None or not (adapter / MANIFEST).is_file():
        return []
    manifest = manifest_of(adapter)
    try:
        wanted = declarations(manifest)
    except KitError as error:
        return [f"{recipe_id}: {error}"]
    if not wanted:
        return []
    label = f"{recipe_id}"
    kit = kit_of(recipe)
    if kit is None:
        return [
            f"{label}: the recipe does not pin a kit commit (provenance.source_reference)"
        ]
    problems: list[str] = []
    if "source" in manifest and declares_base(wanted):
        problems.append(
            f"{label}: {MANIFEST} keeps a hand-authored 'source' pin beside the kit "
            "declaration; remove it (the lock derives it), or use an explicit override"
        )
    lock = read_lock(adapter)
    if lock is None:
        return [*problems, f"{label}: no {LOCK}; run tools/kit-pins lock {recipe_id}"]
    # A withdrawn kit (GitHub no longer serves its commit) is not re-read: the lock's own
    # declaration stands, and everything below still checks it against the recipe.
    gone = withdrawn(root, recipe)
    try:
        # Offline in the resolving direction: the lock's own value is used for the pin, so
        # only the declaring file at the pinned kit commit is fetched.
        pins = []
        for declaration in wanted:
            locked = lock.get("dependencies", {}).get(declaration.name)
            if locked is None:
                problems.append(f"{label}: {LOCK} lacks '{declaration.name}'")
                continue
            repo, ref, reason = (
                (locked["repo"], locked["ref"], locked.get("overridden"))
                if gone
                else read_declared(source, kit[0], kit[1], declaration)
            )
            pins.append(
                Pin(
                    declaration.name,
                    repo,
                    ref,
                    locked.get(IMMUTABLE_KEYS[declaration.datasource], ""),
                    "",
                    locked.get("kind", ""),
                    reason,
                    declaration.datasource,
                )
            )
        expected = lock_document(kit[0], kit[1], pins)
        if lock != expected:
            problems.append(
                f"{label}: {LOCK} is stale: the recipe pins kit {kit[0]}@{kit[1][:12]}, "
                f"{lock_difference(lock, expected)}; run tools/kit-pins lock {recipe_id}"
            )
    except KitError as error:
        problems.append(f"{label}: {error}")
        return problems
    by_name = {d.name: d for d in wanted}
    for pin in pins:
        problems.extend(
            f"{label}: {p}"
            for p in consumers(root, adapter, recipe, by_name[pin.name], pin)
        )
    return problems
