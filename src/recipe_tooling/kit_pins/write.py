"""Writing a lock and moving its consumers."""

from __future__ import annotations

import hashlib
import json
import re
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

from recipe_tooling.adapter_mirror import undeclare
from recipe_tooling.catalog_model import file_id
from recipe_tooling.kit_pins.archive import (
    archive_url,
    declare_archive,
    download_archive,
    recipe_text,
)
from recipe_tooling.kit_pins.check import declares_base
from recipe_tooling.kit_pins.consumers import (
    ARCHIVE_DIGEST,
    ARCHIVE_NAME,
    FROM_LINE,
    bound_model,
    image_reference,
    model_matches,
    split_image,
)
from recipe_tooling.kit_pins.declarations import declarations
from recipe_tooling.kit_pins.lock import (
    adapter_of,
    kit_of,
    lock_document,
    lock_text,
    locked_base,
    manifest_of,
)
from recipe_tooling.kit_pins.model import (
    LOCK,
    MANIFEST,
    MAX_SELECTION_ID,
    Declaration,
    KitError,
    Pin,
)
from recipe_tooling.kit_pins.resolve import derive
from recipe_tooling.kit_pins.upstream import Upstream
from recipe_tooling.paths import REPO_ROOT
from recipe_tooling.qualification.documents import model_display
from vonk_forge_contracts import ModelDefinition, document_sha256, published_json

ROOT = REPO_ROOT


def apply_lock(
    root: Path,
    recipe_id: str,
    recipe: dict[str, Any],
    source: Upstream,
    fetch_archive: Callable[[str, str], bytes] = download_archive,
    declare: Callable[[Path, Path, str], str] = declare_archive,
) -> list[str]:
    """Write ``kit-lock.json`` and move the consumers to it; returns what changed."""
    adapter = adapter_of(root, recipe)
    kit = kit_of(recipe)
    if adapter is None or kit is None:
        raise KitError(f"{recipe_id}: no adapter directory or kit commit")
    manifest = manifest_of(adapter)
    wanted = declarations(manifest)
    if not wanted:
        raise KitError(f"{recipe_id}: {MANIFEST} declares no kit dependencies")
    pins = derive(kit[0], kit[1], wanted, source)
    changed: list[str] = []
    previous = locked_base(adapter) or (
        (manifest["source"]["repo"], manifest["source"]["commit"])
        if "source" in manifest
        else None
    )
    new = lock_text(lock_document(kit[0], kit[1], pins))
    lock_path = adapter / LOCK
    if not lock_path.is_file() or lock_path.read_text(encoding="utf-8") != new:
        lock_path.write_text(new, encoding="utf-8")
        changed.append(LOCK)
    if "source" in manifest and declares_base(wanted):
        del manifest["source"]
        (adapter / MANIFEST).write_text(
            json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        changed.append(f"{MANIFEST} (hand-authored source pin removed)")
    by_name = {d.name: d for d in wanted}
    current = recipe
    for pin in pins:
        if pin.datasource == "git-ref" and pin.name == "base":
            changed.extend(
                _move_consumers(
                    root, adapter, previous, pin, kit, fetch_archive, declare
                )
            )
        elif pin.datasource == "hf-revision":
            current, moved = _rebind_model(
                root, current, by_name[pin.name], pin, source
            )
            changed.extend(moved)
        elif pin.datasource == "oci-image":
            current, moved = _move_image(adapter, current, pin)
            changed.extend(moved)
    if current != recipe:
        (root / "recipes" / f"{recipe_id}.json").write_text(
            recipe_text(current), encoding="utf-8"
        )
        changed.append(f"recipes/{recipe_id}.json")
    return changed


def _move_consumers(
    root: Path,
    adapter: Path,
    previous: tuple[str, str] | None,
    base: Pin,
    kit: tuple[str, str],
    fetch_archive: Callable[[str, str], bytes],
    declare: Callable[[Path, Path, str], str],
) -> list[str]:
    changed: list[str] = []
    dockerfile = adapter / "Dockerfile"
    if not dockerfile.is_file():
        return changed
    text = dockerfile.read_text(encoding="utf-8")
    original = text
    vendor = adapter / "vendor"
    old_archives = sorted(vendor.glob("*.tar.gz")) if vendor.is_dir() else []
    if old_archives:
        named = next((a for a in old_archives if base.immutable in a.name), None)
        digest = ""
        if named is None:
            template = ARCHIVE_NAME.match(old_archives[0].name)
            stem = template["stem"] if template else old_archives[0].name.split(".")[0]
            data = fetch_archive(base.repo, base.immutable)
            named = vendor / f"{stem}-{base.immutable}.tar.gz"
            named.write_bytes(data)
            digest = declare(root, named, archive_url(base.repo, base.immutable))
            changed.append(f"vendor/{named.name} (new base archive, mirrored)")
        digest = digest or hashlib.sha256(named.read_bytes()).hexdigest()
        for old in old_archives:
            if old == named:
                continue
            old_digest = hashlib.sha256(old.read_bytes()).hexdigest()
            old_commit = ARCHIVE_NAME.match(old.name)
            text = text.replace(old.name, named.name).replace(old_digest, digest)
            if old_commit:
                text = text.replace(old_commit["commit"], base.immutable)
            old.unlink()

            undeclare([old.relative_to(root).as_posix()], root=root)
            changed.append(f"vendor/{old.name} (removed)")
        found = ARCHIVE_DIGEST.search(text)
        if found and found["digest"] != digest:
            text = text.replace(found["digest"], digest)
    if previous is not None and previous[1] != base.immutable:
        text = text.replace(previous[1], base.immutable)
    # the kit's own commit, where the Dockerfile cites it (a recipe-patches source label)
    cited = re.search(
        rf"github\.com/{re.escape(kit[0])}/tree/(?P<commit>[0-9a-f]{{40}})", text
    )
    if cited and cited["commit"] != kit[1]:
        text = text.replace(cited["commit"], kit[1])
    if text != original:
        dockerfile.write_text(text, encoding="utf-8")
        changed.append(
            "Dockerfile (pins; review its header comment and licence labels)"
        )
    return changed


def moved_model(
    old: dict[str, Any], inventory: list[dict[str, Any]], revision: str
) -> dict[str, Any]:
    """The Model document ``old`` moved to the snapshot ``inventory`` of ``revision``.

    The curation is kept: identity and the files the document kept (a file the snapshot
    no longer has is dropped). Its licence, capabilities and modalities are display
    facts and travel with the Model's display file (``moved_model_display``). A file the snapshot adds is
    catalogued unless it is metadata (a README), as the old document may have left such
    files out on purpose. The slug, version and variant follow the revision.
    """

    by_path = {f["path"]: f for f in inventory}
    document = json.loads(json.dumps(old))
    old_revision = old["source"]["revision"]
    document["source"]["revision"] = revision
    identity = document["identity"]
    slug = identity["slug"]
    if slug.endswith(old_revision[:8]):
        identity["slug"] = slug[: -len(old_revision[:8])] + revision[:8]
    elif not slug.endswith(revision[:8]):
        identity["slug"] = f"{slug}-{revision[:8]}"
    for key in ("version", "variant"):
        if isinstance(identity.get(key), str) and old_revision[:8] in identity[key]:
            identity[key] = identity[key].replace(old_revision[:8], revision[:8])
    known = {entry["path"] for entry in old["files"]}
    files = [dict(entry) for entry in old["files"] if entry["path"] in by_path]
    files.extend(
        {"path": path, "roles": by_path[path]["roles"]}
        for path in sorted(set(by_path) - known)
        if by_path[path]["roles"] != ["metadata"]
    )
    for entry in files:
        fresh = by_path[entry["path"]]
        entry["sha256"] = fresh["sha256"]
        entry["size_bytes"] = fresh["installed_bytes"]
        # Parts follow the snapshot: a file the source now publishes split (or whole)
        # changes shape, and the whole file's digest above stays the identity.
        if "parts" in fresh:
            entry["parts"] = fresh["parts"]
        else:
            entry.pop("parts", None)
        entry["id"] = file_id(entry["path"], fresh["sha256"])
    document["files"] = sorted(files, key=lambda entry: entry["path"])
    ModelDefinition.model_validate(document)
    return document


def moved_model_display(root: Path, old_slug: str, new_slug: str) -> bytes:
    """The display of ``old_slug`` written for the Model that replaces it as ``new_slug``."""

    display = model_display(old_slug, root)
    moved = display.model_copy(
        update={"identity": display.identity.model_copy(update={"slug": new_slug})}
    )
    return (
        json.dumps(published_json(moved), indent=2, sort_keys=True, ensure_ascii=False)
        + "\n"
    ).encode()


def catalogued_model(root: Path, repo: str, revision: str) -> dict[str, Any] | None:
    """The Model document of a Hugging Face repository at a revision, when catalogued."""
    for path in sorted((root / "models").glob("*.json")):
        document = json.loads(path.read_text(encoding="utf-8"))
        if model_matches(document, repo, revision):
            return document
    return None


def selection_id(model_id: str, example: dict[str, Any], file_id: str) -> str:
    """The id of a selected file, spelled as the recipe spells its other ones.

    A selected file's id is its file id behind a prefix (often the model's role), cut to
    the contract's bound; the example selected file shows the prefix.
    """
    old = example["file_id"]
    for length in range(len(old), 0, -1):
        if example["id"].endswith(old[:length]):
            prefix = example["id"][: len(example["id"]) - length]
            return (prefix + file_id)[:MAX_SELECTION_ID]
    return f"{model_id}-{file_id}"[:MAX_SELECTION_ID]


def rebind(
    recipe: dict[str, Any], model_id: str, old: dict[str, Any], new: dict[str, Any]
) -> dict[str, Any]:
    """The recipe with its Model ``model_id`` moved from the ``old`` document to ``new``.

    A recipe that mounts every file of the old Model mounts every file of the new one (a
    repository switch has no file in common); otherwise each selected file is followed by
    its path and one the new Model lacks is an error. The disk envelope follows the size
    of what is selected.
    """
    document = json.loads(json.dumps(recipe))
    reference = next(m for m in document["models"] if m["id"] == model_id)
    old_by_id = {f["id"]: f for f in old["files"]}
    chosen = [old_by_id[f["file_id"]] for f in reference["files"]]
    new_by_path = {f["path"]: f for f in new["files"]}
    if {f["id"] for f in chosen} == set(old_by_id):
        shapes = {
            json.dumps([f["mount"], f["roles"]], sort_keys=True)
            for f in reference["files"]
        }
        if len(shapes) != 1:
            raise KitError(
                f"Model '{model_id}' mounts every file, but not all the same way; "
                "rebind it by hand"
            )
        template = reference["files"][0]
        targets = list(new["files"])
    else:
        missing = [f["path"] for f in chosen if f["path"] not in new_by_path]
        if missing:
            raise KitError(
                f"the recipe selects files the new Model lacks: {', '.join(missing[:5])}"
            )
        template = None
        targets = [new_by_path[f["path"]] for f in chosen]
    selected = []
    for item in targets:
        if template is not None:
            entry = {
                "file_id": item["id"],
                "id": selection_id(model_id, template, item["id"]),
                "mount": template["mount"],
                "roles": template["roles"],
            }
        else:
            old_entry = next(
                e
                for e in reference["files"]
                if old_by_id[e["file_id"]]["path"] == item["path"]
            )
            entry = dict(
                old_entry,
                file_id=item["id"],
                id=selection_id(model_id, old_entry, item["id"]),
            )
        selected.append(entry)
    if len({e["id"] for e in selected}) != len(selected):
        raise KitError(
            f"Model '{model_id}': selected file ids collide once cut to 64 characters"
        )
    reference["files"] = selected
    reference["model"]["publisher"] = new["identity"]["publisher"]
    reference["model"]["slug"] = new["identity"]["slug"]
    reference["model"]["content_sha256"] = model_digest(new)
    delta = sum(f["size_bytes"] for f in targets) - sum(f["size_bytes"] for f in chosen)
    for role in document["topology"]["roles"]:
        role["resources"]["disk"]["artifact_bytes"] += delta
    return document


def model_digest(document: dict[str, Any]) -> str:
    sys.path.insert(0, str(ROOT / "contracts" / "src"))

    return document_sha256(document)


def _rebind_model(
    root: Path,
    recipe: dict[str, Any],
    declaration: Declaration,
    pin: Pin,
    source: Upstream,
) -> tuple[dict[str, Any], list[str]]:
    assert declaration.binds is not None
    found = bound_model(root, recipe, declaration.binds)
    if found is None:
        raise KitError(
            f"the recipe selects no catalogued Model '{declaration.binds}' for the kit's "
            f"{declaration.name}"
        )
    _, old = found
    if model_matches(old, pin.repo, pin.immutable):
        return recipe, []
    changed: list[str] = []
    new = catalogued_model(root, pin.repo, pin.immutable)
    if new is None:
        old_repo = str((old.get("source") or {}).get("repository", ""))
        if old_repo.lower() != f"https://huggingface.co/{pin.repo}".lower():
            raise KitError(
                f"the kit locks {pin.repo}@{pin.immutable}, another repository than the "
                f"recipe's Model: run tools/catalog-hf-model --repository {pin.repo} "
                f"--revision {pin.immutable} (with its identity, licence and capability "
                "flags), then lock again"
            )
        # another revision of the same repository: the old document's curation moves along
        new = moved_model(
            old, source.hf().inventory(pin.repo, pin.immutable), pin.immutable
        )
        path = root / "models" / f"{new['identity']['slug']}.json"
        path.write_text(
            json.dumps(new, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        changed.append(f"models/{new['identity']['slug']}.json (catalogued)")
        display_path = root / "display" / "models" / f"{new['identity']['slug']}.json"
        display_path.write_bytes(
            moved_model_display(root, old["identity"]["slug"], new["identity"]["slug"])
        )
        changed.append(f"display/models/{new['identity']['slug']}.json (catalogued)")
    moved = rebind(recipe, declaration.binds, old, new)
    changed.append(f"Model '{declaration.binds}' -> {new['identity']['slug']}")
    slug = old["identity"]["slug"]
    if slug != new["identity"]["slug"] and not any(
        m["model"]["slug"] == slug
        for path in sorted((root / "recipes").glob("*.json"))
        for other in [json.loads(path.read_text(encoding="utf-8"))]
        if other["identity"]["slug"] != recipe["identity"]["slug"]
        for m in other["models"]
    ):
        (root / "models" / f"{slug}.json").unlink()
        (root / "display" / "models" / f"{slug}.json").unlink(missing_ok=True)
        changed.append(f"models/{slug}.json (removed: no recipe uses it)")
    return moved, changed


def _move_image(
    adapter: Path, recipe: dict[str, Any], pin: Pin
) -> tuple[dict[str, Any], list[str]]:
    changed: list[str] = []
    dockerfile = adapter / "Dockerfile"
    if dockerfile.is_file():
        text = dockerfile.read_text(encoding="utf-8")

        def move(match: re.Match[str]) -> str:
            image = match["image"]
            if "$" in image or split_image(image)[0].lower() != pin.repo.lower():
                return match[0]
            return match["lead"] + image_reference(pin, image)

        moved = FROM_LINE.sub(move, text)
        if moved != text:
            dockerfile.write_text(moved, encoding="utf-8")
            changed.append("Dockerfile (FROM digest)")
    document = json.loads(json.dumps(recipe))
    base = document.get("execution", {}).get("build", {}).get("base_image")
    if base and str(base.get("repository", "")).lower() == pin.repo.lower():
        base["digest"] = pin.immutable.removeprefix("sha256:")
    return document, changed
