"""Assessing one recipe against its upstream."""

from __future__ import annotations

import difflib
import json
import re
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any
from urllib.parse import quote

from recipe_tooling import base_image_holds as base_holds
from recipe_tooling import catalog_hf_model as catalog
from recipe_tooling import model_revision_holds as model_holds
from recipe_tooling import protocol_wheel
from recipe_tooling import vendored_upstream as vendored
from recipe_tooling.kit_pins import consumers as kit_consumers
from recipe_tooling.kit_pins import lock as kit_lock
from recipe_tooling.kit_pins import write as kit_write
from recipe_tooling.refresh_upstream import catalog_view as refresh_catalog_view
from recipe_tooling.refresh_upstream import classification as refresh_classification
from recipe_tooling.refresh_upstream import evidence as refresh_evidence
from recipe_tooling.refresh_upstream import http as refresh_http
from recipe_tooling.refresh_upstream import items as refresh_items
from recipe_tooling.refresh_upstream import model as refresh_model
from recipe_tooling.refresh_upstream import pinning as refresh_pinning
from recipe_tooling.upstream_http import UpstreamBusy
from vonk_forge_contracts import RecipeDefinition, document_sha256

# ---------------------------------------------------------------------------
# Assessment: decide, and prepare the edits for a mechanical refresh
# ---------------------------------------------------------------------------


class Assessor:
    def __init__(
        self,
        catalog_: refresh_catalog_view.Catalog,
        http: refresh_http.Http,
        cache_dir: Path | None,
        embedded: dict[str, Any],
    ) -> None:
        self.catalog = catalog_
        self.http = http
        self.cache_dir = cache_dir
        self.targets: dict[tuple[str, str], refresh_model.Target] = {}
        self.trees: dict[tuple[str, str], list[Any]] = {}
        self.target_lock = threading.Lock()
        self.embedded_candidates: dict[str, list[dict[str, Any]]] = {}
        for record in embedded.get("recipes", []):
            if not isinstance(record, dict) or not isinstance(
                record.get("recipe_id"), str
            ):
                continue
            for item in record.get("inputs", []):
                if (
                    isinstance(item, dict)
                    and item.get("candidate_status") == "advanced"
                ):
                    self.embedded_candidates.setdefault(record["recipe_id"], []).append(
                        item
                    )

    def target(self, provider: str, repo: str) -> refresh_model.Target:
        key = (provider, repo.lower())
        with self.target_lock:
            found = self.targets.get(key)
        if found is None:
            found = refresh_pinning.resolve(self.http, provider, repo)
            with self.target_lock:
                self.targets[key] = found
        return found

    def assess(self, recipe_id: str) -> refresh_model.Assessment:
        result = refresh_items.collect_items(
            self.catalog, recipe_id, self.http, self.target
        )
        for item in self.embedded_candidates.get(recipe_id, []):
            repository = item.get("repository")
            pinned = item.get("revision")
            candidate = item.get("candidate_revision")
            candidate_url = item.get("candidate_url")
            result.evidence.append(
                f"### Embedded Git source candidate: {repository}\n\n"
                f"- embedded commit: `{pinned}`\n"
                f"- advertised default-branch candidate: `{candidate}` - {candidate_url}\n"
                f"- candidate observed at: {item.get('candidate_observed_at')}\n"
                "- this is review evidence, not a configured upgrade channel or compatibility recommendation; no pin was changed.\n"
            )
        if not result.drifted or result.unreachable:
            return result
        if result.reasons:
            for item in result.items:
                result.evidence.append(
                    f"### Held source candidate: {item.label}\n\n"
                    f"- current pin: `{item.current}`\n"
                    f"- proposed upstream pin: `{item.target.sha}` - {item.target.url}\n"
                    "- no automatic retarget was prepared because this recipe has another review blocker.\n"
                )
            return result
        recipe = self.catalog.recipes[recipe_id]
        edits: dict[str, bytes | None] = {}
        new_recipe = json.loads(json.dumps(recipe))
        hold = base_holds.holds(self.catalog.root).get(recipe_id)
        model_hold = model_holds.holds(self.catalog.root).get(recipe_id)
        for item in result.items:
            moved = item.image[0] if item.image else item.repo
            if (
                (item.image or item.kind == "image")
                and base_holds.moves_held_image(hold, moved)
            ) or (
                item.provider == "github"
                and base_holds.follows_held_source(hold, item.repo)
            ):
                # A held base image stays (policy/base-image-holds.json): no move, no review.
                item.noop = True
                result.declined.append(item.label)
                result.evidence.append(
                    f"### {item.label}\n\n- newest `{item.target.sha}` is not followed: "
                    f"{base_holds.describe(hold)}\n"  # pyright: ignore[reportArgumentType]
                )
                continue
            if item.provider == "huggingface" and model_holds.declines(
                model_hold, item.repo, item.target.sha
            ):
                # The declined revision stays declined (policy/model-revision-holds.json); a
                # newer one is assessed again.
                item.noop = True
                result.declined.append(item.label)
                result.evidence.append(
                    f"### {item.label}\n\n- newest `{item.target.sha}` is not followed: "
                    f"{model_holds.describe(model_hold)}\n"  # pyright: ignore[reportArgumentType]
                )
                continue
            try:
                if item.unverified is not None or item.unmark or item.uncompared:
                    self._mark(item, recipe, new_recipe, result, edits)
                elif item.kind == "model":
                    self._model(item, recipe, new_recipe, result, edits)
                elif item.kind == "image":
                    self._image(item, recipe, new_recipe, result, edits)
                elif item.provider == "github":
                    self._github(item, recipe, new_recipe, result, edits)
                else:
                    # A Hugging Face source reference is a label: what runs is the Model,
                    # judged as its own item. It rides along with a move that runs.
                    item.noop = True
                    result.evidence.append(
                        f"### {item.label}\n\n- pinned `{item.current}`\n- newest `{item.target.sha}`"
                        f" ({item.target.date}) - {item.target.url}\n"
                        "- the source reference alone is not a refresh.\n"
                    )
                    self._retarget_reference(new_recipe, item)
            except UpstreamBusy as error:
                result.unreachable.append(f"{item.label}: {str(error)[:200]}")
            except (RuntimeError, SystemExit, KeyError, ValueError, OSError) as error:
                result.reasons.append(
                    f"{item.label}: could not be assessed: {str(error)[:300]}"
                )
        result.items = [item for item in result.items if not item.noop]
        if result.prepared:
            moves = [item for item in result.items if not item.unmark]
            if moves:  # dropping a provenance mark alone is no new release
                version, date = refresh_classification.released(recipe, moves)
                new_recipe["release"] = {"version": version, "released_at": date}
                result.version = (version, date)
            if new_recipe != recipe:
                if new_recipe.get("runtime", {}).get("engine") == "vllm":
                    # A moved document meets today's margins (never lowered), unless
                    # hardware measured its peaks under no smaller budget: evidence
                    # rows, or the proven document it replaces.
                    try:
                        result.peaks_raised = refresh_evidence.raise_role_peaks(
                            new_recipe["topology"]["roles"],
                            RecipeDefinition.model_validate(new_recipe),
                            RecipeDefinition.model_validate(recipe),
                            document_sha256(recipe),
                        )
                    except (KeyError, TypeError, ValueError) as error:
                        result.reasons.append(
                            f"memory envelope: peaks could not be raised: {str(error)[:300]}"
                        )
                edits[f"recipes/{recipe_id}.json"] = refresh_pinning.canonical(
                    new_recipe
                )
            edits.update(protocol_wheel.template_edits(self.catalog.root, edits))
            result.edits = edits
        return result

    def _mark(
        self,
        item: refresh_model.Item,
        recipe: dict[str, Any],
        new_recipe: dict[str, Any],
        result: refresh_model.Assessment,
        edits: dict[str, bytes | None],
    ) -> None:
        """Follow an image whose build commit is not on GitHub, or drop that mark.

        The image moves by digest and every place that states the commit states the
        recorded one, with ``source.unverified`` beside it in ``vendored-upstream.json``.
        No changed file is judged: the commit cannot be read (the move goes to review).
        """
        pin = refresh_catalog_view.adapter_pin(self.catalog.root, recipe)
        if pin is None or not (pin.directory / vendored.MANIFEST).is_file():
            raise RuntimeError(f"no {vendored.MANIFEST} source to carry the mark")
        if not item.unmark:
            assert item.image is not None
            repository, digest = item.image
            self._retarget_reference(new_recipe, item)
            self._retarget_adapter(pin, item, edits, result)
            self._move_image(
                pin.directory,
                repository,
                digest,
                lambda written: refresh_items._pinned(written, digest),
                new_recipe,
                edits,
                published=True,
                result=result,
            )
            result.evidence.append(
                f"The recipe runs {repository} as published; it moves to `{digest}`, "
                f"which records {item.repo} at `{item.target.sha}`"
                + (
                    " (not on GitHub: provenance marked unverified).\n"
                    if item.unverified
                    else " (on GitHub; the unpublished pin it leaves is not compared).\n"
                )
            )
        else:
            result.evidence.append(
                f"- {item.repo} `{item.current[:12]}` is on GitHub now: the "
                "`source.unverified` mark goes; the image is unchanged.\n"
            )
        self._set_mark(pin.directory, item.unverified, edits)

    def _set_mark(
        self, directory: Path, mark: str | None, edits: dict[str, bytes | None]
    ) -> None:
        """Write (or, with ``None``, drop) ``source.unverified`` in the adapter manifest."""
        path = directory / vendored.MANIFEST
        if not path.is_file():
            return
        document = json.loads(self._pending(path, edits))
        source = document.get("source")
        if not isinstance(source, dict) or source.get("unverified") == mark:
            return
        if mark is None:
            del source["unverified"]
        else:
            source["unverified"] = mark
        edits[path.relative_to(self.catalog.root).as_posix()] = (
            json.dumps(document, indent=2) + "\n"
        ).encode()

    # A source_reference names "<repo>/tree/<sha>"; move only the commit.
    @staticmethod
    def _retarget_reference(
        new_recipe: dict[str, Any], item: refresh_model.Item
    ) -> None:
        provenance = new_recipe["provenance"]
        reference = provenance.get("source_reference")
        if isinstance(reference, str) and item.current in reference:
            provenance["source_reference"] = reference.replace(
                item.current, item.target.sha
            )

    def _github(
        self,
        item: refresh_model.Item,
        recipe: dict[str, Any],
        new_recipe: dict[str, Any],
        result: refresh_model.Assessment,
        edits: dict[str, bytes | None],
    ) -> None:
        comparison = item.comparison or refresh_items.compare_commits(
            self.http, item.repo, item.current, item.target.sha
        )
        pin = refresh_catalog_view.adapter_pin(self.catalog.root, recipe)
        owns_adapter = pin is not None and pin.repo.lower() == item.repo.lower()
        patched: set[str] = set()
        problems: list[str] = []
        stale_docs: list[str] = []  # vendored copies of files no recipe runs (a README)
        if owns_adapter:
            assert pin is not None
            patched = refresh_classification.patch_targets(pin.directory)
            tree = vendored.fetch_tree(item.repo, item.target.sha)
            report = vendored.check_adapter(
                pin.directory,
                self.catalog.root,
                lambda _repo, _commit: tree,
                at=item.target.sha,
            )
            stale = {problem: (local, up) for problem, local, up in report.stale}
            for p in report.problems:
                why = None
                if p in stale:
                    local, upstream = stale[p]
                    why = self._review_vendored(
                        pin,
                        item,
                        local,
                        upstream,
                        tree.blobs[upstream],
                        patched,
                        result,
                    )
                    upstream_runs = refresh_model.runs(upstream)
                else:
                    upstream_runs = True
                (problems if upstream_runs else stale_docs).append(
                    f"vendored file check at the new commit: {p}"
                    + ("" if why is None else f" ({why})")
                )
        context = self.catalog.root / recipe["execution"]["build"]["context"]["path"]
        declared_kit = refresh_catalog_view.kit_directory(self.catalog.root, recipe)
        scope: tuple[str, ...] = ()
        if declared_kit is None and not owns_adapter:
            cited = refresh_catalog_view.kit_citation_scope(context, item.repo)
            scope = cited or ()
        else:
            cited = None
        is_kit = (
            item.kind == "source"
            and not owns_adapter
            and (
                refresh_catalog_view.whole_kit(declared_kit)
                or (declared_kit is None and cited is not None)
            )
        )
        mirrored: set[str] = set()
        judged = comparison
        release_only = (
            owns_adapter
            and pin is not None
            and not patched
            and not refresh_catalog_view.active_mods(pin.directory)
            and refresh_catalog_view.release_asset_only(pin.directory, item.repo)
        )
        if is_kit:
            # The adapter is built from this repository's patches and scripts, so a
            # change to what runs is a reason even though no file is named by URL.
            used_changes = None
            mirrored = refresh_catalog_view.adapter_files(context)
            judged = refresh_catalog_view.scoped(comparison, scope)
            result.evidence.append(
                f"- {item.label}: the adapter is built from this kit"
                + (
                    f"'s `{'`, `'.join(scope)}`; only changes there are judged"
                    if scope
                    else "; every changed file is judged"
                )
                + " (docs, tests and client-side helper scripts are not runtime; "
                "patches, launch scripts, configuration and files the adapter copies are).\n"
            )
        elif release_only:
            # The adapter runs the creator's published release asset, not the source:
            # the release (its asset and verified digest) is what moves.
            used_changes = []
            result.evidence.append(
                f"- {item.label}: the adapter uses this repository only through its "
                "release downloads; the move is judged by the release, not by source "
                "changes.\n"
            )
        else:
            used_changes = self._used_changes(item, recipe, pin, patched, result)
        library = (
            refresh_model.LIBRARY_HARMLESS_DIRS
            if refresh_catalog_view.installs_library(context, item.repo)
            else frozenset()
        )
        reasons = refresh_classification.classify_changes(
            judged,
            patched=patched,
            vendor_problems=problems,
            used_changes=used_changes,
            mirrored=mirrored,
            harmless=library,
        )
        result.evidence.append(
            refresh_evidence.github_evidence(item, comparison, patched)
        )
        if (
            not reasons
            and item.image is None
            and not release_only
            and (
                used_changes == []
                or (
                    used_changes is None
                    and refresh_model.nothing_runs(judged, mirrored, library)
                )
            )
        ):
            # Nothing the recipe runs changed (only docs, benchmarks, tests or a vendored
            # README copy): the move would only rewrite commit references (a new build
            # for identical content). It rides along with a change that does run, and is
            # never proposed on its own.
            item.noop = True
            result.evidence.append(
                f"- {item.label}: no file this recipe runs changed, so this move alone "
                "is not a refresh.\n"
            )
        else:
            # A move that does happen must keep every vendored copy byte-identical.
            reasons += stale_docs
        result.reasons.extend(reasons)
        self._retarget_reference(new_recipe, item)
        if owns_adapter:
            assert pin is not None
            self._retarget_adapter(pin, item, edits, result)
        elif is_kit:
            self._retarget_kit(context, item, edits)
        if declared_kit is not None and item.kind == "source":
            self._retarget_lock(declared_kit, item, edits)
        if item.image is not None:
            repository, digest = item.image
            self._move_image(
                context,
                repository,
                digest,
                lambda written: refresh_items._pinned(written, digest),
                new_recipe,
                edits,
                published=True,
                result=result,
            )
            result.evidence.append(
                f"The recipe runs {repository} as published; it moves to `{digest}`, "
                f"which records {item.repo} at `{item.target.sha}`.\n"
            )

    def _used_changes(
        self,
        item: refresh_model.Item,
        recipe: dict[str, Any],
        pin: refresh_catalog_view.AdapterPin | None,
        patched: set[str],
        result: refresh_model.Assessment,
    ) -> list[str] | None:
        """Used upstream files that differ between the pin and the target.

        ``None`` means the whole repository is used (or a tree is unavailable) and the
        caller falls back to judging every changed file.
        """
        context = recipe.get("execution", {}).get("build", {}).get("context", {})
        directory = (
            pin.directory
            if pin is not None
            else (self.catalog.root / context["path"] if "path" in context else None)
        )
        named, whole = refresh_catalog_view.adapter_usage(
            directory, item.repo, item.current
        )
        if whole:
            result.evidence.append(
                f"- {item.label}: the adapter uses the whole upstream repository "
                "(clone, archive or commit-named file); every changed file is judged.\n"
            )
            return None
        mods = refresh_catalog_view.active_mods(directory)
        used = named | patched | mods
        changed: list[str] = []
        if used:  # nothing named needs no upstream tree to know nothing changed
            try:
                old = vendored.fetch_tree(item.repo, item.current)
                new = vendored.fetch_tree(item.repo, item.target.sha)
            except UpstreamBusy:
                raise
            except RuntimeError:
                return None
            if not (old.complete and new.complete):
                return None
            under = [m + "/" for m in mods]
            names = set(old.blobs) | set(new.blobs)
            changed = sorted(
                p
                for p in names
                if (p in used or p.startswith(tuple(under)))
                and old.blobs.get(p) != new.blobs.get(p)
            )
        result.evidence.append(
            f"- {item.label}: {len(used)} referenced upstream file(s) checked by blob, "
            f"{len(changed)} changed; vendored files are compared byte for byte; other "
            "upstream changes are not mirrored by this recipe.\n"
        )
        return changed

    def _review_vendored(
        self,
        pin: refresh_catalog_view.AdapterPin,
        item: refresh_model.Item,
        local: str,
        upstream: str,
        blob: str,
        patched: set[str],
        result: refresh_model.Assessment,
    ) -> str:
        """Why a vendored creator file that changed upstream needs a person.

        No byte difference is judged harmless: the diff goes into the evidence and the
        copy is not rewritten by the refresh.
        """
        edited = refresh_classification.edited_locally(
            pin.directory, local, upstream, patched
        )
        if edited is not None:
            return f"not vendored verbatim: {edited}"
        old = (pin.directory / local).read_bytes()
        data = refresh_classification.upstream_blob(self.http, item.repo, blob)
        diff = "".join(
            difflib.unified_diff(
                old.decode("utf-8", "replace").splitlines(keepends=True),
                data.decode("utf-8", "replace").splitlines(keepends=True),
                f"a/{upstream}",
                f"b/{upstream}",
            )
        )
        result.evidence.append(
            f"### Vendored `{local}` changed upstream\n\n`{upstream}` changed "
            f"({item.repo} `{item.current[:12]}` -> `{item.target.sha[:12]}`). A reviewer "
            "judges what it now launches.\n\n"
            + refresh_evidence.fence(refresh_evidence.clip(diff, 4000), "diff")
            + "\n"
        )
        return f"the creator's `{upstream}` changed"

    def _retarget_adapter(
        self,
        pin: refresh_catalog_view.AdapterPin,
        item: refresh_model.Item,
        edits: dict[str, bytes | None],
        result: refresh_model.Assessment,
    ) -> None:
        root = self.catalog.root
        for name in (vendored.MANIFEST, "Dockerfile", "NOTICE"):
            path = pin.directory / name
            if not path.is_file():
                continue
            text = path.read_text(encoding="utf-8")
            changed = item.current in text
            if changed:
                text = text.replace(item.current, item.target.sha)
            if name == "Dockerfile":
                try:
                    text, relabelled = refresh_classification.retarget_archive_labels(
                        text, item
                    )
                except UpstreamBusy:
                    raise
                except (RuntimeError, OSError) as error:
                    result.reasons.append(
                        f"{item.label}: could not refresh the archive digest label: "
                        f"{str(error)[:200]}"
                    )
                    continue
                try:
                    text, released_asset = (
                        refresh_classification.retarget_release_assets(
                            text, item, self.http
                        )
                    )
                except refresh_classification.ReleaseAssetError as error:
                    result.reasons.append(
                        f"{item.label}: {refresh_classification.RELEASE_ASSET_REVIEW} ({str(error)[:200]})"
                    )
                    continue
                changed = changed or relabelled or released_asset
            if changed:
                edits[path.relative_to(root).as_posix()] = text.encode()
        if item.unverified is None:
            self._set_mark(pin.directory, None, edits)

    def _retarget_kit(
        self, directory: Path, item: refresh_model.Item, edits: dict[str, bytes | None]
    ) -> None:
        """Move the kit's own commit where the adapter cites it (labels, the lock)."""
        dockerfile = directory / "Dockerfile"
        if dockerfile.is_file():
            text = dockerfile.read_text(encoding="utf-8")
            moved = refresh_catalog_view.kit_citation(item.repo).sub(
                lambda m: m.group(1) + item.target.sha, text
            )
            if moved != text:
                edits[dockerfile.relative_to(self.catalog.root).as_posix()] = (
                    moved.encode()
                )
        self._retarget_lock(directory, item, edits)

    def _retarget_lock(
        self, directory: Path, item: refresh_model.Item, edits: dict[str, bytes | None]
    ) -> None:
        """Move the kit commit the adapter's lock was derived from."""
        lock = self._lock(directory, edits)
        if lock is not None:
            lock["kit"]["commit"] = item.target.sha
            self._write_lock(directory, lock, edits)

    def _lock(self, directory: Path, edits: dict[str, bytes | None]) -> Any:
        """The adapter's kit lock with the edits already prepared for it."""
        pending = edits.get(self._lock_path(directory))
        if pending is not None:
            return json.loads(pending)
        return kit_lock.read_lock(directory)

    def _lock_path(self, directory: Path) -> str:
        return (directory / vendored.KIT_LOCK).relative_to(self.catalog.root).as_posix()

    def _write_lock(
        self, directory: Path, lock: Any, edits: dict[str, bytes | None]
    ) -> None:
        edits[self._lock_path(directory)] = kit_lock.lock_text(lock).encode()

    def _image(
        self,
        item: refresh_model.Item,
        recipe: dict[str, Any],
        new_recipe: dict[str, Any],
        result: refresh_model.Assessment,
        edits: dict[str, bytes | None],
    ) -> None:
        """Move the lock, the Dockerfile's FROM and the recorded base to the new digest."""
        pin = item.kit_pin
        directory = self.catalog.root / recipe["execution"]["build"]["context"]["path"]
        lock = self._lock(directory, edits)
        if lock is not None:
            lock["dependencies"][pin.name] = kit_lock.lock_document(
                lock["kit"]["repo"], lock["kit"]["commit"], [pin]
            )["dependencies"][pin.name]
            self._write_lock(directory, lock, edits)
        self._move_image(
            directory,
            pin.repo,
            pin.immutable,
            lambda written: kit_consumers.image_reference(pin, written),
            new_recipe,
            edits,
            result=result,
        )
        result.evidence.append(
            f"The recipe runs the kit's published image {pin.repo}; it moves to "
            f"`{pin.immutable}` (the kit's declaration), and the thin layer on top is "
            f"rebuilt on it. {refresh_items.REVIEW_HEADER}\n"
        )

    def _move_image(
        self,
        directory: Path,
        repository: str,
        digest: str,
        reference: Callable[[str], str],
        new_recipe: dict[str, Any],
        edits: dict[str, bytes | None],
        *,
        published: bool = False,
        result: refresh_model.Assessment,
    ) -> None:
        """Move the Dockerfile's FROM of ``repository`` and the recorded base image.

        A ``published`` image (the recipe runs it as the creator pushed it) also gets
        its header prose replaced by a note that names no version; a header the note
        cannot replace is named in ``result`` for review, as the kit path does.
        """
        dockerfile = directory / "Dockerfile"
        if dockerfile.is_file():
            text = self._pending(dockerfile, edits)

            def move(match: re.Match[str]) -> str:
                image = match["image"]
                if (
                    "$" in image
                    or kit_consumers.split_image(image)[0].lower() != repository.lower()
                ):
                    return match[0]
                return match["lead"] + reference(image)

            moved = kit_consumers.FROM_LINE.sub(move, text)
            if moved != text:
                if published:
                    moved, noted = refresh_items.IMAGE_NOTE.subn(
                        lambda m: (
                            m["head"]
                            + f"# The published {repository} image, built from the revision "
                            "above; the digest is what runs.\n"
                        ),
                        moved,
                        count=1,
                    )
                    if not noted:
                        result.evidence.append(f"{refresh_items.REVIEW_HEADER}\n")
                edits[dockerfile.relative_to(self.catalog.root).as_posix()] = (
                    moved.encode()
                )
        base = new_recipe["execution"]["build"].get("base_image")
        if base and str(base.get("repository", "")).lower() == repository.lower():
            base["digest"] = digest.removeprefix("sha256:")

    def _pending(self, path: Path, edits: dict[str, bytes | None]) -> str:
        """A file's text with the edits already prepared for it."""
        pending = edits.get(path.relative_to(self.catalog.root).as_posix())
        return path.read_text(encoding="utf-8") if pending is None else pending.decode()

    def _model(
        self,
        item: refresh_model.Item,
        recipe: dict[str, Any],
        new_recipe: dict[str, Any],
        result: refresh_model.Assessment,
        edits: dict[str, bytes | None],
    ) -> None:
        old = self.catalog.models[str(item.model_slug)]
        fresh = self.catalog.catalogued(item.repo, item.target.sha)
        declared = item.kit_pin is not None
        mounted = (
            None if declared else refresh_classification.mounted_paths(recipe, old)
        )
        new_files: list[dict[str, Any]] | None = None
        if fresh is None:
            try:
                new_files = self._curated(
                    old, item, self._inventory(item.repo, item.target.sha)
                )
            except catalog.AccessDenied:
                if declared or not refresh_classification.gated_payload_identical(
                    old,
                    self._tree(item.repo, item.current),
                    self._tree(item.repo, item.target.sha),
                ):
                    raise
                item.noop = True
                result.evidence.append(
                    f"### {item.label}\n\nThe repository is gated: its tree shows every "
                    "catalogued non-metadata file with the same git id and size at "
                    f"`{item.target.sha[:12]}`, so the catalogued revision keeps serving "
                    "identical weights; no move is proposed.\n"
                )
                return
            if (
                mounted is not None
                and refresh_classification.mounted_unchanged(old, new_files, mounted)
            ) or (
                not declared
                and refresh_classification.only_sibling_weights_added(old, new_files)
            ):
                self._mounted_noop(item, result)
                return
            problems = refresh_classification.model_problems(
                old, new_files, mounted, declared=declared
            )
            result.reasons.extend(f"{item.label}: {p}" for p in problems)
            result.evidence.append(
                refresh_evidence.model_evidence(item, old, new_files)
            )
            if problems:
                return
            if (
                refresh_classification.payload_identical(old, new_files)
                and not declared
            ):
                # Same bytes in every file that is not metadata (a README edit): a
                # second catalogued revision would only duplicate the payload.
                item.noop = True
                result.evidence.append(
                    "The newest revision differs only in metadata files; the catalogued "
                    "revision keeps serving identical weights, so no move is proposed.\n"
                )
                return
            fresh = refresh_classification.refreshed_model(old, new_files, item.target)
            edits[f"models/{fresh['identity']['slug']}.json"] = (
                refresh_pinning.canonical(fresh)
            )
            edits[f"display/models/{fresh['identity']['slug']}.json"] = (
                kit_write.moved_model_display(
                    self.catalog.root,
                    old["identity"]["slug"],
                    fresh["identity"]["slug"],
                )
            )
        else:
            if (
                mounted is not None
                and refresh_classification.mounted_unchanged(
                    old, fresh["files"], mounted
                )
            ) or (
                not declared
                and refresh_classification.only_sibling_weights_added(
                    old, fresh["files"]
                )
            ):
                self._mounted_noop(item, result)
                return
            result.evidence.append(
                refresh_evidence.model_evidence(item, old, None)
                + "\nThe newest revision is already catalogued; reusing it.\n"
            )
            problems = refresh_classification.model_problems(
                old, fresh["files"], mounted, declared=declared
            )
            if problems:
                result.reasons.extend(f"{item.label}: {p}" for p in problems)
                return
        self._retarget_model(new_recipe, old, fresh, result)
        if declared:
            directory = (
                self.catalog.root / recipe["execution"]["build"]["context"]["path"]
            )
            lock = self._lock(directory, edits)
            if lock is not None:
                lock["dependencies"][item.kit_pin.name] = kit_lock.lock_document(
                    lock["kit"]["repo"], lock["kit"]["commit"], [item.kit_pin]
                )["dependencies"][item.kit_pin.name]
                self._write_lock(directory, lock, edits)
        remaining = [
            rid
            for rid in self.catalog.model_references(old["identity"]["slug"])
            if rid != recipe["identity"]["slug"]
        ]
        if not remaining:
            edits[f"models/{old['identity']['slug']}.json"] = None
            edits[f"display/models/{old['identity']['slug']}.json"] = None

    def _curated(
        self,
        old: dict[str, Any],
        item: refresh_model.Item,
        new_files: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """The new snapshot's files the Model document is about.

        A document may catalogue only some files of its repository (one checkpoint of
        several); a file it left out that the catalogued revision already had is neither
        new nor part of the move. Files the repository added since are judged as new.
        """
        try:
            before = {f["path"] for f in self._inventory(item.repo, item.current)}
        except UpstreamBusy:
            raise  # not read this run: every file would look new
        except (RuntimeError, SystemExit, KeyError, ValueError, OSError):
            return new_files  # unknown history: judge every file
        return refresh_classification.curate(old, before, new_files)

    @staticmethod
    def _mounted_noop(
        item: refresh_model.Item, result: refresh_model.Assessment
    ) -> None:
        # The recipe mounts some files of the Model and none of them changed: the move
        # would only relabel identical inputs (other files concern other recipes).
        item.noop = True
        result.evidence.append(
            f"- {item.label}: every file this recipe mounts is the same bytes at "
            f"`{item.target.sha[:12]}`, so this move alone is not a refresh.\n"
        )

    def _tree(self, repo: str, revision: str) -> list[Any]:
        """One snapshot's recursive tree, fetched once per run."""
        key = (repo, revision)
        if key not in self.trees:
            self.trees[key] = catalog._get_tree(
                f"https://huggingface.co/api/models/{quote(repo, safe='/')}"
                f"/tree/{revision}?recursive=true&expand=true&limit=100"
            )
        return self.trees[key]

    def _inventory(self, repo: str, revision: str) -> list[dict[str, Any]]:
        cache = (
            None
            if self.cache_dir is None
            else self.cache_dir / "hf" / f"{repo.replace('/', '__')}@{revision}.json"
        )
        if cache is not None and cache.is_file():
            return json.loads(cache.read_text(encoding="utf-8"))
        artifacts = catalog.inventory(repo, revision, self._tree(repo, revision))
        if cache is not None:
            cache.parent.mkdir(parents=True, exist_ok=True)
            cache.write_text(json.dumps(artifacts), encoding="utf-8")
        return artifacts

    def _retarget_model(
        self,
        new_recipe: dict[str, Any],
        old: dict[str, Any],
        fresh: dict[str, Any],
        result: refresh_model.Assessment,
    ) -> None:
        old_ids = {f["path"]: f for f in old["files"]}
        new_ids = {f["path"]: f for f in fresh["files"]}
        by_old_id = {
            f["id"]: new_ids[path] for path, f in old_ids.items() if path in new_ids
        }
        new_ids_by_id = {f["id"] for f in fresh["files"]} - {
            f["id"] for f in old["files"]
        }
        content = document_sha256(fresh)
        old_total = new_total = 0
        for reference in new_recipe["models"]:
            model = reference["model"]
            if model["slug"] != old["identity"]["slug"]:
                continue
            model["slug"] = fresh["identity"]["slug"]
            model["content_sha256"] = content
            # A recipe that mounts every file of the Model mounts every file of the new
            # one: a file the revision adds (a chat template) reaches it, one it drops
            # (a README) leaves it.
            everything = {s["file_id"] for s in reference["files"]} == {
                f["id"] for f in old["files"]
            }
            if everything:
                template = reference["files"][0]
                shapes = {
                    json.dumps([s.get("mount"), s.get("roles")], sort_keys=True)
                    for s in reference["files"]
                }
                if len(shapes) != 1:
                    result.reasons.append(
                        "the recipe mounts every file of the Model, but not all the "
                        "same way; rebind it by hand"
                    )
                    return
                for selected in list(reference["files"]):
                    if selected["file_id"] not in by_old_id:
                        reference["files"].remove(selected)
                        old_total += next(
                            f["size_bytes"]
                            for f in old["files"]
                            if f["id"] == selected["file_id"]
                        )
                kept = {f["path"] for f in old["files"]}
                for entry in fresh["files"]:
                    if entry["path"] in kept:
                        continue
                    reference["files"].append(
                        {
                            "file_id": entry["id"],
                            "id": kit_write.selection_id(
                                reference["id"], template, entry["id"]
                            ),
                            "mount": template["mount"],
                            "roles": template["roles"],
                        }
                    )
                    new_total += entry["size_bytes"]
            for selected in reference["files"]:
                before = selected["file_id"]
                if before in new_ids_by_id:
                    continue  # added above for a recipe that mounts everything
                if before not in by_old_id:
                    result.reasons.append(
                        f"the recipe selects a file the newest revision no longer has "
                        f"(id `{before}`)"
                    )
                    return
                replacement = by_old_id[before]
                old_entry = next(f for f in old["files"] if f["id"] == before)
                old_total += old_entry["size_bytes"]
                new_total += replacement["size_bytes"]
                selected["id"] = kit_write.selection_id(
                    reference["id"], selected, replacement["id"]
                )
                selected["file_id"] = replacement["id"]
            if len({s["id"] for s in reference["files"]}) != len(reference["files"]):
                result.reasons.append(
                    f"selected file ids of Model '{reference['id']}' collide once cut "
                    "to 64 characters; rebind it by hand"
                )
                return
        if old_total != new_total:
            delta = new_total - old_total
            roles = new_recipe["topology"]["roles"]
            if any(
                abs(delta) * 100 > role["resources"]["disk"]["artifact_bytes"]
                for role in roles
            ):
                result.reasons.append(
                    f"model sizes changed ({old_total} -> {new_total} bytes) and the recipe's "
                    f"disk envelope would change by more than {refresh_model.MATERIAL_ENVELOPE_PERCENT}%"
                )
                return
            # The artifact envelope follows the selected files' sizes (the sum over
            # the recipe's models, or that sum plus a declared allowance), so shift
            # it by the size difference.
            for role in roles:
                role["resources"]["disk"]["artifact_bytes"] += delta
