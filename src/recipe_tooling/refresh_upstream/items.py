"""The items a refresh collects and assesses."""

from __future__ import annotations

import re
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from recipe_tooling import vendored_upstream as vendored
from recipe_tooling.kit_pins import consumers as kit_consumers
from recipe_tooling.kit_pins import declarations as kit_declarations
from recipe_tooling.kit_pins import lock as kit_lock
from recipe_tooling.kit_pins import model as kit_model
from recipe_tooling.kit_pins import resolve as kit_resolve
from recipe_tooling.refresh_upstream import catalog_view as refresh_catalog_view
from recipe_tooling.refresh_upstream import evidence as refresh_evidence
from recipe_tooling.refresh_upstream import http as refresh_http
from recipe_tooling.refresh_upstream import model as refresh_model
from recipe_tooling.refresh_upstream import pinning as refresh_pinning
from recipe_tooling.upstream_drift import inventory as drift_inventory
from recipe_tooling.upstream_http import UpstreamBusy


def kit_declared_models(
    recipe: dict[str, Any], directory: Path | None
) -> dict[str, Any]:
    """Slug of each Model a kit declaration pins -> that declaration (``tools/kit-pins``)."""
    if directory is None:
        return {}
    try:
        wanted = kit_declarations.declarations(
            refresh_catalog_view.adapter_manifest(directory)
        )
    except kit_model.KitError:
        return {}
    found: dict[str, Any] = {}
    for declaration in wanted:
        if declaration.datasource != "hf-revision":
            continue
        reference = next(
            (m for m in recipe["models"] if m["id"] == declaration.binds), None
        )
        if reference is not None:
            found[reference["model"]["slug"]] = declaration
    return found


def check_kit_dependencies(
    catalog_: refresh_catalog_view.Catalog,
    recipe: dict[str, Any],
    directory: Path,
    result: refresh_model.Assessment,
    kit_head: str,
    kit_source: Any,
) -> None:
    """What the kit declares at ``kit_head`` against what the recipe is locked to.

    A kit-built recipe pins only the kit; everything the kit declares (its base engine
    ref, its model revision, its image) is whatever the kit says at that commit
    (``tools/kit-pins``). The dependency's own newest release, revision or tag is
    information, never the proposed pin. A declaration that differs from the lock means
    the kit moved it: a review, except a revision of the same Hugging Face repository
    whose files are the catalogued ones (see ``Assessor._model``). ``tools/kit-pins lock``
    then derives the new pin.
    """
    kit = kit_lock.kit_of(recipe)
    if kit is None:
        return
    slug = recipe["identity"]["slug"]
    try:
        wanted = kit_declarations.declarations(
            refresh_catalog_view.adapter_manifest(directory)
        )
        pins = kit_resolve.derive(kit[0], kit_head, wanted, kit_source)
    except UpstreamBusy as error:
        result.unreachable.append(f"kit dependencies: {str(error)[:200]}")
        return
    except kit_model.KitError as error:
        result.reasons.append(
            f"the kit's declared dependencies cannot be read at {kit_head[:12]}: "
            f"{str(error)[:300]}"
        )
        return
    lock = kit_lock.read_lock(directory)
    if lock is None:
        result.reasons.append(
            f"{vendored.KIT_LOCK} is missing; run `tools/kit-pins lock {slug}`"
        )
        return
    declared = {d.name: d for d in wanted}
    for pin in pins:
        locked = lock.get("dependencies", {}).get(pin.name) or {}
        newest = kit_resolve.newest_information(kit_source, pin)
        info = (
            f"{newest} is shown for information; the pin follows the kit's declaration"
            if newest
            else f"{pin.repo}: no release information"
        )
        if pin.overridden:
            result.evidence.append(
                f"### Kit dependency `{pin.name}` is overridden\n\n- {pin.repo} is "
                f"locked to `{pin.ref}` on purpose: {pin.overridden}\n- {info}\n"
            )
            continue
        if locked.get(pin.key) == pin.immutable:
            result.evidence.append(
                f"### Kit dependency `{pin.name}`\n\n- the kit declares {pin.repo} "
                f"`{pin.ref}` (`{pin.immutable[:12]}`), as locked\n- {info}\n"
            )
            continue
        if pin.datasource == "hf-revision":
            kit_model_moved(
                catalog_, recipe, declared[pin.name], pin, locked, info, result
            )
            continue
        if pin.datasource == "oci-image":
            kit_image_moved(recipe, pin, locked, info, result)
            continue
        if pin.kind == "branch" and locked.get("ref") == pin.ref:
            # the kit still names the same branch; its head moved since the lock
            result.reasons.append(
                f"the branch `{pin.ref}` of {pin.repo} that the kit declares moved from "
                f"`{str(locked.get('commit', '?'))[:12]}` to `{pin.immutable[:12]}` "
                f"({pin.date}): a kit dependency changed; review the branch's new "
                f"commits, then run `tools/kit-pins lock {slug}`"
            )
            result.evidence.append(
                f"### Kit dependency `{pin.name}` (branch `{pin.ref}`) moved\n\n"
                f"- locked: `{locked.get('commit', '?')}`\n- branch head now: "
                f"`{pin.immutable}` ({pin.date})\n- compare: "
                f"https://github.com/{pin.repo}/compare/{locked.get('commit', '?')}...{pin.immutable}\n"
                f"- {info}\n"
            )
            continue
        result.reasons.append(
            f"the kit declares {pin.repo} `{pin.ref}` (`{pin.immutable[:12]}`) at "
            f"{kit_head[:12]}, but the recipe is locked to {locked.get('repo', '?')} "
            f"`{locked.get('ref', '?')}` "
            f"(`{str(locked.get('commit', '?'))[:12]}`): the kit moved its base, so its "
            f"patches were rebased; review them, then run "
            f"`tools/kit-pins lock {slug}`"
        )
        result.evidence.append(
            f"### Kit dependency `{pin.name}` moved\n\n- locked: `{locked.get('ref', '?')}`"
            f"\n- declared now: `{pin.ref}` (`{pin.immutable}`)\n- {info}\n"
        )


def kit_image_moved(
    recipe: dict[str, Any],
    pin: Any,
    locked: dict[str, Any],
    info: str,
    result: refresh_model.Assessment,
) -> None:
    """The image the kit declares is another digest than the lock holds.

    The recipe runs the kit's prebuilt image as published, so a creator who republishes
    it (a moved tag, or a kit that pins a newer digest) moves the recipe to the new digest
    of the same repository mechanically: the lock, the Dockerfile's ``FROM`` and the
    recipe's ``base_image`` follow, and the thin layer on top is rebuilt on it. Another
    repository is a review.
    """
    slug = recipe["identity"]["slug"]
    result.evidence.append(
        f"### Kit dependency `{pin.name}` (image) moved\n\n- locked: "
        f"`{locked.get('ref', '?')}` `{locked.get('digest', '?')}`\n- declared now: "
        f"`{pin.ref}` `{pin.immutable}`\n- {info}\n"
    )
    if str(locked.get("repo", "")).lower() != pin.repo.lower():
        result.reasons.append(
            f"the kit now runs the image {pin.repo} instead of {locked.get('repo', '?')}: "
            f"another image repository is a review, then run `tools/kit-pins lock {slug}`"
        )
        return
    result.items.append(
        refresh_model.Item(
            "image",
            "oci",
            pin.repo,
            str(locked.get("digest", "")),
            refresh_model.Target(
                provider="oci",
                repo=pin.repo,
                sha=pin.immutable,
                date=datetime.now(UTC).date().isoformat(),
                ref=None,
                version=None,
                url=f"https://{pin.repo}",
            ),
            kit_pin=pin,
        )
    )


def kit_model_moved(
    catalog_: refresh_catalog_view.Catalog,
    recipe: dict[str, Any],
    declaration: Any,
    pin: Any,
    locked: dict[str, Any],
    info: str,
    result: refresh_model.Assessment,
) -> None:
    """The kit declares another model repository or revision than the lock holds.

    Another repository is a review. Another revision of the same repository becomes a
    model move at exactly the kit's revision (never the repository's newest), which
    ``Assessor._model`` makes mechanical only when every file but the metadata is the
    catalogued bytes.
    """
    slug = recipe["identity"]["slug"]
    was = f"{locked.get('repo', '?')}@{str(locked.get('revision', '?'))[:12]}"
    result.evidence.append(
        f"### Kit dependency `{pin.name}` (model) moved\n\n- locked: {was}\n- declared "
        f"now: {pin.repo}@{pin.immutable} ({pin.date})\n- {info}\n"
    )
    if str(locked.get("repo", "")).lower() != pin.repo.lower():
        result.reasons.append(
            f"the kit now declares the model {pin.repo}@{pin.immutable[:12]} instead of "
            f"{was}: another model repository is a review; catalogue it with "
            f"`tools/catalog-hf-model`, then run `tools/kit-pins lock {slug}`"
        )
        return
    found = kit_consumers.bound_model(catalog_.root, recipe, declaration.binds)
    if found is None:
        result.reasons.append(
            f"the recipe's Model '{declaration.binds}' for the kit's {pin.name} is not catalogued"
        )
        return
    document = found[1]
    result.items.append(
        refresh_model.Item(
            "model",
            "huggingface",
            pin.repo,
            document["source"]["revision"],
            refresh_model.Target(
                provider="huggingface",
                repo=pin.repo,
                sha=pin.immutable,
                date=pin.date,
                ref=None,
                version=None,
                url=f"https://huggingface.co/{pin.repo}/tree/{pin.immutable}",
            ),
            document["identity"]["slug"],
            kit_pin=pin,
        )
    )


def follow_image(
    result: refresh_model.Assessment,
    http: refresh_http.Http,
    repo: str,
    current: str,
    image: tuple[str, str, str],
    source: Any,
    marked: str | None = None,
) -> None:
    """The source pin of a recipe that runs its source's published image.

    What runs is the image, so the pin moves only when the tag upstream publishes as
    current names another digest, and then to the commit that image records (never the
    repository's newest commit or release, which the image may not contain). An image
    that records no build commit is a review: the recipe would claim a revision it does
    not run.

    An image built from a commit GitHub does not serve is still followed by its digest
    (the digest is the identity, the commit provenance): the move is prepared with the
    commit marked unverified (``marked``, the adapter's ``source.unverified``) and goes to
    review, since what changed in the files the recipe uses cannot be judged. Once GitHub
    serves the commit the mark goes, and a later move is judged as any other.
    """
    repository, tag, pinned = image
    try:
        digest = source.oci().resolve(repository, tag)
        found = (
            None
            if digest == pinned
            else source.oci().provenance(repository, digest, repo)
        )
    except (kit_model.KitError, RuntimeError, KeyError, ValueError) as error:
        result.unreachable.append(f"oci:{repository}:{tag}: {str(error)[:200]}")
        return
    if digest == pinned:
        result.evidence.append(
            f"### Published image {repository}:{tag}\n\n- unchanged at `{digest}`; the "
            f"recipe runs that image, so {repo} commits alone move nothing\n"
        )
        if marked is not None and commit_on_github(http, repo, current) is True:
            result.items.append(
                refresh_model.Item(
                    "source",
                    "github",
                    repo,
                    current,
                    _commit_target(repo, current, ""),
                    unmark=True,
                )
            )
        return
    if found is None:
        result.reasons.append(
            f"{repository}:{tag} moved to `{digest}`, but that image records no build "
            f"commit of {repo} (no org.opencontainers.image.revision label, no "
            "build-metadata.yaml); review which revision it runs"
        )
        return
    commit, built = found
    try:
        comparison = compare_commits(http, repo, current, commit)
    except (RuntimeError, KeyError, TypeError) as error:
        served = commit_on_github(http, repo, commit)
        if served is False or (served is True and marked is not None):
            # The new commit is unpublished, or the pin it moves from never was (its
            # mark says so): either way the changed files cannot be compared.
            unverified_image(
                result,
                repo,
                current,
                commit,
                built,
                repository,
                digest,
                source,
                published=served is True,
            )
            return
        # GitHub could not be asked (rate limit, outage): retried on the next run.
        result.unreachable.append(
            f"{repository}@{digest} records {repo} `{commit[:12]}`, which cannot be "
            f"compared upstream: {str(error)[:200]}"
        )
        return
    result.items.append(
        refresh_model.Item(
            "source",
            "github",
            repo,
            current,
            _commit_target(repo, commit, built),
            None,
            comparison,
            image=(repository, digest),
        )
    )


def _commit_target(repo: str, commit: str, built: str) -> refresh_model.Target:
    return refresh_model.Target(
        provider="github",
        repo=repo,
        sha=commit,
        date=built or datetime.now(UTC).date().isoformat(),
        ref=None,
        version=None,
        url=f"https://github.com/{repo}/commit/{commit}",
    )


def commit_on_github(http: refresh_http.Http, repo: str, commit: str) -> bool | None:
    """Whether GitHub serves the commit; ``None`` when it cannot be told (404/422: no)."""
    try:
        http.github(f"repos/{repo}/commits/{commit}")
    except refresh_http.NotFound:
        return False
    except RuntimeError as error:
        return False if str(error).startswith("HTTP 422") else None
    return True


def unverified_image(
    result: refresh_model.Assessment,
    repo: str,
    current: str,
    commit: str,
    built: str,
    repository: str,
    digest: str,
    source: Any,
    published: bool = False,
) -> None:
    """Follow an image for review when its commit cannot be compared with the pin.

    Either GitHub does not serve the image's build commit (the move carries the
    unverified mark) or the pin it moves from was never published (``published``: the
    new commit is served, so the mark goes, but there is no old side to compare).
    """
    mark = None if published else vendored.unverified_mark(repo, commit)
    result.items.append(
        refresh_model.Item(
            "source",
            "github",
            repo,
            current,
            _commit_target(repo, commit, built),
            image=(repository, digest),
            unverified=mark,
            uncompared=published,
        )
    )
    result.held.append(
        f"{repository}@{digest} records {repo} `{commit[:12]}`, but the pinned "
        f"`{current[:12]}` was never published on GitHub: the pull request follows the "
        "image by digest and drops the unverified mark; what changed in the files the "
        "recipe uses cannot be compared"
        if published
        else f"{repository}@{digest} records {repo} `{commit[:12]}`, which is not on "
        "GitHub: the pull request follows the image by digest with its provenance "
        f"marked unverified ({mark}); what changed in the files the recipe uses cannot "
        "be judged until upstream publishes that commit"
    )
    try:
        metadata = source.oci().build_metadata(repository, digest)
    except kit_model.KitError, RuntimeError, KeyError, ValueError:
        metadata = None
    result.evidence.append(
        f"### What the image {repository}@{digest} shows\n\n"
        + (
            "Its `build-metadata.yaml` (image data, evidence only):\n\n"
            + refresh_evidence.fence(refresh_evidence.clip(metadata, 4000), "yaml")
            if metadata
            else "The image carries no readable `build-metadata.yaml` beyond its build commit."
        )
        + "\n"
    )


#: The reminder a moved image's Dockerfile header gets when it cannot be rewritten.
REVIEW_HEADER = "Review the Dockerfile's header comment."

#: The comment lines between a published image's LABEL block and the "Upstream recipes
#: start it with" line describe the image; FROM and LABEL own its identity, so they name none.
IMAGE_NOTE = re.compile(
    r"(?P<head>^FROM[^\n]*\n\n?LABEL[^\n]*(?:\\\n[^\n]*)*\n\n?)"
    r"(?:#[^\n]*\n)+?(?=# Upstream recipes start it with)",
    re.MULTILINE,
)


def _pinned(written: str, digest: str) -> str:
    """``written`` (``repo[:tag]@digest``) at another digest, keeping any tag it names."""
    repository, tag, _ = kit_consumers.split_image(written)
    return f"{repository}{'' if tag is None else ':' + tag}@{digest}"


def collect_items(
    catalog_: refresh_catalog_view.Catalog,
    recipe_id: str,
    http: refresh_http.Http,
    targets: Callable[[str, str], refresh_model.Target],
    kit_source: Any = None,
) -> refresh_model.Assessment:
    """Compare every pin a recipe carries with what the pinning rule selects."""
    recipe = catalog_.recipes[recipe_id]
    result = refresh_model.Assessment(recipe_id)
    seen: set[tuple[str, str]] = set()

    def consider(
        kind: str,
        provider: str,
        repo: str,
        current: str,
        slug: str | None = None,
        line: str | None = None,
    ) -> None:
        key = (provider, repo.lower())
        if (kind != "model" and key in seen) or not refresh_model.SHA.match(current):
            return
        seen.add(key) if kind != "model" else None
        try:
            target = targets(provider, repo)
            if line is not None:
                note = refresh_pinning.newer_line_note(repo, line, target)
                if note is not None:
                    result.notes.append(note)
                target = refresh_pinning.resolve_github_line(http, repo, line)
        except refresh_http.NotFound:
            model = catalog_.models.get(slug or "", {})
            if kind == "model" and model.get("requires_token") is False:
                result.reasons.append(
                    f"model {repo} is no longer anonymously resolvable; review or withdraw this public Model rather than replacing it automatically."
                )
                return
            result.unreachable.append(f"{provider}:{repo} was not found upstream")
            return
        except refresh_http.AccessRestricted as error:
            model = catalog_.models.get(slug or "", {})
            if kind == "model" and model.get("requires_token") is False:
                result.reasons.append(
                    f"model {repo} provider now requires access ({error!s}); review or withdraw this public Model. The updater will not retrieve credentials or author a replacement."
                )
            else:
                result.evidence.append(
                    f"### Model access status: {repo}\n\nAnonymous check was restricted ({error!s}); the authored requires_token flag is not provider evidence.\n"
                )
                result.unreachable.append(
                    f"{provider}:{repo}: anonymous access status could not be verified ({error!s}); no credentials were requested"
                )
            return
        except (RuntimeError, KeyError, TypeError) as error:
            result.unreachable.append(f"{provider}:{repo}: {str(error)[:200]}")
            return
        if kind == "model" and target.access_status is not None:
            model = catalog_.models.get(slug or "", {})
            token_required = model.get("requires_token") is True
            if target.access_status in {"private", "disabled"}:
                result.reasons.append(
                    f"model {repo} provider access is {target.access_status}; review or withdraw this Model. The updater will not retrieve access credentials or author a replacement."
                )
            elif target.access_status == "gated" and not token_required:
                result.reasons.append(
                    f"model {repo} provider access changed to {target.access_status}; the catalog declares anonymous access. Review or withdraw this Model; do not replace it automatically."
                )
            elif target.access_status == "public" and token_required:
                result.reasons.append(
                    f"model {repo} is publicly accessible but the catalog requires a token; review the access declaration before changing it."
                )
            elif target.access_status == "unknown":
                result.evidence.append(
                    f"### Model access status: {repo}\n\nProvider gating/private status was not explicit in the anonymous metadata response; the authored requires_token flag is not treated as provider evidence.\n"
                )
                result.unreachable.append(
                    f"{provider}:{repo}: provider access status is unknown from anonymous metadata; no credentials were requested"
                )
        if target.sha == current:
            return
        compared: refresh_model.Comparison | None = None
        if provider == "huggingface":
            followed = refresh_pinning.follow_huggingface_branch(
                http, repo, current, target
            )
            if followed is not None:
                target = followed
                if target.sha == current:
                    return  # at the head of the branch it follows
        if provider == "github":
            try:
                compared = compare_commits(http, repo, current, target.sha)
            except RuntimeError, KeyError, TypeError:
                compared = None  # assessed (and reported) later
            if compared is not None and compared.status == "diverged" and line is None:
                followed = refresh_pinning.follow_github_branch(http, repo, current)
                if followed is not None:
                    target = followed
                    if target.sha == current:
                        return  # at the head of the branch it follows
                    try:
                        compared = compare_commits(http, repo, current, target.sha)
                    except RuntimeError, KeyError, TypeError:
                        compared = None
            if compared is not None and (
                pin_is_current(compared)
                or (
                    compared.status == "diverged"
                    and pin_is_newer_by_date(http, repo, current, target.sha)
                )
            ):
                return  # never downgrade: the pin is at or ahead of the newest
        result.items.append(
            refresh_model.Item(kind, provider, repo, current, target, slug, compared)
        )

    source = refresh_catalog_view.source_pin(recipe)
    context = catalog_.root / recipe["execution"]["build"]["context"]["path"]
    pin = refresh_catalog_view.adapter_pin(catalog_.root, recipe)
    line = (
        None
        if pin is None or pin.derived
        else refresh_pinning.release_line(pin.directory)
    )
    followed = (
        None
        if source is None or source[0] != "github"
        else kit_consumers.followed_image(catalog_.root, context, source[1])
    )
    if followed is not None:
        assert source is not None
        seen.add(("github", source[1].lower()))
        follow_image(
            result,
            http,
            source[1],
            source[2],
            followed,
            kit_source or refresh_http.kit_upstream(http),
            (refresh_catalog_view.adapter_manifest(context).get("source") or {}).get(
                "unverified"
            ),
        )
    elif source is not None:
        same = pin is not None and pin.repo.lower() == source[1].lower()
        consider("source", *source, line=line if same else None)
    kit_dir = refresh_catalog_view.kit_directory(catalog_.root, recipe)
    if kit_dir is not None:
        # What the kit declares follows the kit commit this run proposes (or keeps); the
        # dependency's own newest release, revision or tag is not a pin candidate.
        kit_head = next(
            (i.target.sha for i in result.items if i.kind == "source"),
            source[2] if source is not None else None,
        )
        if kit_head is not None:
            check_kit_dependencies(
                catalog_,
                recipe,
                kit_dir,
                result,
                kit_head,
                kit_source or refresh_http.kit_upstream(http),
            )
    if (
        pin is not None
        and ("github", pin.repo.lower()) not in seen
        and refresh_catalog_view.image_label_claim(context, pin)
    ):
        result.evidence.append(
            f"### Image label {pin.repo}\n\n- the adapter's `org.opencontainers.image."
            f"revision` `{pin.commit[:12]}` describes the digest-pinned image it runs "
            "FROM; upstream releases do not move it, only a new image digest does\n"
        )
    elif pin is not None and not refresh_catalog_view.declares_adapter_source(kit_dir):
        consider("adapter", "github", pin.repo, pin.commit, line=line)
    kit_models = kit_declared_models(recipe, kit_dir)
    for reference in recipe["models"]:
        slug = reference["model"]["slug"]
        document = catalog_.models.get(slug)
        model_source = (document or {}).get("source")
        if not model_source:
            continue
        location = drift_inventory.parse_source(model_source["repository"])
        if slug in kit_models and location.provider == "huggingface":
            # The kit pins this model: the repository's newest revision is information,
            # never an independent proposal (the kit's own declaration is checked above).
            try:
                newest = targets("huggingface", location.repository)
                result.evidence.append(
                    f"### Model {location.repository} follows the kit\n\n- catalogued "
                    f"revision `{model_source['revision']}`\n- the repository's newest "
                    f"revision `{newest.sha}` ({newest.date}) is shown for information; the "
                    "pin follows the kit's declaration\n"
                )
            except RuntimeError, KeyError, TypeError:
                pass
            continue
        if location.provider == "huggingface":
            consider(
                "model",
                "huggingface",
                location.repository,
                model_source["revision"],
                slug,
            )
    return result


def compare_commits(
    http: refresh_http.Http, repo: str, old: str, new: str
) -> refresh_model.Comparison:
    commits: list[dict[str, Any]] = []
    files: list[dict[str, Any]] = []
    status = "unknown"
    url = f"https://github.com/{repo}/compare/{old[:12]}...{new[:12]}"
    truncated = False
    for page in range(1, 4):
        try:
            document = http.github(
                f"repos/{repo}/compare/{old}...{new}?per_page=100&page={page}"
            )
        except refresh_http.NotFound as error:
            if "no common ancestor" not in error.message.lower():
                raise
            # History was rewritten (a force push, a squashed mirror): the pin is not
            # in the new history. That is a pin move to review, not an unassessable one.
            return refresh_model.Comparison(
                "diverged", [], [], False, url, rewritten=new
            )
        status = document.get("status", status)
        commits.extend(document.get("commits", []))
        files.extend(document.get("files", []))
        url = document.get("html_url", url)
        if (
            len(document.get("commits", [])) < 100
            and len(document.get("files", [])) < 100
        ):
            break
    else:
        truncated = True
    return refresh_model.Comparison(status, commits, files, truncated, url)


def pin_is_current(comparison: refresh_model.Comparison) -> bool:
    """The pin is the newest release/head or already ahead of it (never downgrade)."""
    return comparison.status in {"behind", "identical"}


def pin_is_newer_by_date(
    http: refresh_http.Http, repo: str, pin: str, release: str
) -> bool:
    """Diverged histories (a release branch against a main pin): the newer commit wins.

    Moving from a pin committed after the release commit to that release would be a
    downgrade in everything except Git ancestry.
    """
    try:
        dates = [
            http.github(f"repos/{repo}/commits/{sha}")["commit"]["committer"]["date"]
            for sha in (pin, release)
        ]
    except RuntimeError, KeyError, TypeError, AttributeError:
        return False
    return dates[0] >= dates[1]
