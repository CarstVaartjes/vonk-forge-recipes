"""Select, look up and record the prebuilt runtime images of the recipe library.

Every distinct recipe image is built on a GitHub-hosted ARM64 runner and pushed
to GHCR as ``ghcr.io/<owner>/vonk-forge-recipe-<slug>:build-v2-<key>``, where
``<key>`` is the recipe-owned build key ``tools/build-plan`` derives from recipe
inputs and ``policy/build-policy.json`` only. The signed index pins each recipe's
digest; the Controller runs that digest and never derives a key.

``select``  considers every recipe (or the slugs asked for) whose ``build-v2-<key>``
            tag is missing. A recipe whose published pin is a legacy image of the
            same content and source bundle in its own package is adopted: its
            digest is copied to the new tag, no build, so its evidence (judged by
            digest) still applies. Otherwise a key another recipe already has is
            copied, a PR's ``candidate-v2-<key>`` is promoted by digest, or the key
            is built once. ``--base-plan`` limits PRs to changed keys.
            ``--rebuild-adopted`` rebuilds the named recipes whose tag holds an
            adopted (legacy-labelled) image, never a v2-built one.
``resolve`` looks up every recipe's ``build-v2-<key>`` digest for the signed index.
            When it finds none, a recipe whose content digest and source bundle
            equal the current published release's keeps that release's pin
            (counted in the summary), whatever its key: the pin names the exact
            bytes that ran, and status is judged on that digest.

``resolve`` reports missing images; publication leaves those image pins absent
and reports them as pending while deferred/scheduled builds reconcile them.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tarfile
import tempfile
import urllib.error
import urllib.request
from collections.abc import Callable, Iterable, Mapping
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from recipe_tooling.paths import REPO_ROOT
from recipe_tooling.upstream_http import fetch

ROOT = REPO_ROOT

REGISTRY = "ghcr.io"
BUNDLE = "recipe-library.tar"
INDEX_MEMBER = "catalog-index.json"
PACKAGE_PREFIX = "vonk-forge-recipe-"
# A matrix may hold at most 256 jobs; the rest wait for the next run.
MAX_BUILDS = 256
_MANIFEST_TYPES = (
    "application/vnd.oci.image.manifest.v1+json, "
    "application/vnd.docker.distribution.manifest.v2+json, "
    "application/vnd.oci.image.index.v1+json, "
    "application/vnd.docker.distribution.manifest.list.v2+json"
)

# (owner, package, tag) -> manifest digest or None
Lookup = Callable[[str, str, str], "str | None"]
# pinned reference -> the image's config labels
Labels = Callable[[str], Mapping[str, str]]
# Set on every image the platform adaptation stage built (before the build key v2):
# such an image is an adopted one.
LEGACY_LABEL = "ai.vonkforge.runtime-adapter"


def package(slug: str) -> str:
    return f"{PACKAGE_PREFIX}{slug}"


def repository(owner: str, slug: str) -> str:
    return f"{REGISTRY}/{owner.lower()}/{package(slug)}"


def build_tag(key: str) -> str:
    return f"build-v2-{key}"


def candidate_tag(key: str) -> str:
    return f"candidate-v2-{key}"


def ghcr_digest(owner: str, name: str, tag: str) -> str | None:
    """Anonymous manifest lookup; ``None`` when the tag does not exist."""

    scope = f"repository:{owner.lower()}/{name}:pull"
    try:
        token = fetch(
            urllib.request.Request(f"https://{REGISTRY}/token?scope={scope}"),
            lambda response: json.load(response)["token"],
            timeout=30,
        )
        request = urllib.request.Request(
            f"https://{REGISTRY}/v2/{owner.lower()}/{name}/manifests/{tag}",
            method="HEAD",
            headers={"Authorization": f"Bearer {token}", "Accept": _MANIFEST_TYPES},
        )
        digest = fetch(
            request,
            lambda response: response.headers.get("Docker-Content-Digest"),
            timeout=30,
        )
    except urllib.error.HTTPError as error:
        # A package that was never pushed denies the token or the manifest.
        if error.code in {401, 403, 404}:
            return None
        raise
    return digest if isinstance(digest, str) else None


def skopeo_labels(reference: str) -> Mapping[str, str]:
    """The config labels of a pinned image (anonymous, the ARM64 image of an index)."""

    inspected = subprocess.run(
        ["skopeo", "inspect", "--no-creds", "--override-arch", "arm64",
         f"docker://{reference}"],
        capture_output=True, text=True, check=True, timeout=120,
    )  # fmt: skip
    return json.loads(inspected.stdout).get("Labels") or {}


def _recipes(plan: Mapping[str, Any]) -> Iterable[tuple[str, Mapping[str, Any]]]:
    for image in plan.get("images", []):
        for recipe in image.get("recipes", []):
            yield str(image["build_key"]), recipe


def _same_source(
    owner: str, recipe: Mapping[str, Any], pin: Mapping[str, str | None]
) -> bool:
    """A published pin of this exact recipe input, in the recipe's own package.

    The content digest names the recipe document only, so the source bundle must
    match too. Only our own package counts: the release asset is unverified.
    """

    bundle = recipe.get("source_bundle_sha256")
    return (
        bundle is not None
        and pin.get("source_bundle_sha256") == bundle
        and str(pin.get("reference", "")).startswith(
            f"{repository(owner, str(recipe['slug']))}@"
        )
    )


def _select_image(
    image: Mapping[str, Any],
    *,
    owner: str,
    candidates: set[str] | None,
    lookup: Lookup,
    published: Mapping[str, Mapping[str, str | None]],
    rebuild: set[str],
    labels: Labels,
) -> dict[str, Any]:
    key = str(image["build_key"])
    recipes = list(image.get("recipes", []))
    selected: dict[str, Any] = {
        "build": None, "copies": [], "adopted": [], "kept": [], "busy": False,
    }  # fmt: skip
    wanted = [
        recipe
        for recipe in recipes
        if candidates is None or str(recipe["slug"]) in candidates
    ]
    if not wanted:
        return selected
    tag = build_tag(key)
    try:
        present = {
            str(recipe["slug"]): lookup(owner, package(str(recipe["slug"])), tag)
            for recipe in recipes
        }
        build: list[Mapping[str, Any]] = []
        missing: list[Mapping[str, Any]] = []
        copies: list[dict[str, str]] = []
        adopted: list[str] = []
        kept: list[str] = []
        for recipe in wanted:
            slug = str(recipe["slug"])
            digest = present[slug]
            if slug in rebuild:
                # Only an adopted (adaptation-stage) image is replaced; an
                # unreadable label is no evidence and keeps the image.
                if digest is None or _is_adopted(
                    labels, f"{repository(owner, slug)}@{digest}"
                ):
                    build.append(recipe)
                else:
                    kept.append(slug)
                continue
            if digest is not None:
                continue
            pin = published.get(str(recipe.get("content_sha256")))
            if (
                pin is not None
                and _same_source(owner, recipe, pin)
                # A legacy pin: its own key has no v2 tag. A v2-built pin whose
                # key moved (the build policy changed) is rebuilt instead.
                and lookup(owner, package(slug), build_tag(str(pin["build_key"])))
                is None
            ):
                copies.append(
                    {
                        "from": str(pin["reference"]),
                        "to": f"{repository(owner, slug)}:{tag}",
                    }
                )
                adopted.append(slug)
                continue
            missing.append(recipe)
        if missing:
            source = next(
                (
                    slug
                    for slug, digest in present.items()
                    if digest and slug not in rebuild
                ),
                None,
            )
            # Only a successful build + smoke run writes a candidate tag. Promote
            # its immutable digest; recipe versions and commits do not affect reuse.
            if source is None:
                for recipe in recipes:
                    slug = str(recipe["slug"])
                    digest = lookup(owner, package(slug), candidate_tag(key))
                    if digest:
                        source = slug
                        present[slug] = digest
                        break
            if source is None:
                build.extend(missing)
            else:
                copies.extend(
                    {
                        "from": f"{repository(owner, source)}@{present[source]}",
                        "to": f"{repository(owner, str(recipe['slug']))}:{tag}",
                        "version_tag": f"{repository(owner, str(recipe['slug']))}:"
                        f"{recipe.get('version') or 'latest'}",
                    }
                    for recipe in missing
                )
    except (OSError, RuntimeError) as error:  # incl. UpstreamBusy
        # Not looked up this run: neither built nor copied; the next run asks again.
        print(f"{key}: registry busy, not planned: {error}", file=sys.stderr)
        selected["busy"] = True
        return selected
    if build:
        selected["build"] = {
            "build_key": key,
            "recipes": [
                {
                    "slug": str(recipe["slug"]),
                    "version": recipe.get("version"),
                    "repository": repository(owner, str(recipe["slug"])),
                }
                for recipe in build
            ],
        }
    selected.update(copies=copies, adopted=adopted, kept=kept)
    return selected


def _is_adopted(labels: Labels, reference: str) -> bool:
    try:
        return LEGACY_LABEL in labels(reference)
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        print(f"{reference}: labels unreadable, kept: {error}", file=sys.stderr)
        return False


def select(
    plan: Mapping[str, Any],
    *,
    owner: str,
    candidates: set[str] | None,
    lookup: Lookup = ghcr_digest,
    published: Mapping[str, Mapping[str, str | None]] | None = None,
    rebuild: set[str] | None = None,
    labels: Labels = skopeo_labels,
) -> dict[str, Any]:
    """What to build and what to copy for the candidate recipes.

    ``candidates`` of ``None`` means every recipe. Per recipe without its
    ``build-v2-<key>`` tag, in order: adopt its own published legacy image of the
    same input (``published``, by content digest); copy the key from a recipe that
    has it; promote a PR candidate; build the key once. ``rebuild`` names recipes
    whose adopted image is rebuilt from source.
    """

    with ThreadPoolExecutor(max_workers=8) as pool:
        selected = list(
            pool.map(
                lambda image: _select_image(
                    image,
                    owner=owner,
                    candidates=candidates,
                    lookup=lookup,
                    published=published or {},
                    rebuild=rebuild or set(),
                    labels=labels,
                ),
                plan.get("images", []),
            )
        )
    images = list(plan.get("images", []))
    ordered = sorted(
        (item["build"] for item in selected if item["build"]),
        key=lambda item: item["recipes"][0]["slug"],
    )
    return {
        "builds": ordered[:MAX_BUILDS],
        "deferred": [item["build_key"] for item in ordered[MAX_BUILDS:]],
        "deferred_slugs": [
            recipe["slug"]
            for item in ordered[MAX_BUILDS:]
            for recipe in item["recipes"]
        ],
        "copies": [copy for item in selected for copy in item["copies"]],
        "adopted": sorted(slug for item in selected for slug in item["adopted"]),
        "kept": sorted(slug for item in selected for slug in item["kept"]),
        "busy": [
            str(image["build_key"])
            for image, item in zip(images, selected, strict=True)
            if item["busy"]
        ],
    }


def published_pins(
    repository: str,
    *,
    gh: Callable[[list[str]], Any] = lambda arguments: subprocess.run(
        ["gh", *arguments], capture_output=True, text=True, check=True, timeout=300
    ),
) -> dict[str, dict[str, str | None]] | None:
    """The pins the current published release holds, by recipe content digest.

    Each pin carries its row's source bundle digest. ``None`` when the release
    cannot be read.
    """

    try:
        with tempfile.TemporaryDirectory() as directory:
            bundle = Path(directory) / BUNDLE
            gh(
                [
                    "release",
                    "download",
                    "--repo",
                    repository,
                    "--pattern",
                    BUNDLE,
                    "--output",
                    str(bundle),
                ]
            )
            with tarfile.open(bundle) as archive:
                member = archive.extractfile(INDEX_MEMBER)
                if member is None:
                    return None
                index = json.load(member)
        return {
            str(row["content_sha256"]): {
                "reference": str(row["prebuilt_image"]["reference"]),
                "build_key": str(row["prebuilt_image"]["build_key"]),
                "source_bundle_sha256": row.get("source_bundle_sha256"),
            }
            for row in index["recipes"]
            if row.get("prebuilt_image")
        }
    except (
        OSError,
        ValueError,
        LookupError,
        TypeError,
        AttributeError,
        tarfile.TarError,
        subprocess.SubprocessError,
    ) as error:
        print(f"published release not readable: {error}", file=sys.stderr)
        return None


def _published(repository: str | None) -> Mapping[str, Mapping[str, str | None]]:
    """The published pins, or end the run: without them adoption and carry cannot
    tell a legacy image from a missing one, and would rebuild or unpin proven
    recipes. The live release stays as it is; the next run reads it again."""

    if repository is None:
        return {}
    pins = published_pins(repository)
    if pins is None:
        raise SystemExit(
            "the published release is not readable: nothing planned or pinned; "
            "the next run asks again"
        )
    return pins


def resolve(
    plan: Mapping[str, Any],
    *,
    owner: str,
    lookup: Lookup = ghcr_digest,
    published: Mapping[str, Mapping[str, str | None]] | None = None,
) -> tuple[dict[str, dict[str, str]], list[str], list[str]]:
    """Map each recipe's content digest to its pinned prebuilt image.

    A recipe whose lookup found no digest (busy, failed or no such tag) keeps the
    pin the current release holds for the same content and source bundle, in its
    own package, whatever that pin's key: the digest names exactly the bytes that
    ran, and status is judged on it. Changed input stays unpinned.
    Returns the pins, the slugs without one and the slugs whose pin carried over.
    """

    entries = list(_recipes(plan))
    published = published or {}

    def one(entry: tuple[str, Mapping[str, Any]]) -> tuple[str, str, str | None]:
        key, recipe = entry
        slug = str(recipe["slug"])
        try:
            digest = lookup(owner, package(slug), build_tag(key))
        except (OSError, RuntimeError) as error:  # incl. UpstreamBusy
            print(f"{slug}: registry lookup failed: {error}", file=sys.stderr)
            digest = None
        return key, slug, digest

    with ThreadPoolExecutor(max_workers=8) as pool:
        found = list(pool.map(one, entries))
    images: dict[str, dict[str, str]] = {}
    missing: list[str] = []
    carried: list[str] = []
    for (key, recipe), (_key, slug, digest) in zip(entries, found, strict=True):
        content = str(recipe["content_sha256"])
        if digest is not None:
            images[content] = {
                "reference": f"{repository(owner, slug)}@{digest}",
                "build_key": key,
            }
            continue
        previous = published.get(content)
        if previous is not None and _same_source(owner, recipe, previous):
            images[content] = {
                "reference": str(previous["reference"]),
                "build_key": str(previous["build_key"]),
            }
            carried.append(slug)
            continue
        missing.append(slug)
    return images, sorted(missing), sorted(carried)


def candidate_slugs(requested: list[str]) -> set[str] | None:
    """Reconcile every current key, unless the run explicitly names recipes.

    A release tag is not evidence of image availability. In particular, a new
    contract version has no tag yet, and successful publication may advance a
    tag while builds are still pending.
    """

    return set(requested) if requested and requested != ["all"] else None


def changed_candidates(
    plan: Mapping[str, Any], base_plan: Mapping[str, Any]
) -> set[str]:
    """Compare executable identities, not paths or recipe document revisions.

    Evidence, Model metadata and recipe options that do not affect the build
    must not enqueue missing historical images. Shared contexts still enqueue
    every recipe using a new key, deduplicated by ``select``.
    """

    previous = {str(recipe["slug"]): key for key, recipe in _recipes(base_plan)}
    return {
        str(recipe["slug"])
        for key, recipe in _recipes(plan)
        if previous.get(str(recipe["slug"])) != key
    }


def missing_reasons(plan: Mapping[str, Any], missing: list[str]) -> dict[str, str]:
    """Why each recipe without a prebuilt image has none."""

    reasons = {
        str(item["slug"]): f"not planned: {item['reason']}"
        for item in plan.get("skipped", [])
    }
    for slug in missing:
        reasons[slug] = (
            "no image in the registry for its build key: the build failed, "
            "has not run or the lookup failed (see the image workflow)"
        )
    return dict(sorted(reasons.items()))


def main() -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    choose = commands.add_parser("select")
    choose.add_argument("--plan", type=Path, required=True)
    choose.add_argument("--owner", required=True)
    choose.add_argument(
        "--slugs",
        default="",
        help="space-separated recipe ids; defaults to every missing image",
    )
    choose.add_argument("--output", type=Path, required=True)
    choose.add_argument("--base-plan", type=Path)
    choose.add_argument(
        "--repository",
        help="owner/name whose current release supplies the legacy pins to adopt",
    )
    choose.add_argument(
        "--rebuild-adopted",
        action="store_true",
        help="rebuild the named recipes whose tag holds an adopted image",
    )
    look = commands.add_parser("resolve")
    look.add_argument("--plan", type=Path, required=True)
    look.add_argument("--owner", required=True)
    look.add_argument("--output", type=Path, required=True)
    look.add_argument(
        "--repository",
        help="owner/name whose current release supplies pins to carry forward",
    )
    arguments = parser.parse_args()
    plan = json.loads(arguments.plan.read_text(encoding="utf-8"))
    if arguments.command == "select":
        if arguments.base_plan is not None:
            if plan.get("skipped"):
                raise SystemExit(f"PR image planner refused recipes: {plan['skipped']}")
            candidates = changed_candidates(
                plan, json.loads(arguments.base_plan.read_text(encoding="utf-8"))
            )
        else:
            candidates = candidate_slugs(arguments.slugs.split())
        if arguments.rebuild_adopted and candidates is None:
            # A rebuild re-checks every recipe it touches: never fleet-wide.
            raise SystemExit("--rebuild-adopted needs the recipes named in --slugs")
        selection = select(
            plan,
            owner=arguments.owner,
            candidates=candidates,
            published=_published(arguments.repository),
            rebuild=candidates if arguments.rebuild_adopted else set(),
        )
        if arguments.base_plan is not None and selection["deferred"]:
            raise SystemExit("PR image matrix exceeds 256 keys; split the change")
        if arguments.base_plan is not None and selection["busy"]:
            # A skipped key would leave the PR green without its build and smoke proof.
            raise SystemExit(f"registry busy for {selection['busy']}; rerun")
        arguments.output.write_text(json.dumps(selection, indent=2), encoding="utf-8")
        print(
            f"{len(selection['builds'])} image build(s), "
            f"{len(selection['copies'])} copy(ies), "
            f"{len(selection['adopted'])} adopted, "
            f"{len(selection['deferred'])} deferred, "
            f"{len(selection['busy'])} busy (skipped)"
        )
        return 0
    images, missing, carried = resolve(
        plan, owner=arguments.owner, published=_published(arguments.repository)
    )
    arguments.output.write_text(
        json.dumps(images, indent=2, sort_keys=True), encoding="utf-8"
    )
    reasons = missing_reasons(plan, missing)
    print(
        f"{len(images)} recipe(s) have a prebuilt image; {len(reasons)} do not; "
        f"{len(carried)} pin(s) carried forward from the current release"
    )
    for slug, reason in reasons.items():
        print(f"  no prebuilt image: {slug}: {reason}")
    return 0
