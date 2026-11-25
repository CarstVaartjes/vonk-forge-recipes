"""Classifying what an upstream change touches."""

from __future__ import annotations

import base64
import hashlib
import re
import urllib.error
import urllib.request
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.parse import quote

from recipe_tooling import vendored_upstream as vendored
from recipe_tooling.catalog_model import roles_for
from recipe_tooling.kit_pins import write as kit_write
from recipe_tooling.refresh_upstream import catalog_view as refresh_catalog_view
from recipe_tooling.refresh_upstream import evidence as refresh_evidence
from recipe_tooling.refresh_upstream import http as refresh_http
from recipe_tooling.refresh_upstream import model as refresh_model
from recipe_tooling.upstream_http import fetch
from vonk_forge_contracts.recipe import RecipeRelease

ARCHIVE_LABEL = re.compile(r'([\w.-]*archive-sha256="?)([0-9a-f]{64})("?)')
MAX_ARCHIVE_BYTES = 400_000_000


def archive_sha256(repo: str, commit: str) -> str:
    """SHA-256 of GitHub's source archive of one commit (verified at ingress only)."""
    digest = hashlib.sha256()
    total = 0
    url = f"https://codeload.github.com/{repo}/tar.gz/{commit}"
    request = urllib.request.Request(url, headers={"User-Agent": "vonk-forge-recipes"})

    def hash_stream(response: Any) -> None:
        nonlocal total
        while chunk := response.read(1 << 20):
            total += len(chunk)
            if total > MAX_ARCHIVE_BYTES:
                raise RuntimeError(f"{url} is larger than {MAX_ARCHIVE_BYTES} bytes")
            digest.update(chunk)

    try:
        fetch(request, hash_stream, timeout=120)
    except (urllib.error.HTTPError, urllib.error.URLError) as error:
        raise RuntimeError(f"cannot fetch {url}: {error}") from error
    return digest.hexdigest()


def retarget_archive_labels(
    dockerfile: str, item: refresh_model.Item
) -> tuple[str, bool]:
    """Move the pinned commit's archive digest wherever the Dockerfile states it.

    An ``*archive-sha256`` label whose value equals the archive digest of the old pin
    marks the Dockerfile as fetching that archive; every occurrence of that digest (the
    label and the ``sha256sum --check`` line that verifies the download) then moves to
    the new archive's digest. Other digests (other sources, wheels) are left alone.
    """
    matches = list(ARCHIVE_LABEL.finditer(dockerfile))
    if not matches:
        return dockerfile, False
    old = archive_sha256(item.repo, item.current)
    if not any(m.group(2) == old for m in matches):
        return dockerfile, False
    new = archive_sha256(item.repo, item.target.sha)
    return dockerfile.replace(old, new), True


# An adapter that installs a creator's release asset (TensorFold's native archive) pins
# it by version and sha256: a move re-pins the URL, the asset name, the version checks
# and the digest together, or it is a review.
RELEASE_ASSET_REVIEW = "release asset pinned by version and sha; re-pin by hand"
MAX_DIGEST_FILE_BYTES = 4096


class ReleaseAssetError(RuntimeError):
    """A release asset the adapter pins cannot be re-pinned at the new release."""


def release_downloads(repo: str) -> re.Pattern[str]:
    """``github.com/<repo>/releases/download/<tag>/<asset>`` URLs (tag, asset)."""
    return re.compile(
        rf"https://github\.com/{re.escape(repo)}/releases/download/"
        r"([^/\s\"'\\]+)/([^/\s\"'\\]+)",
        re.IGNORECASE,
    )


DOCKERFILE_ARG = re.compile(r"^\s*ARG\s+(\w+)=(\S+)", re.MULTILINE)
ARG_REFERENCE = re.compile(r"\$\{(\w+)\}|\$(\w+)")


def expand_args(text: str, dockerfile: str) -> str:
    """``text`` with ``${NAME}``/``$NAME`` replaced by the Dockerfile's ``ARG NAME=default``."""
    defaults = {
        m.group(1): m.group(2).strip("\"'") for m in DOCKERFILE_ARG.finditer(dockerfile)
    }
    return ARG_REFERENCE.sub(
        lambda m: defaults.get(m.group(1) or m.group(2), m.group(0)), text
    )


def version_token(version: str) -> re.Pattern[str]:
    """``version`` as a whole token (``1.2.3`` in ``-1.2.3-`` or ``v1.2.3``, not ``11.2.3``)."""
    return re.compile(rf"(?<![\w.])(v?){re.escape(version)}(?![\w.])")


def download_text(url: str) -> str:
    """A small published text file (a release's ``.sha256``), bounded in size."""
    request = urllib.request.Request(url, headers={"User-Agent": "vonk-forge-recipes"})
    try:
        data = fetch(request, lambda response: response.read(MAX_DIGEST_FILE_BYTES + 1))
    except (urllib.error.URLError, OSError) as error:
        raise ReleaseAssetError(f"{url}: {error}") from error
    if len(data) > MAX_DIGEST_FILE_BYTES:
        raise ReleaseAssetError(f"{url} is larger than {MAX_DIGEST_FILE_BYTES} bytes")
    return data.decode("utf-8", errors="replace")


def published_sha256(http: Any, repo: str, tag: str, asset: str) -> str:
    """The sha256 the creator publishes beside ``asset`` (``<asset>.sha256``), verified.

    Ingress read: the published digest file must name the asset and agree with the
    digest GitHub records for the uploaded asset, when GitHub records one.
    """
    try:
        release = http.github(f"repos/{repo}/releases/tags/{quote(tag, safe='')}")
    except (RuntimeError, KeyError, TypeError) as error:
        raise ReleaseAssetError(f"{repo} release {tag}: {error}") from error
    assets = {a.get("name"): a for a in release.get("assets", [])}
    if asset not in assets:
        raise ReleaseAssetError(f"{repo} release {tag} publishes no `{asset}`")
    published = assets.get(f"{asset}.sha256")
    if published is None:
        raise ReleaseAssetError(f"{repo} release {tag} publishes no `{asset}.sha256`")
    digests = {
        fields[0].lower()
        for line in download_text(published["browser_download_url"]).splitlines()
        if len(fields := line.split()) >= 2
        and re.fullmatch(r"[0-9a-fA-F]{64}", fields[0])
        and fields[-1].lstrip("*").rsplit("/", 1)[-1] == asset
    }
    if len(digests) != 1:
        raise ReleaseAssetError(f"`{asset}.sha256` of {tag} names no single digest")
    (digest,) = digests
    recorded = str(assets[asset].get("digest") or "")
    if recorded and recorded.lower() != f"sha256:{digest}":
        raise ReleaseAssetError(
            f"`{asset}.sha256` of {tag} disagrees with GitHub's recorded digest"
        )
    return digest


def retarget_release_assets(
    dockerfile: str, item: refresh_model.Item, http: Any
) -> tuple[str, bool]:
    """Move the creator's release assets the Dockerfile installs to the target release.

    Each ``releases/download/<tag>/<asset>`` URL of the pinned repository moves to the
    target tag, the old version in its asset name to the new one, and the old asset's
    published sha256 (wherever the Dockerfile states it) to the new asset's. Lines that
    check a ``VERSION`` file (``ARG TF_VERSION=`` included), and comments naming the
    project, carry the new version. A URL written with ``${ARG}`` references is read
    through the ``ARG`` defaults and left as written: the defaults move instead.
    Anything that cannot be re-pinned raises ``ReleaseAssetError``.
    """
    pattern = release_downloads(item.repo)
    matches = list(pattern.finditer(dockerfile))
    if not matches:
        return dockerfile, False
    new_tag, new_version = item.target.ref, item.target.version
    if not new_tag or not new_version:
        raise ReleaseAssetError("the target is not a published release")
    old_versions: set[str] = set()
    templated: list[tuple[str, str, str, str]] = []
    seen: set[tuple[str, str]] = set()
    for match in matches:
        raw_tag, raw_asset = match.group(1), match.group(2)
        old_tag = expand_args(raw_tag, dockerfile)
        old_asset = expand_args(raw_asset, dockerfile)
        if old_tag == new_tag or (old_tag, old_asset) in seen:
            continue
        seen.add((old_tag, old_asset))
        old_version = old_tag.removeprefix("v")
        old_versions.add(old_version)
        new_asset = version_token(old_version).sub(
            lambda m: m.group(1) + new_version, old_asset
        )
        old_digest = published_sha256(http, item.repo, old_tag, old_asset)
        if old_digest not in dockerfile:
            raise ReleaseAssetError(
                f"the Dockerfile does not pin `{old_asset}` by sha256"
            )
        new_digest = published_sha256(http, item.repo, new_tag, new_asset)
        dockerfile = dockerfile.replace(old_digest, new_digest)
        if (raw_tag, raw_asset) == (old_tag, old_asset):
            dockerfile = dockerfile.replace(
                match.group(0),
                f"https://github.com/{item.repo}/releases/download/{new_tag}/{new_asset}",
            )
        else:
            # A URL built from ARG defaults stays as written; the defaults move below.
            templated.append((raw_tag, raw_asset, new_tag, new_asset))
    if not old_versions:
        return dockerfile, False
    project = item.repo.split("/", 1)[1].lower()
    lines = []
    for original in dockerfile.splitlines(keepends=True):
        line = original
        if "VERSION" in line or (
            line.lstrip().startswith("#") and project in line.lower()
        ):
            for old_version in old_versions:
                line = version_token(old_version).sub(
                    lambda m: m.group(1) + new_version, line
                )
        lines.append(line)
    moved = "".join(lines)
    for raw_tag, raw_asset, new_tag, new_asset in templated:
        if (expand_args(raw_tag, moved), expand_args(raw_asset, moved)) != (
            new_tag,
            new_asset,
        ):
            raise ReleaseAssetError(
                f"the ARG defaults do not name {new_tag}/{new_asset} after the move"
            )
    return moved, True


def patch_targets(adapter: Path) -> set[str]:
    """Paths named by ``+++ b/<path>`` in the adapter's ``*.patch`` and ``*.diff`` files."""
    targets: set[str] = set()
    for path in sorted(adapter.rglob("*")):
        if path.suffix not in {".patch", ".diff"} or not path.is_file():
            continue
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            if line.startswith("+++ ") and not line.startswith("+++ /dev/null"):
                name = line[4:].split("\t")[0].strip()
                targets.add(re.sub(r"^[ab]/", "", name))
    return targets


# A creator's own vendored file that changed upstream is always a review: any byte of it
# can change what launches (a flag, a continuation, a heredoc, a kernel macro).
EDIT_COMMAND = re.compile(r"\b(sed|perl|awk|patch)\b|\bgit\s+apply\b")


def mentions(text: str, token: str) -> bool:
    """Whether ``token`` (a flag, env var or name) occurs in ``text`` as a whole word."""
    return re.search(rf"(?<![\w-]){re.escape(token)}(?![\w-])", text) is not None


def edited_locally(
    directory: Path, local: str, upstream: str, patched: set[str]
) -> str | None:
    """How the adapter changes a vendored file (a patch, an overlay, sed), if it does."""
    if local in patched or upstream in patched:
        return "an adapter patch edits it"
    name = PurePosixPath(local).name
    ours = [
        str(entry)
        for entry in refresh_catalog_view.adapter_manifest(directory).get("ours", [])
    ]
    for entry in ours:
        if entry != local and PurePosixPath(entry).name == name:
            return f"the adapter's own `{entry}` overlays it"
    redirect = re.compile(rf">>?\s*\S*{re.escape(name)}(?![\w.-])")
    for owned in ["Dockerfile", *ours]:
        path = directory / owned
        if not path.is_file():
            continue
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        text = "\n".join(line for line in lines if not line.lstrip().startswith("#"))
        for command in re.split(r"&&|\|\||;|\n", text.replace("\\\n", " ")):
            if mentions(command, name) and (
                EDIT_COMMAND.search(command) or redirect.search(command)
            ):
                return f"`{owned}` edits it: `{refresh_evidence.clip(command.strip(), 160)}`"
    return None


def upstream_blob(http: refresh_http.Http, repo: str, sha: str) -> bytes:
    """One upstream file's bytes by git blob sha, verified against that sha (ingress)."""
    document = http.github(f"repos/{repo}/git/blobs/{sha}")
    data = base64.b64decode(document["content"])
    if vendored.git_blob_sha(data) != sha:
        raise RuntimeError(f"{repo} blob {sha[:12]} does not hash to its sha")
    return data


def classify_changes(
    comparison: refresh_model.Comparison,
    *,
    patched: set[str],
    vendor_problems: list[str],
    used_changes: list[str] | None = None,
    mirrored: set[str] | frozenset[str] = frozenset(),
    harmless: frozenset[str] = frozenset(),
) -> list[str]:
    """Why a GitHub pin move is not mechanical; empty means it is.

    ``mirrored`` names upstream paths whose copy ships in the adapter (a kit's patches and
    scripts); a change to one is always a reason. ``used_changes`` is ``None`` when the
    files the recipe uses could not be told (the adapter uses the whole repository, or
    the repository is the kit the adapter is built from): every changed upstream file is
    then judged.
    Otherwise it lists the used files that differ between the two commits, and nothing
    else upstream matters: unrelated changes, a truncated comparison and a diverged
    history (the caller has rejected a pin that is ahead) are not reasons.
    """
    rewritten = (
        [
            (
                "upstream history was rewritten (GitHub finds no common ancestor): the "
                f"pinned commit is not in the history of the new head `{comparison.rewritten}`"
            )
        ]
        if comparison.rewritten
        else []
    )
    if used_changes is not None:
        # The used files are compared blob by blob whatever the history (a rewritten
        # one included, #1711), so a rewrite is no reason of its own here.
        used_reasons: list[str] = []
        if comparison.status not in {"ahead", "identical", "diverged", "behind"}:
            used_reasons.append(f"upstream comparison reported '{comparison.status}'")
        if used_changes:
            used_reasons.append(
                "files this recipe uses changed upstream: "
                + ", ".join(f"`{name}`" for name in used_changes[:20])
                + (
                    f" and {len(used_changes) - 20} more"
                    if len(used_changes) > 20
                    else ""
                )
            )
        return used_reasons + vendor_problems
    reasons: list[str] = list(rewritten)
    if comparison.status == "behind":
        reasons.append(
            "the pin is newer than the version the pinning rule selects; "
            "moving to it would be a downgrade"
        )  # unreachable through assess(): such pins are current
    elif comparison.status == "diverged":
        if not rewritten:
            reasons.append(
                "the selected version is not a descendant of the pinned commit"
            )
    elif comparison.status not in {"ahead", "identical"}:
        reasons.append(f"upstream comparison reported '{comparison.status}'")
    if comparison.truncated:
        reasons.append("too many upstream changes to inspect (300+ files or commits)")
    changed = {f["filename"] for f in comparison.files}
    changed |= {
        f["previous_filename"] for f in comparison.files if "previous_filename" in f
    }
    blocking = sorted(
        name
        for name in changed
        if refresh_model.is_blocking(name, harmless) or name in mirrored
    )
    if blocking:
        reasons.append(
            "launch scripts, configuration, patches or the README changed: "
            + ", ".join(f"`{name}`" for name in blocking[:20])
            + (f" and {len(blocking) - 20} more" if len(blocking) > 20 else "")
        )
    hit = sorted(patched & changed)
    if hit:
        reasons.append(
            "a patch's target file changed upstream, so it may no longer apply: "
            + ", ".join(f"`{name}`" for name in hit)
        )
    reasons.extend(vendor_problems)
    return reasons


# A kit-declared model revision is followed mechanically only when it changes nothing but
# metadata: the weights, configuration and tokenizer stay the catalogued bytes.
KIT_MODEL_FILES_CHANGED = (
    "the kit declares a revision whose files differ from the catalogued ones beyond "
    "metadata (weights, configuration or tokenizer); it is a different model payload, "
    "so review it"
)


def model_shape_problems(
    old: dict[str, Any], new_files: list[dict[str, Any]]
) -> list[str]:
    """Why a Model's new snapshot does not keep the old file set's shape."""
    problems: list[str] = []
    by_path = {f["path"]: f for f in new_files}
    old_by = {f["path"]: f for f in old["files"]}
    problems.extend(
        f"file removed or renamed upstream: `{removed}`"
        for removed in sorted(set(old_by) - set(by_path))
        if not set(old_by[removed]["roles"]) <= refresh_model.METADATA_ONLY
    )
    for added in sorted(set(by_path) - set(old_by)):
        roles = by_path[added]["roles"]
        if roles != ["metadata"]:
            problems.append(f"new {'/'.join(roles)} file: `{added}`")
    for path in sorted(set(old_by) & set(by_path)):
        entry = old_by[path]
        if entry["sha256"] != by_path[path]["sha256"] and not set(entry["roles"]) <= {
            "weights",
            "metadata",
        }:
            problems.append(
                f"{'/'.join(entry['roles'])} file changed: `{path}` "
                "(a weights-only digest change is mechanical; this is not)"
            )
    return problems


def mounted_paths(recipe: dict[str, Any], old: dict[str, Any]) -> set[str] | None:
    """Paths of the Model ``old`` that the recipe mounts; ``None`` when it mounts all."""
    by_id = {f["id"]: f["path"] for f in old["files"]}
    mounted = {
        by_id[selected["file_id"]]
        for reference in recipe["models"]
        if reference["model"]["slug"] == old["identity"]["slug"]
        for selected in reference["files"]
        if selected["file_id"] in by_id
    }
    return None if mounted == set(by_id.values()) else mounted


def mounted_unchanged(
    old: dict[str, Any], new_files: list[dict[str, Any]], mounted: set[str]
) -> bool:
    """Whether every file a recipe mounts is the same bytes at the same path."""
    before = {f["path"]: f["sha256"] for f in old["files"] if f["path"] in mounted}
    after = {f["path"]: f["sha256"] for f in new_files if f["path"] in mounted}
    return before == after


def only_sibling_weights_added(
    old: dict[str, Any], new_files: list[dict[str, Any]]
) -> bool:
    """Whether the newest revision only adds weights files beside byte-identical files.

    Another checkpoint of the repository (#2749): a Model whose every catalogued file is
    unchanged; a re-shard of a catalogued checkpoint changes or removes a catalogued file.
    """
    catalogued = {f["path"] for f in old["files"]}
    return mounted_unchanged(old, new_files, catalogued) and all(
        f["roles"] == ["weights"] for f in new_files if f["path"] not in catalogued
    )


def model_problems(
    old: dict[str, Any],
    new_files: list[dict[str, Any]],
    mounted: set[str] | None,
    *,
    declared: bool,
) -> list[str]:
    """Why a recipe cannot follow its Model's newest revision; empty means it follows.

    A re-export of the same repository (re-sharded weights, an added ``HYBRID.json``,
    changed configuration) is the newest revision: a recipe that mounts every file mounts
    the new set, one that mounts some moves unless a file it mounts by name is gone. A
    kit-declared revision keeps the stricter shape rule: the kit names the payload.
    """
    if declared:
        problems = model_shape_problems(old, new_files)
        if not problems and not payload_identical(old, new_files):
            problems = [KIT_MODEL_FILES_CHANGED]
        return problems
    present = {f["path"] for f in new_files}
    return [
        f"file this recipe mounts is gone upstream: `{path}`"
        for path in sorted((mounted or set()) - present)
    ]


def gated_payload_identical(
    old: dict[str, Any],
    before_tree: list[Any],
    new_tree: list[Any],
) -> bool:
    """``payload_identical`` from tree entries alone, for a repository we cannot read.

    A gated repository serves neither non-LFS bytes nor LFS digests without access, but
    its tree lists each file's git blob id (for an LFS file the pointer, which holds the
    digest) and size. ``<id>:<size>`` stands in for the digest on both sides; the
    stored document keeps its roles, and the curation and the comparison are the
    ungated ones (``curate``, ``payload_identical``).
    """

    def records(tree: list[Any]) -> dict[str, str]:
        return {
            e["path"]: f"{e.get('oid')}:{e.get('size')}"
            for e in tree
            if isinstance(e, dict) and e.get("type") == "file" and "path" in e
        }

    before, after = records(before_tree), records(new_tree)
    stored = [{**f, "sha256": before.get(f["path"], "missing")} for f in old["files"]]
    new_files = [
        {"path": path, "roles": roles_for(path), "sha256": digest}
        for path, digest in sorted(after.items())
    ]
    return payload_identical(
        {**old, "files": stored}, curate(old, set(before), new_files)
    )


def curate(
    old: dict[str, Any], before: set[str], new_files: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """The new snapshot's files the Model document is about (see ``Assessor._curated``)."""
    known = {entry["path"] for entry in old["files"]}
    return [f for f in new_files if f["path"] in known or f["path"] not in before]


def payload_identical(old: dict[str, Any], new_files: list[dict[str, Any]]) -> bool:
    """Whether every payload file is the same bytes at the same path.

    A file is payload when either its stored role or ``roles_for`` says so, judged
    identically on both sides by path: a disagreement between the two classifiers can
    neither fake a move (README-only revisions) nor hide one (merges, .gguf, tokenizer
    files stored as metadata). A path the document does not know uses its own roles.
    """
    stored = {f["path"]: f["roles"] for f in old["files"]}

    def payload(path: str, own: list[str]) -> bool:
        return stored.get(path, own) != ["metadata"] or roles_for(path) != ["metadata"]

    before = {
        f["path"]: f["sha256"] for f in old["files"] if payload(f["path"], f["roles"])
    }
    after = {
        f["path"]: f["sha256"] for f in new_files if payload(f["path"], f["roles"])
    }
    return before == after


def refreshed_model(
    old: dict[str, Any], new_files: list[dict[str, Any]], target: refresh_model.Target
) -> dict[str, Any]:
    """The old Model document moved to the new snapshot, keeping its curation."""
    return kit_write.moved_model(old, new_files, target.sha)


def released(
    recipe: dict[str, Any], items: list[refresh_model.Item]
) -> tuple[str, str]:
    """``(version, released_at)`` for a refreshed recipe (docs/recipe-authoring.md section 4)."""
    current = recipe["release"]["version"]
    published = next(
        (i for i in items if i.kind in {"source", "adapter"} and i.target.version), None
    )
    newest = max(items, key=lambda i: i.target.date)
    date = published.target.date if published else newest.target.date
    if published is not None:
        try:
            RecipeRelease.model_validate(
                {"version": published.target.version, "released_at": date}
            )
            return str(published.target.version), date
        except ValueError:
            pass
    match = refresh_model.OWN_VERSION.match(current)
    if match is None:
        return current, max(date, recipe["release"]["released_at"])
    major, minor, patch = (int(part) for part in match.groups())
    return f"{major}.{minor}.{patch + 1}", max(date, recipe["release"]["released_at"])
