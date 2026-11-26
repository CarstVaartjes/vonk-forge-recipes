"""The catalog adapters the refresh reads."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from recipe_tooling import vendored_upstream as vendored
from recipe_tooling.kit_pins import check as kit_check
from recipe_tooling.kit_pins import consumers as kit_consumers
from recipe_tooling.kit_pins import declarations as kit_declarations
from recipe_tooling.kit_pins import model as kit_model
from recipe_tooling.refresh_upstream import model as refresh_model
from recipe_tooling.upstream_drift import inventory as drift_inventory
from recipe_tooling.upstream_drift import model as drift_model
from recipe_tooling.vendored_manifest import VendoredManifest


@dataclass(slots=True)
class Catalog:
    root: Path
    recipes: dict[str, dict[str, Any]] = field(default_factory=dict)  # id -> document
    models: dict[str, dict[str, Any]] = field(default_factory=dict)  # slug -> document

    @classmethod
    def load(cls, root: Path) -> Catalog:
        catalog_ = cls(root)
        for path in sorted((root / "recipes").glob("*.json")):
            document = json.loads(path.read_text(encoding="utf-8"))
            catalog_.recipes[document["identity"]["slug"]] = document
        for path in sorted((root / "models").glob("*.json")):
            document = json.loads(path.read_text(encoding="utf-8"))
            catalog_.models[document["identity"]["slug"]] = document
        return catalog_

    def recipe_path(self, recipe_id: str) -> Path:
        return self.root / "recipes" / f"{recipe_id}.json"

    def model_path(self, slug: str) -> Path:
        return self.root / "models" / f"{slug}.json"

    def model_references(self, slug: str) -> list[str]:
        return [
            recipe_id
            for recipe_id, document in self.recipes.items()
            if any(m["model"]["slug"] == slug for m in document["models"])
        ]

    def catalogued(self, repo: str, revision: str) -> dict[str, Any] | None:
        for document in self.models.values():
            source = document.get("source") or {}
            if (
                source.get("revision") == revision
                and source.get("repository", "").lower()
                == f"https://huggingface.co/{repo}".lower()
            ):
                return document
        return None


@dataclass(frozen=True, slots=True)
class AdapterPin:
    repo: str
    commit: str
    directory: Path
    derived: bool = False  # the commit is locked from what the recipe's kit declares


def adapter_manifest(directory: Path) -> dict[str, Any]:
    path = directory / vendored.MANIFEST
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def kit_directory(root: Path, recipe: dict[str, Any]) -> Path | None:
    """The adapter directory when its manifest declares kit dependencies."""
    directory = root / recipe["execution"]["build"]["context"]["path"]
    if not (directory / "Dockerfile").is_file():
        return None
    return directory if "kit" in adapter_manifest(directory) else None


def declares_adapter_source(directory: Path | None) -> bool:
    """Whether the kit declares what the adapter's source pin would only duplicate.

    A declared base (the engine the kit's patches apply to) or image (what the Dockerfile's
    labels name the source of) is derived from the kit, so the adapter's own pin is not a
    second thing to follow. A kit that declares only models leaves the adapter pin alone.
    """
    if directory is None:
        return False
    try:
        wanted = kit_declarations.declarations(adapter_manifest(directory))
    except kit_model.KitError:
        return False  # reported by check_kit_dependencies
    return any(
        d.datasource == "oci-image" or kit_check.declares_base([d]) for d in wanted
    )


def image_label_claim(directory: Path | None, pin: AdapterPin | None) -> bool:
    """Whether the adapter's pin is only a label about the image it runs ``FROM``.

    An adapter with no ``vendored-upstream.json`` takes its pin from the
    Dockerfile's ``org.opencontainers.image.source`` / ``.revision`` labels. When the
    Dockerfile builds ``FROM`` a digest-pinned image of that repository (its name is in
    the image's) and never uses the repository (``adapter_usage``: no clone, archive, URL or
    commit-named file; labels and a NOTICE only cite it), the label is a claim about what the image is built
    from, not a git pin the refresh may advance: the upstream repository's newest release
    is not what the digest contains. The refresh never moves such a label; it is
    corrected only by a change that also states what the image records.
    """
    if directory is None or pin is None or adapter_manifest(directory):
        return (
            False  # a manifest declares a source, a kit or vendored files: a real pin
        )
    dockerfile = directory / "Dockerfile"
    name = pin.repo.rsplit("/", 1)[-1].lower()
    pinned = any(
        digest and name in image.lower()
        for image, _, digest in (
            kit_consumers.split_image(match["image"])
            for match in kit_consumers.FROM_LINE.finditer(
                dockerfile.read_text(encoding="utf-8")
            )
        )
    )
    return pinned and adapter_usage(directory, pin.repo, pin.commit) == (set(), False)


def whole_kit(directory: Path | None) -> bool:
    """Whether every changed file of the kit repository is judged (the default).

    A kit that is one directory of a creator's monorepo sets ``"whole": false``: only its
    declarations are checked, and the repository's other changes are judged as for any
    source.
    """
    return directory is not None and adapter_manifest(directory)["kit"].get(
        "whole", True
    )


def adapter_pin(root: Path, recipe: dict[str, Any]) -> AdapterPin | None:
    directory = root / recipe["execution"]["build"]["context"]["path"]
    if not (directory / "Dockerfile").is_file():
        return None
    manifest = adapter_manifest(directory)
    try:
        found = vendored._source(directory, VendoredManifest.model_validate(manifest))
    except (KeyError, OSError, ValueError):
        return None
    if found is None:
        return None
    return AdapterPin(found[0], found[1], directory, "kit" in manifest)


def kit_citation(repo: str) -> re.Pattern[str]:
    return re.compile(
        rf"(github\.com/{re.escape(repo)}/tree/)([0-9a-f]{{40}})", re.IGNORECASE
    )


def kit_citation_scope(directory: Path | None, repo: str) -> tuple[str, ...] | None:
    """Where the adapter's files cite the repository as their source; None when they do not.

    An adapter that records ``github.com/<repo>/tree/<commit>`` (a recipe-patches source
    label, a NOTICE) was built from the repository: its patches and scripts are that
    repository's, so the repository is a kit, not a model card. A citation that names a
    subdirectory (``tree/<commit>/models/vllm-sr``) makes only that subdirectory the kit;
    the empty tuple means the whole repository.
    """
    if directory is None or not directory.is_dir():
        return None
    needle = re.compile(
        kit_citation(repo).pattern + r"(/[^\s)\"'`<>]*)?", re.IGNORECASE
    )
    found = False
    scopes: set[str] = set()
    for path in sorted(directory.rglob("*")):
        if not path.is_file() or path.is_symlink() or path.stat().st_size > 1_000_000:
            continue
        if path.suffix in {".gz", ".zip", ".tar", ".patch", ".diff"}:
            continue
        for match in needle.finditer(
            path.read_text(encoding="utf-8", errors="replace")
        ):
            found = True
            subdirectory = (match.group(3) or "").strip("/").rstrip(".,;:")
            if not subdirectory:
                return ()
            scopes.add(subdirectory)
    return tuple(sorted(scopes)) if found else None


def cites_kit(directory: Path | None, repo: str) -> bool:
    return kit_citation_scope(directory, repo) is not None


def in_scope(path: str, scope: tuple[str, ...]) -> bool:
    """Whether a path lies in one of the kit's subdirectories (all of it when unscoped)."""
    return not scope or any(path == s or path.startswith(s + "/") for s in scope)


def scoped(
    comparison: refresh_model.Comparison, scope: tuple[str, ...]
) -> refresh_model.Comparison:
    """The comparison restricted to the kit's subdirectories."""
    if not scope:
        return comparison
    files = [
        f
        for f in comparison.files
        if in_scope(f["filename"], scope)
        or in_scope(f.get("previous_filename", ""), scope)
    ]
    return replace(comparison, files=files)


MOD_PATH = re.compile(r"(?<![\w-])(mods/[\w.-]+)")


def active_mods(directory: Path | None) -> set[str]:
    """Upstream mod paths a vendored creator yaml names that the adapter lacks.

    A creator's launcher applies the ``mods/<name>`` directories its recipe file lists
    from the creator's repository (a mod's script may download what the recipe mirrors),
    so they are used even though no URL names them. No yaml is parsed: every
    ``mods/<name>`` token counts, commented ones too, in any list spelling. Over-review
    is harmless.
    """
    found: set[str] = set()
    if directory is None or not directory.is_dir():
        return found
    for path in sorted(directory.rglob("*.y*ml")):
        text = path.read_text(encoding="utf-8", errors="replace")
        for name in MOD_PATH.findall(text):
            if not (directory / name).exists():
                found.add(name)
    return found


def maps_subset(directory: Path | None) -> bool:
    """Whether the adapter takes a mapped subset of the repository (renamed files).

    Its manifest names which upstream paths its vendored copies come from; mentions of
    the repository elsewhere in its files do not make the rest of the repository used.
    """
    if directory is None:
        return False
    manifest = adapter_manifest(directory)
    return bool(manifest.get("renamed") or manifest.get("renamed_dirs"))


def installs_library(directory: Path | None, repo: str) -> bool:
    """Whether the adapter pip-installs the repository as a library."""
    if directory is None or not directory.is_dir():
        return False
    escaped = re.escape(repo)
    install = re.compile(
        rf"\bpip3?\s+install\b[^\n]*(?:git\+https?://github\.com/{escaped}\b"
        rf"|github\.com/{escaped}/archive/)"
        rf"|^\s*git\+https?://github\.com/{escaped}\b",
        re.IGNORECASE | re.MULTILINE,
    )
    for path in sorted(directory.rglob("*")):
        if not path.is_file() or path.is_symlink() or path.stat().st_size > 1_000_000:
            continue
        if path.name != "Dockerfile" and not path.name.startswith("requirements"):
            continue
        text = path.read_text(encoding="utf-8", errors="replace").replace("\\\n", " ")
        if install.search(text):
            return True
    return False


def release_asset_only(directory: Path | None, repo: str) -> bool:
    """Whether the adapter uses ``repo`` only through its published release downloads.

    ``github.com/<repo>/releases/download/`` is the only way it fetches the repository:
    no clone, archive, codeload, raw URL, bare fetch/checkout or file named after a
    commit. The download is in the Dockerfile (what the refresh rewrites and checks).
    What it runs is the release asset, so a move is judged by the release, not by
    source changes. A ``github.com/<repo>/blob/...`` link is a web page, not a fetch.
    """
    if directory is None or not directory.is_dir():
        return False
    escaped = re.escape(repo)
    other = re.compile(
        "|".join(
            (
                *whole_repo_patterns(escaped)[:-1],
                rf"raw\.githubusercontent\.com/{escaped}/",
                rf"github\.com/{escaped}/raw/",
            )
        ),
        re.IGNORECASE,
    )
    mention = re.compile(rf"github\.com/{escaped}", re.IGNORECASE)
    download = re.compile(rf"github\.com/{escaped}/releases/download/", re.IGNORECASE)
    downloads = False
    for path in sorted(directory.rglob("*")):
        if not path.is_file() or path.is_symlink():
            continue
        if re.search(r"[0-9a-f]{12,}", path.name):
            return False
        if path.stat().st_size > 1_000_000 or path.suffix in {".gz", ".zip", ".tar"}:
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        if other.search(text) or (
            mention.search(text) and re.search(WEAK_WHOLE_REPO, text)
        ):
            return False
        if download.search(text):
            # Only the Dockerfile's download is moved and verified by the refresh.
            if path.name != "Dockerfile" or path.parent != directory:
                return False
            downloads = True
    return downloads


def adapter_files(directory: Path) -> set[str]:
    """Relative paths of the adapter's files that could be copies of upstream files."""
    own = {"Dockerfile", vendored.MANIFEST, vendored.KIT_LOCK}
    return {
        path.relative_to(directory).as_posix()
        for path in directory.rglob("*")
        if path.is_file() and path.relative_to(directory).as_posix() not in own
    }


def source_pin(recipe: dict[str, Any]) -> tuple[str, str, str] | None:
    """(provider, repository, commit) of ``provenance.source_reference``."""
    reference = recipe.get("provenance", {}).get("source_reference")
    if not isinstance(reference, str):
        return None
    try:
        location = drift_inventory.parse_source(reference)
    except drift_model.DriftInputError:
        return None
    if location.embedded_revision is None or location.provider not in {
        "github",
        "huggingface",
    }:
        return None
    return location.provider, location.repository, location.embedded_revision


# Everything that makes an adapter depend on the whole upstream repository rather than on
# the individual files it vendors or fetches by URL.
WEAK_WHOLE_REPO = (
    r"git\s+(fetch|checkout)\b"  # names no repository: a mention elsewhere
)


def whole_repo_patterns(escaped: str) -> tuple[str, ...]:
    """Ways of fetching one repository whole; each names that repository's own URL.

    A Dockerfile that labels the creator's repository and downloads another repository's
    tarball (TensorFold, ComfyUI) uses the creator's repository only where it says so.
    """
    return (
        rf"git\s+clone\b[^\n]*github\.com[/:]{escaped}\b",
        rf"git\+https?://github\.com/{escaped}\b",
        rf"github\.com/{escaped}/archive/",
        rf"codeload\.github\.com/{escaped}/",
        rf"github\.com/{escaped}/releases/download/",
    )


def adapter_usage(
    directory: Path | None, repo: str, commit: str
) -> tuple[set[str], bool]:
    """``(paths named by URL, whole repository used)`` for one upstream repository.

    Vendored files are verified byte for byte by ``check-vendored-upstream``; this adds
    the files an adapter fetches by explicit ``raw.githubusercontent.com`` /
    ``github.com/.../blob|raw/`` URL. An adapter that clones the repository, installs an
    archive of it, or carries a file named after the pinned commit uses all of it.
    """
    if directory is None or not directory.is_dir():
        return set(), False
    escaped = re.escape(repo)
    urls = re.compile(
        rf"(?:raw\.githubusercontent\.com/{escaped}/[^/\s\"'`)]+/"
        rf"|github\.com/{escaped}/(?:blob|raw)/[^/\s\"'`)]+/)([^\s\"'`)\\?#]+)",
        re.IGNORECASE,
    )
    mentions = re.compile(rf"github\.com/{escaped}", re.IGNORECASE)
    paths: set[str] = set()
    whole = weak = False
    for path in sorted(directory.rglob("*")):
        if not path.is_file() or path.is_symlink():
            continue
        if commit[:12] in path.name:
            whole = True
        if path.stat().st_size > 1_000_000 or path.suffix in {".gz", ".zip", ".tar"}:
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        paths.update(match for match in urls.findall(text))
        if mentions.search(text):
            if any(
                re.search(pattern, text, re.IGNORECASE)
                for pattern in whole_repo_patterns(escaped)
            ):
                whole = True
            weak = weak or re.search(WEAK_WHOLE_REPO, text) is not None
    # A bare git fetch/checkout beside a mention of the repository does not make the
    # rest of a mapped subset used; a clone, archive or download of it does.
    return paths, whole or (weak and not maps_subset(directory))
