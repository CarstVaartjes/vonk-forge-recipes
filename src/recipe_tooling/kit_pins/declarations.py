"""The kit declarations a recipe makes."""

from __future__ import annotations

import re
from typing import Any

from recipe_tooling.kit_pins.model import (
    DATASOURCES,
    OVERRIDE_VALUES,
    Declaration,
    KitError,
)
from recipe_tooling.kit_pins.upstream import Upstream

# ---------------------------------------------------------------------------
# Declarations
# ---------------------------------------------------------------------------


def declarations(manifest: dict[str, Any]) -> list[Declaration]:
    """The kit's declared dependencies from an adapter manifest (empty: not a kit adapter)."""
    block = manifest.get("kit")
    if block is None:
        return []
    found: list[Declaration] = []
    for name, entry in sorted(block.get("dependencies", {}).items()):
        required = {"datasource", "file", "pattern"}
        missing = required - set(entry)
        if missing:
            raise KitError(
                f"kit dependency '{name}' lacks {', '.join(sorted(missing))}"
            )
        if entry["datasource"] not in DATASOURCES:
            raise KitError(
                f"kit dependency '{name}': unknown datasource '{entry['datasource']}'"
            )
        override = entry.get("override")
        if override is not None and (
            not override.get("reason")
            or not any(override.get(key) for key in OVERRIDE_VALUES)
        ):
            raise KitError(
                f"kit dependency '{name}': an override needs a ref (or commit, revision, "
                "digest) and a reason"
            )
        try:
            compiled = re.compile(entry["pattern"], re.MULTILINE)
        except re.error as error:
            raise KitError(
                f"kit dependency '{name}': invalid pattern: {error}"
            ) from error
        if "version" not in compiled.groupindex:
            raise KitError(
                f"kit dependency '{name}': the pattern has no (?P<version>...) group"
            )
        # The repository is declared in exactly one place: the manifest, a repo_pattern,
        # or a repo group of the pattern itself.
        places = [
            place
            for place, given in (
                ("repo", "repo" in entry),
                ("repo_pattern", "repo_pattern" in entry),
                ("a (?P<repo>...) group in pattern", "repo" in compiled.groupindex),
            )
            if given
        ]
        if len(places) > 1:
            raise KitError(
                f"kit dependency '{name}': the repository is given by {' and '.join(places)}; "
                "give one"
            )
        if not places:
            raise KitError(
                f"kit dependency '{name}' lacks repo (or repo_pattern, or a "
                "(?P<repo>...) group in pattern)"
            )
        if "repo_pattern" in entry:
            try:
                repo_compiled = re.compile(entry["repo_pattern"], re.MULTILINE)
            except re.error as error:
                raise KitError(
                    f"kit dependency '{name}': invalid repo_pattern: {error}"
                ) from error
            if "repo" not in repo_compiled.groupindex:
                raise KitError(
                    f"kit dependency '{name}': the repo_pattern has no (?P<repo>...) group"
                )
        binds = entry.get("binds")
        if entry["datasource"] == "hf-revision" and not binds:
            raise KitError(
                f"kit dependency '{name}': an hf-revision names the recipe model it pins "
                "(binds: the models[].id)"
            )
        if binds and entry["datasource"] != "hf-revision":
            raise KitError(
                f"kit dependency '{name}': binds is for an hf-revision, not "
                f"{entry['datasource']}"
            )
        found.append(
            Declaration(
                name,
                entry.get("repo", ""),
                entry["datasource"],
                entry["file"],
                entry["pattern"],
                override,
                entry.get("repo_pattern"),
                binds,
            )
        )
    return found


def parse_pairs(text: str, declaration: Declaration) -> set[tuple[str | None, str]]:
    """Every distinct ``(repo or None, version)`` the pattern matches in the text."""
    return {
        (m.groupdict().get("repo"), m.group("version"))
        for m in re.finditer(declaration.pattern, text, re.MULTILINE)
    }


def parse_declared(text: str, declaration: Declaration) -> str:
    matches = {version for _, version in parse_pairs(text, declaration)}
    if not matches:
        raise KitError(
            f"kit dependency '{declaration.name}': nothing in {declaration.file} matches "
            "its pattern at the pinned kit commit"
        )
    if len(matches) > 1:
        raise KitError(
            f"kit dependency '{declaration.name}': {declaration.file} declares several "
            f"values ({', '.join(sorted(matches))}); narrow the pattern"
        )
    return next(iter(matches))


def parse_repo(text: str, declaration: Declaration) -> str:
    """The dependency's repository: the manifest's ``repo``, or what the kit declares."""
    if declaration.repo_in_pattern:
        repos = {repo for repo, _ in parse_pairs(text, declaration)}
    elif declaration.repo_pattern is not None:
        repos = {
            m.group("repo")
            for m in re.finditer(declaration.repo_pattern, text, re.MULTILINE)
        }
    else:
        return declaration.repo
    if not repos:
        raise KitError(
            f"kit dependency '{declaration.name}': nothing in {declaration.file} matches "
            "its repo_pattern at the pinned kit commit"
        )
    named = {_repository(declaration.datasource, str(repo)) for repo in repos}
    if len(named) > 1:
        raise KitError(
            f"kit dependency '{declaration.name}': {declaration.file} declares several "
            f"repositories ({', '.join(sorted(named))}); narrow the "
            f"{'pattern' if declaration.repo_in_pattern else 'repo_pattern'}"
        )
    return next(iter(named))


def _repository(datasource: str, value: str) -> str:
    """The canonical repository name of a declared value, for its datasource."""
    value = value.strip().rstrip("/")
    if datasource == "git-ref":
        return _owner_name(value)
    if datasource == "hf-revision":
        value = re.sub(r"^https?://huggingface\.co/", "", value)
        if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", value):
            raise KitError(f"'{value}' is not a Hugging Face repository (owner/name)")
        return value
    value = re.sub(r"^(?:docker://|https?://)", "", value)
    if not re.fullmatch(r"[a-z0-9][a-z0-9._:/-]*[a-z0-9]", value) or "@" in value:
        raise KitError(f"'{value}' is not an image repository (registry/name)")
    return value


def _owner_name(value: str) -> str:
    """``owner/name`` from a GitHub URL (``.git`` and trailing slash allowed) or itself."""
    value = value.strip().rstrip("/").removesuffix(".git")
    value = re.sub(r"^(?:https?://|git@)github\.com[/:]", "", value)
    if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", value):
        raise KitError(f"'{value}' is not a GitHub repository")
    return value


def read_declared(
    source: Upstream, kit_repo: str, kit_commit: str, declaration: Declaration
) -> tuple[str, str, str | None]:
    """``(repo, ref, override reason)`` the kit declares at ``kit_commit``."""
    text = source.github.read(kit_repo, kit_commit, declaration.file)
    override = declaration.override
    if override is not None:
        repo = override.get("repo")
        repo = (
            _repository(declaration.datasource, repo)
            if repo
            else parse_repo(text, declaration)
        )
        ref = next(override[key] for key in OVERRIDE_VALUES if override.get(key))
        return repo, ref, override["reason"]
    return parse_repo(text, declaration), parse_declared(text, declaration), None
