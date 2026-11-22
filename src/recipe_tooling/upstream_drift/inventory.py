"""The watches the drift check discovers."""

from __future__ import annotations

import json
import re
from dataclasses import replace
from pathlib import Path
from urllib.parse import urlsplit

from vonk_forge_contracts import ModelDefinition, RecipeDefinition
from vonk_forge_contracts.model import GitHubReleaseSource

from recipe_tooling.qualification.documents import model_display
from recipe_tooling.upstream_drift.model import (
    ENTITY_DIRECTORIES,
    POLICIES,
    REPOSITORY_PART,
    SHA,
    DriftInputError,
    SourceLocation,
    Watch,
)


def _repository_parts(path: str) -> tuple[str, str] | None:
    parts = [part for part in path.split("/") if part]
    if len(parts) < 2:
        return None
    owner, repository = parts[:2]
    repository = repository.removesuffix(".git")
    if not REPOSITORY_PART.fullmatch(owner) or not REPOSITORY_PART.fullmatch(
        repository
    ):
        return None
    return owner, repository


def parse_source(value: str) -> SourceLocation:
    """Normalize the GitHub and Hugging Face forms present in catalog metadata."""

    if re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", value):
        return SourceLocation("huggingface", value)
    parsed = urlsplit(value)
    if parsed.scheme != "https" or parsed.username or parsed.password:
        raise DriftInputError(f"source must be a public HTTPS URL: {value}")
    if parsed.query or parsed.fragment:
        raise DriftInputError(
            f"source URL must not contain a query or fragment: {value}"
        )
    host = (parsed.hostname or "").lower()
    parts = [part for part in parsed.path.split("/") if part]
    if host == "github.com":
        supported_length = len(parts) in {2, 4} or (
            len(parts) >= 5 and parts[2] in {"blob", "tree"}
        )
        if not supported_length or not REPOSITORY_PART.fullmatch(parts[0]):
            raise DriftInputError(f"GitHub source repository is invalid: {value}")
        owner, repository = parts[:2]
        repository = repository.removesuffix(".git")
        embedded: str | None = None
        legacy = re.fullmatch(r"(.+)@([0-9a-f]{40})", repository)
        if legacy is not None:
            repository, embedded = legacy.groups()
            repository = repository.removesuffix(".git")
        elif (
            len(parts) == 4
            and parts[2] in {"tree", "commit"}
            and SHA.fullmatch(parts[3])
        ) or (
            len(parts) >= 5 and parts[2] in {"blob", "tree"} and SHA.fullmatch(parts[3])
        ):
            embedded = parts[3]
        elif len(parts) != 2:
            raise DriftInputError(f"GitHub source URL form is unsupported: {value}")
        if not REPOSITORY_PART.fullmatch(repository):
            raise DriftInputError(f"GitHub source repository is invalid: {value}")
        return SourceLocation("github", f"{owner}/{repository}", embedded)
    if host == "huggingface.co":
        repository_parts = _repository_parts(parsed.path)
        if repository_parts is None:
            raise DriftInputError(f"Hugging Face source repository is invalid: {value}")
        owner, repository = repository_parts
        embedded = None
        tree_revision = (
            len(parts) == 4 and parts[2] == "tree" and SHA.fullmatch(parts[3])
        )
        resolved_file = (
            len(parts) >= 5 and parts[2] == "resolve" and SHA.fullmatch(parts[3])
        )
        if tree_revision or resolved_file:
            embedded = parts[3]
        elif len(parts) != 2:
            raise DriftInputError(
                f"Hugging Face source URL form is unsupported: {value}"
            )
        return SourceLocation("huggingface", f"{owner}/{repository}", embedded)
    return SourceLocation(host or "unsupported", value)


def _identity(document: object, path: Path, expected_kind: str) -> tuple[str, str]:
    if not isinstance(document, dict):
        raise DriftInputError(f"catalog kind is invalid: {path}")
    actual_kind = document.get("kind")
    if actual_kind != expected_kind:
        raise DriftInputError(f"catalog kind is invalid: {path}")
    identity = document.get("identity")
    if not isinstance(identity, dict):
        raise DriftInputError(f"catalog identity is missing: {path}")
    publisher, slug = identity.get("publisher"), identity.get("slug")
    if not isinstance(publisher, str) or not isinstance(slug, str):
        raise DriftInputError(f"catalog identity is invalid: {path}")
    return publisher, slug


def load_manifest(path: Path) -> tuple[dict[str, str], dict[str, dict[str, str]]]:
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise DriftInputError(
            f"cannot read upstream watch manifest: {path}: {error}"
        ) from error
    if not isinstance(document, dict) or set(document) != {
        "schema_version",
        "defaults",
        "overrides",
    }:
        raise DriftInputError("upstream watch manifest has unexpected fields")
    defaults, overrides = document.get("defaults"), document.get("overrides")
    if (
        document.get("schema_version") != 1
        or not isinstance(defaults, dict)
        or not isinstance(overrides, dict)
    ):
        raise DriftInputError("upstream watch manifest is invalid")
    clean_defaults: dict[str, str] = {}
    for provider, policy in defaults.items():
        if not isinstance(provider, str) or policy not in POLICIES:
            raise DriftInputError(f"invalid default watch policy for {provider}")
        clean_defaults[provider] = policy
    clean_overrides: dict[str, dict[str, str]] = {}
    for entity, override in overrides.items():
        if not isinstance(entity, str) or not isinstance(override, dict):
            raise DriftInputError("upstream watch override is invalid")
        if set(override) - {"policy", "ref"}:
            raise DriftInputError(
                f"upstream watch override has unexpected fields: {entity}"
            )
        policy, ref = override.get("policy"), override.get("ref")
        if policy not in POLICIES or (ref is not None and not isinstance(ref, str)):
            raise DriftInputError(f"upstream watch override is invalid: {entity}")
        if policy == "latest-release" and ref is not None:
            raise DriftInputError(
                f"latest-release override cannot select a ref: {entity}"
            )
        clean_overrides[entity] = {
            "policy": policy,
            **({"ref": ref} if ref is not None else {}),
        }
    return clean_defaults, clean_overrides


def _watch(
    *,
    entity: str,
    source_path: Path,
    source_value: str,
    revision: str,
    version: str | None,
    root: Path,
    defaults: dict[str, str],
    overrides: dict[str, dict[str, str]],
) -> Watch:
    if not SHA.fullmatch(revision):
        raise DriftInputError(
            f"source revision is not an immutable Git commit: {source_path}"
        )
    location = parse_source(source_value)
    if (
        location.embedded_revision is not None
        and location.embedded_revision != revision
    ):
        raise DriftInputError(f"source URL and revision disagree: {source_path}")
    override = overrides.get(entity, {})
    policy = override.get("policy", defaults.get(location.provider, "manual"))
    ref = override.get("ref")
    if policy == "latest-release" and location.provider != "github":
        raise DriftInputError(f"latest-release is only supported for GitHub: {entity}")
    return Watch(
        entity=entity,
        source_path=source_path.relative_to(root).as_posix(),
        provider=location.provider,
        repository=location.repository,
        pinned_revision=revision,
        pinned_version=version,
        policy=policy,
        ref=ref,
    )


def discover_watches(
    root: Path,
    defaults: dict[str, str],
    overrides: dict[str, dict[str, str]],
) -> list[Watch]:
    watches: list[Watch] = []
    discovered_entities: set[str] = set()
    # A Model's family is its display lineage's; a helper has none.
    family_of: dict[str, str] = {}
    for path in sorted((root / "display" / "models").glob("*.json")):
        display = model_display(path.stem, root)
        if display.lineage is not None:
            identity = display.identity
            family_of[f"model/{identity.publisher}/{identity.slug}"] = (
                display.lineage.family
            )
    models_of: dict[str, tuple[str, ...]] = {}
    for directory, expected_kind in ENTITY_DIRECTORIES.items():
        for path in sorted((root / directory).glob("*.json")):
            document = json.loads(path.read_text(encoding="utf-8"))
            try:
                if expected_kind == "model":
                    validated_model = ModelDefinition.model_validate(document)
                    document = validated_model.model_dump(mode="json")
                else:
                    document = RecipeDefinition.model_validate(document).model_dump(
                        mode="json"
                    )
            except Exception as error:
                raise DriftInputError(
                    f"catalog contract is invalid: {path}: {error}"
                ) from error
            publisher, slug = _identity(document, path, expected_kind)
            entity = f"{expected_kind}/{publisher}/{slug}"
            discovered_entities.add(entity)
            source = document.get("source")
            if not isinstance(source, dict):
                continue
            if source.get("provider") == "github-release":
                release_source = GitHubReleaseSource.model_validate(source)
                policy = overrides.get(entity, {}).get("policy", "latest-release")
                if policy not in {"latest-release", "manual"} or overrides.get(
                    entity, {}
                ).get("ref"):
                    raise DriftInputError(
                        f"release assets require latest-release or manual policy without a ref: {entity}"
                    )
                watches.append(
                    Watch(
                        entity=entity,
                        source_path=path.relative_to(root).as_posix(),
                        provider="github",
                        repository=parse_source(release_source.repository).repository,
                        pinned_revision=str(release_source.release_id),
                        pinned_version=None,
                        policy=policy,
                        ref=None,
                        release_source=release_source,
                    )
                )
                continue
            repository, revision = source.get("repository"), source.get("revision")
            if not isinstance(repository, str) or not isinstance(revision, str):
                raise DriftInputError(f"catalog source is incomplete: {path}")
            location = parse_source(repository)
            watches.append(
                _watch(
                    entity=entity,
                    source_path=path,
                    source_value=repository,
                    revision=revision,
                    version=None,
                    root=root,
                    defaults=defaults,
                    overrides=overrides,
                )
            )
    for path in sorted((root / "recipes").glob("*.json")):
        document = json.loads(path.read_text(encoding="utf-8"))
        try:
            validated_recipe = RecipeDefinition.model_validate(document)
            document = validated_recipe.model_dump(mode="json")
        except Exception as error:
            raise DriftInputError(
                f"catalog contract is invalid: {path}: {error}"
            ) from error
        publisher, slug = _identity(document, path, "recipe")
        entity = f"recipe/{publisher}/{slug}"
        discovered_entities.add(entity)
        models_of[entity] = tuple(
            f"model/{selection.model.publisher}/{selection.model.slug}"
            for selection in validated_recipe.models
        )
        provenance = document.get("provenance")
        reference = (
            provenance.get("source_reference") if isinstance(provenance, dict) else None
        )
        if not isinstance(reference, str):
            continue
        location = parse_source(reference)
        if location.embedded_revision is None:
            raise DriftInputError(
                f"recipe provenance does not pin a Git revision: {path}"
            )
        watches.append(
            _watch(
                entity=entity,
                source_path=path,
                source_value=reference,
                revision=location.embedded_revision,
                version=None,
                root=root,
                defaults=defaults,
                overrides=overrides,
            )
        )
    unknown_overrides = sorted(set(overrides) - discovered_entities)
    if unknown_overrides:
        raise DriftInputError(
            f"watch override names unknown entity: {unknown_overrides[0]}"
        )
    return sorted(
        (
            replace(
                watch,
                family=family_of.get(watch.entity),
                models=models_of.get(watch.entity, ()),
            )
            for watch in watches
        ),
        key=lambda watch: (watch.entity, watch.repository),
    )
