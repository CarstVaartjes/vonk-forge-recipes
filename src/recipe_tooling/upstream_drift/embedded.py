"""The upstream references embedded in adapters."""

from __future__ import annotations

import hashlib
import json
import re
import shlex
from collections import Counter
from pathlib import Path
from urllib.parse import urlsplit

from vonk_forge_contracts import ModelDefinition, RecipeDefinition

from recipe_tooling.upstream_drift.inventory import _identity
from recipe_tooling.upstream_drift.model import SHA, DriftInputError, Watch


def _logical_dockerfile_lines(text: str) -> list[tuple[int, str]]:
    """Join Dockerfile continuations while retaining their first source line."""
    result: list[tuple[int, str]] = []
    start = 0
    current = ""
    for number, raw in enumerate(text.splitlines(), 1):
        line = raw.strip()
        if not current:
            start = number
        current += (" " if current else "") + line.rstrip("\\").strip()
        if line.endswith("\\"):
            continue
        if current:
            result.append((start, current))
        current = ""
    if current:
        result.append((start, current))
    return result


def embedded_inventory(root: Path, watches: list[Watch]) -> dict[str, object]:
    """Inventory recipe inputs that the upstream-ref watch cannot resolve.

    This is deliberately evidence-only: digest pins and embedded Git commits
    have no implied "latest" channel. A source with no configured watch is
    reported with a reason instead of being counted as current.
    """
    watch_entities = {watch.entity for watch in watches if watch.policy != "manual"}
    recipes: list[dict[str, object]] = []
    counts: Counter[str] = Counter()

    def add(inputs: list[dict[str, object]], item: dict[str, object]) -> None:
        inputs.append(item)
        counts[str(item["status"])] += 1

    for path in sorted((root / "recipes").glob("*.json")):
        try:
            document = RecipeDefinition.model_validate(
                json.loads(path.read_text(encoding="utf-8"))
            ).model_dump(mode="json")
        except Exception as error:
            raise DriftInputError(
                f"catalog contract is invalid: {path}: {error}"
            ) from error
        publisher, slug = _identity(document, path, "recipe")
        entity = f"recipe/{publisher}/{slug}"
        inputs: list[dict[str, object]] = []
        build = document.get("execution", {}).get("build", {})
        dockerfile_value = build.get("dockerfile") if isinstance(build, dict) else None
        dockerfile = (
            root / dockerfile_value if isinstance(dockerfile_value, str) else None
        )
        if dockerfile is None or not dockerfile.is_file():
            add(
                inputs,
                {
                    "kind": "dockerfile",
                    "path": dockerfile_value,
                    "status": "unknown",
                    "reason": "the recipe's build Dockerfile is unavailable for local inspection",
                },
            )
        else:
            dockerfile_rel = dockerfile.relative_to(root).as_posix()
            args: dict[str, str] = {}
            stages: set[str] = set()
            from_count = 0
            saw_from = False
            for line_number, line in _logical_dockerfile_lines(
                dockerfile.read_text(encoding="utf-8")
            ):
                arg = re.match(
                    r"^ARG\s+([A-Za-z_][A-Za-z0-9_]*)(?:=(.*))?$", line, re.IGNORECASE
                )
                if arg:
                    if not saw_from:
                        args.setdefault(arg.group(1), (arg.group(2) or "").strip())
                    continue
                if not re.match(r"^FROM\s+", line, re.IGNORECASE):
                    continue
                from_count += 1
                saw_from = True
                try:
                    tokens = shlex.split(line, comments=True)
                except ValueError:
                    tokens = []
                image_index = 1
                while image_index < len(tokens) and tokens[image_index].startswith(
                    "--"
                ):
                    image_index += 1
                raw_image = tokens[image_index] if image_index < len(tokens) else ""
                image = re.sub(
                    r"\$\{?([A-Za-z_][A-Za-z0-9_]*)\}?",
                    lambda match, mapping=args: mapping.get(
                        match.group(1), match.group(0)
                    ),
                    raw_image,
                )
                alias = None
                if "AS" in [token.upper() for token in tokens]:
                    as_index = [token.upper() for token in tokens].index("AS")
                    if as_index + 1 < len(tokens):
                        alias = tokens[as_index + 1].lower()
                if raw_image.lower() in stages:
                    stages.add(alias.lower() if alias else raw_image.lower())
                    continue
                if raw_image.lower() == "scratch":
                    status = "builtin"
                    reason = "Docker's built-in scratch stage has no external registry source"
                elif "$" in image:
                    status = "unknown"
                    reason = "FROM uses an ARG without a locally resolvable default"
                elif "@sha256:" in image:
                    status = "pin_only"
                    reason = "immutable OCI digest has no configured moving channel; availability is not drift"
                else:
                    status = "mutable"
                    reason = "FROM is not resolved to an immutable digest"
                add(
                    inputs,
                    {
                        "kind": "docker-image",
                        "path": dockerfile_rel,
                        "line": line_number,
                        "reference": image or raw_image,
                        "status": status,
                        "reason": reason,
                    },
                )
                if alias:
                    stages.add(alias.lower())
            if from_count == 0:
                add(
                    inputs,
                    {
                        "kind": "dockerfile",
                        "path": dockerfile_rel,
                        "status": "unknown",
                        "reason": "no FROM instruction was recognized by the inventory parser",
                    },
                )

            for line_number, line in _logical_dockerfile_lines(
                dockerfile.read_text(encoding="utf-8")
            ):
                if re.match(r"^ARG\s+", line, re.IGNORECASE):
                    match = re.match(
                        r"^ARG\s+([A-Za-z_][A-Za-z0-9_]*)(?:=(.*))?$",
                        line,
                        re.IGNORECASE,
                    )
                    if match and re.search(
                        r"(?:COMMIT|REVISION|REF|VERSION|SHA256)$",
                        match.group(1),
                        re.IGNORECASE,
                    ):
                        value = (match.group(2) or "").strip()
                        status = "pinned_untracked" if value else "unknown"
                        add(
                            inputs,
                            {
                                "kind": "docker-arg",
                                "path": dockerfile_rel,
                                "line": line_number,
                                "name": match.group(1),
                                "value": value or None,
                                "status": status,
                                "reason": "build ARG is embedded in the Dockerfile and has no independent watch policy",
                            },
                        )
                if re.search(
                    r"\b(?:apt(?:-get)?\s+install|pip(?:3)?\s+install|uv\s+pip\s+install|git\s+clone)\b",
                    line,
                    re.IGNORECASE,
                ):
                    kind = (
                        "git-source"
                        if re.search(r"\bgit\s+clone\b", line, re.IGNORECASE)
                        else "native-or-python-package"
                    )
                    add(
                        inputs,
                        {
                            "kind": kind,
                            "path": dockerfile_rel,
                            "line": line_number,
                            "command_sha256": hashlib.sha256(line.encode()).hexdigest(),
                            "status": "unknown",
                            "reason": "the command's resolved source/version closure is not derived from recipe provenance",
                        },
                    )
                if re.search(r"\b(?:curl|wget|git\s+clone)\b", line, re.IGNORECASE):
                    urls = re.findall(r"https?://[^\s\"'<>]+", line)
                    for url in urls:
                        parsed_url = urlsplit(url)
                        safe_url = (
                            f"{parsed_url.scheme}://{parsed_url.hostname or ''}"
                            f"{parsed_url.path}"
                        )
                        resolved = re.sub(
                            r"\$\{?([A-Za-z_][A-Za-z0-9_]*)\}?",
                            lambda match, mapping=args: mapping.get(
                                match.group(1), match.group(0)
                            ),
                            safe_url,
                        )
                        pinned_commit = bool(
                            re.search(r"[0-9a-f]{40}", resolved, re.IGNORECASE)
                        )
                        source_parts = [
                            part for part in parsed_url.path.split("/") if part
                        ]
                        embedded_repository = None
                        embedded_revision = None
                        if (parsed_url.hostname or "").lower() in {
                            "codeload.github.com",
                            "github.com",
                        } and len(source_parts) >= 3:
                            embedded_repository = "/".join(
                                source_parts[:2]
                            ).removesuffix(".git")
                            revision_match = re.search(
                                r"[0-9a-f]{40}", resolved, re.IGNORECASE
                            )
                            if revision_match:
                                embedded_revision = revision_match.group(0).lower()
                            if embedded_revision is None:
                                for variable in re.findall(
                                    r"\$\{?([A-Za-z_][A-Za-z0-9_]*)\}?", line
                                ):
                                    candidate = args.get(variable, "")
                                    if SHA.fullmatch(candidate):
                                        embedded_revision = candidate
                                        break
                        add(
                            inputs,
                            {
                                "kind": (
                                    "git-source"
                                    if re.search(
                                        r"\bgit\s+clone\b", line, re.IGNORECASE
                                    )
                                    else "download-source"
                                ),
                                "path": dockerfile_rel,
                                "line": line_number,
                                "url": resolved,
                                "repository": embedded_repository,
                                "revision": embedded_revision,
                                "status": "pinned_untracked"
                                if pinned_commit
                                else "unknown",
                                "reason": (
                                    "URL resolves to an embedded Git commit but has no independent watch policy"
                                    if pinned_commit
                                    else "download URL does not resolve to an immutable source revision"
                                ),
                            },
                        )

        runtime = document.get("runtime")
        options = document.get("options", [])
        for label, value in (
            ("launch-defaults", runtime),
            ("runtime-options", options),
        ):
            if value:
                digest = hashlib.sha256(
                    json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
                ).hexdigest()
                add(
                    inputs,
                    {
                        "kind": label,
                        "path": path.relative_to(root).as_posix(),
                        "sha256": digest,
                        "status": "catalogued",
                        "reason": "recipe-owned launch inputs are preserved as evidence; upstream mapping is not inferred",
                    },
                )
        patches = build.get("patches", []) if isinstance(build, dict) else []
        for patch in patches if isinstance(patches, list) else []:
            patch_path = patch.get("path") if isinstance(patch, dict) else None
            context_path = build.get("context", {}).get("path", "")
            patch_file = (
                root / context_path / patch_path
                if isinstance(patch_path, str)
                else None
            )
            if patch_file is not None and patch_file.is_file():
                patch_display_path = patch_file.relative_to(root).as_posix()
                patch_digest = hashlib.sha256(patch_file.read_bytes()).hexdigest()
                patch_status = "catalogued"
                patch_reason = "local patch bytes are inventoried, but no upstream mapping is inferred"
            else:
                patch_display_path = patch_path
                patch_digest = None
                patch_status = "unknown"
                patch_reason = (
                    "declared patch file is not present under the build context"
                )
            add(
                inputs,
                {
                    "kind": "build-patch",
                    "path": patch_display_path,
                    "sha256": patch_digest,
                    "status": patch_status,
                    "reason": patch_reason,
                },
            )
        model_entities: set[str] = set()
        for selection in document.get("models", []):
            model = selection.get("model", {}) if isinstance(selection, dict) else {}
            model_slug = model.get("slug") if isinstance(model, dict) else None
            if not isinstance(model_slug, str):
                continue
            model_path = root / "models" / f"{model_slug}.json"
            if not model_path.is_file():
                add(
                    inputs,
                    {
                        "kind": "companion-model",
                        "slug": model_slug,
                        "status": "unknown",
                        "reason": "selected Model document is unavailable locally",
                    },
                )
                continue
            try:
                model_document = ModelDefinition.model_validate(
                    json.loads(model_path.read_text(encoding="utf-8"))
                ).model_dump(mode="json")
            except Exception as error:
                raise DriftInputError(
                    f"catalog contract is invalid: {model_path}: {error}"
                ) from error
            identity = model_document.get("identity", {})
            model_entity = f"model/{identity.get('publisher')}/{identity.get('slug')}"
            model_entities.add(model_entity)
            source = model_document.get("source")
            tracked = model_entity in watch_entities
            add(
                inputs,
                {
                    "kind": "companion-model",
                    "slug": model_slug,
                    "source": source,
                    "status": "watched" if tracked else "untracked",
                    "reason": None
                    if tracked
                    else "Model has no automated source watch policy",
                },
            )
        recipes.append({"entity": entity, "recipe_id": slug, "inputs": inputs})
    return {
        "coverage": {
            "scope": "derived local inventory; not a claim that every transitive dependency has a resolvable upstream channel",
            "count_semantics": "records are per-recipe references; a shared image, Model, or Dockerfile can appear in multiple recipes",
            "counts_by_status": dict(sorted(counts.items())),
            "counts_by_candidate_status": {},
            "recipe_count": len(recipes),
        },
        "recipes": recipes,
    }
