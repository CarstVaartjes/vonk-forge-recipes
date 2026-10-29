from __future__ import annotations

import json
import re
import shlex
import sys
import unittest
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]
RECIPES = ROOT / "recipes"
sys.path.insert(0, str(ROOT / "contracts" / "src"))

from catalog_documents import catalog_models
from documents import recipe_display
from vonk_forge_contracts import ModelDefinition, RecipeDefinition
from vonk_forge_contracts.recipe import RecipeBuildDefinition
from vonk_forge_contracts.resolver import (
    validate_recipe_models,
    validate_recipe_package_paths,
)

FORBIDDEN_TAGS = frozenset(
    {
        "placeholder",
        "non-executable",
        "integration-required",
    }
)

# Only current user-facing metadata is inspected. Historical release changelogs
# intentionally remain free to explain why an older release was not executable.
FORBIDDEN_METADATA_CLAIMS = (
    re.compile(r"\bmetadata[\s-]+only\b", re.IGNORECASE),
    re.compile(r"\bnon[\s-]+executable\b", re.IGNORECASE),
    re.compile(r"\bintegration[\s-]+required\b", re.IGNORECASE),
    re.compile(r"\bno\s+adapter\b", re.IGNORECASE),
    re.compile(
        r"\buncatalog(?:ed|ued)[\s-]+runtime[\s-]+closure\b",
        re.IGNORECASE,
    ),
)


def recipe_documents() -> list[tuple[Path, dict[str, object]]]:
    paths = sorted(RECIPES.glob("*.json"))
    return [(path, json.loads(path.read_text(encoding="utf-8"))) for path in paths]


class RecipeExecutabilityTests(unittest.TestCase):
    def test_user_facing_catalog_titles_are_unique_per_entity_kind(self) -> None:
        for directory in ("recipes",):
            seen: dict[str, Path] = {}
            for path in sorted((ROOT / directory).glob("*.json")):
                title = recipe_display(path.stem).title
                normalized = " ".join(title.split()).casefold()
                self.assertNotIn(
                    normalized,
                    seen,
                    f"{path.name}: title duplicates {seen.get(normalized)}",
                )
                seen[normalized] = path

    def test_models_and_recipes_are_current_v2_documents(self) -> None:
        models: dict[tuple[str, str], ModelDefinition] = {}
        for path in sorted((ROOT / "models").glob("*.json")):
            model = ModelDefinition.model_validate(
                json.loads(path.read_text(encoding="utf-8"))
            )
            key = (model.identity.publisher, model.identity.slug)
            self.assertNotIn(key, models, path.name)
            models[key] = model
        self.assertTrue(models)
        self.assertTrue(list(RECIPES.glob("*.json")))
        for path in sorted(RECIPES.glob("*.json")):
            recipe = RecipeDefinition.model_validate(
                json.loads(path.read_text(encoding="utf-8"))
            )
            validate_recipe_models(recipe, catalog_models())
            package_paths = {
                item.relative_to(ROOT).as_posix()
                for item in ROOT.rglob("*")
                if (item.is_file() or item.is_dir())
                and "__pycache__" not in item.parts
                and item.suffix != ".pyc"
            }
            validate_recipe_package_paths(recipe, package_paths)

    def test_every_recipe_has_an_executable_contract(self) -> None:
        recipes = recipe_documents()
        self.assertTrue(recipes, "recipes/*.json must contain at least one recipe")

        for path, document in recipes:
            with self.subTest(recipe=path.name):
                display = recipe_display(path.stem)
                tags = display.tags
                self.assertTrue(
                    all(tag.strip() for tag in tags),
                    f"{path.name}: display tags must contain nonempty strings",
                )
                normalized_tags = {tag.casefold() for tag in tags}
                self.assertFalse(
                    normalized_tags & FORBIDDEN_TAGS,
                    f"{path.name}: executable recipes cannot use placeholder, "
                    "non-executable, or integration-required tags",
                )

                for field in ("title", "description"):
                    value = getattr(display, field)
                    matched_claims = [
                        pattern.pattern
                        for pattern in FORBIDDEN_METADATA_CLAIMS
                        if pattern.search(value)
                    ]
                    self.assertFalse(
                        matched_claims,
                        f"{path.name}: current display {field} claims the recipe "
                        f"is not executable: {matched_claims}",
                    )

                runtime = document.get("runtime")
                self.assertIsInstance(runtime, dict, f"{path.name}: missing runtime")
                assert isinstance(runtime, dict)
                entrypoint = runtime.get("entrypoint")
                self.assertIsInstance(
                    entrypoint,
                    list,
                    f"{path.name}: runtime.entrypoint must be a list",
                )
                assert isinstance(entrypoint, list)
                self.assertTrue(
                    entrypoint,
                    f"{path.name}: runtime.entrypoint must not be empty",
                )
                self.assertTrue(
                    all(isinstance(part, str) and part.strip() for part in entrypoint),
                    f"{path.name}: runtime.entrypoint must contain nonempty strings",
                )
                executable = entrypoint[0]
                assert isinstance(executable, str)
                executable_path = PurePosixPath(executable)
                self.assertTrue(
                    executable_path.is_absolute() and executable_path.name,
                    f"{path.name}: runtime.entrypoint must name a concrete "
                    "absolute executable",
                )
                self.assertNotIn(
                    "/bin/false",
                    entrypoint,
                    f"{path.name}: /bin/false is not an executable recipe entrypoint",
                )

    def test_entrypoint_copied_from_the_build_context_is_directly_executable(
        self,
    ) -> None:
        # The runtime execs runtime.entrypoint[0] itself. A script without a
        # shebang fails with ENOEXEC and falls back to /bin/sh, which then
        # reads Python as shell (the ComfyUI job adapters failed exactly so).
        # Entry points written by RUN or copied from a build stage are out of
        # static reach and are left to the build.
        for path in sorted(RECIPES.glob("*.json")):
            recipe = RecipeDefinition.model_validate(
                json.loads(path.read_text(encoding="utf-8"))
            )
            executable = recipe.runtime.entrypoint[0]
            for source in copied_context_sources(recipe.execution.build, executable):
                with self.subTest(recipe=path.name, source=source):
                    head = (ROOT / source).read_bytes()[:4]
                    self.assertTrue(
                        head.startswith(b"#!") or head == b"\x7fELF",
                        f"{path.name}: {source} is installed as the entrypoint "
                        f"{executable} but has no #! line",
                    )


def copied_context_sources(build: RecipeBuildDefinition, destination: str) -> list[str]:
    """Build-context files a Dockerfile COPY/ADD installs at ``destination``."""
    context = PurePosixPath(build.context.path)
    text = (ROOT / build.dockerfile).read_text(encoding="utf-8")
    sources: list[str] = []
    for line in text.replace("\\\n", " ").splitlines():
        instruction = line.split(maxsplit=1)
        if not instruction or instruction[0].upper() not in ("COPY", "ADD"):
            continue
        words = shlex.split(line, comments=True)
        if any(word.startswith("--from") for word in words[1:]):
            continue
        *items, target = [word for word in words[1:] if not word.startswith("--")]
        for item in items:
            installed = PurePosixPath(target)
            if target.endswith("/"):
                installed /= PurePosixPath(item).name
            if installed == PurePosixPath(destination):
                sources.append((context / item).as_posix())
    return sources


if __name__ == "__main__":
    unittest.main()
