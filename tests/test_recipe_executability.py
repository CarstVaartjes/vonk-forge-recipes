from __future__ import annotations

import json
import re
import sys
import unittest
from pathlib import Path, PurePosixPath


ROOT = Path(__file__).resolve().parents[1]
RECIPES = ROOT / "recipes"
sys.path.insert(0, str(ROOT / "contracts" / "src"))

from vonk_forge_contracts import ModelDefinition, RecipeDefinition  # noqa: E402
from vonk_forge_contracts.canonical import content_sha256  # noqa: E402
from vonk_forge_contracts.resolver import validate_recipe_models, validate_recipe_package_paths  # noqa: E402

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
    return [
        (path, json.loads(path.read_text(encoding="utf-8"))) for path in paths
    ]


class RecipeExecutabilityTests(unittest.TestCase):
    def test_user_facing_catalog_titles_are_unique_per_entity_kind(self) -> None:
        for directory in ("recipes",):
            seen: dict[str, Path] = {}
            for path in sorted((ROOT / directory).glob("*.json")):
                document = json.loads(path.read_text(encoding="utf-8"))
                metadata = document.get("metadata")
                self.assertIsInstance(metadata, dict, f"{path.name}: missing metadata")
                assert isinstance(metadata, dict)
                title = metadata.get("title")
                self.assertIsInstance(title, str, f"{path.name}: missing title")
                assert isinstance(title, str)
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
            model = ModelDefinition.model_validate(json.loads(path.read_text(encoding="utf-8")))
            key = (model.identity.publisher, model.identity.slug)
            self.assertNotIn(key, models, path.name)
            models[key] = model
        self.assertEqual(len(models), 92)
        self.assertEqual(len(list(RECIPES.glob("*.json"))), 84)
        for path in sorted(RECIPES.glob("*.json")):
            recipe = RecipeDefinition.model_validate(json.loads(path.read_text(encoding="utf-8")))
            validate_recipe_models(recipe, models.values())
            package_paths = {
                item.relative_to(ROOT).as_posix()
                for item in ROOT.rglob("*")
                if (item.is_file() or item.is_dir()) and "__pycache__" not in item.parts and item.suffix != ".pyc"
            }
            validate_recipe_package_paths(recipe, package_paths)
            self.assertEqual(content_sha256(recipe), content_sha256(recipe))

    def test_every_recipe_is_an_executable_candidate(self) -> None:
        recipes = recipe_documents()
        self.assertTrue(recipes, "recipes/*.json must contain at least one recipe")

        for path, document in recipes:
            with self.subTest(recipe=path.name):
                metadata = document.get("metadata")
                self.assertIsInstance(metadata, dict, f"{path.name}: missing metadata")
                assert isinstance(metadata, dict)

                tags = metadata.get("tags")
                self.assertIsInstance(
                    tags,
                    list,
                    f"{path.name}: metadata.tags must be a list",
                )
                assert isinstance(tags, list)
                self.assertTrue(
                    all(isinstance(tag, str) and tag.strip() for tag in tags),
                    f"{path.name}: metadata.tags must contain nonempty strings",
                )
                normalized_tags = {tag.casefold() for tag in tags}
                self.assertIn(
                    "candidate",
                    normalized_tags,
                    f"{path.name}: every recipe must retain the Candidate tag",
                )
                self.assertIn(
                    "executable",
                    normalized_tags,
                    f"{path.name}: every published recipe must be installable",
                )
                self.assertFalse(
                    normalized_tags & FORBIDDEN_TAGS,
                    f"{path.name}: executable recipes cannot use placeholder, "
                    "non-executable, or integration-required tags",
                )

                for field in ("title", "description"):
                    value = metadata.get(field)
                    self.assertIsInstance(
                        value,
                        str,
                        f"{path.name}: metadata.{field} must be text",
                    )
                    assert isinstance(value, str)
                    matched_claims = [
                        pattern.pattern
                        for pattern in FORBIDDEN_METADATA_CLAIMS
                        if pattern.search(value)
                    ]
                    self.assertFalse(
                        matched_claims,
                        f"{path.name}: current metadata.{field} claims the recipe "
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


if __name__ == "__main__":
    unittest.main()
