"""Model-mount rules that only surface once a recipe compiles to a launch plan.

Schema validation accepts both violations below, so light CI enforces them on
every recipe document:

- every role (Spark rank) of a multi-node recipe mounts the model files, so a
  worker rank can load its tensor-parallel shard;
- every model file mounts under ``/models``, the platform's model root.
"""

from __future__ import annotations

import json
import unittest
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]
RECIPES = ROOT / "recipes"
MODEL_ROOT = PurePosixPath("/models")


def _recipes() -> list[tuple[str, dict[str, object]]]:
    return [
        (path.name, json.loads(path.read_text(encoding="utf-8")))
        for path in sorted(RECIPES.glob("*.json"))
    ]


class RecipeModelMountTests(unittest.TestCase):
    def test_model_files_mount_under_the_model_root(self) -> None:
        for name, recipe in _recipes():
            with self.subTest(recipe=name):
                for selection in recipe["models"]:  # type: ignore[attr-defined]
                    for file in selection["files"]:
                        target = PurePosixPath(file["mount"]["target"])
                        self.assertTrue(
                            target == MODEL_ROOT or MODEL_ROOT in target.parents,
                            f"{name}: {selection['id']}/{file['id']} mounts at "
                            f"{target}, outside {MODEL_ROOT}",
                        )

    def test_every_rank_of_a_multi_node_recipe_mounts_the_model_files(self) -> None:
        for name, recipe in _recipes():
            topology = recipe["topology"]
            if topology["node_count"] < 2:  # type: ignore[index,operator]
                continue
            roles = {role["name"] for role in topology["roles"]}  # type: ignore[index]
            with self.subTest(recipe=name):
                for selection in recipe["models"]:  # type: ignore[attr-defined]
                    mounted = {
                        role for file in selection["files"] for role in file["roles"]
                    }
                    self.assertFalse(
                        roles - mounted,
                        f"{name}: model {selection['id']} has no files for "
                        f"role(s) {sorted(roles - mounted)}",
                    )


if __name__ == "__main__":
    unittest.main()
