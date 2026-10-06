"""A recipe that lets vLLM read local media must ship that directory.

vLLM refuses every request that references a local file when
``--allowed-local-media-path`` names a directory the container lacks
("Invalid ``--allowed-local-media-path``: The path ... does not exist").
Service recipes get no platform mount for it, so the recipe's own Dockerfile
has to create the directory.
"""

from __future__ import annotations

import json
import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FLAG = "allowed-local-media-path"


def _created_directories(dockerfile: str) -> set[str]:
    text = dockerfile.replace("\\\n", " ")
    created: set[str] = set()
    for line in text.splitlines():
        for command in re.split(r"&&|;|\|\|", line):
            words = command.split()
            if ("install" in words and "--directory" in words) or "mkdir" in words:
                created.update(w for w in words if w.startswith("/"))
        words = line.split()
        if words[:1] == ["WORKDIR"] and len(words) > 1:
            created.add(words[1])
    return {p.rstrip("/") or "/" for p in created}


class LocalMediaPathTests(unittest.TestCase):
    def test_every_local_media_path_exists_in_the_image(self) -> None:
        checked = 0
        for recipe_path in sorted((ROOT / "recipes").glob("*.json")):
            recipe = json.loads(recipe_path.read_text(encoding="utf-8"))
            values = [
                a.get("value")
                for a in recipe.get("runtime", {}).get("arguments", [])
                if a.get("name") == FLAG
            ]
            if not values:
                continue
            dockerfile = ROOT / recipe["execution"]["build"]["dockerfile"]
            created = _created_directories(dockerfile.read_text(encoding="utf-8"))
            for value in values:
                checked += 1
                with self.subTest(recipe=recipe_path.stem, path=value):
                    self.assertIsInstance(value, str)
                    self.assertTrue(value.startswith("/"), "must be absolute")
                    self.assertIn(
                        value.rstrip("/"),
                        created,
                        f"{dockerfile.relative_to(ROOT)} must create {value}",
                    )
        self.assertGreater(checked, 0)

    def test_checker_detects_a_missing_directory(self) -> None:
        self.assertNotIn(
            "/inputs", _created_directories("RUN install --directory /state\n")
        )
        self.assertIn(
            "/inputs",
            _created_directories(
                "RUN a \\\n && install --directory --mode=0755 /inputs \\\n && b\n"
            ),
        )


if __name__ == "__main__":
    unittest.main()
