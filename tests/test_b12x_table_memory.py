"""A b12x-image recipe must not leave the PLE/Engram table on a disk transport.

The b12x DiskTable offers io_uring (blocked by the sandbox seccomp profile) and
direct GDS (needs a real nvidia-fs; compat mode is rejected with cuFile error
5027). Models with a BF16 PLE table (or Engram) default to disk, so a recipe
must set VLLM_PLE_TABLE_MEMORY to shared or ram. See docs/recipe-authoring.md.
"""

from __future__ import annotations

import json
import re
import unittest
from pathlib import Path
from typing import Any

RECIPES = Path(__file__).resolve().parents[1] / "recipes"
TABLE_ENV = "VLLM_PLE_TABLE_MEMORY"
SAFE = {"shared", "ram"}
# Model families whose checkpoint carries a PLE/Engram table on the b12x stack.
TABLE_MODEL_PREFIXES = ("qwen3-8-flash-next", "deepseek-v4", "glm-5-3-flash")

# Recipes that still leave the table on disk, each with the reason. Remove an
# entry when the recipe is fixed; do not add one for a new recipe.
ALLOWED: dict[str, str] = {
    "deepseek-v4-flash-0731-eugr-vllm-dual": "no table env; unproven on hardware, follow-up",
    "deepseek-v4-flash-vision-exp-eugr-vllm-dual": "no table env; unproven on hardware, follow-up",
    "deepseek-v4-flash-vision-exp-sfxnz-vllm-dual": "no table env; unproven on hardware, follow-up",
    "glm-5-3-flash-nvfp4-spark-eugr-vllm-dual": "no table env; unproven on hardware, follow-up",
    "qwen3-8-flash-next-fp4-local-vcruz-vllm-single": "disk with io_uring; unproven, follow-up",
    "qwen3-8-flash-next-nvfp4-eugr-vllm-dual": "no table env (BF16 default); unproven, follow-up",
}


def env_values(node: Any, found: dict[str, str]) -> None:
    if isinstance(node, dict):
        name = node.get("name")
        if isinstance(name, str) and isinstance(node.get("value"), str):
            found[name] = node["value"]
        for child in node.values():
            env_values(child, found)
    elif isinstance(node, list):
        for child in node:
            env_values(child, found)


def flagged() -> set[str]:
    result: set[str] = set()
    for path in sorted(RECIPES.glob("*.json")):
        document = json.loads(path.read_text())
        env: dict[str, str] = {}
        env_values(document, env)
        b12x = (
            "b12x" in document["metadata"].get("tags", []) or "B12X_DISK_BACKEND" in env
        )
        slugs = set(re.findall(r'"slug": "([^"]+)"', json.dumps(document)))
        has_table = any(s.startswith(TABLE_MODEL_PREFIXES) for s in slugs)
        if b12x and has_table and env.get(TABLE_ENV) not in SAFE:
            result.add(path.stem)
    return result


class B12xTableMemoryTest(unittest.TestCase):
    def test_table_memory_is_not_disk(self) -> None:
        unexpected = flagged() - ALLOWED.keys()
        self.assertEqual(
            set(),
            unexpected,
            f"set {TABLE_ENV}=shared (or ram) on b12x images: {sorted(unexpected)}",
        )

    def test_allowlist_has_no_stale_entries(self) -> None:
        self.assertEqual(set(), ALLOWED.keys() - flagged())


if __name__ == "__main__":
    unittest.main()
