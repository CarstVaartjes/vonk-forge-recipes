from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import tarfile
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RECIPE = ROOT / "recipes/deepseek-v4-flash-0731-ds4-single.json"


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def digest(path: Path) -> str:
    body = json.dumps(load(path), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(body.encode()).hexdigest()


def compiled_engine_argv(recipe: dict, *, target: str, drafter: str | None = None) -> list[str]:
    """Render the recipe's opaque engine arguments exactly as authored."""
    settings = recipe["settings"]
    argv: list[str] = []
    for argument in recipe["runtime"]["arguments"]:
        value = argument.get("value")
        if value is None:
            value = settings[argument["setting"]]["value"]
        if argument["name"] == "model":
            value = target
        elif argument["name"] == "mtp-model":
            value = drafter
        argv.extend((f"--{argument['name']}", str(value)))
    return [*argv, "--cpu", "--host", "127.0.0.1", "--port", "18080"]


class DeepseekDs4RecipeTests(unittest.TestCase):
    def test_cuda_profile_describes_ordered_two_session_fallback(self) -> None:
        recipe = load(RECIPE)
        self.assertIn("two-session concurrency", recipe["metadata"]["title"])
        self.assertEqual(next(a["setting"] for a in recipe["runtime"]["arguments"] if a["name"] == "batched-session"), "concurrency")
        self.assertEqual(recipe["settings"]["kind"], "generation")
        self.assertEqual(recipe["topology"]["node_count"], 1)
        names = [argument["name"] for argument in recipe["runtime"]["arguments"]]
        self.assertEqual(names, ["model", "ctx", "batched-session"])

    def test_dspark_uses_pinned_parser_option_names(self) -> None:
        recipe = load(ROOT / "recipes/deepseek-v4-flash-0731-ds4-dspark-latency-single.json")
        names = [argument["name"] for argument in recipe["runtime"]["arguments"]]
        self.assertEqual(names, ["model", "mtp-model", "ctx"])
        self.assertEqual(recipe["release"]["version"], "1.2.5")

    def test_pinned_cpu_parser_accepts_compiled_corrected_argv(self) -> None:
        if shutil.which("make") is None or shutil.which("cc") is None:
            self.skipTest("native C toolchain is unavailable")
        archive_path = (
            ROOT
            / "adapters/deepseek/ds4/vendor/"
            "ds4-f4d03f6cf9f11c1e7b630bcb160853acfba7c52a.tar.gz"
        )
        with tempfile.TemporaryDirectory(prefix="ds4-parser-") as temporary:
            extraction_root = Path(temporary)
            with tarfile.open(archive_path, mode="r:gz") as archive:
                archive.extractall(extraction_root, filter="data")
            source_root = extraction_root / "ds4-f4d03f6"
            subprocess.run(["make", "cpu", "-j2"], cwd=source_root, check=True)
            binary = source_root / "ds4-server"
            cases = (
                (
                    ROOT / "recipes/deepseek-v4-flash-0731-ds4-dspark-latency-single.json",
                    "/missing/target.gguf",
                    "/missing/drafter.gguf",
                ),
                (
                    ROOT / "recipes/deepseek-v4-flash-0731-ds4-single.json",
                    "/missing/target.gguf",
                    None,
                ),
            )
            for recipe_path, target, drafter in cases:
                with self.subTest(recipe=recipe_path.name):
                    result = subprocess.run(
                        [str(binary), *compiled_engine_argv(load(recipe_path), target=target, drafter=drafter)],
                        check=False,
                        capture_output=True,
                        text=True,
                    )
                    output = f"{result.stdout}\n{result.stderr}".lower()
                    self.assertNotEqual(result.returncode, 0)
                    self.assertNotIn("unknown option", output)
                    self.assertIn("cannot open model", output)

    def test_release_binds_the_current_recipe_digest(self) -> None:
        index = load(ROOT / "catalog-index.json")
        entry = next(item for item in index["recipes"] if item["source_path"] == f"recipes/{RECIPE.name}")
        self.assertEqual(entry["package"]["recipe_content_sha256"], digest(RECIPE))


if __name__ == "__main__":
    unittest.main()
