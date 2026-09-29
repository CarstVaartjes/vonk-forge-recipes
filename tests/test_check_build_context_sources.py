from __future__ import annotations

import importlib.machinery
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "tools/check-build-context-sources"
LOADER = importlib.machinery.SourceFileLoader(
    "check_build_context_sources", str(SCRIPT)
)
SPEC = importlib.util.spec_from_loader(LOADER.name, LOADER)
assert SPEC is not None
checker = importlib.util.module_from_spec(SPEC)
sys.modules[LOADER.name] = checker
LOADER.exec_module(checker)


class BuildContextSourcesTest(unittest.TestCase):
    def _recipe(self, root: Path, dockerfile: str, files: list[str]) -> Path:
        context = root / "adapters/x"
        context.mkdir(parents=True)
        (context / "Dockerfile").write_text(dockerfile)
        for name in files:
            target = context / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text("")
        (root / "recipes").mkdir()
        recipe = root / "recipes/r.json"
        recipe.write_text(
            json.dumps(
                {
                    "execution": {
                        "build": {
                            "context": {"path": "adapters/x"},
                            "dockerfile": "adapters/x/Dockerfile",
                        }
                    }
                }
            )
        )
        return recipe

    def test_reports_missing_source_and_accepts_present_ones(self) -> None:
        dockerfile = (
            "FROM scratch\n"
            "COPY a.py \\\n  b/c.py /dst/\n"
            "COPY tests/missing.py /dst/missing.py\n"
            "COPY --from=build /x /y\n"
            "COPY tests/*.txt /dst/\n"
            "ADD https://example.com/f.tgz /f.tgz\n"
        )
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            recipe = self._recipe(root, dockerfile, ["a.py", "b/c.py", "tests/one.txt"])
            problems = checker.check_recipe(recipe, root)
        self.assertEqual(len(problems), 1, problems)
        self.assertIn("tests/missing.py", problems[0])
        self.assertIn("Dockerfile:4", problems[0])

    def test_glob_without_match_is_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            recipe = self._recipe(root, "COPY none/*.py /dst/\n", [])
            self.assertEqual(len(checker.check_recipe(recipe, root)), 1)

    def test_catalog_has_no_missing_sources(self) -> None:
        problems: list[str] = []
        for recipe in sorted((ROOT / "recipes").glob("*.json")):
            problems.extend(checker.check_recipe(recipe))
        self.assertEqual(problems, [])


if __name__ == "__main__":
    unittest.main()
