from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any

from documents import recipe_display
from repo_root import ROOT


def load(path: str) -> dict[str, Any]:
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


class MiaSglangSingleRecipeTests(unittest.TestCase):
    def _capture(self, wrapper: Path, arguments: list[str]) -> list[str]:
        with tempfile.TemporaryDirectory() as directory:
            launcher = Path(directory) / "sglang" / "launch_server.py"
            launcher.parent.mkdir()
            launcher.write_text(
                "import json, os, sys\nopen(os.environ['CAPTURE'], 'w').write(json.dumps(sys.argv[1:]))"
            )
            capture = Path(directory) / "arguments.json"
            result = subprocess.run(
                [sys.executable, str(wrapper), *arguments],
                check=False,
                env={**os.environ, "PYTHONPATH": directory, "CAPTURE": str(capture)},
            )
            assert result.returncode == 0
            return json.loads(capture.read_text())

    def test_ling_profile_binds_two_model_slots_and_wrapper_mounts(self) -> None:
        recipe = load("recipes/ling-3-0-flash-dspark-sglang-single.json")
        assert len(recipe["models"]) == 2
        mounts = {
            item["mount"]["target"]
            for selection in recipe["models"]
            for item in selection["files"]
        }
        assert mounts == {"/models/target", "/models/draft"}
        captured = self._capture(
            ROOT / "adapters/ling/flash-dspark-single/sglang-serve",
            [
                "--model-path",
                "/models",
                "--served-model-name",
                "ling-3-0-flash",
                "--tensor-parallel-size",
                "1",
                "--context-length",
                "262144",
                "--mem-fraction-static",
                "0.80",
                "--host",
                "0.0.0.0",
                "--port",
                "30000",
            ],
        )
        assert captured[captured.index("--model-path") + 1] == "/models/target"
        assert (
            captured[captured.index("--speculative-draft-model-path") + 1]
            == "/models/draft"
        )
        assert "DSPARK" in captured

    def test_ling_profile_has_the_declared_admission_envelope(self) -> None:
        recipe = load("recipes/ling-3-0-flash-dspark-sglang-single.json")
        assert recipe["settings"]["context_tokens"]["value"] == 262144
        memory = recipe["topology"]["roles"][0]["resources"]["memory"]
        assert memory["peak_bytes"] == 119000000000
        assert (
            "262144 context" in recipe_display(recipe["identity"]["slug"]).description
        )

    def test_qwen27_profile_binds_external_dspark_draft(self) -> None:
        recipe = load("recipes/qwen3-8-27b-nvfp4-dspark-sglang-single.json")
        assert recipe["topology"]["node_count"] == 1
        assert len(recipe["models"]) == 2
        captured = self._capture(
            ROOT / "adapters/qwen/qwen38-27b-dspark-single/sglang-serve",
            [
                "--model-path",
                "/models",
                "--served-model-name",
                "qwen3-8-27b",
                "--tensor-parallel-size",
                "1",
                "--context-length",
                "262144",
                "--mem-fraction-static",
                "0.80",
                "--host",
                "0.0.0.0",
                "--port",
                "30000",
            ],
        )
        assert "DSPARK" in captured
        assert "--enable-torch-compile" in captured
        assert "--continuous-decode-steps" not in captured

    def test_sglang_recipes_bind_immutable_runtime_arguments(self) -> None:
        for path in (
            "recipes/ling-3-0-flash-dspark-sglang-single.json",
            "recipes/qwen3-8-27b-nvfp4-dspark-sglang-single.json",
        ):
            recipe = load(path)
            assert recipe["runtime"]["engine"] == "sglang"
            assert recipe["runtime"]["entrypoint"]
            assert recipe["execution"]["build"]["context"]["path"]


if __name__ == "__main__":
    unittest.main()
