"""Every adapter on the SparkInfer base image installs an xgrammar that has normalize_tool_choice.

The base pins xgrammar 0.1.27 (PIP_CONSTRAINT) while its vLLM needs >= 0.2.1; an adapter that
forgets the build-time install fails every tool-call request with HTTP 500.
"""

from __future__ import annotations

import json
import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MIA_RECIPE = ROOT / "recipes/deepseek-v4-flash-0731-mia-sparkinfer-single.json"
MIA_DOCKERFILE = ROOT / "adapters/deepseek/mia-sparkinfer-single/Dockerfile"
BASE_DIGEST = json.loads(MIA_RECIPE.read_text(encoding="utf-8"))["execution"]["build"][
    "base_image"
]["digest"]
INSTALLED = re.compile(r"xgrammar[-=]+0\.(?:[2-9]|\d{2,})\.\d+")
ASSERTED = "from xgrammar import normalize_tool_choice"


class SparkInferBaseXgrammarTests(unittest.TestCase):
    def test_every_adapter_on_the_base_installs_a_new_enough_xgrammar(self) -> None:
        dockerfiles = [
            path
            for path in sorted((ROOT / "adapters").rglob("Dockerfile"))
            if re.search(
                rf"^FROM\s.*@sha256:{BASE_DIGEST}",
                path.read_text(encoding="utf-8"),
                re.MULTILINE,
            )
        ]
        self.assertIn(MIA_DOCKERFILE, dockerfiles)
        for path in dockerfiles:
            text = path.read_text(encoding="utf-8")
            where = str(path.relative_to(ROOT))
            self.assertRegex(text, INSTALLED, f"{where}: no xgrammar >= 0.2 install")
            self.assertIn(ASSERTED, text, f"{where}: no normalize_tool_choice assert")

    def test_mia_adapter_install_is_hash_pinned_and_keeps_transformers(self) -> None:
        text = (ROOT / "adapters/deepseek/mia-sparkinfer-single/Dockerfile").read_text(
            encoding="utf-8"
        )
        self.assertIn("--require-hashes", text)
        self.assertIn(
            "--hash=sha256:b1d2bef13aec8122dfa6b96e3ece2cc0a42a732685f1d44f6248c305274e8eae",
            text,
        )
        self.assertIn('m.version("transformers") == "5.13.1"', text)


if __name__ == "__main__":
    unittest.main()
