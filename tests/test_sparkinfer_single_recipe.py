from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import unittest
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "contracts" / "src"))
from generated_catalog import GENERATED
from vonk_forge_contracts import (
    document_sha256,
)

ADAPTER_ROOT = ROOT / "adapters/deepseek/sparkinfer-single"
RECIPE_PATH = ROOT / "recipes/deepseek-v4-flash-0731-sparkinfer-single.json"
EXECUTABLE_PAYLOAD_REVISION = "22f28d32b9b29b4352eaa380ff8c2c170b2847ab"
RUNTIME_REVISION = "d370c0f0806a9ac994dc5f354715576cc23840b9"
IMAGE_DIGEST = "2e077489a83a0360952828051fe7f7a32c1801e5ce8436d85f7267583d614ff4"


def _document(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _model_path() -> Path:
    """The Model the recipe selects: the kit declares its revision, so no slug is pinned here."""
    recipe: Any = _document(RECIPE_PATH)
    return ROOT / "models" / f"{recipe['models'][0]['model']['slug']}.json"


def _canonical_digest(path: Path) -> str:
    return document_sha256(_document(path))


def _catalog_entry(slug: str) -> dict[str, object]:
    catalog = _document(GENERATED / "catalog-index.json")
    return next(
        item
        for item in catalog["recipes"]
        if item["document"]["identity"]["slug"] == slug
    )


class SparkInferSingleRecipeTests(unittest.TestCase):
    def test_complete_immutable_authority_closure(self) -> None:
        recipe = _document(RECIPE_PATH)
        model = _document(_model_path())
        self.assertEqual(
            recipe["models"][0]["model"]["content_sha256"],
            _canonical_digest(_model_path()),
        )
        # the wrapper keeps its state under the payload revision: the Model is that revision
        self.assertEqual(model["source"]["revision"], EXECUTABLE_PAYLOAD_REVISION)
        self.assertEqual(len(model["files"]), 190)
        self.assertEqual(
            len({item["path"] for item in model["files"]}), len(model["files"])
        )
        self.assertTrue(all(len(item["sha256"]) == 64 for item in model["files"]))

    def test_adapter_is_offline_and_uses_the_published_launch_path(self) -> None:
        dockerfile = (ADAPTER_ROOT / "Dockerfile").read_text(encoding="utf-8")
        wrapper = (ADAPTER_ROOT / "vllm-wrapper.sh").read_text(encoding="utf-8")
        recipe_text = RECIPE_PATH.read_text(encoding="utf-8")

        self.assertIn(f"@sha256:{IMAGE_DIGEST}", dockerfile)
        self.assertIn(
            f'org.opencontainers.image.revision="{RUNTIME_REVISION}"', dockerfile
        )
        self.assertIn("ENTRYPOINT []", dockerfile)
        self.assertIn("/opt/recipe/scripts/coalesce_rank_sliced_exl3.py", wrapper)
        self.assertIn("/opt/recipe/scripts/verify_tp1_manifest.py", wrapper)
        self.assertIn("/opt/recipe/scripts/build_dspark_draft.py", wrapper)
        self.assertIn("/opt/recipe/scripts/selftest.py", wrapper)
        self.assertIn("exec /opt/vllm/serve-ds4-flash.sh", wrapper)
        self.assertIn(
            f"readonly executable_payload_revision={EXECUTABLE_PAYLOAD_REVISION}",
            wrapper,
        )
        self.assertIn(
            "readonly state_root=/outputs/${executable_payload_revision}", wrapper
        )

        for forbidden in (
            "snapshot_download",
            "huggingface_hub",
            "curl ",
            "wget ",
            "git clone",
        ):
            self.assertNotIn(forbidden, wrapper)
        for forbidden in (
            "metadata-only",
            "non-executable",
            "integration-required",
            "/bin/false",
            "exit 78",
        ):
            self.assertNotIn(forbidden, recipe_text)
        for forbidden in (
            "non-executable",
            "integration-required",
            "/bin/false",
            "exit 78",
        ):
            self.assertNotIn(forbidden, recipe_text)

        subprocess.run(
            ["bash", "-n", str(ADAPTER_ROOT / "vllm-wrapper.sh")],
            check=True,
        )

    def test_catalog_package_digests_match_the_recipe(self) -> None:
        recipe = _document(RECIPE_PATH)
        context = recipe["execution"]["build"]["context"]
        self.assertEqual(context["path"], "adapters/deepseek/sparkinfer-single")
        recipe_digest = _canonical_digest(RECIPE_PATH)
        entry = _catalog_entry(recipe["identity"]["slug"])
        self.assertEqual(
            entry["content_sha256"],
            recipe_digest,
        )
        package = entry["package"]
        self.assertEqual(package["recipe_content_sha256"], recipe_digest)
        payload = (GENERATED / package["path"]).read_bytes()
        self.assertEqual(len(payload), package["expected_bytes"])
        self.assertEqual(hashlib.sha256(payload).hexdigest(), package["sha256"])


if __name__ == "__main__":
    unittest.main()
