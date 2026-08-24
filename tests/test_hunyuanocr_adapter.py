from __future__ import annotations

import hashlib
import json
import runpy
import tempfile
import types
import unittest
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ADAPTER_ROOT = ROOT / "adapters/ocr/hunyuanocr-1-5-vllm-dflash"
ADAPTER_PATH = ADAPTER_ROOT / "run.py"
MODEL_PATH = ROOT / "model-versions/hunyuanocr-1-5-449e7d47.json"
RUNTIME_PATH = ROOT / "runtime-distributions/hunyuanocr-1-5-vllm-dflash-arm64.json"
RECIPE_PATH = ROOT / "recipes/hunyuanocr-1-5-vllm-dflash-single.json"
RELEASE_PATH = ROOT / "recipe-releases/hunyuanocr-1-5-vllm-dflash-single.json"
MODEL_REVISION = "449e7d471a8a1ef5bd5d652e4881183d7252cbc7"
SOURCE_REVISION = "c55965d3da1e6f41987abec8068f2e70851318bc"


def canonical_digest(path: Path) -> str:
    document = json.loads(path.read_text(encoding="utf-8"))
    payload = json.dumps(
        document,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def adapter_module():
    module = types.ModuleType("hunyuanocr_adapter")
    module.__file__ = str(ADAPTER_PATH)
    exec(
        compile(ADAPTER_PATH.read_text(encoding="utf-8"), str(ADAPTER_PATH), "exec"),
        module.__dict__,
    )
    return module


class HunyuanOCRAuthorityTests(unittest.TestCase):
    def test_recipe_resolves_exact_authorities(self) -> None:
        recipe = json.loads(RECIPE_PATH.read_text(encoding="utf-8"))
        model = json.loads(MODEL_PATH.read_text(encoding="utf-8"))
        runtime = json.loads(RUNTIME_PATH.read_text(encoding="utf-8"))
        release = json.loads(RELEASE_PATH.read_text(encoding="utf-8"))

        self.assertEqual(recipe["model"]["content_sha256"], canonical_digest(MODEL_PATH))
        self.assertEqual(
            recipe["runtime"]["distribution"]["content_sha256"],
            canonical_digest(RUNTIME_PATH),
        )
        self.assertEqual(release["history"][0]["recipe_content_sha256"], canonical_digest(RECIPE_PATH))
        self.assertEqual(model["source"]["revision"], MODEL_REVISION)
        self.assertEqual(runtime["source"]["revision"], SOURCE_REVISION)
        self.assertEqual(recipe["artifacts"][0]["revision"], MODEL_REVISION)
        self.assertEqual(recipe["topology"]["node_count"], 1)
        self.assertEqual(recipe["interfaces"][0]["adapter"], "artifact-job")
        self.assertTrue(recipe["interfaces"][0]["input"]["required"])
        self.assertIn(
            {"source": "inputs", "target": "/inputs", "read_only": True},
            recipe["runtime"]["security"]["mounts"],
        )

    def test_model_inventory_is_complete_and_size_bound(self) -> None:
        model = json.loads(MODEL_PATH.read_text(encoding="utf-8"))
        artifacts = model["artifacts"]
        self.assertEqual(sum(item["download_bytes"] for item in artifacts), model["sizes"]["download_bytes"])
        paths = {item["path"] for item in artifacts}
        for required in (
            "model.safetensors",
            "dflash/model.safetensors",
            "dflash/dflash.py",
            "v1.0/model-00004-of-00004.safetensors",
        ):
            self.assertIn(required, paths)

    def test_signed_source_bundle_matches_recipe(self) -> None:
        recipe = json.loads(RECIPE_PATH.read_text(encoding="utf-8"))
        source_bundle = runpy.run_path(str(ROOT / "tools/build-catalog-index"))["source_bundle"]
        archive, _, digest = source_bundle(ADAPTER_ROOT)
        context = recipe["build"]["context"]
        self.assertEqual(context["sha256"], digest)
        self.assertEqual(context["expected_bytes"], len(archive))

    def test_wrapper_keeps_official_dflash_serving_contract(self) -> None:
        source = ADAPTER_PATH.read_text(encoding="utf-8")
        dockerfile = (ADAPTER_ROOT / "Dockerfile").read_text(encoding="utf-8")
        for fixed_flag in (
            '"--no-enable-prefix-caching"',
            '"--mm-processor-cache-gb"',
            '"--allowed-local-media-path"',
            '"--limit-mm-per-prompt"',
            '"--speculative-config"',
            '"num_speculative_tokens": 15',
        ):
            self.assertIn(fixed_flag, source)
        self.assertIn(SOURCE_REVISION, dockerfile)
        self.assertIn("sha256sum --check --strict", dockerfile)
        self.assertIn("from utils.tasks import get_prompt", source)
        self.assertIn("normalize_doc_parse_markdown", source)
        self.assertNotIn("huggingface.co/", source)


class HunyuanOCRInputTests(unittest.TestCase):
    def setUp(self) -> None:
        self.module = adapter_module()
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.module.INPUTS = Path(self.temporary.name)

    def test_manifest_selects_only_bounded_official_tasks(self) -> None:
        image = self.module.INPUTS / "page.png"
        image.write_bytes(b"not-decoded-by-contract-test")
        (self.module.INPUTS / "job.json").write_text(
            json.dumps({"images": ["page.png"], "task_type": "table", "max_tokens": 2048}),
            encoding="utf-8",
        )
        images, task_type, max_tokens = self.module.read_job()
        self.assertEqual(images, [image])
        self.assertEqual(task_type, "table")
        self.assertEqual(max_tokens, 2048)

        (self.module.INPUTS / "job.json").write_text(
            json.dumps({"images": ["page.png"], "task_type": "free_prompt"}),
            encoding="utf-8",
        )
        with self.assertRaisesRegex(SystemExit, "official task_type"):
            self.module.read_job()

    def test_manifest_rejects_traversal_and_unknown_fields(self) -> None:
        (self.module.INPUTS / "job.json").write_text(
            json.dumps({"images": ["../page.png"]}), encoding="utf-8"
        )
        with self.assertRaisesRegex(SystemExit, "plain filenames"):
            self.module.read_job()
        (self.module.INPUTS / "job.json").write_text(
            json.dumps({"images": ["page.png"], "prompt": "override"}), encoding="utf-8"
        )
        with self.assertRaisesRegex(SystemExit, "unsupported fields"):
            self.module.read_job()

    def test_output_bundle_contains_results_and_provenance(self) -> None:
        source = self.module.INPUTS / "page.png"
        source.write_bytes(b"fixture")
        with tempfile.TemporaryDirectory() as output:
            target = self.module.write_bundle(
                Path(output), [(source, "# Parsed\n", False)], "doc_parse"
            )
            with zipfile.ZipFile(target) as archive:
                manifest = json.loads(archive.read("manifest.json"))
                self.assertEqual(manifest["model_revision"], MODEL_REVISION)
                self.assertEqual(manifest["inference"], "vllm-dflash")
                self.assertEqual(archive.read("documents/001-page.md"), b"# Parsed\n")


if __name__ == "__main__":
    unittest.main()
