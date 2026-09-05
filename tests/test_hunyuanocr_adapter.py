from __future__ import annotations

import json
import tempfile
import types
import unittest
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ADAPTER_ROOT = ROOT / "adapters/ocr/hunyuanocr-1-5-vllm-dflash"
ADAPTER_PATH = ADAPTER_ROOT / "run.py"
MODEL_PATH = ROOT / "models/hunyuanocr-1-5-47644ecc.json"
RECIPE_PATH = ROOT / "recipes/hunyuanocr-1-5-vllm-dflash-single.json"
RELEASE_PATH = ROOT / "recipe-releases/hunyuanocr-1-5-vllm-dflash-single.json"
SOURCE_REVISION = "c55965d3da1e6f41987abec8068f2e70851318bc"


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def adapter_module():
    module = types.ModuleType("hunyuanocr_adapter")
    module.__file__ = str(ADAPTER_PATH)
    exec(compile(ADAPTER_PATH.read_text(encoding="utf-8"), str(ADAPTER_PATH), "exec"), module.__dict__)
    return module


class HunyuanOCRAuthorityTests(unittest.TestCase):
    def test_recipe_resolves_exact_model_and_source_bundle(self) -> None:
        model, recipe, release = map(load, (MODEL_PATH, RECIPE_PATH, RELEASE_PATH))
        reference = recipe["models"][0]["model"]
        from vonk_forge_contracts import ModelDefinition
        model_digest = __import__("hashlib").sha256(json.dumps(ModelDefinition.model_validate(model).model_dump(mode="json"), sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        self.assertEqual(reference["content_sha256"], model_digest)
        self.assertEqual(model["source"]["revision"], "47644ecc4fc854efa4f505155158831f36773ee4")
        self.assertEqual(recipe["topology"]["node_count"], 1)
        self.assertEqual(recipe["interfaces"][0]["adapter"], "artifact-job")
        self.assertEqual(release["history"][0]["recipe_content_sha256"], __import__("hashlib").sha256(json.dumps(recipe, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest())

    def test_model_inventory_license_and_runtime_source_are_closed(self) -> None:
        model, recipe = load(MODEL_PATH), load(RECIPE_PATH)
        paths = {item["path"] for item in model["files"]}
        for required in ("model.safetensors", "dflash/model.safetensors", "dflash/dflash.py"):
            self.assertIn(required, paths)
        self.assertTrue(model["license"]["operator_acceptance_required"])
        self.assertEqual(recipe["runtime"]["engine"], "pytorch-pipeline")
        selected = {item["file_id"] for item in recipe["models"][0]["files"]}
        selected_paths = {item["path"] for item in model["files"] if item["id"] in selected}
        self.assertTrue(selected_paths)
        self.assertFalse(any(path.startswith("v1.0/") for path in selected_paths))
        self.assertEqual(len({item["id"] for item in model["files"]}), len(model["files"]))
        self.assertTrue(all(len(item["sha256"]) == 64 for item in model["files"]))

    def test_signed_source_bundle_matches_recipe(self) -> None:
        tool = __import__("runpy").run_path(str(ROOT / "tools/build-catalog-index"))
        _, _, digest = tool["source_bundle"](ADAPTER_ROOT)
        context = load(RECIPE_PATH)["execution"]["build"]["context"]
        self.assertEqual(context["path"], "adapters/ocr/hunyuanocr-1-5-vllm-dflash")
        self.assertTrue(digest)
        dockerfile = (ADAPTER_ROOT / "Dockerfile").read_text()
        self.assertIn(SOURCE_REVISION, dockerfile)
        self.assertIn("sha256sum --check --strict", dockerfile)

    def test_wrapper_keeps_official_dflash_serving_contract(self) -> None:
        source = ADAPTER_PATH.read_text(encoding="utf-8")
        for fixed_flag in ('"--no-enable-prefix-caching"', '"--limit-mm-per-prompt"', '"--speculative-config"'):
            self.assertIn(fixed_flag, source)
        self.assertNotIn("huggingface.co/", source)


class HunyuanOCRInputTests(unittest.TestCase):
    def setUp(self) -> None:
        self.module = adapter_module()
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.module.INPUTS = Path(self.temporary.name)

    def manifest(self, files: list[dict[str, object]]) -> None:
        (self.module.INPUTS / "manifest.json").write_text(json.dumps({"schema_version": 1, "total_bytes": 0, "files": files}))

    def test_manifest_selects_bounded_official_tasks(self) -> None:
        image = self.module.INPUTS / "page.png"
        image.write_bytes(b"fixture")
        (self.module.INPUTS / "job.json").write_text(json.dumps({"images": ["page.png"], "task_type": "table", "max_tokens": 2048}))
        self.manifest([{"slot": "document", "name": "page.png"}, {"slot": "config", "name": "job.json"}])
        images, task_type, max_tokens = self.module.read_job()
        self.assertEqual(images, [image]); self.assertEqual(task_type, "table"); self.assertEqual(max_tokens, 2048)

        (self.module.INPUTS / "job.json").write_text(json.dumps({"images": ["page.png"], "task_type": "free_prompt"}))
        with self.assertRaisesRegex(SystemExit, "official task_type"):
            self.module.read_job()

    def test_manifest_rejects_traversal_and_unknown_fields(self) -> None:
        self.manifest([{"slot": "document", "name": "../page.png"}])
        with self.assertRaisesRegex(SystemExit, "unsafe name"):
            self.module.read_job()
        image = self.module.INPUTS / "page.png"
        image.write_bytes(b"fixture")
        (self.module.INPUTS / "job.json").write_text(json.dumps({"images": ["page.png"], "prompt": "override"}))
        self.manifest([{"slot": "document", "name": "page.png"}, {"slot": "config", "name": "job.json"}])
        with self.assertRaisesRegex(SystemExit, "unsupported fields"):
            self.module.read_job()

    def test_output_bundle_contains_results_and_provenance(self) -> None:
        source = self.module.INPUTS / "page.png"; source.write_bytes(b"fixture")
        with tempfile.TemporaryDirectory() as output:
            target = self.module.write_bundle(Path(output), [(source, "# Parsed\n", False)], "doc_parse")
            with zipfile.ZipFile(target) as archive:
                manifest = json.loads(archive.read("manifest.json"))
                self.assertEqual(manifest["inference"], "vllm-dflash")
                self.assertEqual(manifest["model_revision"], "47644ecc4fc854efa4f505155158831f36773ee4")
                self.assertEqual(archive.read("documents/001-page.md"), b"# Parsed\n")


if __name__ == "__main__":
    unittest.main()
