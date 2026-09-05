from __future__ import annotations

import hashlib
import json
import runpy
import subprocess
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "contracts" / "src"))
from vonk_forge_contracts import ModelDefinition, RecipeDefinition, content_sha256  # noqa: E402
ADAPTER_ROOT = ROOT / "adapters/deepseek/sparkinfer-target-only-single"
MODEL_PATH = ROOT / "models/deepseek-v4-flash-0731-sparkinfer-exl3-k216.json"
ORIGINAL_RECIPE_PATH = ROOT / "recipes/deepseek-v4-flash-0731-sparkinfer-single.json"
RECIPE_PATH = ROOT / "recipes/deepseek-v4-flash-0731-sparkinfer-target-only-canary-single.json"
RELEASE_PATH = ROOT / "recipe-releases/deepseek-v4-flash-0731-sparkinfer-target-only-canary-single.json"
MODEL_REVISION = "ce5ff0f1efb2e184aafc759d281bfae47d3a359c"
EXECUTABLE_PAYLOAD_REVISION = "22f28d32b9b29b4352eaa380ff8c2c170b2847ab"
RUNTIME_REVISION = "590d2172394dd83c1f36ff29f0dc9ec6032ea9e2"
IMAGE_DIGEST = "2e077489a83a0360952828051fe7f7a32c1801e5ce8436d85f7267583d614ff4"
LOWER_SPARK_BASELINE_BYTES = 126_946_283_520


def _document(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _canonical_digest(path: Path) -> str:
    contract = ModelDefinition if path.parent.name == "models" else RecipeDefinition
    return content_sha256(contract.model_validate(_document(path)))


class SparkInferTargetOnlyCanaryRecipeTests(unittest.TestCase):
    def test_original_speculative_recipe_is_unchanged(self) -> None:
        release = _document(ROOT / "recipe-releases/deepseek-v4-flash-0731-sparkinfer-single.json")
        historical = {entry["version"]: entry["recipe_content_sha256"] for entry in release["history"]}
        self.assertEqual(
            historical["0.1.2"],
            "0c4c4e3235f8d694956d0d4b9d7d9b05d542ab5e23782b4bbe8fe552aa39a879",
        )

    def test_exact_target_only_contract_and_authority_closure(self) -> None:
        recipe = _document(RECIPE_PATH)
        model = _document(MODEL_PATH)
        self.assertEqual(recipe["models"][0]["model"]["content_sha256"], _canonical_digest(MODEL_PATH))
        self.assertEqual(model["source"]["revision"], MODEL_REVISION)
        self.assertEqual(len(model["files"]), 190)

        arguments = {
            item["name"]: item["value"] for item in recipe["runtime"]["arguments"]
        }
        self.assertEqual(_document(RECIPE_PATH)["settings"]["context_tokens"]["value"], 262_144)
        self.assertEqual(_document(RECIPE_PATH)["settings"]["concurrency"]["value"], 4)
        self.assertEqual(arguments["max-num-batched-tokens"], 8_192)
        self.assertEqual(arguments["max-cudagraph-capture-size"], 4)
        self.assertEqual(arguments["gpu-memory-utilization"], "0.95")
        self.assertEqual(arguments["kv-cache-dtype"], "nvfp4_ds_mla")

    def test_declared_envelope_fits_the_lower_live_baseline(self) -> None:
        memory = _document(RECIPE_PATH)["topology"]["roles"][0]["resources"]["memory"]
        runtime_maximum = max(
            memory["startup_peak_bytes"],
            memory["steady_state_bytes"] + memory["runtime_growth_bytes"],
        )
        admission = runtime_maximum + memory["system_reserve_bytes"]

        self.assertEqual(memory["startup_peak_bytes"], 119_000_000_000)
        self.assertEqual(runtime_maximum, 119_000_000_000)
        self.assertEqual(admission, 126_000_000_000)
        self.assertEqual(LOWER_SPARK_BASELINE_BYTES - admission, 946_283_520)

    def test_adapter_selects_target_only_mode_and_never_builds_a_draft(self) -> None:
        dockerfile = (ADAPTER_ROOT / "Dockerfile").read_text(encoding="utf-8")
        notice = (ADAPTER_ROOT / "NOTICE").read_text(encoding="utf-8")
        wrapper = (ADAPTER_ROOT / "vllm-wrapper.sh").read_text(encoding="utf-8")

        self.assertIn(f"@sha256:{IMAGE_DIGEST}", dockerfile)
        self.assertIn(f'org.opencontainers.image.revision="{RUNTIME_REVISION}"', dockerfile)
        self.assertIn("ENTRYPOINT []", dockerfile)
        self.assertIn(
            f"readonly executable_payload_revision={EXECUTABLE_PAYLOAD_REVISION}",
            wrapper,
        )
        self.assertIn("export MODE=mtp0", wrapper)
        self.assertIn("export CUDAGRAPH_CAPTURE_SIZES=1,2,4", wrapper)
        self.assertIn("exec /opt/vllm/serve-ds4-flash.sh", wrapper)
        self.assertNotIn("build_dspark_draft.py", wrapper)
        self.assertNotIn("export SPEC_MODEL_PATH", wrapper)
        self.assertIn("92.13 GiB", notice)
        self.assertIn("95.39 GiB", notice)

        for forbidden in (
            "snapshot_download",
            "huggingface_hub",
            "curl ",
            "wget ",
            "git clone",
        ):
            self.assertNotIn(forbidden, wrapper)

        subprocess.run(
            ["bash", "-n", str(ADAPTER_ROOT / "vllm-wrapper.sh")],
            check=True,
        )

    def test_release_pins_the_upstream_measurements(self) -> None:
        release = _document(RELEASE_PATH)
        evidence = next(
            change
            for entry in release["history"]
            for change in entry.get("changes", [])
            if "references" in change
        )
        self.assertIn("92.13 GiB", evidence["details"])
        self.assertIn("95.39 GiB", evidence["details"])
        self.assertEqual(
            evidence["references"],
            [
                f"https://github.com/0xSero/deepseek-v4-flash-0731-spark-sparkinfer/blob/{RUNTIME_REVISION}/results/acceptance.json",
                f"https://github.com/0xSero/deepseek-v4-flash-0731-spark-sparkinfer/blob/{RUNTIME_REVISION}/results/clean-image-acceptance.json",
                f"https://github.com/0xSero/deepseek-v4-flash-0731-spark-sparkinfer/blob/{RUNTIME_REVISION}/scripts/entrypoint.sh",
            ],
        )

    def test_source_bundle_and_release_digests_match(self) -> None:
        recipe = _document(RECIPE_PATH)
        release = _document(RELEASE_PATH)
        index_tool = runpy.run_path(str(ROOT / "tools/build-catalog-index"))
        archive, _files, digest = index_tool["source_bundle"](ADAPTER_ROOT)
        context = recipe["execution"]["build"]["context"]
        self.assertEqual(context["path"], "adapters/deepseek/sparkinfer-target-only-single")
        self.assertRegex(digest, r"^[a-f0-9]{64}$")
        self.assertEqual(
            release["history"][0]["recipe_content_sha256"],
            _canonical_digest(RECIPE_PATH),
        )


if __name__ == "__main__":
    unittest.main()
