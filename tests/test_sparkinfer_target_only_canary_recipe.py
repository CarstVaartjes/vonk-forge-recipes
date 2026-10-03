from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import unittest
from pathlib import Path
from typing import Any, cast

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "contracts" / "src"))
from generated_catalog import GENERATED
from vonk_forge_contracts import (
    document_sha256,
)

from qualification.definitions_loader import load_definitions

ADAPTER_ROOT = ROOT / "adapters/deepseek/sparkinfer-target-only-single"
ORIGINAL_RECIPE_PATH = ROOT / "recipes/deepseek-v4-flash-0731-sparkinfer-single.json"
RECIPE_PATH = (
    ROOT / "recipes/deepseek-v4-flash-0731-sparkinfer-target-only-canary-single.json"
)
EXECUTABLE_PAYLOAD_REVISION = "22f28d32b9b29b4352eaa380ff8c2c170b2847ab"
RUNTIME_REVISION = "d370c0f0806a9ac994dc5f354715576cc23840b9"
IMAGE_DIGEST = "2e077489a83a0360952828051fe7f7a32c1801e5ce8436d85f7267583d614ff4"
XGRAMMAR_VERSION = "0.2.3"
XGRAMMAR_WHEEL_SHA256 = (
    "11255f184971489fc72b948b096e2917f482ba2dca975177f5411562cedb9c6d"
)
LOWER_SPARK_BASELINE_BYTES = 126_946_283_520


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


class SparkInferTargetOnlyCanaryRecipeTests(unittest.TestCase):
    def test_exact_target_only_contract_and_authority_closure(self) -> None:
        recipe = _document(RECIPE_PATH)
        model = _document(_model_path())
        self.assertEqual(
            recipe["models"][0]["model"]["content_sha256"],
            _canonical_digest(_model_path()),
        )
        # the wrapper keeps its state under the payload revision: the Model is that revision
        self.assertEqual(model["source"]["revision"], EXECUTABLE_PAYLOAD_REVISION)
        self.assertEqual(len(model["files"]), 190)

        arguments = {
            item["name"]: item["value"] for item in recipe["runtime"]["arguments"]
        }
        self.assertEqual(
            _document(RECIPE_PATH)["settings"]["context_tokens"]["value"], 262_144
        )
        self.assertEqual(_document(RECIPE_PATH)["settings"]["concurrency"]["value"], 4)
        self.assertEqual(arguments["max-num-batched-tokens"], 8_192)
        self.assertEqual(arguments["max-cudagraph-capture-size"], 4)
        self.assertEqual(arguments["gpu-memory-utilization"], "0.95")
        self.assertEqual(arguments["kv-cache-dtype"], "nvfp4_ds_mla")

    def test_declared_envelope_fits_the_lower_live_baseline(self) -> None:
        memory = _document(RECIPE_PATH)["topology"]["roles"][0]["resources"]["memory"]
        admission = memory["peak_bytes"] + memory["reserve_bytes"]

        self.assertEqual(memory["peak_bytes"], 119_000_000_000)
        self.assertEqual(admission, 126_000_000_000)
        self.assertEqual(LOWER_SPARK_BASELINE_BYTES - admission, 946_283_520)

    def test_adapter_selects_target_only_mode_and_never_builds_a_draft(self) -> None:
        dockerfile = (ADAPTER_ROOT / "Dockerfile").read_text(encoding="utf-8")
        notice = (ADAPTER_ROOT / "NOTICE").read_text(encoding="utf-8")
        wrapper = (ADAPTER_ROOT / "vllm-wrapper.sh").read_text(encoding="utf-8")

        self.assertIn(f"@sha256:{IMAGE_DIGEST}", dockerfile)
        self.assertIn(
            f'org.opencontainers.image.revision="{RUNTIME_REVISION}"', dockerfile
        )
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

        self.assertIn("COPY vendor/ /tmp/xgrammar-wheel-parts/", dockerfile)
        self.assertIn(XGRAMMAR_WHEEL_SHA256, dockerfile)
        # pip refuses a wheel whose file name is not a valid wheel name.
        self.assertRegex(
            dockerfile,
            r"--no-deps --no-index /tmp/xgrammar-0\.2\.3-cp312-cp312-[\w.]+\.whl",
        )
        self.assertIn("from xgrammar import normalize_tool_choice", dockerfile)
        self.assertIn(f'version("xgrammar") == "{XGRAMMAR_VERSION}"', dockerfile)
        self.assertIn(XGRAMMAR_WHEEL_SHA256, notice)
        wheel_parts = sorted((ADAPTER_ROOT / "vendor").glob("xgrammar-0.2.3.part-*"))
        self.assertEqual(
            [path.name[-2:] for path in wheel_parts],
            ["aa", "ab", "ac", "ad", "ae"],
        )
        self.assertTrue(all(path.stat().st_size <= 10_000_000 for path in wheel_parts))
        wheel_digest = hashlib.sha256()
        for path in wheel_parts:
            wheel_digest.update(path.read_bytes())
        self.assertEqual(wheel_digest.hexdigest(), XGRAMMAR_WHEEL_SHA256)

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

    def test_target_only_smoke_exercises_the_tool_parser(self) -> None:
        definitions = cast(dict[str, Any], load_definitions(ROOT / "qualification"))
        services = definitions["service_recipes"]
        contract = services[
            "vonk-forge/deepseek-v4-flash-0731-sparkinfer-target-only-canary-single"
        ]
        self.assertEqual(contract["smoke_cases"], ["M0", "A391", "T_REPORT"])
        tool_case = definitions["service_case_templates"]["T_REPORT"]
        self.assertEqual(len(tool_case["body"]["tools"]), 1)
        self.assertEqual(
            tool_case["body"]["tool_choice"]["function"]["name"],
            "report_temperature",
        )
        self.assertEqual(
            tool_case["assertions"][-1],
            {
                "kind": "path.equals",
                "path": "choices.0.finish_reason",
                "value": "tool_calls",
            },
        )

    def test_catalog_package_digests_match_the_recipe(self) -> None:
        recipe = _document(RECIPE_PATH)
        context = recipe["execution"]["build"]["context"]
        self.assertEqual(
            context["path"], "adapters/deepseek/sparkinfer-target-only-single"
        )
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
