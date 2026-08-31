from __future__ import annotations

import hashlib
import json
import runpy
import tempfile
import types
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CANARY_SLUG = "ltx-2-5-22b-distilled-fp8-cast-diffusers-single"
BF16_SLUG = "ltx-2-5-22b-distilled-bf16-diffusers-single"
CANARY_ADAPTER = ROOT / "adapters/video/ltx25-diffusers-fp8-canary"
CANARY_RECIPE = ROOT / f"recipes/{CANARY_SLUG}.json"
CANARY_RELEASE = ROOT / f"recipe-releases/{CANARY_SLUG}.json"
BF16_RECIPE = ROOT / f"recipes/{BF16_SLUG}.json"
MODEL_VERSION = ROOT / "model-versions/ltx-2-5-22b-distilled-bf16-diffusers.json"
RUNTIME = ROOT / "runtime-distributions/diffusers-0-40-0-cuda13-arm64.json"
SPARK_ALLOCATABLE_BYTES = 126_946_283_520


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def digest(path: Path) -> str:
    return hashlib.sha256(
        json.dumps(
            load(path),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def adapter_module():
    path = CANARY_ADAPTER / "run.py"
    module = types.ModuleType("ltx25_fp8_canary_adapter")
    module.__file__ = str(path)
    exec(  # noqa: S102 - isolate the adapter without importing heavy dependencies.
        compile(path.read_text(encoding="utf-8"), str(path), "exec"),
        module.__dict__,
    )
    return module


class Ltx25Fp8CanaryTests(unittest.TestCase):
    def test_bf16_recipe_retains_its_runtime_contract(self) -> None:
        recipe = load(BF16_RECIPE)
        self.assertEqual(digest(BF16_RECIPE), "889ca27825623bac876490e6bd83492437ae64dbd64327d69757c92e1bc1f470")
        self.assertNotIn("nvcr.io", recipe["build"]["network"]["hosts"])
        self.assertEqual(
            recipe["build"]["context"]["path"], "adapters/video/ltx25-diffusers"
        )
        memory = recipe["topology"]["roles"][0]["resources"]["memory"]
        required = max(
            memory["startup_peak_bytes"],
            memory["steady_state_bytes"] + memory["runtime_growth_bytes"],
        ) + memory["system_reserve_bytes"]
        self.assertEqual(required, 128_000_000_000)

    def test_canary_reuses_exact_model_and_runtime_authorities(self) -> None:
        canary = load(CANARY_RECIPE)
        bf16 = load(BF16_RECIPE)
        model = load(MODEL_VERSION)
        runtime = load(RUNTIME)

        self.assertEqual(canary["model"], bf16["model"])
        self.assertEqual(canary["artifacts"], bf16["artifacts"])
        self.assertEqual(canary["runtime"]["distribution"], bf16["runtime"]["distribution"])
        self.assertEqual(canary["model"]["content_sha256"], digest(MODEL_VERSION))
        self.assertEqual(
            canary["runtime"]["distribution"]["content_sha256"], digest(RUNTIME)
        )
        self.assertEqual(
            model["source"]["revision"],
            "426936f8b22dc28e4def61e515478b0b7e4a53cc",
        )
        self.assertEqual(
            runtime["source"]["revision"],
            "d035dcd7cc7c88e0a154609b62887d50bba9fdc2",
        )

    def test_canary_profile_cannot_fall_back_to_bf16(self) -> None:
        adapter = adapter_module()
        self.assertEqual(adapter.PROFILES, {"fp8-cast-sequential-offload"})
        self.assertEqual(adapter.DEFAULT_PROFILE, "fp8-cast-sequential-offload")
        self.assertEqual(adapter.ALLOWED_REQUEST_KEYS, {"seed"})
        self.assertEqual(adapter._profile(None), "fp8-cast-sequential-offload")
        with self.assertRaises(ValueError):
            adapter._profile("bf16-model-offload")

        with tempfile.TemporaryDirectory() as directory:
            adapter.INPUT_ROOT = Path(directory)
            (adapter.INPUT_ROOT / "request.json").write_text(
                json.dumps({"profile": "bf16-model-offload"}), encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "unsupported fields"):
                adapter._load_request()

        source = (CANARY_ADAPTER / "run.py").read_text(encoding="utf-8")
        self.assertIn("storage_dtype=torch.float8_e4m3fn", source)
        self.assertIn('enable_sequential_cpu_offload(device="cuda")', source)
        self.assertNotIn('enable_model_cpu_offload(device="cuda")', source)

    def test_declared_savings_are_limited_and_canary_is_admissible(self) -> None:
        adapter = adapter_module()
        self.assertEqual(adapter.TRANSFORMER_BF16_BYTES, 37_976_221_088)
        self.assertEqual(adapter.DECLARED_MINIMUM_FP8_SAVINGS_BYTES, 8_000_000_000)

        recipe = load(CANARY_RECIPE)
        target = next(item for item in recipe["artifacts"] if item["id"] == "target")
        self.assertTrue(set(adapter.TRANSFORMER_BF16_SHARD_BYTES) <= set(target["include_paths"]))
        memory = recipe["topology"]["roles"][0]["resources"]["memory"]
        self.assertEqual(
            (
                memory["startup_peak_bytes"],
                memory["steady_state_bytes"],
                memory["runtime_growth_bytes"],
                memory["system_reserve_bytes"],
            ),
            (110_000_000_000, 96_000_000_000, 6_000_000_000, 10_000_000_000),
        )
        required = max(
            memory["startup_peak_bytes"],
            memory["steady_state_bytes"] + memory["runtime_growth_bytes"],
        ) + memory["system_reserve_bytes"]
        self.assertEqual(required, 120_000_000_000)
        self.assertLessEqual(required, SPARK_ALLOCATABLE_BYTES)
        tags = set(recipe["metadata"]["tags"])
        self.assertTrue({"executable", "candidate", "canary", "fp8-cast"} <= tags)
        self.assertNotIn("accepted", tags)

    def test_source_bundle_release_and_ledger_are_exact(self) -> None:
        source_bundle = runpy.run_path(str(ROOT / "tools/build-catalog-index"))[
            "source_bundle"
        ]
        archive, _, bundle_digest = source_bundle(CANARY_ADAPTER)
        recipe = load(CANARY_RECIPE)
        context = recipe["build"]["context"]
        self.assertEqual(context["path"], "adapters/video/ltx25-diffusers-fp8-canary")
        self.assertEqual(context["expected_bytes"], len(archive))
        self.assertEqual(context["sha256"], bundle_digest)

        release = load(CANARY_RELEASE)
        self.assertEqual(release["recipe"]["slug"], CANARY_SLUG)
        self.assertEqual(release["version"], "1.0.1")
        self.assertEqual(release["history"][0]["recipe_content_sha256"], digest(CANARY_RECIPE))
        self.assertEqual(release["history"][0]["upgrade_effect"], "rebuild")

        ledger = load(ROOT / "model-targets/video.json")
        target = next(
            item
            for item in ledger["targets"]
            if item.get("catalog_model_version")
            == "ltx-2-5-22b-distilled-bf16-diffusers"
        )
        self.assertEqual(target["recipe_slugs"], [BF16_SLUG, CANARY_SLUG])
        self.assertIn("120 GB", target["notes"])
        self.assertIn("Physical", target["notes"])


if __name__ == "__main__":
    unittest.main()
