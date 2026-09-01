from __future__ import annotations

import hashlib
import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LOWER_FLEET_BASELINE_BYTES = 126_946_283_520
CANARY_ADMISSION_BYTES = 126_000_000_000

LAGUNA_ORIGINAL = "laguna-s-2-1-nvfp4-vllm-single"
LAGUNA_CANARY = "laguna-s-2-1-nvfp4-vllm-low-memory-canary-single"
NEMOTRON_ORIGINAL = (
    "nemotron-3-5-lightning-30b-a3b-vllm-dspark-latency-single"
)
NEMOTRON_CANARY = "nemotron-3-5-lightning-dspark-lowmem-canary-single"


def load(relative: str) -> dict[str, object]:
    return json.loads((ROOT / relative).read_text(encoding="utf-8"))


def recipe(slug: str) -> dict[str, object]:
    return load(f"recipes/{slug}.json")


def arguments(document: dict[str, object]) -> dict[str, object]:
    return {
        item["name"]: item["value"]
        for item in document["runtime"]["arguments"]
    }


def canonical_digest(document: dict[str, object]) -> str:
    payload = json.dumps(
        document,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def admission_bytes(document: dict[str, object]) -> int:
    memory = document["topology"]["roles"][0]["resources"]["memory"]
    return max(
        memory["startup_peak_bytes"],
        memory["steady_state_bytes"] + memory["runtime_growth_bytes"],
    ) + memory["system_reserve_bytes"]


class LowMemoryCanaryRecipeTests(unittest.TestCase):
    def test_original_profiles_are_preserved_and_canaries_are_distinct(self) -> None:
        original_laguna = recipe(LAGUNA_ORIGINAL)
        canary_laguna = recipe(LAGUNA_CANARY)
        original_nemotron = recipe(NEMOTRON_ORIGINAL)
        canary_nemotron = recipe(NEMOTRON_CANARY)

        self.assertEqual(admission_bytes(original_laguna), 130_000_000_000)
        self.assertEqual(admission_bytes(original_nemotron), 128_000_000_000)
        self.assertNotEqual(
            original_laguna["identity"]["slug"], canary_laguna["identity"]["slug"]
        )
        self.assertNotEqual(
            original_nemotron["identity"]["slug"],
            canary_nemotron["identity"]["slug"],
        )

    def test_canaries_reuse_exact_artifacts_runtime_and_build_inputs(self) -> None:
        for original_slug, canary_slug in (
            (LAGUNA_ORIGINAL, LAGUNA_CANARY),
            (NEMOTRON_ORIGINAL, NEMOTRON_CANARY),
        ):
            with self.subTest(canary=canary_slug):
                original = recipe(original_slug)
                canary = recipe(canary_slug)
                self.assertEqual(canary["model"]["kind"], original["model"]["kind"])
                self.assertEqual(canary["model"]["publisher"], original["model"]["publisher"])
                self.assertEqual(canary.get("dependencies"), original.get("dependencies"))
                self.assertEqual(canary["artifacts"], original["artifacts"])
                self.assertEqual(
                    canary["runtime"]["distribution"],
                    original["runtime"]["distribution"],
                )
                self.assertEqual(canary["build"], original["build"])
                self.assertEqual(canary["provenance"], original["provenance"])

    def test_canaries_use_explicit_supported_memory_controls(self) -> None:
        laguna = arguments(recipe(LAGUNA_CANARY))
        self.assertEqual(laguna["max-model-len"], 32_768)
        self.assertEqual(laguna["max-num-seqs"], 1)
        self.assertEqual(laguna["max-num-batched-tokens"], 4_096)
        self.assertIs(laguna["enable-chunked-prefill"], True)
        self.assertEqual(laguna["kv-cache-dtype"], "fp8")
        self.assertEqual(laguna["kv-cache-memory-bytes"], 3_000_000_000)
        self.assertIs(laguna["enforce-eager"], True)

        nemotron = arguments(recipe(NEMOTRON_CANARY))
        self.assertEqual(nemotron["max-model-len"], 262_144)
        self.assertEqual(nemotron["max-num-seqs"], 1)
        self.assertEqual(nemotron["max-num-batched-tokens"], 4_096)
        self.assertIs(nemotron["enable-chunked-prefill"], True)
        self.assertEqual(nemotron["kv-cache-dtype"], "fp8")
        self.assertEqual(nemotron["kv-cache-memory-bytes"], 4_000_000_000)
        self.assertEqual(nemotron["mamba-ssm-cache-dtype"], "float16")
        self.assertIs(nemotron["enforce-eager"], True)

        for values in (laguna, nemotron):
            self.assertNotIn("gpu-memory-utilization", values)
            self.assertNotIn("cpu-offload-gb", values)
            self.assertNotIn("swap-space", values)

    def test_canary_envelopes_fit_without_claiming_acceptance(self) -> None:
        for slug in (LAGUNA_CANARY, NEMOTRON_CANARY):
            with self.subTest(recipe=slug):
                document = recipe(slug)
                self.assertEqual(admission_bytes(document), CANARY_ADMISSION_BYTES)
                self.assertLess(admission_bytes(document), LOWER_FLEET_BASELINE_BYTES)
                self.assertEqual(
                    LOWER_FLEET_BASELINE_BYTES - admission_bytes(document),
                    946_283_520,
                )
                tags = set(document["metadata"]["tags"])
                self.assertTrue(
                    {"candidate", "canary", "physical-oom-gated", "not-default"}
                    <= tags
                )
                description = document["metadata"]["description"].lower()
                self.assertIn("canary hypothesis", description)
                self.assertIn("physical", description)

    def test_release_history_pins_recipe_and_runtime_evidence(self) -> None:
        for slug in (LAGUNA_CANARY, NEMOTRON_CANARY):
            with self.subTest(recipe=slug):
                document = recipe(slug)
                release = load(f"recipe-releases/{slug}.json")
                expected_version = "1.0.2" if slug == LAGUNA_CANARY else "1.0.1"
                self.assertEqual(release["version"], expected_version)
                expected_date = "2026-09-01"
                self.assertEqual(release["released_at"], expected_date)
                self.assertEqual(
                    release["history"][0]["recipe_content_sha256"],
                    canonical_digest(document),
                )
                references = {
                    reference
                    for entry in release["history"]
                    for change in entry.get("changes", [])
                    for reference in change.get("references", [])
                }
                self.assertIn(
                    "https://github.com/vllm-project/vllm/tree/"
                    "6e448d0ea9bf3d88d898b65449ca6dc2aec170ac",
                    references,
                )
                self.assertIn(
                    "https://docs.vllm.ai/en/v0.27.1/cli/serve/", references
                )

    def test_target_inventory_exposes_both_canaries(self) -> None:
        targets = load("model-targets/language.json")["targets"]
        by_model = {target["model"]: target for target in targets}
        self.assertIn(
            LAGUNA_CANARY, by_model["Laguna S"]["recipe_slugs"]
        )
        self.assertIn(
            NEMOTRON_CANARY,
            by_model["Nemotron 3.5 Lightning 30B-A3B"]["recipe_slugs"],
        )


if __name__ == "__main__":
    unittest.main()
