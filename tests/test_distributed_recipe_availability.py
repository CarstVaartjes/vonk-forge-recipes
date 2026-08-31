from __future__ import annotations

import hashlib
import json
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

RECIPES = {
    "glm-5-2-aqlm-vllm-triple": {
        "nodes": 3,
        "version": "2.0.5",
        "released_at": "2026-09-01",
        "effect": "reinstall",
        "alternative": "glm-5-3-flash-exl3-dflash2-vllm-dual",
        "tp2_weight_floor": 146_299_574_266,
    },
    "glm-5-2-quanttrio-vllm-four": {
        "nodes": 4,
        "version": "2.0.3",
        "released_at": "2026-08-30",
        "effect": "metadata-only",
        "alternative": "glm-5-3-flash-exl3-dflash2-vllm-dual",
        "tp2_weight_floor": 202_759_255_945,
    },
    "glm-5-3-flash-nvfp4-vllm-four": {
        "nodes": 4,
        "version": "1.1.0",
        "released_at": "2026-08-31",
        "effect": "reinstall",
        "alternative": "glm-5-3-flash-exl3-dflash2-vllm-dual",
    },
    "inkling-975b-a41b-nvfp4-sglang-eight": {
        "nodes": 8,
        "version": "1.0.4",
        "released_at": "2026-08-30",
        "effect": "metadata-only",
        "alternative": "Inkling Small NVFP4 dual",
        "tp2_weight_floor": 296_018_668_559,
    },
}

FLEET_FREE_MEMORY_BYTES = 126_900_000_000
FLEET_FREE_DISK_BYTES = 3_500_000_000_000


def load(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def canonical_digest(document: dict[str, object]) -> str:
    payload = json.dumps(
        document,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode()
    return hashlib.sha256(payload).hexdigest()


class DistributedRecipeAvailabilityTests(unittest.TestCase):
    def test_larger_topologies_are_explicitly_unsupported_on_current_fleet(
        self,
    ) -> None:
        for slug, expected in RECIPES.items():
            with self.subTest(recipe=slug):
                recipe = load(ROOT / f"recipes/{slug}.json")
                self.assertEqual(recipe["topology"]["node_count"], expected["nodes"])
                description = recipe["metadata"]["description"]
                self.assertIn(
                    "unsupported on Vonk's current one- or two-Spark configuration",
                    description,
                )
                self.assertIn(expected["alternative"], description)
                self.assertIn("candidate", recipe["metadata"]["tags"])
                self.assertIn("executable", recipe["metadata"]["tags"])

    def test_checkpoint_lower_bounds_reject_unprovable_tp2_variants(self) -> None:
        for slug, expected in RECIPES.items():
            weight_floor = expected.get("tp2_weight_floor")
            if weight_floor is None:
                continue
            with self.subTest(recipe=slug):
                recipe = load(ROOT / f"recipes/{slug}.json")
                checkpoint_bytes = recipe["topology"]["roles"][0]["resources"]["disk"][
                    "artifact_bytes"
                ]
                self.assertEqual(checkpoint_bytes // 2, weight_floor)
                self.assertGreater(weight_floor, FLEET_FREE_MEMORY_BYTES)

    def test_all_historical_contracts_fit_the_per_node_disk_inventory(self) -> None:
        for slug in RECIPES:
            with self.subTest(recipe=slug):
                recipe = load(ROOT / f"recipes/{slug}.json")
                disk = recipe["topology"]["roles"][0]["resources"]["disk"]
                envelope = sum(disk.values())
                self.assertLess(envelope, FLEET_FREE_DISK_BYTES)

    def test_glm53_fit_guidance_uses_controller_admission_envelopes(self) -> None:
        four = load(ROOT / "recipes/glm-5-3-flash-nvfp4-vllm-four.json")
        ray = load(ROOT / "recipes/glm-5-3-flash-nvfp4-vllm-dual.json")
        exl3 = load(ROOT / "recipes/glm-5-3-flash-exl3-dflash2-vllm-dual.json")

        def admission_bytes(recipe: dict[str, object]) -> int:
            memory = recipe["topology"]["roles"][0]["resources"]["memory"]
            required = max(
                memory["startup_peak_bytes"],
                memory["steady_state_bytes"] + memory["runtime_growth_bytes"],
            )
            return required + memory["system_reserve_bytes"]

        self.assertEqual(admission_bytes(ray), 132_000_000_000)
        self.assertEqual(admission_bytes(exl3), 126_000_000_000)
        self.assertIn(
            "132 GB Controller admission envelope", four["metadata"]["description"]
        )
        self.assertIn(
            "126 GB-envelope fleet Candidate", four["metadata"]["description"]
        )
        self.assertEqual(
            four["artifacts"][0]["revision"],
            "92d8bfb91c19ceb6fb530dfb538a3a24eceb6ef7",
        )
        self.assertIn(
            "unsupported on Vonk's current one- or two-Spark configuration",
            four["metadata"]["description"],
        )
        self.assertIn(
            "collective.two-ranks",
            four["validation"]["validators"][0]["checks"],
        )

    def test_current_releases_bind_the_exact_current_recipes(self) -> None:
        for slug, expected in RECIPES.items():
            with self.subTest(recipe=slug):
                recipe = load(ROOT / f"recipes/{slug}.json")
                release = load(ROOT / f"recipe-releases/{slug}.json")
                self.assertEqual(release["version"], expected["version"])
                self.assertEqual(release["released_at"], expected["released_at"])
                self.assertEqual(
                    release["history"][0]["upgrade_effect"], expected["effect"]
                )
                self.assertEqual(
                    release["history"][0]["recipe_content_sha256"],
                    canonical_digest(recipe),
                )

    def test_quanttrio_is_superseded_without_mutating_historical_contract(self) -> None:
        recipe = load(ROOT / "recipes/glm-5-2-quanttrio-vllm-four.json")
        self.assertTrue(
            {"historical", "superseded", "tp4"} <= set(recipe["metadata"]["tags"])
        )
        self.assertEqual(
            recipe["artifacts"][0]["repository"],
            "QuantTrio/GLM-5.2-Int4-Int8Mix",
        )
        self.assertEqual(
            recipe["artifacts"][0]["revision"],
            "1d3bcfe5ec549ecd000fd80b37f191183842e983",
        )
        self.assertEqual(
            recipe["runtime"]["distribution"]["slug"],
            "glm-5-2-quanttrio-four-spark",
        )
        release = load(ROOT / "recipe-releases/glm-5-2-quanttrio-vllm-four.json")
        references = release["history"][0]["changes"][0]["references"]
        self.assertTrue(any("keys-latest-GLM-5.2" in item for item in references))

    def test_target_ledger_does_not_recommend_unavailable_topologies(self) -> None:
        target_set = load(ROOT / "model-targets/language.json")
        targets = target_set["targets"]
        by_recipe = {
            slug: target
            for target in targets
            for slug in target.get("recipe_slugs", [])
        }
        self.assertIn(
            "146.3 GB-per-node checkpoint lower bound",
            by_recipe["glm-5-2-aqlm-vllm-triple"]["notes"],
        )
        self.assertIn(
            "Superseded historical TP4",
            by_recipe["glm-5-2-quanttrio-vllm-four"]["notes"],
        )
        self.assertIn(
            "132 GB per node", by_recipe["glm-5-3-flash-nvfp4-vllm-four"]["notes"]
        )
        self.assertIn(
            "128 GB Controller admission envelope",
            by_recipe["inkling-small-nvfp4-sglang-dual"]["notes"],
        )


if __name__ == "__main__":
    unittest.main()
