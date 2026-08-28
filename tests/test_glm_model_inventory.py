from __future__ import annotations

import hashlib
import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load(path: str) -> dict[str, object]:
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def canonical_digest(document: dict[str, object]) -> str:
    payload = json.dumps(
        document,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode()
    return hashlib.sha256(payload).hexdigest()


class GlmModelInventoryTests(unittest.TestCase):
    def test_aqlm_inventory_closes_the_full_pinned_snapshot(self) -> None:
        model = load("model-versions/glm-5-2-nvfp4-aqlm-hybrid-53e0082e.json")
        recipe = load("recipes/glm-5-2-aqlm-vllm-triple.json")
        artifacts = model["artifacts"]
        traces = [item for item in artifacts if item["path"].startswith("traces/")]

        self.assertEqual(len(artifacts), 751)
        self.assertEqual(len(traces), 603)
        self.assertEqual(sum(item["download_bytes"] for item in traces), 83_970_453)
        self.assertEqual(
            sum(item["download_bytes"] for item in artifacts),
            292_599_148_533,
        )
        self.assertEqual(model["sizes"]["download_bytes"], 292_599_148_533)
        self.assertEqual(recipe["artifacts"][0]["download_bytes"], 292_599_148_533)
        self.assertEqual(len({item["id"] for item in artifacts}), len(artifacts))
        self.assertEqual(recipe["model"]["content_sha256"], canonical_digest(model))

    def test_gated_abliterated_inventory_fails_closed_without_fake_artifact(self) -> None:
        model = load(
            "model-versions/glm-5-3-flash-nvfp4-abliterated-d7f8afa8.json"
        )
        recipe = load(
            "recipes/glm-5-3-flash-nvfp4-kv-1m-abliterated-vllm-dual.json"
        )
        artifacts = model["artifacts"]

        self.assertEqual(model["availability"], "withdrawn")
        self.assertEqual(len(artifacts), 126)
        self.assertNotIn("snapshot", {item["path"] for item in artifacts})
        self.assertEqual(len({item["id"] for item in artifacts}), len(artifacts))
        self.assertEqual(
            sum(item["download_bytes"] for item in artifacts),
            194_692_710_561,
        )
        self.assertEqual(model["sizes"]["download_bytes"], 194_692_710_561)
        self.assertEqual(
            recipe["artifacts"][0]["download_bytes"]
            - model["sizes"]["download_bytes"],
            50_928,
        )
        self.assertIn("13 Git files", model["metadata"]["description"])
        self.assertIn("HTTP 401", model["metadata"]["description"])
        self.assertIn("inventory-blocked", recipe["metadata"]["tags"])
        self.assertEqual(recipe["model"]["content_sha256"], canonical_digest(model))


if __name__ == "__main__":
    unittest.main()
