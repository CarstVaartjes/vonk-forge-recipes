from __future__ import annotations

import json
import os
import runpy
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RECIPE = ROOT / "recipes/glm-5-3-flash-nvfp4-ablit-l15-43-dflash2-vllm-dual.json"
ADAPTER = ROOT / "adapters/glm/tonyd2wild-dflash2-dual"


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


class DrowzeysGlm53Dflash2DualRecipeTests(unittest.TestCase):
    def test_model_selections_are_immutable_and_dual(self) -> None:
        recipe = load(RECIPE)
        self.assertEqual(recipe["topology"]["node_count"], 2)
        self.assertEqual(recipe["topology"]["parallelism"]["backend"], "mp")
        self.assertEqual(len(recipe["models"]), 1)
        self.assertTrue({"candidate", "executable"} <= set(recipe["metadata"]["tags"]))
        self.assertNotIn("accepted", recipe["metadata"]["tags"])

    def test_exact_serving_profile_and_thinking_off_contract(self) -> None:
        recipe = load(RECIPE)
        arguments = {item["name"]: item for item in recipe["runtime"]["arguments"]}
        self.assertEqual(arguments["gpu-memory-utilization"]["value"], "0.85")
        self.assertEqual(json.loads(arguments["default-chat-template-kwargs"]["value"]), {"enable_thinking": False})
        self.assertEqual(recipe["topology"]["start_order"], ["worker", "entrypoint"])

    def test_source_security_and_controller_resource_envelope_are_exact(self) -> None:
        recipe = load(RECIPE)
        build = recipe["execution"]["build"]
        self.assertEqual(build["network"], {"mode": "none", "hosts": []})
        self.assertEqual(build["base_image"]["digest"], "4def0ef644cb2e9814136dcffd5e385e21bc594f48f3b292234051904abe85a6")
        self.assertEqual(build["context"]["path"], "adapters/glm/tonyd2wild-dflash2-dual")
        for role in recipe["topology"]["roles"]:
            self.assertEqual(role["resources"]["disk"]["artifact_bytes"], 200_223_714_003)
            self.assertEqual(role["resources"]["memory"]["startup_peak_bytes"], 126_000_000_000)
        dockerfile = (ADAPTER / "Dockerfile").read_text()
        self.assertIn('org.opencontainers.image.revision="3eef46632c45ffb6c397de0716c23b3d2d594798"', dockerfile)
        self.assertIn("USER 10001:10001", dockerfile)
        self.assertIn("HF_HUB_OFFLINE=1", dockerfile)

    def test_adapter_bundle_is_source_pinned_and_has_no_ssh_rollout(self) -> None:
        tool = runpy.run_path(str(ROOT / "tools/build-catalog-index"))
        _, _, bundle_digest = tool["source_bundle"](ADAPTER)
        self.assertTrue(bundle_digest)
        self.assertNotIn("ssh", "\n".join(path.read_text(errors="ignore") for path in ADAPTER.iterdir() if path.is_file()).lower())

    def test_wrapper_rejects_unbound_invocation(self) -> None:
        result = subprocess.run(
            [sys.executable, str(ADAPTER / "vllm-wrapper.py")],
            check=False,
            capture_output=True,
            text=True,
            env={"PATH": os.environ.get("PATH", "")},
            timeout=10,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("exact Controller rendezvous", result.stderr)

    def test_catalog_package_tracks_recipe_content(self) -> None:
        recipe = load(RECIPE)
        import hashlib
        digest = hashlib.sha256(json.dumps(recipe, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()
        index = load(ROOT / "catalog-index.json")
        entry = next(item for item in index["recipes"] if item["source_path"] == f"recipes/{RECIPE.name}")
        self.assertEqual(entry["package"]["recipe_content_sha256"], digest)


if __name__ == "__main__": unittest.main()
