from __future__ import annotations

import json
import os
import runpy
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
RECIPE = ROOT / "recipes/glm-5-3-flash-exl3-dflash2-vllm-dual.json"
ADAPTER = ROOT / "adapters/glm/mia-exl3-dflash2-dual"


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


class Glm53Exl3DualRecipeTests(unittest.TestCase):
    def test_model_and_drafter_selections_are_closed(self) -> None:
        recipe = load(RECIPE)
        self.assertEqual(recipe["topology"]["node_count"], 2)
        self.assertEqual(recipe["topology"]["parallelism"]["backend"], "mp")
        self.assertGreaterEqual(len(recipe["models"]), 1)
        self.assertTrue({"candidate", "executable"} <= set(recipe["metadata"]["tags"]))
        self.assertTrue(all(selection["files"] for selection in recipe["models"]))
        drafter = next(
            selection
            for selection in recipe["models"]
            if selection["id"] == "dependency-glm-5-3-flash-dflash2-bf582e4e"
        )
        self.assertEqual(
            {file["file_id"] for file in drafter["files"]},
            {
                "metadata-70e0ed421d65-618cd5b83d",
                "readme-2014434d3e36-b335630551",
                "dflash2-figure-6d8dcc9a9472-e520c8f797",
                "config-c4aeac010119-587cb980af",
                "model-b038e1d9d1e7-9d75c1098f",
            },
        )
        self.assertEqual(
            {file["mount"]["target"] for file in drafter["files"]},
            {"/models/drafter"},
        )

    def test_recipe_has_bounded_speculative_profile(self) -> None:
        arguments = {item["name"]: item for item in load(RECIPE)["runtime"]["arguments"]}
        specification = json.loads(arguments["speculative-config"]["value"])
        self.assertEqual(specification["model"], "/models/drafter")
        self.assertEqual(specification["num_speculative_tokens"], 7)
        self.assertEqual(load(RECIPE)["topology"]["start_order"], ["worker", "entrypoint"])

    def test_runtime_tracks_current_upstream_defaults(self) -> None:
        recipe = load(RECIPE)
        arguments = {item["name"]: item for item in recipe["runtime"]["arguments"]}
        self.assertEqual(
            recipe["provenance"]["source_reference"],
            "https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks/tree/3021f24c88a0904c768c46ff22a508407e31360a",
        )
        self.assertEqual(recipe["execution"]["build"]["base_image"]["digest"], "905c02933be6021301db2dc284e24e3727467aa3a0f63b41d609885778a07bce")
        self.assertEqual(arguments["max-num-batched-tokens"]["value"], 7168)
        self.assertEqual(arguments["kv-cache-dtype"]["value"], "fp8")
        self.assertEqual(recipe["models"][0]["model"]["slug"], "glm-5-3-flash-exl3-tr3-4bpw-dflash2-25a44fdb")

    def test_wrapper_preserves_authored_engine_arguments(self) -> None:
        original = [
            "serve",
            "/models/target",
            "--nnodes",
            "2",
            "--node-rank",
            "0",
            "--distributed-executor-backend",
            "mp",
            "--quantization",
            "exl3",
            "--chat-template",
            "/models/target/chat-template.jinja",
            "--cudagraph-capture-sizes",
            "1,2,4",
            "--opaque-runtime-option",
            "preserve-me",
        ]
        captured: list[tuple[str, ...]] = []

        def fake_execv(path: str, argv: tuple[str, ...]) -> None:
            captured.append(argv)
            raise RuntimeError("captured")

        env = {
            "VONK_LOCAL_ADDR": "10.0.0.1",
            "VONK_MASTER_ADDR": "10.0.0.2",
            "VONK_MASTER_PORT": "29500",
            "NCCL_SOCKET_IFNAME": "eth0",
            "NCCL_IB_HCA": "roce0",
            "NCCL_IB_GID_INDEX": "3",
            "TP_SOCKET_IFNAME": "eth0",
            "GLOO_SOCKET_IFNAME": "eth0",
        }
        with patch.object(sys, "argv", [str(ADAPTER / "vllm-wrapper.py"), *original]), patch.dict(
            os.environ, env, clear=False
        ), patch("pathlib.Path.is_file", return_value=True), patch("os.access", return_value=True), patch(
            "os.execv", side_effect=fake_execv
        ):
            with self.assertRaisesRegex(RuntimeError, "captured"):
                runpy.run_path(str(ADAPTER / "vllm-wrapper.py"))

        self.assertEqual(len(captured), 1)
        self.assertEqual(captured[0][1 : 1 + len(original)], tuple(original))
        self.assertEqual(captured[0][-4:], ("--master-addr", "10.0.0.2", "--master-port", "29500"))

    def test_current_source_build_closure_is_vendored_and_uses_no_ssh(self) -> None:
        dockerfile = (ADAPTER / "Dockerfile").read_text()
        self.assertIn("ARG BASE=vllm/vllm-openai:glm53-flash-arm64-cu130@sha256:905c02933be6021301db2dc284e24e3727467aa3a0f63b41d609885778a07bce", dockerfile)
        self.assertIn("COPY overlay/exl3.py", dockerfile)
        self.assertIn("COPY files/chat_template.jinja", dockerfile)
        self.assertTrue((ADAPTER / "upstream-LICENSE").is_file())
        self.assertTrue((ADAPTER / "upstream-start.sh").is_file())
        text = "\n".join((ADAPTER / name).read_text(errors="ignore") for name in ("Dockerfile", "vllm-wrapper.py", "verify-runtime.py"))
        self.assertNotIn("ssh -", text.lower())

    def test_adapter_bundle_is_pinned(self) -> None:
        import runpy
        tool = runpy.run_path(str(ROOT / "tools/build-catalog-index"))
        _, _, digest = tool["source_bundle"](ADAPTER)
        self.assertTrue(digest)


if __name__ == "__main__": unittest.main()
