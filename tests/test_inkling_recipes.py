from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load(path: str) -> dict:
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


class InklingSmallRecipeTests(unittest.TestCase):
    def test_complete_model_inventory_and_two_spark_profile(self) -> None:
        model, recipe = load("models/inkling-small-nvfp4.json"), load("recipes/inkling-small-nvfp4-sglang-dual.json")
        self.assertEqual(model["source"]["revision"], "b6a99534467840620d411e4cd4ad5819b2610d9c")
        self.assertEqual(len({item["id"] for item in model["files"]}), len(model["files"]))
        self.assertEqual(recipe["topology"]["node_count"], 2)
        self.assertEqual(recipe["topology"]["parallelism"]["tensor"], 2)
        self.assertEqual(recipe["topology"]["parallelism"]["backend"], "native")

    def test_runtime_is_offline_and_multimodal_api_is_named(self) -> None:
        recipe = load("recipes/inkling-small-nvfp4-sglang-dual.json")
        interface = recipe["interfaces"][0]
        self.assertEqual(interface["adapter"], "openai")
        self.assertEqual(interface["model_aliases"], ["inkling-small"])
        self.assertEqual(recipe["runtime"]["engine"], "sglang")
        self.assertTrue(recipe["models"])

    def test_wrapper_resolves_only_the_compiler_rendezvous_sentinel(self) -> None:
        wrapper = ROOT / "adapters/inkling/small-dual/sglang-serve"
        with tempfile.TemporaryDirectory() as directory:
            launcher = Path(directory) / "sglang/launch_server.py"; launcher.parent.mkdir()
            launcher.write_text("import json, os, sys\nopen(os.environ['CAPTURE'], 'w').write(json.dumps(sys.argv[1:]))")
            capture = Path(directory) / "arguments.json"
            env = {**os.environ, "PYTHONPATH": directory, "CAPTURE": str(capture), "VONK_MASTER_ADDR": "192.0.2.10", "VONK_MASTER_PORT": "29500", "VONK_LOCAL_ADDR": "192.0.2.11", "NCCL_SOCKET_IFNAME": "eth0", "NCCL_IB_HCA": "roce0", "NCCL_IB_GID_INDEX": "3", "TP_SOCKET_IFNAME": "eth0", "GLOO_SOCKET_IFNAME": "eth0"}
            result = subprocess.run([sys.executable, str(wrapper), "--nnodes", "2", "--node-rank", "1", "--dist-init-addr", "VONK_MASTER_ADDR:VONK_MASTER_PORT"], env=env, check=False)
            self.assertEqual(result.returncode, 0)
            self.assertEqual(json.loads(capture.read_text())[-1], "192.0.2.10:29500")


if __name__ == "__main__": unittest.main()
