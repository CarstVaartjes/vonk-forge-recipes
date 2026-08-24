from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODEL_REVISION = "b6a99534467840620d411e4cd4ad5819b2610d9c"
SGLANG_REVISION = "a74222ef6e690f851e2e4ff1c0be7dc1357be313"


def load(path: str) -> dict[str, object]:
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


class InklingSmallRecipeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.recipe = load("recipes/inkling-small-nvfp4-sglang-dual.json")
        self.model = load("model-versions/inkling-small-nvfp4.json")
        self.distribution = load(
            "runtime-distributions/sglang-inkling-small-dual-spark-a74222ef.json"
        )
        self.patch = load(
            "patch-bundles/sglang-inkling-small-dual-spark-profile.json"
        )

    def test_complete_checkpoint_inventory_is_immutable(self) -> None:
        artifacts = self.model["artifacts"]
        self.assertEqual(self.model["source"]["revision"], MODEL_REVISION)
        self.assertEqual(
            sum(item["download_bytes"] for item in artifacts),
            self.model["sizes"]["download_bytes"],
        )
        self.assertEqual(
            {item["revision"] for item in artifacts}, {MODEL_REVISION}
        )
        self.assertTrue(
            all(len(item["sha256"]) == 64 for item in artifacts)
        )

    def test_recipe_preserves_verified_two_spark_profile(self) -> None:
        arguments = {
            item["name"]: item["value"] for item in self.recipe["runtime"]["arguments"]
        }
        self.assertEqual(self.recipe["topology"]["node_count"], 2)
        self.assertEqual(
            self.recipe["topology"]["parallelism"],
            {
                "world_size": 2,
                "tensor": 2,
                "pipeline": 1,
                "data": 1,
                "backend": "native",
            },
        )
        self.assertEqual(arguments["quantization"], "modelopt_fp4")
        self.assertEqual(arguments["attention-backend"], "triton")
        self.assertEqual(arguments["fp4-gemm-backend"], "marlin")
        self.assertEqual(arguments["moe-runner-backend"], "marlin")
        self.assertEqual(arguments["page-size"], 128)
        self.assertEqual(arguments["context-length"], 65_536)
        self.assertEqual(arguments["served-model-name"], "inkling-small")
        self.assertIs(arguments["enable-multimodal"], True)
        self.assertIs(arguments["disable-prefill-cuda-graph"], True)

    def test_runtime_binds_official_arm64_image_and_native_ranks(self) -> None:
        self.assertEqual(self.distribution["source"]["revision"], SGLANG_REVISION)
        self.assertEqual(
            self.distribution["image"],
            "docker.io/lmsysorg/sglang@sha256:"
            "bbedab8cbf2d209b00f48f1e96ef4e9b638b98771477fa14e0e70d62679f383b",
        )
        capability = self.distribution["capabilities"]["distributed_sglang"]
        self.assertIs(capability["verified"], True)
        self.assertEqual(capability["mechanism"], "sglang-native")
        self.assertEqual(capability["tensor_parallel_size"], 2)
        profiles = capability["launch"]["rank_profiles"]
        self.assertEqual(
            [(profile["rank"], profile["role"]) for profile in profiles],
            [(0, "entrypoint"), (1, "worker")],
        )
        self.assertTrue(
            all(profile["environment"]["NCCL_IB_GID_INDEX"] == "3" for profile in profiles)
        )

    def test_runtime_is_offline_and_multimodal_api_is_named(self) -> None:
        interface = self.recipe["interfaces"][0]
        self.assertEqual(interface["adapter"], "openai")
        self.assertEqual(interface["model_aliases"], ["inkling-small"])
        self.assertTrue(self.recipe["runtime"]["security"]["host_network"])
        self.assertEqual(self.distribution["security"]["network_mode"], "none")
        self.assertIs(self.distribution["build"]["offline_after_installation"], True)
        self.assertEqual(
            self.patch["sha256"],
            "8e479a5ddd5cfe154a2fd75c6e156b1ff07154deaa96834d8c467306140a7acd",
        )

    def test_wrapper_resolves_only_the_compiler_rendezvous_sentinel(self) -> None:
        wrapper = ROOT / "adapters/inkling/small-dual/sglang-serve"
        with tempfile.TemporaryDirectory() as directory:
            launcher = Path(directory) / "sglang" / "launch_server.py"
            launcher.parent.mkdir()
            launcher.write_text(
                "import json, os, sys\n"
                "open(os.environ['CAPTURE'], 'w').write(json.dumps(sys.argv[1:]))\n",
                encoding="utf-8",
            )
            environment = {
                **os.environ,
                "PYTHONPATH": directory,
                "CAPTURE": str(Path(directory) / "arguments.json"),
                "VONK_LOCAL_ADDR": "192.0.2.11",
                "VONK_MASTER_ADDR": "192.0.2.10",
                "VONK_MASTER_PORT": "29500",
                "NCCL_SOCKET_IFNAME": "eth0",
                "NCCL_IB_HCA": "mlx5_0:1",
                "NCCL_IB_GID_INDEX": "3",
                "TP_SOCKET_IFNAME": "eth0",
                "GLOO_SOCKET_IFNAME": "eth0",
            }
            result = subprocess.run(
                [
                    sys.executable,
                    str(wrapper),
                    "--nnodes",
                    "2",
                    "--node-rank",
                    "1",
                    "--dist-init-addr",
                    "VONK_MASTER_ADDR:VONK_MASTER_PORT",
                ],
                check=False,
                env=environment,
            )
            self.assertEqual(result.returncode, 0)
            captured = json.loads(Path(environment["CAPTURE"]).read_text())
            self.assertEqual(captured[-1], "192.0.2.10:29500")


if __name__ == "__main__":
    unittest.main()
