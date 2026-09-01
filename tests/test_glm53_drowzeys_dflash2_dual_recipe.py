from __future__ import annotations

import hashlib
import json
import os
import runpy
import subprocess
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "model-versions/glm-5-3-flash-nvfp4-ablit-l15-43-mtp-l45-80b6d18d.json"
RECIPE = ROOT / "recipes/glm-5-3-flash-nvfp4-ablit-l15-43-dflash2-vllm-dual.json"
RUNTIME = ROOT / "runtime-distributions/vllm-glm53-dflash2-tonyd2wild-3eef4663-arm64.json"
PATCH = ROOT / "patch-bundles/glm53-nvfp4-dflash2-dual-3eef4663-profile.json"
RELEASE = ROOT / "recipe-releases/glm-5-3-flash-nvfp4-ablit-l15-43-dflash2-vllm-dual.json"
ADAPTER = ROOT / "adapters/glm/tonyd2wild-dflash2-dual"
DRAFTER = ROOT / "model-versions/glm-5-3-flash-dflash2-bf582e4e.json"


def load(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def canonical_digest(document: object) -> str:
    payload = json.dumps(
        document,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(payload).hexdigest()


EXPECTED_INVENTORY = {
    ".gitattributes": (1635, "d1a8f4a1d2e3787c5956393d9306365ebefe1912cb76828870682b1ab16f5c27"),
    "ABLIT_META.json": (10190, "0ee97d0d4086ca33eebeb0acd7b0bd113b904e9c8430655609cf1f8243bf7ff6"),
    "CREDITS.md": (2544, "d37711506ba2cc557f66d11e09e4bf8fb21346991822b63e3ffc9b05f9d4dfe6"),
    "LAYER_MAP.json": (10445, "666b2fe0563d061cda91c0f5382b1c5ab3aabeb8984d58c31911a236b44d3420"),
    "LICENSE": (1592, "f4f9c3f443838247ab77ac1e7b7475828ce59dcb252df5d4de08be0c70f28dc4"),
    "LICENSE-KEYS.md": (1592, "f4f9c3f443838247ab77ac1e7b7475828ce59dcb252df5d4de08be0c70f28dc4"),
    "METHOD.md": (6427, "fea52eb6db87af491c067b360a952fb0fd62413b570b76cd9a6573f7f8293a20"),
    "README.md": (11858, "d56e57c9c0fe3e58746cd0d8fcbc0d0d8a5ee60e004b0aa1daf86a5d9b857362"),
    "RESPONSIBLE_USE.md": (3795, "895e5deb2153b95a32a356579513a5ebdafb218ca549448c5777b1b1d5591feb"),
    "VARIATIONS.json": (1869, "869a8d4588af5c6f4d0b12b14fefacc60cf36a7e9eda4eabd583cc75c7543092"),
    "ablit/refusal_direction_glm53_bf_oproj.pt": (18328, "2f0322252b909f01a6cc24cf4bbca26a084353304bfd8f81aea7db2fb5dfcb2d"),
    "ablit/refusal_direction_glm53_dealign_late.pt": (18356, "ef297400d306f2dd83a7df9036af613ed18dbce68c7d4469b23587dc112cee2a"),
    "chat_template.jinja": (10644, "34d5ee66b12fa6446cdae131c352b8f68cd85369e0e6fda115583805fada3891"),
    "chat_template.thinking-off.jinja": (11326, "dbf7ca0b9c264338c9d6f005c4a7135e00fb4f98c05f47d7b076dd4c27ee030d"),
    "config.json": (53876, "29c9f4171196910e99b9c069d6b76c56e3cdcd0f436dc1bacbc9513c9a7529ac"),
    "generation_config.json": (177, "e9e93ee323895fa5e8e160f1f9826fd98b6bd7b3935435f423c3554f88bfa3ac"),
    "model-00001-of-00010.safetensors": (20002894272, "6b6846fb90f68505d54dc143c517cfb6dc5e6e0e83d1335942027a8c54369abb"),
    "model-00002-of-00010.safetensors": (20002803476, "f8316bca081630a5a6b2c8cf6a2a760e9f409649d7759d3dbc9777d9d91bcca2"),
    "model-00003-of-00010.safetensors": (20000726256, "d26cf1a3b639a9d57c0c0b2408d0f5bfaf9fc2e014b4260507aae7b864f1b5e3"),
    "model-00004-of-00010.safetensors": (20000334356, "49aac2eac1603c0c4d94946247c2cc2c2685892b0c4144b3dafea82176001bab"),
    "model-00005-of-00010.safetensors": (20000724512, "2c054fbed9c3aa34d244ac85c04921da43fe737b9dc09dcb1799ccbd7399c093"),
    "model-00006-of-00010.safetensors": (20000331556, "d6705ac4e7b7f5fd0793ee412f129ee6a3f2c12a85cfcda78612e93c95c3d7e9"),
    "model-00007-of-00010.safetensors": (20000725168, "93ac9987d461a8303d710c7c580df06a5fe2c5a6c0365bda7b003878329b7f6d"),
    "model-00008-of-00010.safetensors": (20000332348, "7d61f4c2571edc07e42c5a090525826bfdd6726fe4a1c14bc142d3999d2917d3"),
    "model-00009-of-00010.safetensors": (20000722232, "6ded1f8c734a8436a8d6d2f9559c861ddcf801f48b3e8dbe448378fecb4dff7e"),
    "model-00010-of-00010.safetensors": (10215657876, "2bfd177459f4c1dbeadeded0936ce569ef72f198464dbe974c4e7cda5fb90bbd"),
    "model.safetensors.index.json": (17055736, "015faae91e8189c7553f1d48ec3d0694b8c02b282d7f58af2d7b4064a81ce4c0"),
    "model_mtp.safetensors": (7618560424, "378f887512bffee45555d95856fd37ca5e804e16aed1dfee18da1ff06bbe5af7"),
    "processor_config.json": (909, "aae38374c94b08cc9b0547c6e64f05b951bd9735cea571c6988f5ed552bed3ed"),
    "recipe.yaml": (1200, "c90bdcc6cd502c44af68ed4017740fe448d213f537a1a9509dc6437b9833c673"),
    "tokenizer.json": (20217541, "0cfe2c099a7702a0921abc315ee039deb51e4a34b4818fc509bd27fa3dc4acc1"),
    "tokenizer_config.json": (790, "77af7d4769cd62c107b90495cac9b0ba81573c86486821bfba2980c04285ec7a"),
}


class DrowzeysGlm53Dflash2DualRecipeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.model = load(MODEL)
        cls.recipe = load(RECIPE)
        cls.runtime = load(RUNTIME)
        cls.patch = load(PATCH)
        cls.release = load(RELEASE)
        cls.drafter = load(DRAFTER)

    def test_complete_gated_inventory_is_active_and_fail_closed(self) -> None:
        artifacts = self.model["artifacts"]
        actual = {
            item["path"]: (item["download_bytes"], item["sha256"])
            for item in artifacts
        }
        self.assertEqual(actual, EXPECTED_INVENTORY)
        self.assertEqual(len({item["id"] for item in artifacts}), 32)
        self.assertEqual(sum(size for size, _digest in actual.values()), 197_881_253_306)
        self.assertEqual(self.model["sizes"]["download_bytes"], 197_881_253_306)
        self.assertEqual(self.model["sizes"]["installed_bytes"], 197_881_253_306)
        self.assertEqual(self.model["availability"], "active")
        self.assertEqual(
            self.model["access"],
            {"visibility": "restricted", "gated": True, "authentication": "token"},
        )
        self.assertTrue(self.model["license"]["operator_acceptance_required"])
        self.assertNotIn("snapshot", actual)

    def test_new_identity_and_drafter_dependency_are_immutable(self) -> None:
        self.assertEqual(
            self.model["source"]["revision"],
            "80b6d18d77e3020f2384597081d405f19893f101",
        )
        self.assertEqual(
            self.model["dependencies"],
            [{
                "kind": "model-version",
                "publisher": "incoai",
                "slug": "glm-5-3-flash-dflash2-bf582e4e",
                "content_sha256": "78102f61e07270f24d60ad50dc523530c2d8dfce7bafe4f2db8e4b4ddc8f59da",
            }],
        )
        self.assertEqual(self.recipe["model"]["content_sha256"], canonical_digest(self.model))
        artifacts = {item["id"]: item for item in self.recipe["artifacts"]}
        self.assertEqual(artifacts["target"]["revision"], self.model["source"]["revision"])
        self.assertEqual(artifacts["target"]["download_bytes"], 197_881_253_306)
        self.assertEqual(
            artifacts["drafter"]["revision"],
            "bf582e4eacc1810f76656d1811693ff6c6737d2a",
        )
        self.assertEqual(artifacts["drafter"]["download_bytes"], 2_342_460_697)
        self.assertEqual(artifacts["target"]["roles"], ["entrypoint", "worker"])
        self.assertEqual(artifacts["drafter"]["roles"], ["entrypoint", "worker"])
        self.assertEqual(self.drafter["license"]["spdx"], "CC-BY-NC-ND-4.0")
        self.assertTrue(self.drafter["license"]["operator_acceptance_required"])
        self.assertIn("separate CC-BY-NC-ND-4.0", self.recipe["metadata"]["description"])

    def test_historical_withdrawn_line_is_unchanged(self) -> None:
        expected = {
            "models/glm-5-3-flash-nvfp4-abliterated.json": "cd86c93b67bd8a501b5aaf407b4a1500f2c5aebb557fbf16efaa08177437b149",
            "model-versions/glm-5-3-flash-nvfp4-abliterated-d7f8afa8.json": "206d70bc7470b79349c94fafe89d595cdc0ac771f52875302438a7305f9cf77e",
            "recipes/glm-5-3-flash-nvfp4-kv-1m-abliterated-vllm-dual.json": "9892cf5b96978bdbb736b96ab7f7b2b97b47552138182adccfea2b9c95358694",
            "runtime-distributions/vllm-glm53-drowzeys-1m-c36b5958-arm64.json": "d657b3ddb6c6c99b1693c25df9ad002d94a986a9b54817ba71ba0ec91072a5e2",
            "patch-bundles/drowzeys-glm53-nvfp4-kv-1m-dual-profile.json": "d9e1c1928a9b929cf773111b053e0eac4435e4bd2bfb4040f031b60efdde91fd",
            "recipe-releases/glm-5-3-flash-nvfp4-kv-1m-abliterated-vllm-dual.json": "56e076a92f48761f20fc63be10a9a884357bd22c441ee7e321f8e79c097212b2",
        }
        for path, digest in expected.items():
            self.assertEqual(canonical_digest(load(ROOT / path)), digest, path)
        old_model = load(ROOT / "model-versions/glm-5-3-flash-nvfp4-abliterated-d7f8afa8.json")
        self.assertEqual(old_model["availability"], "withdrawn")

    def test_exact_publisher_serving_profile_and_thinking_off_contract(self) -> None:
        arguments = {
            item["name"]: item["value"] for item in self.recipe["runtime"]["arguments"]
        }
        self.assertLessEqual(float(arguments["gpu-memory-utilization"]), 0.85)
        self.assertEqual(arguments["max-model-len"], 262_144)
        self.assertEqual(arguments["max-num-seqs"], 6)
        self.assertEqual(arguments["max-num-batched-tokens"], 8192)
        self.assertEqual(arguments["block-size"], 2304)
        self.assertEqual(arguments["moe-backend"], "marlin")
        self.assertEqual(arguments["kv-cache-dtype"], "fp8_e4m3")
        self.assertEqual(arguments["kv-cache-memory"], 3_221_225_472)
        self.assertTrue(arguments["enforce-eager"])
        self.assertEqual(
            json.loads(arguments["speculative-config"]),
            {"method": "dflash", "model": "/models/drafter", "num_speculative_tokens": 7},
        )
        self.assertEqual(
            arguments["chat-template"],
            "/models/target/chat_template.thinking-off.jinja",
        )
        self.assertEqual(
            json.loads(arguments["default-chat-template-kwargs"]),
            {"enable_thinking": False},
        )
        self.assertNotIn("quantization", arguments)
        recipe_text = RECIPE.read_text(encoding="utf-8").lower()
        self.assertNotIn("exl3", recipe_text)
        self.assertNotIn("1048576", recipe_text)

    def test_two_phase_topology_matches_runtime_capability(self) -> None:
        topology = self.recipe["topology"]
        distributed = self.runtime["capabilities"]["distributed_vllm"]
        self.assertEqual(topology["parallelism"]["backend"], "mp")
        self.assertEqual(distributed["mechanism"], "vllm-mp")
        self.assertEqual(topology["node_count"], 2)
        self.assertEqual(topology["start_order"], ["worker", "entrypoint"])
        self.assertEqual(
            self.recipe["runtime"]["lifecycle"]["readiness"],
            {"strategy": "endpoint-owner-after-all-ranks", "path": "/health", "timeout_seconds": 3600},
        )
        self.assertEqual(self.recipe["interfaces"][0]["health_path"], "/health")
        self.assertTrue(distributed["rank_loss_withdraws_endpoint"])

    def test_runtime_image_source_security_and_adapter_are_exact(self) -> None:
        self.assertEqual(
            self.runtime["source"],
            {
                "repository": "https://github.com/tonyd2wild/GLM-5.3-Flash-NVFP4-DFlash2-2x-DGX-Spark",
                "revision": "3eef46632c45ffb6c397de0716c23b3d2d594798",
                "archive_sha256": "1b9d7ce8f6546a47d6a4479e837953551852ac4dd6ec33af602e2ec7c5ccc1c2",
                "license": "MIT",
            },
        )
        self.assertEqual(
            self.runtime["image"],
            "ghcr.io/tonyd2wild/vllm-glm53-flash@sha256:4def0ef644cb2e9814136dcffd5e385e21bc594f48f3b292234051904abe85a6",
        )
        self.assertEqual(
            self.runtime["image_manifest"],
            {
                "digest": "4def0ef644cb2e9814136dcffd5e385e21bc594f48f3b292234051904abe85a6",
                "size": 10372,
                "config_digest": "35c6f70ffcba62fd67d7b9d4b4e8300ad177201792ce9cdb1ea18fd449bc23b6",
                "compressed_layers_bytes": 14_204_524_092,
            },
        )
        self.assertEqual(self.runtime["build"]["network_hosts"], [])
        self.assertEqual(self.recipe["build"]["network"], {"mode": "none", "hosts": []})
        self.assertEqual(self.runtime["security"]["user"], "10001:10001")
        self.assertTrue(self.runtime["security"]["no_new_privileges"])
        self.assertFalse(self.recipe["runtime"]["security"]["privileged"])

        source_bundle = runpy.run_path(str(ROOT / "tools/build-catalog-index"))["source_bundle"]
        archive, _files, digest = source_bundle(ADAPTER)
        context = self.recipe["build"]["context"]
        self.assertEqual(context["sha256"], digest)
        self.assertEqual(context["expected_bytes"], len(archive))
        self.assertEqual(self.patch["source_bundle"]["sha256"], digest)
        self.assertEqual(self.patch["applies_to"]["content_sha256"], canonical_digest(self.runtime))
        self.assertEqual(self.recipe["execution"]["patch_bundle"]["content_sha256"], canonical_digest(self.patch))
        indexer = ADAPTER / "sparse_attn_indexer_kpool_sm121.py"
        self.assertEqual(indexer.stat().st_size, 46_355)
        self.assertEqual(
            hashlib.sha256(indexer.read_bytes()).hexdigest(),
            "8a3ecfb0bab2441dd7417ed00a10d142191496149f88e5fe79fcfaea4b160980",
        )
        dockerfile = (ADAPTER / "Dockerfile").read_text(encoding="utf-8")
        self.assertIn(
            "COPY sparse_attn_indexer_kpool_sm121.py /usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/sparse_attn_indexer_kpool.py",
            dockerfile,
        )
        adapter_text = "\n".join(path.read_text(encoding="utf-8") for path in ADAPTER.iterdir())
        self.assertNotIn("ssh", adapter_text.lower())

    def test_controller_disk_envelope_and_release_are_bound(self) -> None:
        artifact_bytes = sum(item["download_bytes"] for item in self.recipe["artifacts"])
        self.assertEqual(artifact_bytes, 200_223_714_003)
        for role in self.recipe["topology"]["roles"]:
            disk = role["resources"]["disk"]
            self.assertEqual(disk["artifact_bytes"], artifact_bytes)
            self.assertGreaterEqual(
                disk["image_bytes"], self.runtime["image_manifest"]["compressed_layers_bytes"]
            )
        tags = set(self.recipe["metadata"]["tags"])
        self.assertTrue({"candidate", "executable", "gated", "dflash2"} <= tags)
        self.assertNotIn("accepted", tags)
        self.assertEqual(self.release["version"], "1.0.0")
        self.assertEqual(self.release["history"][0]["upgrade_effect"], "reinstall")
        self.assertEqual(
            self.release["history"][0]["recipe_content_sha256"],
            canonical_digest(self.recipe),
        )

    def test_wrapper_rejects_unbound_invocation(self) -> None:
        environment = {"PATH": os.environ.get("PATH", "")}
        result = subprocess.run(
            [sys.executable, str(ADAPTER / "vllm-wrapper.py")],
            check=False,
            capture_output=True,
            text=True,
            env=environment,
            timeout=10,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("exact Controller rendezvous", result.stderr)


if __name__ == "__main__":
    unittest.main()
