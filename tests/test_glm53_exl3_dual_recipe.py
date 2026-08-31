from __future__ import annotations

import hashlib
import json
import os
import runpy
import sys
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]


def _document(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _digest(path: Path) -> str:
    payload = json.dumps(
        _document(path), sort_keys=True, separators=(",", ":")
    ).encode()
    return hashlib.sha256(payload).hexdigest()


class Glm53Exl3DualRecipeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.recipe_path = ROOT / "recipes/glm-5-3-flash-exl3-dflash2-vllm-dual.json"
        self.release_path = ROOT / "recipe-releases/glm-5-3-flash-exl3-dflash2-vllm-dual.json"
        self.runtime_path = ROOT / "runtime-distributions/vllm-glm53-exl3-dflash2-mia-493cb88f-arm64.json"
        self.patch_path = ROOT / "patch-bundles/glm-5-3-flash-exl3-dflash2-mia-dual-profile.json"
        self.target_path = ROOT / "model-versions/glm-5-3-flash-exl3-tr3-4bpw-dflash2-25a44fdb.json"
        self.drafter_path = ROOT / "model-versions/glm-5-3-flash-dflash2-bf582e4e.json"
        self.adapter = ROOT / "adapters/glm/mia-exl3-dflash2-dual"

    def test_exact_upstream_runtime_and_image_are_pinned(self) -> None:
        runtime = _document(self.runtime_path)
        self.assertEqual(runtime["source"]["revision"], "493cb88fc69f8ba73ac87404f429d763e2739d89")
        self.assertEqual(runtime["source"]["archive_sha256"], "bdd535e461ab85d16d93a3d2c6970674f560ba1e0a7d27a0d13839c1bc15b88d")
        self.assertEqual(runtime["image_manifest"]["digest"], "9bb1557a4234fce63d59599e44d10747eabd742beb337eebf9e7070be8a0fd58")
        self.assertEqual(runtime["image_manifest"]["config_digest"], "ad0cdd86d1ddd15ee758f519d16da15ac237f7f0648a5c52fbc20f9554944263")
        self.assertEqual(runtime["image_manifest"]["compressed_layers_bytes"], 9_788_994_117)

    def test_target_and_drafter_inventories_and_licenses_are_closed(self) -> None:
        target = _document(self.target_path)
        drafter = _document(self.drafter_path)
        self.assertEqual(target["source"]["revision"], "25a44fdbf16862a46b7cc9921142c6c81350af2f")
        self.assertEqual(len(target["artifacts"]), 144)
        self.assertEqual(sum(item["download_bytes"] for item in target["artifacts"]), 175_715_854_754)
        self.assertEqual(target["license"]["spdx"], "LicenseRef-ShapleyMCG-1.0")
        self.assertTrue(target["license"]["operator_acceptance_required"])
        self.assertEqual(target["dependencies"][0]["content_sha256"], _digest(self.drafter_path))

        self.assertEqual(drafter["source"]["revision"], "bf582e4eacc1810f76656d1811693ff6c6737d2a")
        self.assertEqual(len(drafter["artifacts"]), 5)
        self.assertEqual(sum(item["download_bytes"] for item in drafter["artifacts"]), 2_342_460_697)
        self.assertEqual(
            next(item for item in drafter["artifacts"] if item["path"] == "model.safetensors")["sha256"],
            "b038e1d9d1e7833fa3880c2c0135ba9b673013f03da1b29fb831931584759dac",
        )
        self.assertEqual(drafter["license"]["spdx"], "CC-BY-NC-ND-4.0")
        self.assertTrue(drafter["license"]["operator_acceptance_required"])

    def test_recipe_is_controller_native_candidate_with_exact_profile(self) -> None:
        recipe = _document(self.recipe_path)
        arguments = {item["name"]: item["value"] for item in recipe["runtime"]["arguments"]}
        self.assertIn("candidate", recipe["metadata"]["tags"])
        self.assertIn("pending", recipe["metadata"]["tags"])
        self.assertNotIn("accepted", recipe["metadata"]["tags"])
        self.assertEqual(recipe["model"]["content_sha256"], _digest(self.target_path))
        self.assertEqual(recipe["runtime"]["distribution"]["content_sha256"], _digest(self.runtime_path))
        self.assertEqual(recipe["execution"]["patch_bundle"]["content_sha256"], _digest(self.patch_path))
        self.assertNotIn("quantization", arguments)
        self.assertNotIn("chat-template", arguments)
        self.assertEqual(arguments["max-model-len"], 1_000_000)
        self.assertEqual(arguments["gpu-memory-utilization"], "0.87")
        self.assertEqual(arguments["max-num-seqs"], 4)
        self.assertEqual(arguments["max-num-batched-tokens"], 2048)
        self.assertEqual(arguments["kv-cache-dtype"], "fp8")
        benchmarks = {item["name"]: item for item in recipe["validation"]["benchmarks"]}
        self.assertEqual(
            {
                benchmarks[name]["configuration"]["prompt_tokens"]
                for name in benchmarks
                if name.startswith("cold-prefill-")
            },
            {8_000, 16_000, 100_000, 256_000, 300_000},
        )
        specification = json.loads(arguments["speculative-config"])
        self.assertEqual(specification["model"], "/models/drafter")
        self.assertEqual(specification["num_speculative_tokens"], 7)
        self.assertEqual(specification["draft_tensor_parallel_size"], 2)
        self.assertEqual(specification["kv_cache_dtype"], "auto")
        self.assertEqual(recipe["topology"]["start_order"], ["worker", "entrypoint"])
        self.assertEqual(recipe["topology"]["parallelism"], {"world_size": 2, "tensor": 2, "pipeline": 1, "data": 1, "backend": "mp"})

    def test_admission_fits_both_live_sparks_with_strict_margin(self) -> None:
        recipe = _document(self.recipe_path)
        live = (
            {"disk": 3_490_100_346_880, "memory": 126_899_314_688},
            {"disk": 3_517_418_012_672, "memory": 126_985_986_048},
        )
        for role, snapshot in zip(recipe["topology"]["roles"], live, strict=True):
            disk = role["resources"]["disk"]
            required_disk = sum(disk.values())
            self.assertEqual(required_disk, 297_058_315_451)
            self.assertGreater(snapshot["disk"] - required_disk, 3_193_000_000_000)
            memory = role["resources"]["memory"]
            required_memory = max(
                memory["startup_peak_bytes"],
                memory["steady_state_bytes"] + memory["runtime_growth_bytes"],
            ) + memory["system_reserve_bytes"]
            self.assertEqual(required_memory, 126_000_000_000)
            self.assertGreater(snapshot["memory"] - required_memory, 899_000_000)

    def test_adapter_bundle_is_immutable_and_uses_no_ssh(self) -> None:
        source_bundle = runpy.run_path(str(ROOT / "tools/build-catalog-index"))["source_bundle"]
        archive, _files, digest = source_bundle(self.adapter)
        recipe = _document(self.recipe_path)
        patch = _document(self.patch_path)
        self.assertEqual(len(archive), 30_720)
        self.assertEqual(digest, "7f2796adb97fa3a55e3ff698dca03f35dbe530b2ba540498ae0ef7857580270b")
        self.assertEqual(recipe["build"]["context"]["sha256"], digest)
        self.assertEqual(patch["source_bundle"]["sha256"], digest)
        dockerfile = (self.adapter / "Dockerfile").read_text(encoding="utf-8")
        wrapper = (self.adapter / "vllm-wrapper.py").read_text(encoding="utf-8")
        self.assertIn("@sha256:9bb1557a4234fce63d59599e44d10747eabd742beb337eebf9e7070be8a0fd58", dockerfile)
        self.assertIn("493cb88fc69f8ba73ac87404f429d763e2739d89", dockerfile)
        self.assertIn("python3 /opt/glm53/patch_kpool_tail_slotmap.py", dockerfile)
        self.assertIn("python3 /opt/glm53/test_kpool_tail_slotmap.py", dockerfile)
        self.assertNotIn("ssh", wrapper.lower())
        self.assertIn('os.environ["VLLM_HOST_IP"] = local_address', wrapper)
        compile(wrapper, str(self.adapter / "vllm-wrapper.py"), "exec")

    def test_rank_profiles_own_their_gid_and_kpool_patch_is_fail_closed(self) -> None:
        runtime = _document(self.runtime_path)
        profiles = runtime["capabilities"]["distributed_vllm"]["launch"]["rank_profiles"]
        self.assertEqual([profile["rank"] for profile in profiles], [0, 1])
        self.assertTrue(all(profile["environment"]["NCCL_IB_GID_INDEX"] == "3" for profile in profiles))
        patch = (self.adapter / "patch_kpool_tail_slotmap.py").read_text(encoding="utf-8")
        self.assertIn("tl.minimum(block_indices, block_table_stride - 1)", patch)
        self.assertIn("pinned block_table slot-mapping anchor drifted", patch)

    def test_wrapper_accepts_controller_mounts_and_executes_exact_profile(self) -> None:
        recipe = _document(self.recipe_path)
        arguments = recipe["runtime"]["arguments"]
        command = [*recipe["runtime"]["entrypoint"]]
        for item in arguments:
            name = f"--{item['name']}"
            value = item["value"]
            if value is True:
                command.append(name)
            elif value is not False:
                command.extend((name, str(value)))
        command.extend((
            "--distributed-executor-backend", "mp",
            "--nnodes", "2",
            "--node-rank", "0",
        ))
        environment = {
            "VONK_LOCAL_ADDR": "10.0.0.1",
            "VONK_MASTER_ADDR": "10.0.0.1",
            "VONK_MASTER_PORT": "29500",
            "NCCL_SOCKET_IFNAME": "enp1s0f0np0,enp1s0f1np1",
            "NCCL_IB_HCA": "mlx5_0,mlx5_1",
            "NCCL_IB_GID_INDEX": "3",
            "TP_SOCKET_IFNAME": "enp1s0f0np0,enp1s0f1np1",
            "GLOO_SOCKET_IFNAME": "enp1s0f0np0,enp1s0f1np1",
        }
        captured: list[object] = []

        def capture_exec(*call: object) -> None:
            captured.extend(call)
            raise RuntimeError("exec captured")

        with (
            mock.patch.object(sys, "argv", command),
            mock.patch.dict(os.environ, environment, clear=True),
            mock.patch("pathlib.Path.is_file", return_value=True),
            mock.patch("os.access", return_value=True),
            mock.patch("os.execv", side_effect=capture_exec),
            self.assertRaisesRegex(RuntimeError, "exec captured"),
        ):
            runpy.run_path(str(self.adapter / "vllm-wrapper.py"), run_name="__main__")

        executed = captured[1]
        self.assertIn("/models/target", executed)
        self.assertIn('/models/drafter', next(value for value in executed if '"method":"dflash"' in value))
        self.assertEqual(executed[executed.index("--quantization") + 1], "exl3")
        self.assertEqual(
            executed[executed.index("--chat-template") + 1],
            "/opt/glm53/chat_template.jinja",
        )
        self.assertEqual(executed[-4:], ("--master-addr", "10.0.0.1", "--master-port", "29500"))

    def test_release_tracks_exact_recipe_digest(self) -> None:
        release = _document(self.release_path)
        self.assertEqual(release["version"], "1.2.0")
        self.assertEqual(release["history"][0]["recipe_content_sha256"], _digest(self.recipe_path))


if __name__ == "__main__":
    unittest.main()
