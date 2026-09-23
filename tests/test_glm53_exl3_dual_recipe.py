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
        arguments = {
            item["name"]: item for item in load(RECIPE)["runtime"]["arguments"]
        }
        specification = json.loads(arguments["speculative-config"]["value"])
        self.assertEqual(specification["model"], "/models/drafter")
        self.assertEqual(specification["num_speculative_tokens"], 7)
        self.assertEqual(
            load(RECIPE)["topology"]["start_order"], ["worker", "entrypoint"]
        )

    def test_current_profile_fits_idle_sparks_with_controller_headroom(self) -> None:
        recipe = load(RECIPE)
        # Both physical Sparks reported over 126 GB MemAvailable on 2026-09-13.
        # Keep the Controller's 4 GB floor; the former 0.87 budget blocked the
        # current 0.85 launch even on these otherwise idle nodes.
        available_bytes = 126_000_000_000
        controller_floor_bytes = 4_000_000_000
        utilization = float(
            next(
                item["value"]
                for item in recipe["runtime"]["arguments"]
                if item["name"] == "gpu-memory-utilization"
            )
        )
        # Upstream documents roughly 9 GiB of initialization outside vLLM's
        # utilization budget. Admission must also cover that lower bound.
        estimated_startup_bytes = 130_663_235_584 * utilization + 9 * 1024**3
        for role in recipe["topology"]["roles"]:
            memory = role["resources"]["memory"]
            demand = max(
                memory["startup_peak_bytes"],
                memory["steady_state_bytes"] + memory["runtime_growth_bytes"],
            )
            self.assertGreaterEqual(
                memory["startup_peak_bytes"], estimated_startup_bytes
            )
            self.assertGreaterEqual(
                available_bytes - demand,
                max(controller_floor_bytes, memory["system_reserve_bytes"]),
            )

    def test_runtime_tracks_current_upstream_defaults(self) -> None:
        recipe = load(RECIPE)
        arguments = {item["name"]: item for item in recipe["runtime"]["arguments"]}
        self.assertEqual(
            recipe["provenance"]["source_reference"],
            "https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks/tree/bc68f310f8d5e941227ce5c93e95bca43b50fd6c",
        )
        self.assertEqual(
            recipe["execution"]["build"]["base_image"]["digest"],
            "905c02933be6021301db2dc284e24e3727467aa3a0f63b41d609885778a07bce",
        )
        # The declared envelope is sized so the node can actually cover it: the agent
        # requires MemAvailable >= reserved + 4 GB and the Sparks report ~125.5 GB.
        self.assertEqual(arguments["gpu-memory-utilization"]["value"], "0.84")
        # The engine measures its own ceiling: at gpu-memory-utilization 0.84 it
        # offers 9.05 GiB of KV and estimates a maximum of 225792 tokens, so the
        # declared context is the 192k that budget actually serves.
        self.assertEqual(recipe["settings"]["context_tokens"]["value"], 196608)
        self.assertEqual(arguments["max-num-batched-tokens"]["value"], 7168)
        self.assertEqual(arguments["kv-cache-dtype"]["value"], "fp8")
        self.assertEqual(arguments["quantization"]["value"], "exl3")
        # vLLM spells this --no-enable-flashinfer-autotune; the earlier
        # disable-flashinfer-autotune was not a real flag and must not return.
        self.assertIn("no-enable-flashinfer-autotune", arguments)
        self.assertNotIn("disable-flashinfer-autotune", arguments)
        # The Dockerfile installs instanttensor before this is declared.
        self.assertEqual(arguments["load-format"]["value"], "instanttensor")
        self.assertEqual(arguments["mm-processor-cache-gb"]["value"], 1)
        self.assertEqual(
            json.loads(arguments["mm-processor-kwargs"]["value"]),
            {"max_image_tokens": 2048},
        )
        self.assertEqual(
            arguments["chat-template"]["value"], "/opt/glm53/chat_template.jinja"
        )
        self.assertEqual(
            json.loads(arguments["compilation-config"]["value"])[
                "cudagraph_capture_sizes"
            ],
            [1, 2, 4, 8, 16, 24, 32],
        )
        self.assertEqual(
            recipe["models"][0]["model"]["slug"],
            "glm-5-3-flash-exl3-tr3-4bpw-dflash2-25a44fdb",
        )

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
        with (
            patch.object(sys, "argv", [str(ADAPTER / "vllm-wrapper.py"), *original]),
            patch.dict(os.environ, env, clear=False),
            patch("pathlib.Path.is_file", return_value=True),
            patch("os.access", return_value=True),
            patch("os.execv", side_effect=fake_execv),
            self.assertRaisesRegex(RuntimeError, "captured"),
        ):
            runpy.run_path(str(ADAPTER / "vllm-wrapper.py"))

        self.assertEqual(len(captured), 1)
        self.assertEqual(captured[0][1 : 1 + len(original)], tuple(original))
        self.assertEqual(
            captured[0][-4:], ("--master-addr", "10.0.0.2", "--master-port", "29500")
        )

    def test_current_source_build_closure_is_vendored_and_uses_no_ssh(self) -> None:
        dockerfile = (ADAPTER / "Dockerfile").read_text()
        self.assertIn(
            "FROM docker.io/vllm/vllm-openai@sha256:905c02933be6021301db2dc284e24e3727467aa3a0f63b41d609885778a07bce",
            dockerfile,
        )
        self.assertIn("COPY overlay/exl3.py", dockerfile)
        self.assertIn("COPY files/chat_template.jinja", dockerfile)
        self.assertTrue((ADAPTER / "upstream-LICENSE").is_file())
        self.assertTrue((ADAPTER / "upstream-LICENSE-MIT").is_file())
        self.assertTrue((ADAPTER / "upstream-start.sh").is_file())
        self.assertIn("overlay/exl3_fat_moe.cu", dockerfile)
        self.assertIn("patch_exl3_fat_kernel.py", dockerfile)
        self.assertNotIn("build_exl3_fat_moe_ext.py", dockerfile)
        self.assertIn("instanttensor==0.2.0", dockerfile)
        self.assertIn("patch_adaptive_k.py", dockerfile)
        self.assertIn("patch_dense_fp8.py", dockerfile)
        self.assertIn("EXL3_FAT_GROUPED=1", dockerfile)
        self.assertIn("COPY overlay/patch_mamba_align_chunking.py", dockerfile)
        self.assertIn("COPY overlay/patch_mamba_align_state_free.py", dockerfile)
        self.assertIn("COPY overlay/patch_tool_choice_none.py", dockerfile)
        self.assertLess(
            dockerfile.index("RUN python3 /opt/glm53/patch_scheduler_decode_floor.py"),
            dockerfile.index("RUN python3 /opt/glm53/patch_mamba_align_chunking.py"),
        )
        self.assertIn(
            "RUN python3 /opt/glm53/patch_mamba_align_state_free.py", dockerfile
        )
        self.assertIn("RUN python3 /opt/glm53/patch_tool_choice_none.py", dockerfile)
        self.assertIn("python3 /opt/glm53/test_mamba_align_chunking.py", dockerfile)
        self.assertIn("python3 /opt/glm53/test_tool_choice_none.py", dockerfile)
        verifier = (ADAPTER / "verify-runtime.py").read_text()
        self.assertIn("# [glm53-mamba-align-chunking-v1]", verifier)
        self.assertIn("# [glm53-mamba-align-state-free-v1]", verifier)
        self.assertIn("# [glm53-tool-choice-none]", verifier)
        text = "\n".join(
            (ADAPTER / name).read_text(errors="ignore")
            for name in ("Dockerfile", "vllm-wrapper.py", "verify-runtime.py")
        )
        self.assertNotIn("ssh -", text.lower())

    def test_current_release_backports_only_the_selected_upstream_fixes(self) -> None:
        recipe = load(RECIPE)
        self.assertEqual(recipe["release"]["version"], "1.6.7")
        self.assertEqual(recipe["release"]["history"][0]["upgrade_effect"], "rebuild")
        details = recipe["release"]["history"][0]["changes"][0]
        self.assertIn("Mamba", details["summary"])
        self.assertEqual(
            set(details["references"]),
            {
                "https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks/commit/fd329d248e63de3ddd334c7d2ed3276e8321672b",
                "https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks/commit/771a1867bc862fae5fa8946f85fb66749468679e",
            },
        )
        # These targeted overlays are built on the existing exact lineage and
        # runtime image; they do not claim to consume either upstream repository
        # wholesale or silently float the platform.
        self.assertEqual(
            recipe["provenance"]["source_reference"],
            "https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks/tree/bc68f310f8d5e941227ce5c93e95bca43b50fd6c",
        )
        self.assertEqual(
            recipe["execution"]["build"]["base_image"]["digest"],
            "905c02933be6021301db2dc284e24e3727467aa3a0f63b41d609885778a07bce",
        )

    def test_current_defaults_and_source_license_are_declared(self) -> None:
        recipe = load(RECIPE)
        environment = {
            item["name"]: item["value"] for item in recipe["runtime"]["environment"]
        }
        self.assertEqual(environment["EXL3_FAT_GROUPED"], "1")
        self.assertEqual(environment["EXL3_FAT_KERNEL"], "1")
        self.assertEqual(environment["EXL3_TEMP_ROWS_FUSED"], "32")
        self.assertEqual(environment["GLM53_INDEXER_WORKSPACE"], "rightsize")
        self.assertEqual(environment["GLM53_ADAPTIVE_K"], "off")
        self.assertEqual(environment["GLM53_DENSE_FP8"], "off")
        # Upstream main defaults the mixed-prefill policy to fair; the
        # GLM53_FAIR_PREFILL_* legs only apply under that policy.
        self.assertEqual(environment["GLM53_MIXED_PREFILL_CHUNK"], "fair")
        self.assertEqual(environment["GLM53_APC_NO_STORE"], "1")
        self.assertEqual(environment["GLM53_KV_CAPACITY_LOG"], "1")
        self.assertEqual(environment["GLM53_SPINWAIT_MS"], "stock")
        self.assertIn("AGPL-3.0", recipe["metadata"]["description"])
        self.assertIn("AGPL-3.0", recipe["provenance"]["attribution"][-4])

    def test_the_overlay_that_disables_persistent_topk_is_applied_at_build_time(
        self,
    ) -> None:
        # The workload runs with a read-only root, so the one overlay that
        # rewrites an installed vLLM file cannot be applied at run time the way
        # GLM53_OVERLAY_ORDER applies it upstream. A live worker failure named the
        # guard it disables: "launch_persistent_topk, topk.cu:138,
        # persistent_topk would oversubscribe and the FilteredTopK fallback
        # requires >=128KB smem per block (have 101376). total_ctas=73 >
        # num_sms*occupancy=48".
        dockerfile = (ADAPTER / "Dockerfile").read_text()
        self.assertIn("COPY overlay/patch_glm_video_placeholders.py", dockerfile)
        applied = "RUN python3 /opt/glm53/patch_glm_video_placeholders.py"
        self.assertIn(applied, dockerfile)
        # It rewrites the same file the tail-slotmap overlay does, and
        # GLM53_OVERLAY_ORDER runs it before that one.
        self.assertLess(
            dockerfile.index(applied),
            dockerfile.index("RUN python3 /opt/glm53/patch_kpool_tail_slotmap.py"),
        )
        # The build proves the guard is disabled rather than trusting the RUN.
        verify = (ADAPTER / "verify-runtime.py").read_text()
        self.assertIn('"persistent_topk" not in kpool', verify)
        self.assertIn('"if False and current_platform.is_cuda()" not in kpool', verify)

    def test_the_loader_budget_cannot_refuse_a_buffer_the_node_can_allocate(
        self,
    ) -> None:
        # instanttensor refuses its staging buffer when it exceeds
        # torch.cuda.mem_get_info().free times this fraction, and on a
        # unified-memory Spark that reading excludes the reclaimable page cache of
        # the model the platform just copied: a live cycle reported about 950 MB
        # free on a node reporting 126 GB available and refused the 1,268,776,960 B
        # buffer the checkpoint's largest tensor needs. The fraction must admit a
        # buffer larger than the reading, which keeps the estimate a warning and
        # leaves the allocator the owner of whether the allocation succeeds.
        recipe = load(RECIPE)
        environment = {
            item["name"]: item["value"] for item in recipe["runtime"]["environment"]
        }
        self.assertGreater(float(environment["INSTANTTENSOR_MAX_FREE_MEM_USAGE"]), 1.0)

    def test_adapter_bundle_is_pinned(self) -> None:
        import runpy

        tool = runpy.run_path(str(ROOT / "tools/build-catalog-index"))
        _, _, digest = tool["source_bundle"](ADAPTER)
        self.assertTrue(digest)


if __name__ == "__main__":
    unittest.main()
