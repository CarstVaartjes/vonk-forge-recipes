from __future__ import annotations

import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def document(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


class BuildNetworkHostMetadataTests(unittest.TestCase):
    def test_nvcr_is_not_declared_only_for_a_base_image(self) -> None:
        for recipe_path in sorted((ROOT / "recipes").glob("*.json")):
            recipe = document(recipe_path)
            hosts = recipe["build"]["network"]["hosts"]
            if "nvcr.io" not in hosts:
                continue
            dockerfile = (
                ROOT
                / recipe["build"]["context"]["path"]
                / recipe["build"]["dockerfile"]
            )
            build_steps = "\n".join(
                line
                for line in dockerfile.read_text(encoding="utf-8").splitlines()
                if not line.lstrip().upper().startswith("FROM ")
            )
            self.assertIn(
                "https://nvcr.io",
                build_steps,
                f"{recipe_path.name} declares base-image-only nvcr.io egress",
            )

    def test_native_runtime_host_lists_exclude_base_image_registry(self) -> None:
        for name in (
            "moss-vl-realtime-native-arm64.json",
            "mova-native-arm64.json",
        ):
            runtime = document(ROOT / "runtime-distributions" / name)
            self.assertNotIn("nvcr.io", runtime["build"]["network_hosts"])

    def test_redirect_and_security_hosts_are_explicit(self) -> None:
        ltx = document(
            ROOT / "recipes/ltx-2-19b-dev-fp4-pytorch-single.json"
        )
        self.assertTrue(
            {"codeload.github.com", "download-r2.pytorch.org"}.issubset(
                ltx["build"]["network"]["hosts"]
            )
        )
        for slug in (
            "step1x-3d-geometry-pytorch-single",
            "step1x-3d-label-geometry-pytorch-single",
            "step1x-3d-texture-pytorch-single",
        ):
            recipe = document(ROOT / "recipes" / f"{slug}.json")
            self.assertIn(
                "security.ubuntu.com", recipe["build"]["network"]["hosts"]
            )

    def test_dual_spark_build_hosts_match_outbound_build_steps(self) -> None:
        expected = {
            "glm-5-3-flash-exl3-dflash2-vllm-dual": ("none", []),
            "glm-5-3-flash-nvfp4-kv-1m-abliterated-vllm-dual": ("none", []),
            "glm-5-3-flash-nvfp4-vllm-dual": (
                "public",
                ["files.pythonhosted.org", "flashinfer.ai", "pypi.org"],
            ),
            "qwen3-8-flash-next-nvfp4-sglang-dual": (
                "public",
                ["files.pythonhosted.org"],
            ),
        }
        for slug, (mode, hosts) in expected.items():
            with self.subTest(slug=slug):
                recipe = document(ROOT / "recipes" / f"{slug}.json")
                self.assertEqual(
                    recipe["build"]["network"],
                    {"mode": mode, "hosts": hosts},
                )

        runtime_hosts = {
            "glm-5-3-flash-nvfp4-ray-dual.json": expected[
                "glm-5-3-flash-nvfp4-vllm-dual"
            ][1],
            "vllm-glm53-exl3-dflash2-mia-493cb88f-arm64.json": [],
            "vllm-glm53-drowzeys-1m-c36b5958-arm64.json": [],
            "sglang-qwen38-flash-next-dspark-0f950012-arm64.json": expected[
                "qwen3-8-flash-next-nvfp4-sglang-dual"
            ][1],
        }
        for name, hosts in runtime_hosts.items():
            with self.subTest(runtime=name):
                runtime = document(ROOT / "runtime-distributions" / name)
                self.assertEqual(runtime["build"]["network_hosts"], hosts)


if __name__ == "__main__":
    unittest.main()
