from __future__ import annotations

import hashlib
import importlib.machinery
import importlib.util
import json
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOCKERFILE = ROOT / "adapters/llm/laguna-vllm/Dockerfile"
RECIPE = ROOT / "recipes/laguna-xs-2-1-nvfp4-vllm-single.json"
RELEASE = ROOT / "recipe-releases/laguna-xs-2-1-nvfp4-vllm-single.json"
CACHE_ENVIRONMENT = {
    "HOME": "/outputs/cache/home",
    "XDG_CACHE_HOME": "/outputs/cache/xdg",
    "XDG_CONFIG_HOME": "/outputs/cache/config",
    "HF_HOME": "/outputs/cache/huggingface",
    "VLLM_CACHE_ROOT": "/outputs/cache/vllm",
    "FLASHINFER_WORKSPACE_BASE": "/outputs/cache/flashinfer",
    "VLLM_FLASHINFER_AUTOTUNE_CACHE_DIR": "/outputs/cache/flashinfer/autotune",
    "TRITON_CACHE_DIR": "/outputs/cache/triton",
    "TORCH_HOME": "/outputs/cache/torch",
    "TORCH_EXTENSIONS_DIR": "/outputs/cache/torch_extensions",
    "TORCHINDUCTOR_CACHE_DIR": "/outputs/cache/torchinductor",
    "CUDA_CACHE_PATH": "/outputs/cache/cuda",
    "UV_CACHE_DIR": "/outputs/cache/uv",
}


def docker_environment(source: str) -> dict[str, str]:
    assignments: dict[str, str] = {}
    for raw_line in source.splitlines():
        line = raw_line.strip().removesuffix("\\").strip()
        if line.startswith("ENV "):
            line = line.removeprefix("ENV ")
        if "=" not in line:
            continue
        name, value = line.split("=", 1)
        if name in CACHE_ENVIRONMENT:
            assignments[name] = value
    return assignments


def catalog_index_module():
    loader = importlib.machinery.SourceFileLoader(
        "laguna_catalog_index", str(ROOT / "tools/build-catalog-index")
    )
    spec = importlib.util.spec_from_loader(loader.name, loader)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    loader.exec_module(module)
    return module


class LagunaVllmAdapterRuntimeTests(unittest.TestCase):
    def test_non_root_runtime_survives_a_read_only_root_filesystem(self) -> None:
        source = DOCKERFILE.read_text(encoding="utf-8")
        recipe = json.loads(RECIPE.read_text(encoding="utf-8"))

        self.assertIn("useradd --uid 10001 --gid 10001", source)
        self.assertIn(
            "install --directory --owner=10001 --group=10001 /outputs /outputs/cache",
            source,
        )
        self.assertIn("'set -eu'", source)
        self.assertIn("'mkdir -p ", source)
        for name in CACHE_ENVIRONMENT:
            self.assertIn(f'"${name}"', source)
        self.assertIn("WORKDIR /tmp", source)
        self.assertIn("USER 10001:10001", source)

        mounts = {
            (item["target"], item["read_only"])
            for item in recipe["runtime"]["security"]["mounts"]
        }
        self.assertIn(("/outputs", False), mounts)
        self.assertIn(("/models", True), mounts)
        self.assertEqual(recipe["runtime"]["security"]["user"], "10001:10001")

    def test_every_persistent_runtime_cache_uses_the_writable_output_mount(
        self,
    ) -> None:
        assignments = docker_environment(DOCKERFILE.read_text(encoding="utf-8"))
        self.assertEqual(assignments, CACHE_ENVIRONMENT)
        self.assertTrue(
            all(path.startswith("/outputs/cache/") for path in assignments.values())
        )

    def test_recipe_and_release_bind_the_hardened_source_bundle(self) -> None:
        recipe = json.loads(RECIPE.read_text(encoding="utf-8"))
        release = json.loads(RELEASE.read_text(encoding="utf-8"))
        context = recipe["build"]["context"]
        archive, _, digest = catalog_index_module().source_bundle(
            ROOT / context["path"]
        )
        self.assertEqual(context["sha256"], digest)
        self.assertEqual(context["expected_bytes"], len(archive))

        canonical = json.dumps(
            recipe,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        recipe_digest = hashlib.sha256(canonical).hexdigest()
        self.assertEqual(release["version"], "1.0.2")
        self.assertEqual(release["history"][0]["upgrade_effect"], "rebuild")
        self.assertEqual(release["history"][0]["recipe_content_sha256"], recipe_digest)


if __name__ == "__main__":
    unittest.main()
