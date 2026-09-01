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
LAGUNA_S_RECIPE = ROOT / "recipes/laguna-s-2-1-nvfp4-vllm-single.json"
LAGUNA_S_RELEASE = ROOT / "recipe-releases/laguna-s-2-1-nvfp4-vllm-single.json"
LAGUNA_S_MODEL = ROOT / "model-versions/laguna-s-2-1-nvfp4-826aacdf.json"
LAGUNA_S_DOCKERFILE = ROOT / "adapters/llm/laguna-s-vllm/Dockerfile"
LAGUNA_S_REVISION = "826aacdf6d8b2699d4e367def6f17c83b06044c2"
CURRENT_LAGUNA_S_REVISION = "826aacdf6d8b2699d4e367def6f17c83b06044c2"
LAGUNA_XS_REVISION = "d32afde8b09af1539b49ff96ff5551c674485f8e"
VLLM_REVISION = "6e448d0ea9bf3d88d898b65449ca6dc2aec170ac"
LANGUAGE_TARGETS = ROOT / "model-targets/language.json"
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


def canonical_digest(path: Path) -> str:
    document = json.loads(path.read_text(encoding="utf-8"))
    canonical = json.dumps(
        document,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


class LagunaVllmAdapterRuntimeTests(unittest.TestCase):
    def test_adapter_oci_labels_describe_baked_vllm_not_mounted_weights(self) -> None:
        for dockerfile in (DOCKERFILE, LAGUNA_S_DOCKERFILE):
            with self.subTest(adapter=dockerfile.parent.name):
                source = dockerfile.read_text(encoding="utf-8")
                self.assertIn(
                    'org.opencontainers.image.source="https://github.com/vllm-project/vllm"',
                    source,
                )
                self.assertIn(
                    f'org.opencontainers.image.revision="{VLLM_REVISION}"', source
                )
                self.assertIn('org.opencontainers.image.licenses="Apache-2.0"', source)
                self.assertNotIn("huggingface.co/poolside", source)
                self.assertNotIn(LAGUNA_S_REVISION, source)
                self.assertNotIn(LAGUNA_XS_REVISION, source)

    def test_model_provenance_remains_exact_outside_the_runtime_image(self) -> None:
        expected = {
            RECIPE: ("poolside/Laguna-XS-2.1-NVFP4", LAGUNA_XS_REVISION),
            LAGUNA_S_RECIPE: ("poolside/Laguna-S-2.1-NVFP4", LAGUNA_S_REVISION),
        }
        for recipe_path, (repository, revision) in expected.items():
            with self.subTest(recipe=recipe_path.name):
                recipe = json.loads(recipe_path.read_text(encoding="utf-8"))
                artifact = recipe["artifacts"][0]
                self.assertEqual(artifact["repository"], repository)
                self.assertEqual(artifact["revision"], revision)
                self.assertEqual(
                    recipe["provenance"]["source_reference"],
                    f"https://huggingface.co/{repository}/tree/{revision}",
                )

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
        self.assertEqual(release["version"], "1.0.3")
        self.assertEqual(release["history"][0]["upgrade_effect"], "rebuild")
        self.assertEqual(release["history"][0]["recipe_content_sha256"], recipe_digest)


class LagunaSVllmRecipeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.model = json.loads(LAGUNA_S_MODEL.read_text(encoding="utf-8"))
        self.recipe = json.loads(LAGUNA_S_RECIPE.read_text(encoding="utf-8"))
        self.release = json.loads(LAGUNA_S_RELEASE.read_text(encoding="utf-8"))

    def test_model_manifest_covers_the_complete_pinned_snapshot(self) -> None:
        expected_paths = {
            ".gitattributes",
            "LICENSE.md",
            "README.md",
            "chat_template.jinja",
            "config.json",
            "configuration_laguna.py",
            "generation_config.json",
            "model.safetensors.index.json",
            "modeling_laguna.py",
            "special_tokens_map.json",
            "tokenizer.json",
            "tokenizer_config.json",
            *(f"model-{index:05d}-of-00049.safetensors" for index in range(1, 50)),
        }
        artifacts = self.model["artifacts"]
        self.assertEqual({artifact["path"] for artifact in artifacts}, expected_paths)
        self.assertEqual(len(artifacts), 61)
        self.assertEqual(
            {artifact["revision"] for artifact in artifacts}, {LAGUNA_S_REVISION}
        )
        self.assertEqual(
            sum(artifact["download_bytes"] for artifact in artifacts), 99_717_279_420
        )
        self.assertEqual(
            self.model["sizes"],
            {"download_bytes": 99_717_279_420, "installed_bytes": 99_717_279_420},
        )
        self.assertEqual(self.model["limits"]["context_tokens"], 1_048_576)
        self.assertTrue(
            all(
                artifact["download_bytes"] == artifact["installed_bytes"]
                for artifact in artifacts
            )
        )

    def test_recipe_sizes_and_model_reference_match_the_manifest(self) -> None:
        snapshot = self.recipe["artifacts"][0]
        disk = self.recipe["topology"]["roles"][0]["resources"]["disk"]
        self.assertEqual(snapshot["revision"], LAGUNA_S_REVISION)
        self.assertEqual(snapshot["download_bytes"], 99_717_279_862)
        self.assertEqual(snapshot["installed_bytes"], 99_717_279_862)
        self.assertEqual(disk["artifact_bytes"], snapshot["installed_bytes"])
        self.assertEqual(disk["staging_bytes"], 2 * snapshot["download_bytes"])
        self.assertEqual(
            self.recipe["build"]["resources"]["temporary_bytes"],
            disk["staging_bytes"],
        )
        self.assertEqual(
            self.recipe["model"]["content_sha256"], canonical_digest(LAGUNA_S_MODEL)
        )

    def test_spark_jit_is_bounded_for_cold_start(self) -> None:
        source = LAGUNA_S_DOCKERFILE.read_text(encoding="utf-8")
        self.assertIn("CUTE_DSL_ARCH=sm_121a", source)
        self.assertIn("MAX_JOBS=4", source)
        self.assertEqual(docker_environment(source), CACHE_ENVIRONMENT)
        self.assertEqual(
            {item["name"] for item in self.recipe["runtime"]["environment"]},
            {"HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "VLLM_NO_USAGE_STATS"},
        )
        self.assertEqual(
            self.recipe["runtime"]["distribution"]["slug"],
            "vllm-0-27-1-nvidia-arm64",
        )

    def test_recipe_binds_the_laguna_s_adapter(self) -> None:
        context = self.recipe["build"]["context"]
        archive, _, digest = catalog_index_module().source_bundle(
            ROOT / context["path"]
        )
        self.assertEqual(context["path"], "adapters/llm/laguna-s-vllm")
        self.assertEqual(context["sha256"], digest)
        self.assertEqual(context["expected_bytes"], len(archive))

    def test_release_tracks_the_corrected_recipe(self) -> None:
        self.assertEqual(self.release["version"], "1.0.5")
        self.assertEqual(
            self.release["history"][0]["recipe_content_sha256"],
            canonical_digest(LAGUNA_S_RECIPE),
        )

    def test_current_upstream_is_documentation_only_not_a_new_model_payload(
        self,
    ) -> None:
        target_set = json.loads(LANGUAGE_TARGETS.read_text(encoding="utf-8"))
        target = next(
            item for item in target_set["targets"] if item["model"] == "Laguna S"
        )
        self.assertEqual(target["status"], "candidate")
        self.assertEqual(
            target["recipe_slugs"],
            [
                LAGUNA_S_RECIPE.stem,
                "laguna-s-2-1-nvfp4-vllm-low-memory-canary-single",
            ],
        )
        self.assertIn(CURRENT_LAGUNA_S_REVISION[:8], target["notes"])
        self.assertIn("README.md only", target["notes"])


if __name__ == "__main__":
    unittest.main()
