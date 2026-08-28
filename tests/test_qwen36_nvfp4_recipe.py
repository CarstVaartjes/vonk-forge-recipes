from __future__ import annotations

import importlib.machinery
import importlib.util
import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load(path: str) -> dict[str, object]:
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def catalog_index_module():
    loader = importlib.machinery.SourceFileLoader(
        "qwen36_catalog_index", str(ROOT / "tools/build-catalog-index")
    )
    spec = importlib.util.spec_from_loader(loader.name, loader)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    loader.exec_module(module)
    return module


class Qwen36Nvfp4RecipeTests(unittest.TestCase):
    def test_exact_model_runtime_and_spark_profile(self) -> None:
        version = load("model-versions/qwen3-6-35b-a3b-nvfp4-491c2f1e.json")
        runtime = load("runtime-distributions/vllm-0-28-0-nvidia-arm64.json")
        recipe = load("recipes/qwen3-6-35b-a3b-nvfp4-vllm-single.json")

        self.assertEqual(
            version["source"]["revision"],
            "491c2f1ea524c639598bf8fa787a93fed5a6fbce",
        )
        self.assertEqual(version["parameters"], {"total": 35_000_000_000, "active": 3_000_000_000})
        self.assertEqual(version["sizes"]["download_bytes"], 23_462_477_790)
        self.assertEqual(
            version["metadata"]["tags"],
            [
                "nvidia",
                "nvidia-qwen",
                "multimodal",
                "vision",
                "reasoning",
                "agentic",
                "moe",
                "nvfp4",
                "candidate",
            ],
        )
        self.assertEqual(
            sum(item["download_bytes"] for item in version["artifacts"]),
            23_462_477_790,
        )
        self.assertEqual(
            runtime["image"],
            "docker.io/vllm/vllm-openai@sha256:41b54fb42c66a670a8b27e613ebef05898f24b9ab1bdab28bd00c877bd4935f4",
        )
        self.assertEqual(runtime["dependencies"][0]["version"], "0.28.0")
        arguments = {item["name"]: item["value"] for item in recipe["runtime"]["arguments"]}
        self.assertEqual(arguments["gpu-memory-utilization"], "0.40")
        self.assertEqual(arguments["max-model-len"], 262_144)
        self.assertEqual(arguments["max-num-batched-tokens"], 8192)
        self.assertEqual(arguments["moe-backend"], "marlin")
        self.assertEqual(arguments["tool-call-parser"], "qwen3_xml")
        self.assertEqual(recipe["build"]["resources"]["download_bytes"], 33_163_973_513)

    def test_wrapper_injects_unsupported_official_spark_flags_and_writable_caches(self) -> None:
        wrapper = ROOT / "adapters/nvidia/qwen36-35b-vllm/vllm"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            capture = root / "arguments.json"
            executable = root / "vllm"
            executable.write_text(
                "#!/bin/sh\nprintf '%s\\n' \"$@\" > \"$CAPTURE\"\n",
                encoding="utf-8",
            )
            executable.chmod(0o755)
            cache = root / "cache"
            environment = {
                **os.environ,
                "PATH": f"{root}:{os.environ['PATH']}",
                "CAPTURE": str(capture),
                "HOME": str(cache / "home"),
                "XDG_CACHE_HOME": str(cache / "xdg"),
                "XDG_CONFIG_HOME": str(cache / "config"),
                "HF_HOME": str(cache / "huggingface"),
                "VLLM_CACHE_ROOT": str(cache / "vllm"),
                "TRITON_CACHE_DIR": str(cache / "triton"),
                "TORCH_HOME": str(cache / "torch"),
                "TORCH_EXTENSIONS_DIR": str(cache / "torch_extensions"),
                "TORCHINDUCTOR_CACHE_DIR": str(cache / "torchinductor"),
                "CUDA_CACHE_PATH": str(cache / "cuda"),
                "UV_CACHE_DIR": str(cache / "uv"),
            }
            result = subprocess.run(
                ["/bin/sh", str(wrapper), "serve", "/models"],
                check=False,
                env=environment,
            )
            self.assertEqual(result.returncode, 0)
            arguments = capture.read_text(encoding="utf-8").splitlines()
            self.assertEqual(arguments[:2], ["serve", "/models"])
            self.assertEqual(
                arguments[-4:],
                ["--attention-backend", "flashinfer", "--load-format", "fastsafetensors"],
            )
            for path in environment.values():
                if isinstance(path, str) and path.startswith(str(cache)):
                    self.assertTrue(Path(path).is_dir())

    def test_source_bundle_matches_recipe_contract(self) -> None:
        recipe = load("recipes/qwen3-6-35b-a3b-nvfp4-vllm-single.json")
        context = recipe["build"]["context"]
        archive, _, digest = catalog_index_module().source_bundle(ROOT / context["path"])
        self.assertEqual(digest, context["sha256"])
        self.assertEqual(len(archive), context["expected_bytes"])


if __name__ == "__main__":
    unittest.main()
