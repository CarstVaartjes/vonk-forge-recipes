from __future__ import annotations

import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ADAPTERS = (
    ROOT / "adapters/llm/vllm-openai/Dockerfile",
    ROOT / "adapters/llm/vllm-openai-028/Dockerfile",
    ROOT / "adapters/llm/qwen35-vllm/Dockerfile",
)
CACHE_ENVIRONMENT = {
    "HOME": "/outputs/cache/home",
    "XDG_CACHE_HOME": "/outputs/cache/xdg",
    "XDG_CONFIG_HOME": "/outputs/cache/config",
    "HF_HOME": "/outputs/cache/huggingface",
    "VLLM_CACHE_ROOT": "/outputs/cache/vllm",
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


class QwenVllmAdapterRuntimeTests(unittest.TestCase):
    def test_non_root_runtime_state_is_routed_to_the_writable_output_mount(self) -> None:
        for dockerfile in ADAPTERS:
            with self.subTest(adapter=dockerfile.parent.name):
                source = dockerfile.read_text(encoding="utf-8")
                self.assertIn("useradd --uid 10001 --gid 10001", source)
                self.assertIn(
                    "install --directory --owner=10001 --group=10001 "
                    "/outputs /outputs/cache",
                    source,
                )
                for name, path in CACHE_ENVIRONMENT.items():
                    self.assertIn(f"{name}={path}", source)
                    self.assertIn(f'"${name}"', source)
                self.assertIn("'set -eu'", source)
                self.assertIn("'mkdir -p ", source)
                self.assertIn("WORKDIR /tmp", source)
                self.assertIn("USER 10001:10001", source)

    def test_runtime_cache_paths_do_not_target_the_read_only_image(self) -> None:
        for dockerfile in ADAPTERS:
            with self.subTest(adapter=dockerfile.parent.name):
                source = dockerfile.read_text(encoding="utf-8")
                assignments = docker_environment(source)
                self.assertEqual(assignments, CACHE_ENVIRONMENT)
                self.assertTrue(
                    all(path.startswith("/outputs/cache/") for path in assignments.values())
                )


if __name__ == "__main__":
    unittest.main()
