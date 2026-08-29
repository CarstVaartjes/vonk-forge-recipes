from __future__ import annotations

import hashlib
import json
import runpy
import subprocess
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _document(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _canonical_digest(path: Path) -> str:
    payload = json.dumps(
        _document(path),
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def _arguments(recipe: dict[str, object]) -> dict[str, object]:
    runtime = recipe["runtime"]
    assert isinstance(runtime, dict)
    arguments = runtime["arguments"]
    assert isinstance(arguments, list)
    return {item["name"]: item["value"] for item in arguments}


class OriginAlignedProfileTests(unittest.TestCase):
    def test_drowzeys_glm53_1m_profile_and_gated_contract(self) -> None:
        recipe_path = ROOT / "recipes/glm-5-3-flash-nvfp4-kv-1m-abliterated-vllm-dual.json"
        runtime_path = ROOT / "runtime-distributions/vllm-glm53-drowzeys-1m-c36b5958-arm64.json"
        patch_path = ROOT / "patch-bundles/drowzeys-glm53-nvfp4-kv-1m-dual-profile.json"
        model_path = ROOT / "model-versions/glm-5-3-flash-nvfp4-abliterated-d7f8afa8.json"
        adapter = ROOT / "adapters/glm/drowzeys-glm53-1m"
        recipe = _document(recipe_path)
        runtime = _document(runtime_path)
        patch = _document(patch_path)
        model = _document(model_path)
        arguments = _arguments(recipe)

        self.assertEqual(arguments["max-model-len"], 1048576)
        self.assertEqual(arguments["block-size"], 7168)
        self.assertEqual(arguments["kv-cache-memory"], 6334808064)
        self.assertEqual(arguments["kv-cache-dtype"], "nvfp4_ds_mla")
        self.assertEqual(arguments["max-num-batched-tokens"], 4096)
        self.assertEqual(arguments["max-num-seqs"], 2)
        self.assertEqual(arguments["gpu-memory-utilization"], "0.85")
        self.assertEqual(
            json.loads(arguments["speculative-config"])["num_speculative_tokens"],
            2,
        )
        self.assertEqual(model["access"], {"visibility": "restricted", "gated": True, "authentication": "token"})
        self.assertEqual(runtime["source"]["revision"], "c36b5958412158a69629e7fbed321312e6d0761d")
        self.assertEqual(
            runtime["image"],
            "ghcr.io/drowzeys/keys-vllm-glm53-flash-nvfp4-ablit@sha256:"
            "f722ec19d8260833e948d5bf46949d9ac574841860060caa24213cf550d1a41b",
        )
        self.assertEqual(runtime["build"]["network_hosts"], ["ghcr.io"])
        self.assertEqual(recipe["runtime"]["distribution"]["content_sha256"], _canonical_digest(runtime_path))
        self.assertEqual(patch["applies_to"]["content_sha256"], _canonical_digest(runtime_path))
        self.assertEqual(recipe["execution"]["patch_bundle"]["content_sha256"], _canonical_digest(patch_path))

        source_bundle = runpy.run_path(str(ROOT / "tools/build-catalog-index"))["source_bundle"]
        archive, _files, digest = source_bundle(adapter)
        self.assertEqual(recipe["build"]["context"]["sha256"], digest)
        self.assertEqual(recipe["build"]["context"]["expected_bytes"], len(archive))
        self.assertEqual(patch["sha256"], digest)
        wrapper = (adapter / "vllm-wrapper.py").read_text(encoding="utf-8")
        self.assertIn('os.environ["KV_FP8_ROPE"] = "1"', wrapper)
        self.assertIn('("--attention-backend", "B12X_MLA_SPARSE")', wrapper)
        self.assertIn('text_config["index_topk"] = 2044', wrapper)
        compile(wrapper, str(adapter / "vllm-wrapper.py"), "exec")

    def test_qwen_current_1m_profile_and_closure(self) -> None:
        recipe_path = ROOT / "recipes/qwen3-8-flash-next-nvfp4-sglang-dual.json"
        runtime_path = ROOT / "runtime-distributions/sglang-qwen38-flash-next-dspark-344f9d0d-arm64.json"
        patch_path = ROOT / "patch-bundles/sglang-qwen38-flash-next-dual-profile.json"
        adapter = ROOT / "adapters/qwen/flash-next-sglang-dual"
        recipe = _document(recipe_path)
        runtime = _document(runtime_path)
        patch = _document(patch_path)
        arguments = _arguments(recipe)

        self.assertEqual(arguments["context-length"], 1048576)
        self.assertEqual(arguments["chunked-prefill-size"], 1024)
        self.assertEqual(arguments["max-running-requests"], 28)
        self.assertEqual(arguments["mem-fraction-static"], "0.82")
        self.assertEqual(runtime["source"]["revision"], "344f9d0d5e9523d8398fa2804d5a3e123fd3d21a")
        self.assertEqual(recipe["runtime"]["distribution"]["content_sha256"], _canonical_digest(runtime_path))
        self.assertEqual(patch["applies_to"]["content_sha256"], _canonical_digest(runtime_path))
        self.assertEqual(recipe["execution"]["patch_bundle"]["content_sha256"], _canonical_digest(patch_path))

        source_bundle = runpy.run_path(str(ROOT / "tools/build-catalog-index"))["source_bundle"]
        archive, _files, digest = source_bundle(adapter)
        self.assertEqual(recipe["build"]["context"]["sha256"], digest)
        self.assertEqual(recipe["build"]["context"]["expected_bytes"], len(archive))
        self.assertEqual(patch["sha256"], digest)
        wrapper = (adapter / "sglang-serve").read_text(encoding="utf-8")
        self.assertIn('("--kv-cache-dtype", "nvfp4")', wrapper)
        self.assertIn('"--max-prefill-tokens", "2048"', wrapper)
        self.assertIn('"original_max_position_embeddings":262144', wrapper)
        self.assertIn('os.environ["SGLANG_HOST_IP"] = local_address', wrapper)
        qsa_patch = (adapter / "patch-qsa.py").read_text(encoding="utf-8")
        self.assertIn("SM121 must not use TRT-LLM sparse decode", qsa_patch)
        self.assertIn("qsa.sm121_varlen", qsa_patch)
        fallback = (adapter / "sm121_varlen.py").read_text(encoding="utf-8")
        self.assertIn("qsa_sm121_varlen_attention", fallback)
        compile(wrapper, str(adapter / "sglang-serve"), "exec")
        compile(qsa_patch, str(adapter / "patch-qsa.py"), "exec")
        compile(fallback, str(adapter / "sm121_varlen.py"), "exec")

    def test_mia_deepseek_384k_profile_packages_exact_overlays(self) -> None:
        recipe_path = ROOT / "recipes/deepseek-v4-flash-0731-mia-sparkinfer-single.json"
        runtime_path = ROOT / "runtime-distributions/sparkinfer-dsv4-mia-fdcd538f-single.json"
        patch_path = ROOT / "patch-bundles/sparkinfer-dsv4-mia-384k-single-profile.json"
        adapter = ROOT / "adapters/deepseek/mia-sparkinfer-single"
        recipe = _document(recipe_path)
        runtime = _document(runtime_path)
        patch = _document(patch_path)
        arguments = _arguments(recipe)

        self.assertEqual(arguments["max-model-len"], 384000)
        self.assertEqual(arguments["max-num-seqs"], 1)
        self.assertEqual(arguments["max-cudagraph-capture-size"], 24)
        self.assertEqual(arguments["gpu-memory-utilization"], "0.94")
        self.assertEqual(runtime["source"]["revision"], "fdcd538fbf95fb15b2d6850db9613d22b2c889b8")
        self.assertEqual(recipe["runtime"]["distribution"]["content_sha256"], _canonical_digest(runtime_path))
        self.assertEqual(recipe["execution"]["patch_bundle"]["content_sha256"], _canonical_digest(patch_path))

        source_bundle = runpy.run_path(str(ROOT / "tools/build-catalog-index"))["source_bundle"]
        archive, _files, digest = source_bundle(adapter)
        self.assertEqual(recipe["build"]["context"]["sha256"], digest)
        self.assertEqual(recipe["build"]["context"]["expected_bytes"], len(archive))
        self.assertEqual(patch["sha256"], digest)

        expected_hashes = {
            "coalesce_rank_sliced_exl3.py": "c0c8e28901c4a1c65cf2865d071868bb9f789d4deffbf97f019a5e0b3213c2da",
            "serve-ds4-flash.sh": "0b043f923e6e3a8cbc2f589714c0c4c442cba2701b4bf2111556a6c4999ce4dc",
            "sparkinfer/moe/fused_moe/_impl.py": "d7de3129a6d76ad7b345626644c7b4dae339b285f0091453f7135ddd1d546417",
            "sparkinfer/moe/_shared/kernels/tiny_decode.py": "7b06bdd5cdf261ef9f13340338faf2edb1640df257b6f2459543a17c04a13382",
            "vllm/models/deepseek_v4/nvidia/dspark.py": "5d0545e8271170aab501214e978211e269c9357aa665b562f247567b9e9090de",
            "sparkinfer/attention/_shared/mla/prefill.py": "da97c473a71a1ceb5be414360cd4b6c58d4cbd6af3fa0c158ae6be96866bc166",
            "sparkinfer/attention/_shared/mla/prefill_mg.py": "632690a3f006887c17ecc2651cd801e3b0fe420abad02d5ca82ee0b6cca9f76f",
        }
        for relative, expected in expected_hashes.items():
            self.assertEqual(hashlib.sha256((adapter / "overlays" / relative).read_bytes()).hexdigest(), expected)
        subprocess.run(["bash", "-n", str(adapter / "vllm-wrapper.sh")], check=True)


if __name__ == "__main__":
    unittest.main()
