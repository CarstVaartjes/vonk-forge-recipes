import hashlib
import importlib.util
import json
from pathlib import Path
import struct
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
ADAPTER = ROOT / "adapters" / "media" / "comfyui-core"
RECIPE_SLUGS = (
    "flux-2-klein-4b-comfyui-single",
    "qwen-image-2512-comfyui-single",
    "qwen-image-edit-2511-comfyui-single",
    "wan-2-2-i2v-14b-comfyui-single",
    "wan-2-2-t2v-14b-comfyui-single",
    "wan-2-2-ti2v-5b-comfyui-single",
)
CORE_NODES = {
    "CFGGuider",
    "CFGNorm",
    "CLIPLoader",
    "CLIPTextEncode",
    "ConditioningZeroOut",
    "CreateVideo",
    "DiffusersLoader",
    "EmptyFlux2LatentImage",
    "EmptyHunyuanLatentVideo",
    "EmptySD3LatentImage",
    "Flux2Scheduler",
    "FluxKontextImageScale",
    "FluxKontextMultiReferenceLatentMethod",
    "KSampler",
    "KSamplerAdvanced",
    "KSamplerSelect",
    "LoadImage",
    "ModelSamplingAuraFlow",
    "ModelSamplingSD3",
    "RandomNoise",
    "SamplerCustomAdvanced",
    "SaveImage",
    "SaveVideo",
    "TextEncodeQwenImageEditPlus",
    "UNETLoader",
    "VAEDecode",
    "VAEEncode",
    "VAELoader",
    "Wan22ImageToVideoLatent",
    "WanImageToVideo",
}


class ComfyUICoreRecipeTests(unittest.TestCase):
    def test_all_recipes_bind_hash_locked_core_workflows(self) -> None:
        for slug in RECIPE_SLUGS:
            with self.subTest(slug=slug):
                recipe = json.loads((ROOT / "recipes" / f"{slug}.json").read_text())
                arguments = {
                    item["name"]: item["value"] for item in recipe["runtime"]["arguments"]
                }
                workflow = ADAPTER / Path(arguments["workflow"]).name
                if not workflow.exists():
                    workflow = ADAPTER / "workflows" / Path(arguments["workflow"]).name
                self.assertEqual(
                    hashlib.sha256(workflow.read_bytes()).hexdigest(),
                    arguments["workflow-sha256"],
                )
                document = json.loads(workflow.read_text())
                node_types = {node["class_type"] for node in document["prompt"].values()}
                self.assertLessEqual(node_types, CORE_NODES)
                self.assertTrue(node_types & {"SaveImage", "SaveVideo"})
                artifact_ids = {artifact["id"] for artifact in recipe["artifacts"]}
                workflow_ids = {
                    item["artifact_id"] for item in document.get("models", [])
                } | {
                    item["artifact_id"]
                    for item in document.get("diffusers_snapshots", [])
                } | {
                    item["artifact_id"] for item in document.get("snapshot_models", [])
                }
                self.assertEqual(artifact_ids, workflow_ids)

    def test_streaming_safetensors_merger_preserves_names_and_bytes(self) -> None:
        spec = importlib.util.spec_from_file_location("comfyui_job", ADAPTER / "comfyui_job.py")
        assert spec and spec.loader
        module = importlib.util.module_from_spec(spec)
        previous = sys.dont_write_bytecode
        sys.dont_write_bytecode = True
        try:
            spec.loader.exec_module(module)
        finally:
            sys.dont_write_bytecode = previous

        def write_shard(path: Path, name: str, payload: bytes) -> None:
            header = json.dumps(
                {name: {"dtype": "U8", "shape": [len(payload)], "data_offsets": [0, len(payload)]}},
                separators=(",", ":"),
            ).encode()
            header += b" " * ((8 - len(header) % 8) % 8)
            path.write_bytes(struct.pack("<Q", len(header)) + header + payload)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first, second, merged = root / "a.safetensors", root / "b.safetensors", root / "merged.safetensors"
            write_shard(first, "alpha", b"abc")
            write_shard(second, "beta", b"12345")
            module.merge_safetensors([first, second], merged)
            header_length = struct.unpack("<Q", merged.read_bytes()[:8])[0]
            header = json.loads(merged.read_bytes()[8 : 8 + header_length])
            self.assertEqual(header["alpha"]["data_offsets"], [0, 3])
            self.assertEqual(header["beta"]["data_offsets"], [3, 8])
            self.assertEqual(merged.read_bytes()[8 + header_length :], b"abc12345")

    def test_runtime_is_pinned_and_has_no_custom_node_supply_chain(self) -> None:
        dockerfile = (ADAPTER / "Dockerfile").read_text()
        self.assertIn("7a131a3afadc8200120f67f9236311a2c48b7445", dockerfile)
        self.assertIn("7e123716ae698194b3ded7ecbd8028b792d9015ce56d2318ebf4b8066efc6016", dockerfile)
        self.assertNotIn("ComfyUI-Manager", dockerfile)
        self.assertNotIn("custom_nodes", dockerfile)
        for slug in RECIPE_SLUGS:
            recipe = json.loads((ROOT / "recipes" / f"{slug}.json").read_text())
            self.assertIn("candidate", recipe["metadata"]["tags"])
            self.assertEqual(recipe["execution"]["harness"]["slug"], "comfyui")
            self.assertFalse(recipe["runtime"]["security"]["host_network"])


if __name__ == "__main__":
    unittest.main()
