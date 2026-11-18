"""The overlay builder of the vcruz305 GLM-5.3-Flash SAGE-REXL3 vLLM adapter declares what vllm-exl3 must look up."""

from __future__ import annotations

import json
import os
import struct
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ADAPTER = (
    Path(__file__).resolve().parents[1] / "adapters/glm/vcruz-sage-rexl3-vllm-single"
)
LAYER = "model.language_model.layers"


def trellis(k: int) -> dict:
    return {"dtype": "I16", "shape": [4, 4, 16 * k], "data_offsets": [0, 0]}


def marker() -> dict:
    return {"dtype": "I32", "shape": [], "data_offsets": [0, 0]}


def group(prefix: str, k: int) -> dict:
    return {f"{prefix}.trellis": trellis(k), f"{prefix}.mul1": marker()}


class OverlayTest(unittest.TestCase):
    def test_overlay_declares_the_lookups_the_creators_self_test_makes(self) -> None:
        tensors: dict = {}
        tensors.update(group(f"{LAYER}.0.self_attn.qkv_proj", 8))
        tensors.update(group(f"{LAYER}.0.self_attn.o_proj", 8))
        tensors.update(group(f"{LAYER}.3.self_attn.o_proj", 7))
        tensors.update(group(f"{LAYER}.3.self_attn.q_a_proj", 8))
        tensors.update(group(f"{LAYER}.3.mlp.shared_experts.up_proj", 6))
        for layer, k in ((3, 3), (4, 2), (5, 2)):
            for expert in (0, 1):
                tensors.update(
                    group(f"{LAYER}.{layer}.mlp.experts.{expert}.up_proj", k)
                )
        tensors[f"{LAYER}.0.self_attn.b_proj.weight"] = {
            "dtype": "F16",
            "shape": [4, 4],
            "data_offsets": [0, 0],
        }
        with tempfile.TemporaryDirectory() as root:
            pack, overlay = Path(root, "pack"), Path(root, "overlay")
            pack.mkdir()
            header = json.dumps(tensors).encode()
            (pack / "model-00001-of-00001.safetensors").write_bytes(
                struct.pack("<Q", len(header)) + header
            )
            index = {
                "weight_map": dict.fromkeys(tensors, "model-00001-of-00001.safetensors")
            }
            (pack / "model.safetensors.index.json").write_text(json.dumps(index))
            (pack / "config.json").write_text(
                json.dumps({"quantization_config": {"quant_method": "exl3"}})
            )
            subprocess.run(
                [
                    sys.executable,
                    "-I",
                    str(ADAPTER / "make_overlay.py"),
                    str(pack),
                    str(overlay),
                ],
                check=True,
                capture_output=True,
                env={"PATH": os.environ.get("PATH", "")},
            )
            quantization = json.loads((overlay / "config.json").read_text())[
                "quantization_config"
            ]
            self.assertTrue((overlay / "model-00001-of-00001.safetensors").is_symlink())
        self.assertEqual((quantization["bits"], quantization["codebook"]), (2, "mul1"))
        self.assertEqual(quantization["layer_bits"], {"3": 3})
        layers = quantization["non_routed_exl3"]["layers"]
        for prefix in ("model.layers.0", "language_model.model.layers.0"):
            self.assertEqual(
                layers[f"{prefix}.self_attn.in_proj_qkvbfg_a"],
                {"bits": 8, "bf16_shards": [3, 4, 5]},
            )
        self.assertEqual(layers["model.layers.0.self_attn.o_proj"], {"bits": 8})
        self.assertEqual(layers["model.layers.3.self_attn.o_proj"], {"bits": 7})
        self.assertEqual(
            layers["model.layers.3.self_attn.fused_qkv_a_proj"], {"bits": 8}
        )
        self.assertEqual(
            layers["model.layers.3.mlp.shared_experts.gate_up_proj"], {"bits": 6}
        )
        self.assertNotIn("model.layers.0.self_attn.b_proj", layers)


if __name__ == "__main__":
    unittest.main()
