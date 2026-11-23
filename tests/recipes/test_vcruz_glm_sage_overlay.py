"""The overlay builder of the vcruz305 GLM-5.3-Flash SAGE-REXL3 vLLM adapter declares what vllm-exl3 must look up."""

from __future__ import annotations

import json
import math
import os
import struct
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any

from repo_root import ROOT

ADAPTER = ROOT / "adapters/glm/vcruz-sage-rexl3-vllm-single"
LAYER = "model.language_model.layers"


SIZE = {"I16": 2, "I32": 4, "BF16": 2, "F16": 2}


def tensor(dtype: str, shape: list[int], data: bytes | None = None) -> tuple[Any, ...]:
    return (
        dtype,
        shape,
        data if data is not None else bytes(SIZE[dtype] * math.prod(shape)),
    )


def write_shard(path: Path, tensors: dict[str, Any]) -> None:
    header, body = {}, b""
    for name, (dtype, shape, data) in tensors.items():
        header[name] = {
            "dtype": dtype,
            "shape": shape,
            "data_offsets": [len(body), len(body) + len(data)],
        }
        body += data
    blob = json.dumps(header).encode()
    path.write_bytes(struct.pack("<Q", len(blob)) + blob + body)


def trellis(k: int) -> tuple[Any, ...]:
    return tensor("I16", [4, 3, 16 * k])


def marker() -> tuple[Any, ...]:
    return tensor("I32", [])


def group(prefix: str, k: int) -> dict[str, Any]:
    return {f"{prefix}.trellis": trellis(k), f"{prefix}.mul1": marker()}


def read_shard(path: Path) -> dict[str, Any]:
    raw = path.read_bytes()
    (length,) = struct.unpack("<Q", raw[:8])
    header = json.loads(raw[8 : 8 + length])
    body = raw[8 + length :]
    return {
        name: (meta["shape"], body[slice(*meta["data_offsets"])])
        for name, meta in header.items()
    }


def build(pack: Path, overlay: Path) -> None:
    done = subprocess.run(
        [
            sys.executable,
            "-I",
            str(ADAPTER / "make_overlay.py"),
            str(pack),
            str(overlay),
        ],
        capture_output=True,
        check=False,
        text=True,
        env={"PATH": os.environ.get("PATH", "")},
    )
    if done.returncode:
        raise AssertionError(done.stderr)


class OverlayTest(unittest.TestCase):
    def test_overlay_declares_the_lookups_the_creators_self_test_makes(self) -> None:
        tensors: dict[str, Any] = {}
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
        tensors[f"{LAYER}.0.self_attn.b_proj.weight"] = tensor("F16", [4, 4])
        with tempfile.TemporaryDirectory() as root:
            pack, overlay = Path(root, "pack"), Path(root, "overlay")
            pack.mkdir()
            write_shard(pack / "model-00001-of-00001.safetensors", tensors)
            index = {
                "weight_map": dict.fromkeys(tensors, "model-00001-of-00001.safetensors")
            }
            (pack / "model.safetensors.index.json").write_text(json.dumps(index))
            (pack / "config.json").write_text(
                json.dumps({"quantization_config": {"quant_method": "exl3"}})
            )
            build(pack, overlay)
            quantization = json.loads((overlay / "config.json").read_text())[
                "quantization_config"
            ]
            assert (overlay / "model-00001-of-00001.safetensors").is_symlink()
        assert (quantization["bits"], quantization["codebook"]) == (2, "mul1")
        assert quantization["layer_bits"] == {"3": 3}
        layers = quantization["non_routed_exl3"]["layers"]
        for prefix in ("model.layers.0", "language_model.model.layers.0"):
            assert layers[f"{prefix}.self_attn.in_proj_qkvbfg_a"] == {
                "bits": 8,
                "bf16_shards": [3, 4, 5],
            }
        assert layers["model.layers.0.self_attn.o_proj"] == {"bits": 8}
        assert layers["model.layers.3.self_attn.o_proj"] == {"bits": 7}
        assert layers["model.layers.3.self_attn.fused_qkv_a_proj"] == {"bits": 8}
        assert layers["model.layers.3.mlp.shared_experts.gate_up_proj"] == {"bits": 6}
        assert "model.layers.0.self_attn.b_proj" not in layers


class SplitTest(unittest.TestCase):
    """vLLM loads q/k/v separately; the pack holds ExLlamaV3's fused tensors."""

    def setUp(self) -> None:
        self.root = tempfile.TemporaryDirectory()
        self.addCleanup(self.root.cleanup)
        root = Path(self.root.name)
        self.pack, self.overlay = root / "pack", root / "overlay"
        self.pack.mkdir()
        attn = f"{LAYER}.0.self_attn"
        # conv1d [6,1,4] BF16: row r holds the bytes (r, r, ..., r) so the q/k/v rows are recognisable
        conv = b"".join(bytes([r]) * 8 for r in range(6))
        # trellis [rows=2, cols=6, w=16] I16: word (row, col) is filled with 10 * row + col
        words = b"".join(bytes([10 * r + c]) * 32 for r in range(2) for c in range(6))
        tensors = {
            f"{attn}.conv1d.weight": tensor("BF16", [6, 1, 4], conv),
            f"{attn}.qkv_proj.trellis": tensor("I16", [2, 6, 16], words),
            f"{attn}.qkv_proj.suh": tensor("F16", [2], b"\x01\x02\x03\x04"),
            f"{attn}.qkv_proj.svh": tensor("F16", [6], bytes(range(1, 13))),
            f"{attn}.qkv_proj.mul1": tensor("I32", [], b"\x07\x00\x00\x00"),
            f"{attn}.o_proj.trellis": trellis(1),
            f"{attn}.o_proj.mul1": marker(),
            f"{LAYER}.3.mlp.experts.0.up_proj.trellis": trellis(2),
            f"{LAYER}.3.mlp.experts.0.up_proj.mul1": marker(),
        }
        write_shard(self.pack / "a.safetensors", tensors)
        weight_map = dict.fromkeys(tensors, "a.safetensors")
        (self.pack / "model.safetensors.index.json").write_text(
            json.dumps({"metadata": {"x": 1}, "weight_map": weight_map})
        )
        (self.pack / "config.json").write_text("{}")
        build(self.pack, self.overlay)
        self.split = self.overlay / "vonk-kda-split.safetensors"
        self.attn = attn

    def test_fused_tensors_are_split_in_q_k_v_order(self) -> None:
        got = read_shard(self.split)
        a = self.attn
        for j, letter in enumerate("qkv"):
            shape, data = got[f"{a}.{letter}_conv1d.weight"]
            assert (shape, data) == (
                [2, 1, 4],
                b"".join(bytes([2 * j + r]) * 8 for r in range(2)),
            )
            shape, data = got[f"{a}.{letter}_proj.trellis"]
            assert shape == [2, 2, 16]
            # columns 2j and 2j+1 of every row (an interleaved stride-3 split would give j, j+3)
            assert data == b"".join(
                bytes([10 * r + 2 * j + c]) * 32 for r in range(2) for c in range(2)
            )
            assert got[f"{a}.{letter}_proj.svh"] == (
                [2],
                bytes(range(1 + 4 * j, 5 + 4 * j)),
            )
            assert got[f"{a}.{letter}_proj.suh"] == ([2], b"\x01\x02\x03\x04")
            assert got[f"{a}.{letter}_proj.mul1"] == ([], b"\x07\x00\x00\x00")
        assert len(got) == 3 * 5

    def test_index_is_a_real_file_without_the_fused_names(self) -> None:
        index_path = self.overlay / "model.safetensors.index.json"
        assert not index_path.is_symlink()
        index = json.loads(index_path.read_text())
        weights = index["weight_map"]
        assert index["metadata"] == {"x": 1}
        a = self.attn
        for fused in (
            "conv1d.weight",
            "qkv_proj.trellis",
            "qkv_proj.suh",
            "qkv_proj.svh",
            "qkv_proj.mul1",
        ):
            assert f"{a}.{fused}" not in weights
        assert weights[f"{a}.k_proj.svh"] == "vonk-kda-split.safetensors"
        assert weights[f"{a}.v_conv1d.weight"] == "vonk-kda-split.safetensors"
        assert weights[f"{a}.o_proj.trellis"] == "a.safetensors"
        assert (self.overlay / "a.safetensors").is_symlink()

    def test_split_file_is_rebuilt_and_stale_partial_removed(self) -> None:
        self.split.write_bytes(b"stale model")
        Path(str(self.split) + ".partial").write_bytes(b"stale")
        build(self.pack, self.overlay)
        assert not Path(str(self.split) + ".partial").exists()
        assert len(read_shard(self.split)) == 15


if __name__ == "__main__":
    unittest.main()
