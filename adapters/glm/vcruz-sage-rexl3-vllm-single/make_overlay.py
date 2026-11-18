#!/usr/bin/env python3
"""Build the vllm-exl3 loader overlay for vcruz305/GLM-5.3-Flash-SAGE-REXL3-2.48bpw without editing the pack.

usage: make_overlay.py <pack_dir> <overlay_dir>

The pack's config.json carries only {"quant_method": "exl3"}. vllm-exl3 needs the expert K of every layer
(`bits` and `layer_bits`), the codebook and a `non_routed_exl3` map of the dense EXL3 linears, with the fused vLLM
module names (in_proj_qkvbfg_a, fused_qkv_a_proj, gate_up_proj). The creator served a local overlay directory of that
kind but does not publish how he built it. This script derives the declaration the way the plugin's own native-pack tool
(vllm-exl3 tools/exl3_pack_tools/qwen_pack_config.py) does: it reads only the safetensors headers, takes each tensor's K
from its trellis shape (last dimension / 16), gives a layer the most common K of its routed experts, and emits every
dense linear under each plausible module prefix (the checkpoint's, the VL wrapper's and the fused module's). The
KDA layers' fused qkv_proj goes to in_proj_qkvbfg_a with bf16_shards [3, 4, 5] (b, f_a and g_a stay BF16), the
convention of the plugin's tools/dense_overlay.py for this model. The overlay is a directory of symlinks to the pack's
files plus the new config.json.
"""

from __future__ import annotations

import collections
import json
import os
import struct
import sys

ROOTS = (("model.language_model.", "model."), ("model.language_model.", "language_model.model."))
FUSED_LEAF = {
    "q_a_proj": "fused_qkv_a_proj",
    "kv_a_proj_with_mqa": "fused_qkv_a_proj",
    "gate_proj": "gate_up_proj",
    "up_proj": "gate_up_proj",
}
BF16_SHARDS = {"in_proj_qkvbfg_a": [3, 4, 5]}


def read_header(path: str) -> dict:
    with open(path, "rb") as handle:
        (length,) = struct.unpack("<Q", handle.read(8))
        return json.loads(handle.read(length))


def trellis_k(meta: dict) -> int:
    shape = meta["shape"]
    if meta["dtype"] != "I16" or len(shape) != 3 or shape[2] % 16:
        raise SystemExit(f"unexpected trellis layout {meta}")
    return shape[2] // 16


def scan(pack: str) -> tuple[dict[str, int], str]:
    """K of every EXL3 tensor group, keyed by its module prefix, and the pack's one codebook marker."""
    with open(os.path.join(pack, "model.safetensors.index.json")) as handle:
        weight_map = json.load(handle)["weight_map"]
    headers = {name: read_header(os.path.join(pack, name)) for name in sorted(set(weight_map.values()))}
    ks: dict[str, int] = {}
    markers: set[str] = set()
    for tensor, shard in weight_map.items():
        prefix, _, part = tensor.rpartition(".")
        if part == "trellis":
            ks[prefix] = trellis_k(headers[shard][tensor])
        elif part in ("mul1", "mcg"):
            markers.add(part)
    if len(markers) != 1:
        raise SystemExit(f"expected exactly one codebook marker, found {sorted(markers)}")
    return ks, markers.pop()


def variants(prefix: str, bits: int) -> dict[str, dict]:
    head, _, leaf = prefix.rpartition(".")
    names = {prefix: {"bits": bits}}
    if leaf == "qkv_proj" and head.endswith(".self_attn"):
        names[f"{head}.in_proj_qkvbfg_a"] = {"bits": bits, "bf16_shards": BF16_SHARDS["in_proj_qkvbfg_a"]}
    elif leaf in FUSED_LEAF:
        names[f"{head}.{FUSED_LEAF[leaf]}"] = {"bits": bits}
    for name, entry in list(names.items()):
        for source, target in ROOTS:
            if name.startswith(source):
                names[target + name[len(source):]] = dict(entry)
    return names


def declaration(ks: dict[str, int], codebook: str) -> dict:
    expert_k: dict[int, list[int]] = collections.defaultdict(list)
    dense: dict[str, dict] = {}
    for prefix, bits in ks.items():
        if ".experts." in prefix:
            layer = int(prefix.split(".layers.", 1)[1].split(".", 1)[0])
            expert_k[layer].append(bits)
            continue
        for name, entry in variants(prefix, bits).items():
            old = dense.get(name)
            dense[name] = entry if old is None else {**entry, "bits": max(entry["bits"], old["bits"])}
    layer_k = {layer: collections.Counter(values).most_common(1)[0][0] for layer, values in expert_k.items()}
    if not layer_k:
        raise SystemExit("no routed expert trellis tensors found")
    base = collections.Counter(layer_k.values()).most_common(1)[0][0]
    return {
        "quant_method": "exl3",
        "bits": base,
        "codebook": codebook,
        "layer_bits": {str(layer): bits for layer, bits in sorted(layer_k.items()) if bits != base},
        "non_routed_exl3": {"codebook": codebook, "layers": dict(sorted(dense.items()))},
    }


def main() -> None:
    pack, overlay = (os.path.abspath(path) for path in sys.argv[1:3])
    with open(os.path.join(pack, "config.json")) as handle:
        config = json.load(handle)
    config["native_quantization_config"] = config.get("quantization_config", {})
    config["quantization_config"] = declaration(*scan(pack))
    os.makedirs(overlay, exist_ok=True)
    for name in sorted(os.listdir(pack)):
        if name == "config.json" or name.startswith("."):
            continue
        destination = os.path.join(overlay, name)
        if os.path.lexists(destination):
            os.unlink(destination)
        os.symlink(os.path.join(pack, name), destination)
    target = os.path.join(overlay, "config.json")
    temporary = target + ".tmp"
    with open(temporary, "w") as handle:
        json.dump(config, handle, indent=2)
    os.replace(temporary, target)
    quantization = config["quantization_config"]
    print(
        f"overlay {overlay}: bits={quantization['bits']} layer_bits={len(quantization['layer_bits'])} "
        f"dense modules={len(quantization['non_routed_exl3']['layers'])} codebook={quantization['codebook']}"
    )


if __name__ == "__main__":
    main()
