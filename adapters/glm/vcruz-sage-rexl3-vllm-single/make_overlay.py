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
convention of the plugin's tools/dense_overlay.py for this model.

vLLM's glm5next module loads the KDA layers' short convolution and q/k/v projections as separate tensors, while the
pack holds the ExLlamaV3 fused ones (`conv1d.weight`, `qkv_proj.*`). The overlay therefore also carries one extra
safetensors file with the split tensors (the layout of dense_overlay.py) and a rewritten index that no longer names the
fused ones. The overlay is a directory of symlinks to the pack's shards plus the new config.json, that extra file and
that index.
"""

from __future__ import annotations

import collections
import json
import math
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
INDEX = "model.safetensors.index.json"
SPLIT_FILE = "vonk-kda-split.safetensors"
SPLIT_PARTS = ("trellis", "suh", "svh", "mul1", "mcg")
ITEMSIZE = {"BF16": 2, "F16": 2, "I16": 2, "I32": 4}


def read_header(path: str) -> dict:
    with open(path, "rb") as handle:
        (length,) = struct.unpack("<Q", handle.read(8))
        return json.loads(handle.read(length))


def trellis_k(meta: dict) -> int:
    shape = meta["shape"]
    if meta["dtype"] != "I16" or len(shape) != 3 or shape[2] % 16:
        raise SystemExit(f"unexpected trellis layout {meta}")
    return shape[2] // 16


def scan(weight_map: dict[str, str], headers: dict[str, dict]) -> tuple[dict[str, int], str]:
    """K of every EXL3 tensor group, keyed by its module prefix, and the pack's one codebook marker."""
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


def fused_kind(tensor: str) -> tuple[str, str, str] | None:
    """(prefix, new-name template, kind) of a fused KDA tensor the vLLM module wants split, else None."""
    head, _, part = tensor.rpartition(".")
    if head.endswith(".self_attn.conv1d") and part == "weight":
        return head.removesuffix("conv1d"), "{}_conv1d.weight", "conv1d"
    if head.endswith(".self_attn.qkv_proj") and part in SPLIT_PARTS:
        return head.removesuffix("qkv_proj"), "{}_proj." + part, part
    return None


def block(kind: str, shape: list[int], itemsize: int, blob: bytes, index: int) -> bytes:
    """Bytes of block `index` (q, k, v) of a fused tensor."""
    if kind in ("suh", "mul1", "mcg"):
        return blob
    if kind == "trellis":
        # [in/16, 3*out/16, 16K] row-major: block j is one contiguous run per row (dense_overlay.py download_bytes;
        # the plugin's fused load slices trellis[:, start // 16:(start + size) // 16, :])
        rows, cols, words = shape
        row, width = cols * words * itemsize, (cols // 3) * words * itemsize
        return b"".join(blob[r * row + index * width : r * row + (index + 1) * width] for r in range(rows))
    # svh and conv1d.weight: thirds along dim 0 (exllamav3 gated_delta_net.py torch.cat([q, k, v], dim=0))
    size = len(blob) // 3
    return blob[index * size : (index + 1) * size]


def block_shape(kind: str, shape: list[int]) -> list[int]:
    if kind in ("suh", "mul1", "mcg"):
        return shape
    if shape[1 if kind == "trellis" else 0] % 3:
        raise SystemExit(f"fused tensor of shape {shape} does not split in three")
    return [shape[0], shape[1] // 3, shape[2]] if kind == "trellis" else [shape[0] // 3, *shape[1:]]


def build_split(pack: str, weight_map: dict[str, str], headers: dict[str, dict], target: str) -> dict[str, str]:
    """Write the split tensors to `target`, rebuilt on every start; return {new tensor name: SPLIT_FILE}."""
    plan: list[tuple[str, str, str, str, dict]] = []
    header: dict[str, dict] = {}
    offset = 0
    for tensor in sorted(weight_map):
        fused = fused_kind(tensor)
        if not fused:
            continue
        stem, template, kind = fused
        meta = headers[weight_map[tensor]][tensor]
        shape = block_shape(kind, meta["shape"])
        count = ITEMSIZE[meta["dtype"]] * math.prod(shape)
        for letter in "qkv":
            header[stem + template.format(letter)] = {
                "dtype": meta["dtype"],
                "shape": shape,
                "data_offsets": [offset, offset + count],
            }
            offset += count
        plan.append((tensor, weight_map[tensor], stem, template, meta))
    names = dict.fromkeys(header, SPLIT_FILE)
    # /outputs/cache is not keyed by model identity: never trust an earlier file, and free its space first.
    if os.path.lexists(target):
        os.unlink(target)
    blob = json.dumps(header, separators=(",", ":")).encode()
    blob += b" " * (-len(blob) % 8)
    partial = target + ".partial"
    if os.path.lexists(partial):
        os.unlink(partial)
    with open(partial, "wb") as out:
        out.write(struct.pack("<Q", len(blob)) + blob)
        for tensor, shard, stem, template, meta in plan:
            with open(os.path.join(pack, shard), "rb") as handle:
                (length,) = struct.unpack("<Q", handle.read(8))
                start, end = meta["data_offsets"]
                handle.seek(8 + length + start)
                data = handle.read(end - start)
            kind = fused_kind(tensor)[2]
            for index, letter in enumerate("qkv"):
                out.write(block(kind, meta["shape"], ITEMSIZE[meta["dtype"]], data, index))
    os.replace(partial, target)
    return names


def write_index(pack: str, overlay: str, index: dict, headers: dict[str, dict]) -> None:
    weight_map = index["weight_map"]
    index["weight_map"] = {name: shard for name, shard in weight_map.items() if not fused_kind(name)}
    index["weight_map"].update(build_split(pack, weight_map, headers, os.path.join(overlay, SPLIT_FILE)))
    target = os.path.join(overlay, INDEX)
    temporary = target + ".tmp"
    with open(temporary, "w") as handle:
        json.dump(index, handle)
    os.replace(temporary, target)


def main() -> None:
    pack, overlay = (os.path.abspath(path) for path in sys.argv[1:3])
    with open(os.path.join(pack, "config.json")) as handle:
        config = json.load(handle)
    config["native_quantization_config"] = config.get("quantization_config", {})
    with open(os.path.join(pack, INDEX)) as handle:
        index = json.load(handle)
    weight_map = index["weight_map"]
    headers = {name: read_header(os.path.join(pack, name)) for name in sorted(set(weight_map.values()))}
    config["quantization_config"] = declaration(*scan(weight_map, headers))
    os.makedirs(overlay, exist_ok=True)
    for name in sorted(os.listdir(pack)):
        if name in ("config.json", INDEX) or name.startswith("."):
            continue
        destination = os.path.join(overlay, name)
        if os.path.lexists(destination):
            os.unlink(destination)
        os.symlink(os.path.join(pack, name), destination)
    write_index(pack, overlay, index, headers)
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
