#!/usr/bin/env python3
"""Build the SGLang view of the EXL3 pack without editing the pack.

The pack keeps its per-tensor storage manifest (`tensor_storage`) in quantization_config.json.
The sglang-exl3 loader reads it from config.json's `quantization_config`. The view is a directory
of symlinks to every pack file, plus one real config.json with `tensor_storage` merged in.

usage: make_view.py <pack_dir> <view_dir>
"""
import json
import os
import sys


def main() -> None:
    pack, view = (os.path.abspath(p) for p in sys.argv[1:3])
    os.makedirs(view, exist_ok=True)
    for name in sorted(os.listdir(pack)):
        if name == "config.json" or name.startswith("."):
            continue
        dst = os.path.join(view, name)
        if os.path.islink(dst) or os.path.exists(dst):
            os.unlink(dst)
        os.symlink(os.path.join(pack, name), dst)
    with open(os.path.join(pack, "config.json")) as f:
        cfg = json.load(f)
    with open(os.path.join(pack, "quantization_config.json")) as f:
        qcfg = json.load(f)
    cfg.setdefault("quantization_config", {})["tensor_storage"] = qcfg["tensor_storage"]
    out = os.path.join(view, "config.json")
    if os.path.islink(out) or os.path.exists(out):
        os.unlink(out)
    tmp = out + ".tmp"
    with open(tmp, "w") as f:
        json.dump(cfg, f, indent=2)
    os.replace(tmp, out)
    print(f"view {view}: {len(qcfg['tensor_storage'])} tensor_storage entries merged into config.json")


if __name__ == "__main__":
    main()
