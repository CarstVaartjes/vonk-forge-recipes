"""Add a bounded PyTorch SDPA fallback to pinned TRELLIS-style sources."""

from __future__ import annotations

import argparse
from pathlib import Path


def replace_once(path: Path, old: str, new: str) -> None:
    source = path.read_text()
    count = source.count(old)
    if count != 1:
        raise SystemExit(f"expected one match in {path}, found {count}")
    path.write_text(source.replace(old, new))


def patch_tree(package_root: Path) -> None:
    config = package_root / "modules" / "sparse" / "config.py"
    full = package_root / "modules" / "sparse" / "attention" / "full_attn.py"
    windowed = package_root / "modules" / "sparse" / "attention" / "windowed_attn.py"

    if "'sdpa'" not in config.read_text():
        replace_once(
            config,
            "['xformers', 'flash_attn', 'flash_attn_3']",
            "['xformers', 'flash_attn', 'flash_attn_3', 'sdpa']",
        )

    if "elif config.ATTN == 'sdpa':" not in full.read_text():
        replace_once(full, "import torch\n", "import torch\nimport torch.nn.functional as F\n")
        replace_once(
            full,
            "    else:\n        raise ValueError(f\"Unknown attention module: {config.ATTN}\")\n",
            """    elif config.ATTN == 'sdpa':
        if num_all_args == 1:
            q, k, v = qkv.unbind(dim=1)
        elif num_all_args == 2:
            k, v = kv.unbind(dim=1)
        outputs = []
        q_offset = 0
        kv_offset = 0
        for q_len, kv_len in zip(q_seqlen, kv_seqlen):
            q_item = q[q_offset:q_offset + q_len].transpose(0, 1).unsqueeze(0)
            k_item = k[kv_offset:kv_offset + kv_len].transpose(0, 1).unsqueeze(0)
            v_item = v[kv_offset:kv_offset + kv_len].transpose(0, 1).unsqueeze(0)
            item = F.scaled_dot_product_attention(q_item, k_item, v_item)
            outputs.append(item.squeeze(0).transpose(0, 1))
            q_offset += q_len
            kv_offset += kv_len
        out = torch.cat(outputs, dim=0)
    else:
        raise ValueError(f\"Unknown attention module: {config.ATTN}\")
""",
        )

    replace_once(windowed, "import torch\n", "import torch\nimport torch.nn.functional as F\n")
    replace_once(
        windowed,
        "    return fwd_indices, bwd_indices, seq_lens, attn_func_args\n",
        """    elif config.ATTN == 'sdpa':
        attn_func_args = {}
    else:
        raise ValueError(f\"Unknown sparse attention backend: {config.ATTN}\")

    return fwd_indices, bwd_indices, seq_lens, attn_func_args
""",
    )
    replace_once(
        windowed,
        "    out = out[bwd_indices]      # [T, H, C]\n",
        """    elif config.ATTN == 'sdpa':
        outputs = []
        offset = 0
        for length in seq_lens.tolist():
            q, k, v = qkv_feats[offset:offset + length].unbind(dim=1)
            item = F.scaled_dot_product_attention(
                q.transpose(0, 1).unsqueeze(0),
                k.transpose(0, 1).unsqueeze(0),
                v.transpose(0, 1).unsqueeze(0),
            )
            outputs.append(item.squeeze(0).transpose(0, 1))
            offset += length
        out = torch.cat(outputs, dim=0)
    else:
        raise ValueError(f\"Unknown sparse attention backend: {config.ATTN}\")

    out = out[bwd_indices]      # [T, H, C]
""",
    )
    replace_once(
        windowed,
        "    out = out[q_bwd_indices]      # [T, H, C]\n",
        """    elif config.ATTN == 'sdpa':
        outputs = []
        q_offset = 0
        kv_offset = 0
        for q_len, kv_len in zip(q_seq_lens.tolist(), kv_seq_lens.tolist()):
            q_item = q_feats[q_offset:q_offset + q_len].transpose(0, 1).unsqueeze(0)
            k_item, v_item = kv_feats[kv_offset:kv_offset + kv_len].unbind(dim=1)
            item = F.scaled_dot_product_attention(
                q_item,
                k_item.transpose(0, 1).unsqueeze(0),
                v_item.transpose(0, 1).unsqueeze(0),
            )
            outputs.append(item.squeeze(0).transpose(0, 1))
            q_offset += q_len
            kv_offset += kv_len
        out = torch.cat(outputs, dim=0)
    else:
        raise ValueError(f\"Unknown sparse attention backend: {config.ATTN}\")

    out = out[q_bwd_indices]      # [T, H, C]
""",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("package_roots", nargs="+", type=Path)
    args = parser.parse_args()
    for package_root in args.package_roots:
        patch_tree(package_root)


if __name__ == "__main__":
    main()
