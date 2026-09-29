#!/usr/bin/env python3
"""Head-shard ranges must match vLLM EngramLayout (not an even row split)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
for _d in (HERE, ROOT / "overlay"):
    if (_d / "engram_layout.py").is_file():
        sys.path.insert(0, str(_d))
        break
from engram_layout import (  # noqa: E402
    layer_head_sizes_from_config,
    shard_range,
)

TEXT = {
    "engram_layer_ids": [1, 14],
    "engram_num_embeddings": [384006168, 384016682],
    "engram_vocab_size": 16000000,
    "engram_n_heads": 8,
    "engram_max_ngram_size": 4,
}
_CFG = ROOT / "model" / "config.json"
if _CFG.is_file():
    TEXT = json.loads(_CFG.read_text())["text_config"]


def test_layer_row_counts() -> None:
    for i, layer_id in enumerate(TEXT["engram_layer_ids"]):
        sizes = layer_head_sizes_from_config(TEXT, i)
        assert sum(sizes) == TEXT["engram_num_embeddings"][i], layer_id
        assert len(sizes) == (TEXT["engram_max_ngram_size"] - 1) * TEXT["engram_n_heads"]


def test_tp2_matches_vllm_log() -> None:
    sizes = layer_head_sizes_from_config(TEXT, 0)
    lo0, hi0, hs0, lh0 = shard_range(sizes, 0, 2)
    lo1, hi1, hs1, lh1 = shard_range(sizes, 1, 2)
    assert (lo0, hi0) == (0, 192001740)
    assert (lo1, hi1) == (192001740, 384006168)
    assert hs0 == 0 and lh0 == 12
    assert hs1 == 12 and lh1 == 12
    assert hi0 - lo0 + hi1 - lo1 == sum(sizes)


if __name__ == "__main__":
    test_layer_row_counts()
    test_tp2_matches_vllm_log()
    print("test_engram_layout: ok")
