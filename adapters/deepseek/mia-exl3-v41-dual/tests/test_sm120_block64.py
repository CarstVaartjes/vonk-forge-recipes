#!/usr/bin/env python3
"""The SM12x 64-token-block patch applies once to the real backends (in the image) and is loud on drift."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
for _d in (HERE, ROOT / "overlay"):
    if (_d / "patch_sm120_block64.py").is_file():
        sys.path.insert(0, str(_d))
        break
from patch_sm120_block64 import JOBS, MARK, apply  # noqa: E402


def test_synthetic() -> None:
    for rel, old, new, helper in JOBS:
        head = "from vllm.platforms import current_platform\nMultipleOf = int\n\n\nclass B:\n"
        if not old.lstrip().startswith("def "):
            head += "    def f(self):\n"
        if old.lstrip().startswith("backend_cls="):
            head += "        self.swa_backend_cls = None\n        x = dict(\n"
        if old.lstrip().startswith("return MLAAttentionSpec("):
            head = "import torch\nMLAAttentionSpec = dict\n\n\nclass B:\n    def f(self, vllm_config, uses_fp8_ds_mla_layout):\n"
        if old.lstrip().startswith("self.max_image_tokens = ("):
            head = "\n\nclass B:\n    def f(self, config, hf_config):\n"
        if old.lstrip().startswith("use_persistent_topk = ") or old.lstrip().startswith("if current_platform.is_cuda() and select_k"):
            head = "from vllm.platforms import current_platform\n\n\nclass B:\n    def f(self, topk_tokens, select_k):\n"
            tail = "            pass\n" if old.lstrip().startswith("if current_platform") else ""
        if old.lstrip().startswith("#   40 * 163840"):
            head = "def get_max_prefill_buffer_size(vllm_config):\n    max_model_len = vllm_config.model_config.max_model_len\n"
        if old.lstrip().startswith("kv = self.wkv("):
            head = "class B:\n    def f(self, hidden_states, hash_ids):\n"
        if old.lstrip().startswith("mixed_tokens = "):  # module-level function body
            head = "logger = None\n_SPARSE_MLA_MIXED_WARMUP_TOKENS = 16\n\n\ndef _clamp_warmup_tokens(a, b):\n    return a\n\n\ndef f(max_tokens):\n"
        tail_default = "        )\n" if old.lstrip().startswith("return MLAAttentionSpec(") else ""
        tail = tail if old.lstrip().startswith("if current_platform.is_cuda() and select_k") else tail_default
        src = head + old + tail
        out, st = apply(src, old, new, helper)
        assert st == "applied", (rel, st)
        assert MARK in out and old not in out
        compile(out, rel, "exec")
        out2, st2 = apply(out, old, new, helper)
        assert st2 == "skipped" and out2 == out


def test_real_files() -> None:
    try:
        import vllm
    except Exception:
        print("test_sm120_block64: vllm not importable here; real-file check runs in the image build")
        return
    root = Path(vllm.__file__).resolve().parent
    texts: dict = {}
    for rel, old, new, helper in JOBS:
        src = texts.get(rel) or (root / rel).read_text()
        out, st = apply(src, old, new, helper)
        assert st in ("applied", "skipped"), (rel, st)
        compile(out, rel, "exec")
        texts[rel] = out
    for rel, old, new, helper in JOBS:  # idempotent
        out, st = apply(texts[rel], old, new, helper)
        assert st == "skipped" and out == texts[rel], (rel, st)


def test_drift_is_loud() -> None:
    rel, old, new, helper = JOBS[-1]
    _, st = apply("class B:\n    def f(self):\n        return [128]\n", old, new, helper)
    assert st.startswith("missing:"), st


if __name__ == "__main__":
    test_synthetic()
    test_real_files()
    test_drift_is_loud()
    print("test_sm120_block64: ok")
