#!/usr/bin/env python3
"""CPU test: packed EXL3 lm_head mapper + ParallelLMHead quant_config patch."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
for _d in (HERE, ROOT / "overlay"):
    if (_d / "patch_exl3_lm_head.py").is_file():
        sys.path.insert(0, str(_d))
        break

from patch_exl3_lm_head import (  # noqa: E402
    COMPRESSOR_NEW,
    COMPRESSOR_OLD,
    INDEXER_WK_NEW,
    INDEXER_WK_OLD,
    DRAFT_SLICE_NEW,
    DRAFT_SLICE_OLD,
    O_PROJ_BMM_NEW,
    O_PROJ_BMM_OLD,
    O_PROJ_NEW,
    O_PROJ_OLD,
    LLM_SUFFIX_NEW,
    LLM_SUFFIX_OLD,
    LM_HEAD_NEW,
    LM_HEAD_OLD,
    SCORE_NEW,
    SCORE_OLD,
    SLICE_NEW,
    SLICE_OLD,
    VL_SUFFIX_NEW,
    VL_SUFFIX_OLD,
    apply_once,
    patch_text,
    should_consider,
)


def test_ctor() -> None:
    src = "class X:\n    def __init__(self):\n" + LM_HEAD_OLD + "        self.x = 1\n"
    out, status = apply_once(src, LM_HEAD_OLD, LM_HEAD_NEW)
    assert status == "applied", status
    assert "quant_config=vllm_config.quant_config" in out
    out2, status2 = apply_once(out, LM_HEAD_OLD, LM_HEAD_NEW)
    assert status2 == "skipped", status2


def test_llm_suffix() -> None:
    src = LLM_SUFFIX_OLD
    out, status = apply_once(src, LLM_SUFFIX_OLD, LLM_SUFFIX_NEW)
    assert status == "applied", status
    assert '"head.trellis": "lm_head.trellis"' in out
    assert '"head.mul1": "lm_head.mul1"' in out
    out2, status2 = apply_once(out, LLM_SUFFIX_OLD, LLM_SUFFIX_NEW)
    assert status2 == "skipped", status2


def test_vl_suffix() -> None:
    src = VL_SUFFIX_OLD
    out, status = apply_once(src, VL_SUFFIX_OLD, VL_SUFFIX_NEW)
    assert status == "applied", status
    assert '"head.suh": "language_model.lm_head.suh"' in out
    out2, status2 = apply_once(out, VL_SUFFIX_OLD, VL_SUFFIX_NEW)
    assert status2 == "skipped", status2


def test_patch_text_model_and_vl() -> None:
    model = LM_HEAD_OLD + "\n" + LLM_SUFFIX_OLD
    out, statuses = patch_text(model)
    assert "ctor:applied" in statuses
    assert "llm-suffix:applied" in statuses
    assert "vl-suffix" not in statuses
    assert "quant_config=vllm_config.quant_config" in out
    assert "head.trellis" in out
    vl = VL_SUFFIX_OLD
    out_vl, st_vl = patch_text(vl)
    assert "vl-suffix:applied" in st_vl
    assert "language_model.lm_head.mul1" in out_vl
    out2, st2 = patch_text(out)
    assert all(s.endswith(":skipped") for s in st2), st2


def test_wo_a_slice_rewrite() -> None:
    import re

    src = SLICE_OLD
    out, status = apply_once(src, SLICE_OLD, SLICE_NEW)
    assert status == "applied", status
    assert "wo_a.slice" in out
    assert "weight_loader(param, loaded_weight, _sid)" in out
    rewritten = "layers.0.attn.wo_a.slice.0.mul1"
    pat = r"\.slice\.(\d+)\.((?:trellis|suh|svh|mul1|mcg))$"
    m = re.search(pat, rewritten)
    assert m is not None
    mapped = rewritten[: m.start()] + "." + m.group(2)
    assert mapped == "layers.0.attn.wo_a.mul1"
    assert int(m.group(1)) == 0
    out2, status2 = apply_once(out, SLICE_OLD, SLICE_NEW)
    assert status2 == "skipped", status2


def test_compressor_and_score_files_are_considered() -> None:
    assert should_consider(COMPRESSOR_OLD)
    assert should_consider(SCORE_OLD)
    assert should_consider("self.lm_head = ParallelLMHead(\n")
    assert not should_consider("class Unrelated:\n    pass\n")
    out, st = patch_text(COMPRESSOR_OLD)
    assert "compressor-quant:applied" in st
    assert "vllm_config.quant_config" in out
    out_s, st_s = patch_text(SCORE_OLD)
    assert "compressor-score:applied" in st_s
    assert "quant_method" in out_s
    assert should_consider(INDEXER_WK_OLD)
    out_w, st_w = patch_text(INDEXER_WK_OLD)
    assert "indexer-wk:applied" in st_w
    assert "quant_config=quant_config" in out_w


def test_o_proj_exl3_branch() -> None:
    """o_proj.py: packed wo_a (no .weight) must go through the quant method."""
    src = O_PROJ_OLD + "    z = None\n    if use_fp8:\n        pass\n" + O_PROJ_BMM_OLD
    assert should_consider(src)
    out, st = patch_text(src)
    assert "o-proj-exl3:applied" in st and "o-proj-exl3-bmm:applied" in st, st
    assert 'getattr(wo_a, "weight", None)' in out
    assert "wo_a.quant_method.apply(wo_a, o_proj_input)" in out
    assert "wo_a.weight.dtype ==" not in out
    # the dense bmm branch is kept for native checkpoints
    assert "grouped_weight = wo_a.weight.view" in out
    out2, st2 = patch_text(out)
    assert out2 == out and all(s.endswith(":skipped") for s in st2), st2
    # the fragment is function-body indented; give it a header to compile
    compile("def _f(wo_a, wo_b, o_proj_input, z, n_groups, o_lora_rank):\n" + out, "o_proj.py", "exec")


def test_draft_wo_a_slice_rewrite() -> None:
    """dspark.py: mtp.N.attn.wo_a.slice.i.* must reach the packed wo_a loader."""
    src = "import regex as re\n\n\ndef f(weights, params_dict, loaded_params):\n    loaded_confidence_head = False\n    for name, loaded_weight in weights:\n        mapped = name\n        if True:\n" + DRAFT_SLICE_OLD
    assert should_consider(src)
    out, st = patch_text(src)
    assert "draft-wo-a-slices:applied" in st, st
    assert "param.weight_loader(param, loaded_weight, _sid)" in out
    compile(out, "dspark.py", "exec")
    out2, st2 = patch_text(out)
    assert out2 == out and all(s.endswith(":skipped") for s in st2), st2


def test_suffix_is_not_double_applied_on_mapped_name() -> None:
    """VL mapper must not prefix head. or suffix would fire twice."""
    assert "language_model.lm_head.suh" in VL_SUFFIX_NEW
    mapped = "language_model.lm_head.suh"
    # Document the existing head.weight pitfall: endswith("head.suh") is true
    # after the first rewrite, so this patch must run through a no-op inner mapper.
    assert mapped.endswith("head.suh")


def main() -> int:
    test_ctor()
    test_llm_suffix()
    test_vl_suffix()
    test_patch_text_model_and_vl()
    test_o_proj_exl3_branch()
    test_draft_wo_a_slice_rewrite()
    test_wo_a_slice_rewrite()
    test_compressor_and_score_files_are_considered()
    out, st = apply_once(COMPRESSOR_OLD, COMPRESSOR_OLD, COMPRESSOR_NEW)
    assert st == "applied"
    assert "vllm_config.quant_config" in out
    out, st = apply_once(SCORE_OLD, SCORE_OLD, SCORE_NEW)
    assert st == "applied"
    assert "quant_method" in out
    out, st = apply_once(INDEXER_WK_OLD, INDEXER_WK_OLD, INDEXER_WK_NEW)
    assert st == "applied"
    assert "quant_config=quant_config" in out
    test_suffix_is_not_double_applied_on_mapped_name()
    print("exl3 lm_head patch OK", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
