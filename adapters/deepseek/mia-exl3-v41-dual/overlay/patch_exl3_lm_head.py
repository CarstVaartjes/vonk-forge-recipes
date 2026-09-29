#!/usr/bin/env python3
"""Map packed EXL3 lm_head tensors and pass quant_config to ParallelLMHead.

This checkpoint stores head.{trellis,suh,svh,mul1} (K=head_bits=6), not a dense
head.weight. Stock DeepSeek-V4.1 mappers only rewrite head.weight, so
AutoWeightsLoader dies with: There is no module or parameter named 'head'.
ParallelLMHead is also constructed without quant_config, so even a mapped
trellis would have nowhere to land.
"""
from __future__ import annotations

from pathlib import Path
import sys

MARK = "dsv41-exl3-lm-head"

LM_HEAD_OLD = """            self.lm_head = ParallelLMHead(
                config.vocab_size,
                config.hidden_size,
                prefix=maybe_prefix(prefix, "lm_head"),
            )
"""

LM_HEAD_NEW = """            self.lm_head = ParallelLMHead(
                config.vocab_size,
                config.hidden_size,
                quant_config=vllm_config.quant_config,
                prefix=maybe_prefix(prefix, "lm_head"),
            )
"""

LLM_SUFFIX_OLD = """        orig_to_new_suffix={
            "head.weight": "lm_head.weight",
            "embed.weight": "embed_tokens.weight",
            ".ffn.gate.bias": ".ffn.gate.e_score_correction_bias",
        },
"""

LLM_SUFFIX_NEW = """        orig_to_new_suffix={
            "head.weight": "lm_head.weight",
            "head.trellis": "lm_head.trellis",
            "head.suh": "lm_head.suh",
            "head.svh": "lm_head.svh",
            "head.mul1": "lm_head.mul1",
            "head.mcg": "lm_head.mcg",
            "embed.weight": "embed_tokens.weight",
            ".ffn.gate.bias": ".ffn.gate.e_score_correction_bias",
        },
"""

VL_SUFFIX_OLD = """        orig_to_new_suffix={
            "head.weight": "language_model.lm_head.weight",
            "embed.weight": "embed_tokens.weight",
            ".ffn.gate.bias": ".ffn.gate.e_score_correction_bias",
        },
"""

VL_SUFFIX_NEW = """        orig_to_new_suffix={
            "head.weight": "language_model.lm_head.weight",
            "head.trellis": "language_model.lm_head.trellis",
            "head.suh": "language_model.lm_head.suh",
            "head.svh": "language_model.lm_head.svh",
            "head.mul1": "language_model.lm_head.mul1",
            "head.mcg": "language_model.lm_head.mcg",
            "embed.weight": "embed_tokens.weight",
            ".ffn.gate.bias": ".ffn.gate.e_score_correction_bias",
        },
"""

SLICE_OLD = """        for name, loaded_weight in weights:
            if name.startswith(("vision.", "aligner.", "image_")):
                # Vision weights are loaded by the outer multimodal wrapper.
                logger.warning_once("Skipping non-text weight: %s", name)
                continue
"""

SLICE_NEW = """        for name, loaded_weight in weights:
            if name.startswith(("vision.", "aligner.", "image_")):
                # Vision weights are loaded by the outer multimodal wrapper.
                logger.warning_once("Skipping non-text weight: %s", name)
                continue
            # [dsv41-exl3-lm-head] wo_a is packed as wo_a.slice.{i}.* (o_groups).
            _sl = re.search(r"\\.slice\\.(\\d+)\\.((?:trellis|suh|svh|mul1|mcg))$", name)
            if _sl:
                _sid = int(_sl.group(1))
                mapped = name[: _sl.start()] + "." + _sl.group(2)
                if mapped not in params_dict:
                    head, _, leaf = mapped.rpartition(".")
                    suffixed = f"{head}.base_layer.{leaf}"
                    if suffixed in params_dict:
                        mapped = suffixed
                param = params_dict[mapped]
                param.weight_loader(param, loaded_weight, _sid)
                loaded_params.add(mapped)
                continue
"""

COMPRESSOR_OLD = """            quant_config=None,
            disable_tp=True,
            prefix=f"{prefix}.fused_wkv_wgate",
"""

COMPRESSOR_NEW = """            quant_config=vllm_config.quant_config,
            disable_tp=True,
            prefix=f"{prefix}.fused_wkv_wgate",
"""

SCORE_OLD = """            def compressor_kv_score() -> torch.Tensor:
                return torch.mm(
                    hidden_states,
                    compressor.fused_wkv_wgate.weight.T,
                    out_dtype=torch.float32,
                )
"""

SCORE_NEW = """            def compressor_kv_score() -> torch.Tensor:
                # Packed EXL3 compressor has no dense .weight; apply the
                # fused wkv+wgate LinearEXL3 and match the fp32 mm dtype.
                _layer = compressor.fused_wkv_wgate
                _qm = getattr(_layer, "quant_method", None)
                if _qm is not None and not hasattr(_layer, "weight"):
                    return _qm.apply(_layer, hidden_states).float()
                return torch.mm(
                    hidden_states,
                    _layer.weight.T,
                    out_dtype=torch.float32,
                )
"""

INDEXER_WK_OLD = """            self.wk = ReplicatedLinear(
                main_head_dim,
                self.head_dim,
                bias=False,
                quant_config=None,
                prefix=f"{prefix}.wk",
            )
"""

INDEXER_WK_NEW = """            self.wk = ReplicatedLinear(
                main_head_dim,
                self.head_dim,
                bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.wk",
            )
"""


# models/deepseek_v4/nvidia/ops/o_proj.py: the CUDA o-projection reads
# wo_a.weight to pick FP8 vs bf16 bmm. Packed EXL3 wo_a has no dense weight
# (one LinearEXL3 per local o_group, loaded from wo_a.slice.i.*), so route it
# through the quant method, which applies group g of the input to inner g.
O_PROJ_OLD = """    use_fp8 = wo_a.weight.dtype == torch.float8_e4m3fn
"""

O_PROJ_NEW = """    # [dsv41-exl3-lm-head] packed EXL3 wo_a has no dense .weight.
    _wo_a_weight = getattr(wo_a, "weight", None)
    use_fp8 = _wo_a_weight is not None and _wo_a_weight.dtype == torch.float8_e4m3fn
"""

O_PROJ_BMM_OLD = """    else:
        grouped_weight = wo_a.weight.view(n_groups, o_lora_rank, -1)
        torch.bmm(
            o_proj_input.transpose(0, 1),
            grouped_weight.transpose(1, 2),
            out=z.transpose(0, 1),
        )
    return wo_b(z.flatten(1))
"""

O_PROJ_BMM_NEW = """    elif _wo_a_weight is None:
        # [dsv41-exl3-lm-head] grouped packed wo_a: [T, G, R] -> [T, G, o_lora_rank]
        z = wo_a.quant_method.apply(wo_a, o_proj_input).to(torch.bfloat16)
    else:
        grouped_weight = wo_a.weight.view(n_groups, o_lora_rank, -1)
        torch.bmm(
            o_proj_input.transpose(0, 1),
            grouped_weight.transpose(1, 2),
            out=z.transpose(0, 1),
        )
    return wo_b(z.flatten(1))
"""


# models/deepseek_v4_1/nvidia/dspark.py: the DSpark draft has its own
# load_weights loop; its wo_a is packed as wo_a.slice.{i}.* like the target.
DRAFT_SLICE_OLD = """            name = mapped
            if "confidence_head." in name:
                loaded_confidence_head = True
"""

DRAFT_SLICE_NEW = """            name = mapped
            if "confidence_head." in name:
                loaded_confidence_head = True
            # [dsv41-exl3-lm-head] wo_a is packed as wo_a.slice.{i}.* (o_groups).
            _sl = re.search(r"\\.slice\\.(\\d+)\\.((?:trellis|suh|svh|mul1|mcg))$", name)
            if _sl:
                _sid = int(_sl.group(1))
                _mapped = name[: _sl.start()] + "." + _sl.group(2)
                param = params_dict[_mapped]
                param.weight_loader(param, loaded_weight, _sid)
                loaded_params.add(_mapped)
                continue
"""


def apply_once(text: str, old: str, new: str) -> tuple[str, str]:
    if new in text:
        return text, "skipped"
    if old not in text:
        return text, "missing"
    if text.count(old) != 1:
        return text, f"not-unique:{text.count(old)}"
    return text.replace(old, new, 1), "applied"


def should_consider(text: str) -> bool:
    """model.py/vl_model.py (lm_head) plus compressor.py/attention.py."""
    return any(
        needle in text
        for needle in (
            "ParallelLMHead",
            "orig_to_new_suffix",
            "fused_wkv_wgate",
            "compressor_kv_score",
            "self.wk = ReplicatedLinear",
            "use_fp8 = wo_a.weight.dtype",
            "loaded_confidence_head = True",
        )
    )


def patch_text(text: str) -> tuple[str, list[str]]:
    statuses: list[str] = []
    for old, new, label in (
        (LM_HEAD_OLD, LM_HEAD_NEW, "ctor"),
        (LLM_SUFFIX_OLD, LLM_SUFFIX_NEW, "llm-suffix"),
        (VL_SUFFIX_OLD, VL_SUFFIX_NEW, "vl-suffix"),
        (SLICE_OLD, SLICE_NEW, "wo-a-slices"),
        (DRAFT_SLICE_OLD, DRAFT_SLICE_NEW, "draft-wo-a-slices"),
        (COMPRESSOR_OLD, COMPRESSOR_NEW, "compressor-quant"),
        (SCORE_OLD, SCORE_NEW, "compressor-score"),
        (INDEXER_WK_OLD, INDEXER_WK_NEW, "indexer-wk"),
        (O_PROJ_OLD, O_PROJ_NEW, "o-proj-exl3"),
        (O_PROJ_BMM_OLD, O_PROJ_BMM_NEW, "o-proj-exl3-bmm"),
    ):
        text, status = apply_once(text, old, new)
        if status != "missing":
            statuses.append(f"{label}:{status}")
    return text, statuses


def _vllm_root() -> Path | None:
    try:
        import vllm

        return Path(vllm.__file__).resolve().parent
    except Exception:
        return None


def main() -> int:
    root = _vllm_root()
    if root is None:
        print("WARN: vllm not importable; skip exl3 lm_head patch", file=sys.stderr)
        return 0
    rc = 0
    ds_root = root / "models" / "deepseek_v4_1"
    paths = sorted(ds_root.rglob("*.py")) if ds_root.is_dir() else []
    # The CUDA o-projection helper lives one package up (shared with V4).
    ops_root = root / "models" / "deepseek_v4" / "nvidia" / "ops"
    if ops_root.is_dir():
        paths += sorted(ops_root.glob("o_proj*.py"))
    for path in paths:
        text = path.read_text()
        if not should_consider(text):
            continue
        new, statuses = patch_text(text)
        useful = [s for s in statuses if not s.endswith(":missing")]
        if any(s.endswith(":not-unique") or ":not-unique:" in s for s in statuses):
            print(f"WARN: exl3 lm_head {path.relative_to(root)} {statuses}", file=sys.stderr)
            rc = 1
            continue
        if new == text:
            needs = "head.weight" in text and "head.trellis" not in text
            ctor_needed = LM_HEAD_OLD in text
            if needs or ctor_needed:
                print(f"WARN: exl3 lm_head no-op {path.relative_to(root)} {statuses}", file=sys.stderr)
                rc = 1
            elif useful:
                print(f"exl3 lm_head: {', '.join(useful)} ({path.relative_to(root)})")
            continue
        if MARK not in new and "quant_config=vllm_config.quant_config" in new:
            new = new.replace(
                "quant_config=vllm_config.quant_config,",
                f"quant_config=vllm_config.quant_config,  # [{MARK}]",
                1,
            )
        path.write_text(new)
        print(f"exl3 lm_head: {', '.join(useful) or 'edited'} ({path.relative_to(root)})")
    return rc


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"patch_exl3_lm_head skipped: {exc}", file=sys.stderr)
        raise SystemExit(0)
