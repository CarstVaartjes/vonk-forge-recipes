"""Backport the MXFP4 allocation fix from vLLM PR #47851 at image build.

Upstream 1bf3997eaeb41b1ef5cf25d87faa97b7f0be4788. Preserve the kernel and
scale/bias conversions; replace only the duplicate expert-weight assembly.
The single expected preimage is v0.21.0's MXFP4 repack function (also the
26.05 NGC path reported in #43969). Source drift fails the build.
"""
from __future__ import annotations

import sysconfig
from pathlib import Path

OLD = '''def repack_weight(weight: torch.Tensor, name: str) -> torch.Tensor:
        tensor_list = []
        if "w13" in name:
            size_n, size_k = n * 2, k
        else:
            size_n, size_k = k, n

        assert weight.shape == (e, size_n, size_k // 2)

        for i in range(e):
            qweight = weight[i].view(torch.int32).T.contiguous()
            marlin_qweight = ops.gptq_marlin_repack(
                b_q_weight=qweight,
                perm=perm,
                size_k=size_k,
                size_n=size_n,
                num_bits=4,
                is_a_8bit=is_a_8bit,
            )
            tensor_list.append(marlin_qweight)
        return torch.cat([x.unsqueeze(0) for x in tensor_list], 0)'''
NEW = '''def repack_weight(weight: torch.Tensor, name: str) -> torch.Tensor:
        if "w13" in name:
            size_n, size_k = n * 2, k
        else:
            size_n, size_k = k, n
        assert weight.shape == (e, size_n, size_k // 2)
        return _repack_marlin_experts(weight, size_n, size_k, perm, is_a_8bit)'''
HELPER = '''def _repack_marlin_experts(
    weight: torch.Tensor,
    size_n: int,
    size_k: int,
    perm: torch.Tensor,
    is_a_8bit: bool,
) -> torch.Tensor:
    """Repack each expert to marlin format into a preallocated output."""
    num_experts = weight.shape[0]
    out: torch.Tensor | None = None
    for i in range(num_experts):
        qweight = weight[i].view(torch.int32).T.contiguous()
        marlin_qweight = ops.gptq_marlin_repack(
            b_q_weight=qweight,
            perm=perm,
            size_k=size_k,
            size_n=size_n,
            num_bits=4,
            is_a_8bit=is_a_8bit,
        )
        if out is None:
            out = torch.empty(
                (num_experts, *marlin_qweight.shape),
                dtype=marlin_qweight.dtype,
                device=marlin_qweight.device,
            )
        out[i] = marlin_qweight
    assert out is not None
    return out'''


def patch_source(source: str) -> str:
    anchor = "def prepare_moe_mxfp4_layer_for_marlin("
    if (
        source.count(OLD) != 1
        or source.count(anchor) != 1
        or "def _repack_marlin_experts(" in source
    ):
        raise ValueError("MXFP4 Marlin repack source drift; review the pinned image")
    patched = source.replace(OLD, NEW).replace(anchor, HELPER + "\n\n\n" + anchor)
    compile(patched, "marlin_utils_fp4.py", "exec")
    return patched


if __name__ == "__main__":
    path = Path(sysconfig.get_path("purelib")) / (
        "vllm/model_executor/layers/quantization/utils/marlin_utils_fp4.py"
    )
    path.write_text(patch_source(path.read_text()))
