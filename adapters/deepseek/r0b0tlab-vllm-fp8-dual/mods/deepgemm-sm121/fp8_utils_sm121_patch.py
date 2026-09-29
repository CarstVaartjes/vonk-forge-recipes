# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""SM121 patch for DeepSeek V4 BMM FP8 weight-scale layout.

For DeepSeek V4 wo_a grouped BMM on GB10/SM121, DeepGEMM's SM120 scalar
fp8_einsum expects standard FP32 block scales with shape:
    [groups, ceil(out_rank / 128), ceil(hidden / 128)]
The upstream BMM post-process path pre-transforms those scales into
TMA-packed INT32 UE8M0 layout [groups, out_rank, ceil(hidden / 512)], which
is consumed by SM100 paths but rejected by the SM120 scalar kernel and also
breaks the Python dequant fallback numerically.
"""

import torch

from vllm.model_executor.layers.quantization.utils.fp8_utils import (
    _upcast_e8m0_to_fp32,
    is_deep_gemm_e8m0_used,
    requant_weight_ue8m0_inplace,
)
from vllm.platforms import current_platform
from vllm.utils.deep_gemm import transform_sf_into_required_layout


def deepgemm_post_process_fp8_weight_block(
    wq: torch.Tensor,
    ws: torch.Tensor,
    quant_block_shape: tuple[int],
    use_e8m0: bool,
    is_bmm: bool = False,
    bmm_batch_size: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    assert wq.dtype == torch.float8_e4m3fn, (
        "Expected quantized tensor dtype "
        f"to be torch.float8_e4m3fn, got {wq.dtype} instead."
    )

    if ws.dtype == torch.float8_e8m0fnu:
        ws = _upcast_e8m0_to_fp32(ws)
    else:
        assert ws.dtype == torch.float32, (
            f"Expected tensor scales dtype to be torch.float32 or "
            f"torch.float8_e8m0fnu, got {ws.dtype} instead"
        )
        if use_e8m0:
            requant_weight_ue8m0_inplace(wq, ws, block_size=quant_block_shape)

    if is_bmm:
        # Reshape 2D weight/scale to 3D for grouped BMM (einsum):
        # wq: (g*r, d) -> (g, r, d)
        # ws: (g*r/128, d/128) -> (g, r/128, d/128)
        g = bmm_batch_size
        assert wq.ndim == 2 and ws.ndim == 2
        d = wq.size(1)
        r = wq.size(0) // g
        wq = wq.view(g, r, d)
        ws = ws.view(g, r // quant_block_shape[0], d // quant_block_shape[1])

        if current_platform.is_device_capability_family(120):
            # GB10/SM121 uses DeepGEMM's SM120 scalar einsum, which consumes
            # standard FP32 block scales directly.  Do not TMA-pack to INT32.
            return wq, ws.contiguous()

        dg_ws = transform_sf_into_required_layout(
            sf=ws,
            mn=r,
            k=d,
            recipe=(1, quant_block_shape[0], quant_block_shape[1]),
            num_groups=g,
            is_sfa=False,
        )
        return wq, dg_ws

    original_ndim = wq.ndim
    if wq.ndim == 2:
        assert ws.ndim == 2
        wq = wq.unsqueeze(0)
        ws = ws.unsqueeze(0)

    recipe = (1, 128, 128)
    dg_ws = transform_sf_into_required_layout(
        sf=ws,
        mn=wq.size(1),
        k=wq.size(2),
        recipe=recipe,
        num_groups=wq.size(0),
        is_sfa=False,
    )

    if original_ndim == 2:
        wq = wq.squeeze(0)
        dg_ws = dg_ws.squeeze(0)

    return wq, dg_ws


def prepare_fp8_moe_layer_for_deepgemm(
    w13: torch.Tensor,
    w2: torch.Tensor,
    w13_scale: torch.Tensor,
    w2_scale: torch.Tensor,
    block_shape: tuple[int],
):
    w13, w13_scale = deepgemm_post_process_fp8_weight_block(
        wq=w13,
        ws=w13_scale,
        quant_block_shape=block_shape,
        use_e8m0=is_deep_gemm_e8m0_used(),
    )
    w2, w2_scale = deepgemm_post_process_fp8_weight_block(
        wq=w2,
        ws=w2_scale,
        quant_block_shape=block_shape,
        use_e8m0=is_deep_gemm_e8m0_used(),
    )
    return w13, w2, w13_scale, w2_scale
