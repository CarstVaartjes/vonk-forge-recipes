# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""SM12x Triton FP8 einsum kernels for DeepSeek V4."""

import torch

from vllm.distributed import get_tensor_model_parallel_rank
from vllm.model_executor.layers.quantization.utils.fp8_utils import (
    _upcast_e8m0_to_fp32,
)
from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton
from vllm.utils.deep_gemm import fp8_einsum
from vllm.utils.torch_utils import direct_register_custom_op


@triton.jit
def _deepseek_v4_sm12x_fp8_einsum_kernel(
    a_ptr,
    a_scale_ptr,
    b_ptr,
    b_scale_ptr,
    out_ptr,
    num_tokens: tl.constexpr,
    num_groups: tl.constexpr,
    out_rank: tl.constexpr,
    hidden_size: tl.constexpr,
    a_stride_token: tl.constexpr,
    a_stride_group: tl.constexpr,
    a_stride_hidden: tl.constexpr,
    a_scale_stride_token: tl.constexpr,
    a_scale_stride_group: tl.constexpr,
    a_scale_stride_hidden: tl.constexpr,
    b_stride_group: tl.constexpr,
    b_stride_out: tl.constexpr,
    b_stride_hidden: tl.constexpr,
    b_scale_stride_group: tl.constexpr,
    b_scale_stride_out: tl.constexpr,
    b_scale_stride_hidden: tl.constexpr,
    out_stride_token: tl.constexpr,
    out_stride_group: tl.constexpr,
    out_stride_rank: tl.constexpr,
    BLOCK_TOKENS: tl.constexpr,
    BLOCK_OUT: tl.constexpr,
    BLOCK_HIDDEN: tl.constexpr,
) -> None:
    token_block = tl.program_id(0)
    out_block = tl.program_id(1)
    group = tl.program_id(2)

    token_offsets = token_block * BLOCK_TOKENS + tl.arange(0, BLOCK_TOKENS)
    out_offsets = out_block * BLOCK_OUT + tl.arange(0, BLOCK_OUT)
    hidden_offsets = tl.arange(0, BLOCK_HIDDEN)
    accum = tl.zeros((BLOCK_TOKENS, BLOCK_OUT), dtype=tl.float32)

    for hidden_start in range(0, hidden_size, BLOCK_HIDDEN):
        hidden = hidden_start + hidden_offsets
        a = tl.load(
            a_ptr
            + token_offsets[:, None] * a_stride_token
            + group * a_stride_group
            + hidden[None, :] * a_stride_hidden,
            mask=(token_offsets[:, None] < num_tokens)
            & (hidden[None, :] < hidden_size),
            other=0.0,
        )
        b = tl.load(
            b_ptr
            + group * b_stride_group
            + out_offsets[None, :] * b_stride_out
            + hidden[:, None] * b_stride_hidden,
            mask=(out_offsets[None, :] < out_rank) & (hidden[:, None] < hidden_size),
            other=0.0,
        )
        raw = tl.dot(a, b, out_dtype=tl.float32)
        hidden_scale_block = hidden_start // BLOCK_HIDDEN
        a_scale = tl.load(
            a_scale_ptr
            + token_offsets * a_scale_stride_token
            + group * a_scale_stride_group
            + hidden_scale_block * a_scale_stride_hidden,
            mask=token_offsets < num_tokens,
            other=0.0,
        )
        b_scale = tl.load(
            b_scale_ptr
            + group * b_scale_stride_group
            + (out_offsets // 128) * b_scale_stride_out
            + hidden_scale_block * b_scale_stride_hidden,
            mask=out_offsets < out_rank,
            other=0.0,
        )
        accum += raw * a_scale[:, None] * b_scale[None, :]

    tl.store(
        out_ptr
        + token_offsets[:, None] * out_stride_token
        + group * out_stride_group
        + out_offsets[None, :] * out_stride_rank,
        accum,
        mask=(token_offsets[:, None] < num_tokens) & (out_offsets[None, :] < out_rank),
    )


def deepseek_v4_sm12x_fp8_einsum(
    a: torch.Tensor,
    a_scale: torch.Tensor,
    b: torch.Tensor,
    b_scale: torch.Tensor,
    out: torch.Tensor,
) -> None:
    """Compute ``bhr,hdr->bhd`` with FP32 block scales on SM12x.

    ``a`` is the transposed output of ``fused_inv_rope_fp8_quant`` with shape
    ``[tokens, groups, hidden]``. ``b`` is ``wo_a`` reshaped to
    ``[groups, out_rank, hidden]``.
    """
    num_tokens, num_groups, hidden_size = a.shape
    b_groups, out_rank, b_hidden_size = b.shape
    assert b_groups == num_groups
    assert b_hidden_size == hidden_size
    assert out.shape == (num_tokens, num_groups, out_rank)
    assert hidden_size % 128 == 0
    assert out_rank % 128 == 0
    assert a.dtype == torch.float8_e4m3fn
    assert b.dtype == torch.float8_e4m3fn
    e8m0_dtype = getattr(torch, "float8_e8m0fnu", None)
    if a_scale.dtype == e8m0_dtype:
        a_scale = _upcast_e8m0_to_fp32(a_scale)
    if b_scale.dtype == e8m0_dtype:
        b_scale = _upcast_e8m0_to_fp32(b_scale)
    assert a_scale.dtype == torch.float32
    assert b_scale.dtype == torch.float32

    if num_tokens == 0:
        return

    block_tokens = 16
    block_out = 128
    block_hidden = 128
    grid = (
        triton.cdiv(num_tokens, block_tokens),
        triton.cdiv(out_rank, block_out),
        num_groups,
    )
    _deepseek_v4_sm12x_fp8_einsum_kernel[grid](
        a,
        a_scale,
        b,
        b_scale,
        out,
        num_tokens,
        num_groups,
        out_rank,
        hidden_size,
        a.stride(0),
        a.stride(1),
        a.stride(2),
        a_scale.stride(0),
        a_scale.stride(1),
        a_scale.stride(2),
        b.stride(0),
        b.stride(1),
        b.stride(2),
        b_scale.stride(0),
        b_scale.stride(1),
        b_scale.stride(2),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        BLOCK_TOKENS=block_tokens,
        BLOCK_OUT=block_out,
        BLOCK_HIDDEN=block_hidden,
        num_warps=4,
        num_stages=3,
    )


def deepseek_v4_fp8_einsum_config(
    capability_major: int,
) -> tuple[tuple[int, int, int], bool]:
    if capability_major == 10:
        return (1, 1, 128), True
    return (1, 128, 128), False


def _use_deepseek_v4_sm12x_triton_fp8_einsum(
    equation: str,
    recipe: list[int],
    b_scale: torch.Tensor,
) -> bool:
    capability = current_platform.get_device_capability()
    e8m0_dtype = getattr(torch, "float8_e8m0fnu", None)
    return (
        capability is not None
        and capability.major == 12
        and equation == "bhr,hdr->bhd"
        and tuple(recipe) == (1, 128, 128)
        and b_scale.dtype in (torch.float32, e8m0_dtype)
        and b_scale.dim() == 3
    )


def _unpack_ue8m0_scales(scale: torch.Tensor) -> torch.Tensor:
    """Convert int32-packed UE8M0 scales to float32.

    Each int32 holds 4 packed UE8M0 bytes. Viewing as uint8 unpacks
    them (4× expansion on last dim).  Then shift each byte into the
    IEEE-754 float32 exponent field.
    """
    if scale.dtype != torch.int32:
        return scale
    # int32 [..., K_packed] → uint8 [..., K_packed*4]
    scale_bytes = scale.contiguous().view(torch.uint8)
    # uint8 → int32, shift into float32 exponent bits, reinterpret
    fp32_bits = scale_bytes.to(torch.int32) << 23
    return fp32_bits.view(torch.float32)


def _dequant_einsum_fallback(
    equation: str,
    a: torch.Tensor,
    a_scale: torch.Tensor,
    b: torch.Tensor,
    b_scale: torch.Tensor,
    out: torch.Tensor,
    recipe: list[int],
) -> None:
    """Dequantize FP8 data with block scales and run torch.einsum."""
    gran_k = recipe[2]  # e.g. 128

    # Dequantize a: [b, h, r] with scale [b, h, r//gran_k]
    a_f32 = a.to(torch.float32)
    if a_scale.numel() > 0 and a_scale.shape[-1] == a.shape[-1] // gran_k:
        # Repeat scales along k dim to match data
        a_s = a_scale.repeat_interleave(gran_k, dim=-1)[..., :a.shape[-1]]
        a_f32 = a_f32 * a_s

    # Dequantize b: [h, d, r] with scale [h, d, r//gran_k]
    b_f32 = b.to(torch.float32)
    if b_scale.numel() > 0 and b_scale.shape[-1] == b.shape[-1] // gran_k:
        b_s = b_scale.repeat_interleave(gran_k, dim=-1)[..., :b.shape[-1]]
        b_f32 = b_f32 * b_s

    result = torch.einsum(equation, a_f32, b_f32)
    out.copy_(result.to(out.dtype))


def deepseek_v4_fp8_einsum(
    a: torch.Tensor,
    a_scale: torch.Tensor,
    b: torch.Tensor,
    b_scale: torch.Tensor,
    out: torch.Tensor,
    equation: str,
    recipe: list[int],
) -> None:
    # Unpack int32-packed UE8M0 scales to float32 so both the Triton
    # SM12x kernel and the DeepGEMM C++ SM120 kernel can consume them.
    a_scale_f32 = _unpack_ue8m0_scales(a_scale)
    b_scale_f32 = _unpack_ue8m0_scales(b_scale)
    # Keep this path quiet during decode; this op runs once per layer per step.
    # Use the fallback log below only when we miss all native paths.
    if equation == "bhr,hdr->bhd" and b.dim() >= 2:
        num_groups = out.shape[1]
        out_rank = out.shape[2]
        hidden_size = a.shape[2]

        if b.dim() == 2:
            if b.shape[0] % out_rank != 0:
                raise RuntimeError(
                    "DeepSeek V4 fp8 einsum weight rows must be divisible by "
                    f"out_rank={out_rank}, got {b.shape[0]}"
                )
            b_groups = b.shape[0] // out_rank
            group_start = 0
            if b_groups != num_groups:
                if b_groups % num_groups != 0:
                    raise RuntimeError(
                        "DeepSeek V4 fp8 einsum weight groups must match the "
                        "TP-local output groups or be an integer multiple of "
                        f"them, got weight_groups={b_groups}, "
                        f"output_groups={num_groups}"
                    )
                group_partitions = b_groups // num_groups
                group_start = (
                    get_tensor_model_parallel_rank() % group_partitions
                ) * num_groups
            b = b.view(b_groups, out_rank, hidden_size)
            if group_start != 0 or b_groups != num_groups:
                b = b.narrow(0, group_start, num_groups)
        else:
            # b is already 3D (groups, out_rank, hidden)
            b_groups = b.shape[0]
            group_start = 0
            if b_groups != num_groups:
                if b_groups % num_groups != 0:
                    raise RuntimeError(
                        "DeepSeek V4 fp8 einsum weight groups must match the "
                        "TP-local output groups or be an integer multiple of "
                        f"them, got weight_groups={b_groups}, "
                        f"output_groups={num_groups}"
                    )
                group_partitions = b_groups // num_groups
                group_start = (
                    get_tensor_model_parallel_rank() % group_partitions
                ) * num_groups
                b = b.narrow(0, group_start, num_groups)

        if b_scale_f32.dim() == 2:
            scale_mn = recipe[1]
            scale_k = recipe[2]  # always float32 after unpack
            scale_out_blocks = (out_rank + scale_mn - 1) // scale_mn
            scale_hidden_blocks = (hidden_size + scale_k - 1) // scale_k
            if b_scale_f32.shape[0] % scale_out_blocks != 0:
                raise RuntimeError(
                    "DeepSeek V4 fp8 einsum scale rows must be divisible by "
                    f"scale_out_blocks={scale_out_blocks}, "
                    f"got {b_scale_f32.shape[0]}"
                )
            scale_groups = b_scale_f32.shape[0] // scale_out_blocks
            if scale_groups not in (num_groups, b_groups):
                raise RuntimeError(
                    "DeepSeek V4 fp8 einsum scale groups must match the "
                    "TP-local output groups or weight groups, got "
                    f"scale_groups={scale_groups}, output_groups={num_groups}, "
                    f"weight_groups={b_groups}"
                )
            b_scale_f32 = b_scale_f32.view(
                scale_groups,
                scale_out_blocks,
                scale_hidden_blocks,
            )
            if scale_groups == b_groups and scale_groups != num_groups:
                b_scale_f32 = b_scale_f32.narrow(0, group_start, num_groups)
        elif b_scale_f32.dim() == 3 and b_scale_f32.shape[0] == b_groups:
            if b_groups != num_groups:
                b_scale_f32 = b_scale_f32.narrow(0, group_start, num_groups)

        if _use_deepseek_v4_sm12x_triton_fp8_einsum(equation, recipe, b_scale_f32):
            deepseek_v4_sm12x_fp8_einsum(a, a_scale_f32, b, b_scale_f32, out)
            return

        expected_b_scale_shape = (
            num_groups,
            (out_rank + recipe[1] - 1) // recipe[1],
            (hidden_size + recipe[2] - 1) // recipe[2],
        )
        if (
            b_scale_f32.dtype == torch.float32
            and tuple(b_scale_f32.shape) == expected_b_scale_shape
        ):
            # SM120 DeepGEMM scalar einsum consumes standard FP32 block-scale
            # layout [groups, ceil(out_rank/128), ceil(hidden/128)].  Do not
            # use the Python dequant fallback when the loader has preserved
            # this layout; it is both much slower and numerically wrong for
            # TMA-packed weight scales.
            fp8_einsum(
                equation,
                (a, a_scale_f32),
                (b, b_scale_f32),
                out,
                recipe=tuple(recipe),
            )
            return

    print(f"[einsum-fallback] eq={equation} a={a.shape} a_scale_f32={a_scale_f32.shape} "
          f"b={b.shape} b_scale_f32={b_scale_f32.shape} out={out.shape} "
          f"recipe={recipe} — using dequant torch.einsum", flush=True)
    # Dequantize and use native torch.einsum only as a last-resort fallback.
    _dequant_einsum_fallback(equation, a, a_scale_f32, b, b_scale_f32,
                             out, recipe)


def deepseek_v4_fp8_einsum_fake(
    a: torch.Tensor,
    a_scale: torch.Tensor,
    b: torch.Tensor,
    b_scale: torch.Tensor,
    out: torch.Tensor,
    equation: str,
    recipe: list[int],
) -> None:
    return


direct_register_custom_op(
    op_name="deepseek_v4_fp8_einsum",
    op_func=deepseek_v4_fp8_einsum,
    mutates_args=["out"],
    fake_impl=deepseek_v4_fp8_einsum_fake,
)
