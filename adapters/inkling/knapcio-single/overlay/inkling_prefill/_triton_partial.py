"""Triton kernel of partial_bf16 (imported only on the GPU path).

A copy of sglang.kernels.ops.moe.inkling2_moe._routed_partial whose only change is
the BF16 store. The helpers are the stock jit functions, imported as globals.
"""

import triton
import triton.language as tl

from sglang.kernels.ops.moe.inkling2_moe import PDL, _nvfp4_row_scale, _row_expert  # noqa: F401


@triton.jit
def _routed_partial_bf16(
    X,
    XAmax,
    WAmax,
    WDenominator,
    Offsets,
    Src2Dst,
    Y,
    N: tl.constexpr,
    E: tl.constexpr,
    TOP_K: tl.constexpr,
    SCALED: tl.constexpr,
    BLOCK: tl.constexpr,
    SEARCH: tl.constexpr,
):
    tl.extra.cuda.gdc_wait()
    tl.extra.cuda.gdc_launch_dependents()
    src = tl.program_id(0).to(tl.int64)
    cols = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    lo = tl.load(Offsets)
    hi = tl.load(Offsets + E)
    total = tl.zeros([BLOCK], dtype=tl.float32)
    for k in tl.static_range(TOP_K):
        row = tl.load(Src2Dst + src * TOP_K + k)
        inside = (row >= lo) & (row < hi)
        x = tl.load(X + row.to(tl.int64) * N + cols, (cols < N) & inside, other=0)
        if SCALED:
            x = x * _nvfp4_row_scale(
                XAmax, WAmax, WDenominator, _row_expert(Offsets, row, E, SEARCH), row
            )
        total += tl.where(inside, x.to(tl.bfloat16).to(tl.float32), 0.0)
    tl.store(Y + src * N + cols, total.to(tl.bfloat16), cols < N)
