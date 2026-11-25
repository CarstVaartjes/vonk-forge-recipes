"""vcap's dead-row remap plus noexp's phantom-expert ids in one launch.

v2.7.2 per MoE visit: vcap's fused remap (weights[src[r]], ids[src[r]]) and, inside _sm120_route, four ATen launches
(arange, ==, full, where) building ids_masked[r] = ids[r] if src[r] == r else E. This kernel writes all three in vcap's
single launch: the gate output keeps vcap's exact tensors (every consumer sees what it saw in v2.7.2) and the masked
ids are a side output that only _sm120_route reads.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _remap_mask_kernel(w_in, id_in, src, w_out, id_out, masked_out, w_stride, id_stride, phantom,
                       K: tl.constexpr, BLOCK_K: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    s = tl.load(src + row).to(tl.int64)
    k = tl.arange(0, BLOCK_K)
    mask = k < K
    w = tl.load(w_in + s * w_stride + k, mask=mask)
    i = tl.load(id_in + s * id_stride + k, mask=mask)
    tl.store(w_out + row * K + k, w, mask=mask)
    tl.store(id_out + row * K + k, i, mask=mask)
    tl.store(masked_out + row * K + k, tl.where(s == row, i, phantom).to(masked_out.dtype.element_ty), mask=mask)


def remap_mask(weights, ids, src_index, phantom):
    """(weights[src], ids[src], phantom-masked ids)."""
    m, k = ids.shape
    w_out = torch.empty((m, k), dtype=weights.dtype, device=weights.device)
    id_out = torch.empty((m, k), dtype=ids.dtype, device=ids.device)
    masked = torch.empty((m, k), dtype=ids.dtype, device=ids.device)
    if m:
        _remap_mask_kernel[(m,)](weights, ids, src_index, w_out, id_out, masked, weights.stride(0), ids.stride(0),
                                 phantom, K=k, BLOCK_K=triton.next_power_of_2(k), num_warps=1)
    return w_out, id_out, masked


def reference(weights, ids, src_index, phantom):
    """The v2.7.2 tensors: index_select remap, then noexp's live mask (route_dead_free)."""
    m = ids.shape[0]
    w, i = weights.index_select(0, src_index[:m]), ids.index_select(0, src_index[:m])
    live = src_index[:m] == torch.arange(m, device=ids.device, dtype=src_index.dtype)
    return w, i, torch.where(live[:, None], i, torch.full_like(i, phantom))


def selfcheck(device):
    """None when the kernel matches the reference bit for bit on ``device`` (int32 and int64 ids, 8 / 16 / 128
    rows, dead rows at every position of a request)."""
    g = torch.Generator(device="cpu").manual_seed(7)
    for m in (8, 16, 128):
        for dtype in (torch.int32, torch.int64):
            weights = torch.rand((m, 8), generator=g).to(device)
            ids = torch.randint(0, 128, (m, 8), generator=g, dtype=dtype).to(device)
            lens = torch.randint(1, 9, (m // 8,), generator=g)
            src = torch.tensor([(r // 8) * 8 if r % 8 >= int(lens[r // 8]) else r for r in range(m)],
                               dtype=torch.int64).to(device)
            got, want = remap_mask(weights, ids, src, 128), reference(weights, ids, src, 128)
            torch.cuda.synchronize(device)
            if not all(torch.equal(a, b) for a, b in zip(got, want)):
                return f"fused phantom mask differs at {m} rows, {dtype}"
    return None
