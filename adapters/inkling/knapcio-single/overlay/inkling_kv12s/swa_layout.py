"""Page-major layout for the unified pool's target sliding-window band.

The stock unified arena stores a token's K/V rows for all window visits side by side
(one 160 KiB entry per token at 40 visits x 8 heads). One visit's 1,024-key window
then spans ~80 distinct 2 MiB GPU pages per request, and the compact verify
attention pays a TLB miss per page after the MoE weights evict the TLB (sampled
thinking c24: 22 % slower than the per-layer non-unified pool).

Here each 128-token page of the band keeps every visit's rows together:
``[visit 0: K rows 0..127 | V rows 0..127][visit 1: ...]``. Readers get strided
``(pages, 128, heads, dim)`` views and writers scatter at (page, row). The bytes are
the stock bytes; only their addresses inside the page change. Whole-page moves,
zeroing and frees are layout-agnostic.
"""

import torch
import triton
import triton.language as tl

PAGE = 128
# Snapshot transfers in tiles of ~16 MiB like the stock snapshot (160 KiB per row at 40 visits).
SNAPSHOT_ROWS = 96


@triton.jit(do_not_specialize=["n_rows", "limit"])
def _store_page_major(Loc, K, V, KOut, VOut, n_rows, limit,
                      SK0, SK1, SV0, SV1, SO0: tl.constexpr, SO1: tl.constexpr,
                      PAGE: tl.constexpr, H: tl.constexpr, D: tl.constexpr, BLOCK: tl.constexpr):
    """One visit's K (program 1 = 0) or V (= 1) row i into its page-major slot.
    Like the stock store, slot 0 (padding) and out-of-band ids write nothing."""
    i, kind = tl.program_id(0), tl.program_id(1)
    loc = tl.load(Loc + i).to(tl.int64)
    valid = (loc > 0) & (loc < limit)
    loc = tl.where(valid, loc, 0)
    words = tl.arange(0, BLOCK)
    live = (words < H * D) & valid
    head, dim = (words // D).to(tl.int64), words % D
    dst = (loc // PAGE) * SO0 + (loc % PAGE) * SO1 + words
    if kind == 0:
        row = tl.load(K + i.to(tl.int64) * SK0 + head * SK1 + dim, live)
        tl.store(KOut + dst, row, live)
    else:
        row = tl.load(V + i.to(tl.int64) * SV0 + head * SV1 + dim, live)
        tl.store(VOut + dst, row, live)


class _NoRowView(list):
    """Stock token-row K/V buffers do not exist in this layout: fail loudly on use."""

    def __getitem__(self, index):
        raise RuntimeError("inkling_kv12s: a token-row access reached the page-major window band")


def eligible(spec):
    from sglang.srt.mem_cache.kv.layout.mha import MHASubPoolSpec

    return (
        isinstance(spec, MHASubPoolSpec)
        and spec.name.startswith("window")
        and spec.draft_region is None
        and not spec.is_mxfp8
        and spec.kv_cache_dtype == torch.bfloat16
        and spec.layer_num > 0
    )


def convert(view, arena, spec, page_size):
    """Turn a freshly built stock window MHAView (nothing written yet) into the
    page-major layout in place."""
    from sglang.srt.mem_cache.memory_pool import unwrap_write_loc, write_loc_is_physical

    if page_size != PAGE:
        raise RuntimeError("inkling_kv12s: the page-major window band needs 128-token pages")
    entry = spec.entry_bytes()
    k_row, v_row = spec.k_row_bytes(), spec.v_row_bytes()
    visit = PAGE * (k_row + v_row)
    if spec.layer_num * visit > PAGE * entry or k_row % 16 or v_row % 16:
        raise RuntimeError("inkling_kv12s: window band geometry does not fit a page-major layout")
    pages = arena.extent_entries(spec.name) // PAGE
    heads, dim, v_dim = spec.head_num, spec.head_dim, spec.v_head_dim
    raw = arena.raw[:pages * PAGE * entry].view(torch.bfloat16)
    base = raw.storage_offset()
    page_stride = PAGE * entry // 2

    def lane(layer, offset, d):
        start = base + (layer * visit + offset) // 2
        return torch.as_strided(raw, (pages, PAGE, heads, d), (page_stride, heads * d, d, 1), start)

    view.k4 = [lane(i, 0, dim) for i in range(spec.layer_num)]
    view.v4 = [lane(i, PAGE * k_row, v_dim) for i in range(spec.layer_num)]
    view.page_major = True
    limit = pages * PAGE

    def store(index, loc, cache_k, cache_v, k_scale=None, v_scale=None):
        if cache_k.dtype != torch.bfloat16:
            if k_scale is not None:
                cache_k = cache_k.div(k_scale)
            if v_scale is not None:
                cache_v = cache_v.div(v_scale)
            cache_k, cache_v = cache_k.to(torch.bfloat16), cache_v.to(torch.bfloat16)
        loc = loc.reshape(-1).contiguous()
        if not loc.numel():
            return
        cache_k = cache_k.reshape(-1, heads, dim)
        cache_v = cache_v.reshape(-1, heads, v_dim)
        if (cache_k.shape[0] != loc.numel() or cache_v.shape[0] != loc.numel() or cache_k.stride(-1) != 1
                or cache_v.stride(-1) != 1 or loc.dtype not in (torch.int32, torch.int64) or dim != v_dim):
            raise ValueError("inkling_kv12s: invalid page-major window write")
        k_out, v_out = view.k4[index], view.v4[index]
        _store_page_major[(loc.numel(), 2)](
            loc, cache_k, cache_v, k_out, v_out, loc.numel(), limit,
            cache_k.stride(0), cache_k.stride(1), cache_v.stride(0), cache_v.stride(1),
            k_out.stride(0), k_out.stride(1), PAGE, heads, dim, triton.next_power_of_2(heads * dim), num_warps=4,
        )

    def set_kv_buffer(layer, loc_info, cache_k, cache_v, k_scale=None, v_scale=None,
                      layer_id_override=None, dcp_kv_mask=None):
        if not write_loc_is_physical(loc_info) or dcp_kv_mask is not None:
            raise RuntimeError("inkling_kv12s: page-major window writes need physical, unsharded ids")
        loc, _, _ = unwrap_write_loc(loc_info)
        layer_id = layer.layer_id if layer_id_override is None else layer_id_override
        store(layer_id - view.start_layer, loc, cache_k, cache_v, k_scale, v_scale)

    def set_kv_buffer_prefix_valid(layer, loc_2d, commit_lens, cache_k, cache_v, k_scale=None, v_scale=None,
                                   layer_id_override=None):
        if loc_2d.ndim != 2 or commit_lens.shape != (loc_2d.shape[0],):
            raise ValueError("inkling_kv12s: invalid prefix commit metadata")
        valid = torch.arange(loc_2d.shape[1], device=loc_2d.device)[None, :] < commit_lens[:, None]
        layer_id = layer.layer_id if layer_id_override is None else layer_id_override
        store(layer_id - view.start_layer, torch.where(valid, loc_2d, 0), cache_k, cache_v, k_scale, v_scale)

    def read_rows(rows):
        """(n, layers, 2, heads, dim) BF16 rows at physical ids, for snapshots (keep n small:
        each row is layers x 4 KiB)."""
        rows = torch.as_tensor(rows, dtype=torch.int64, device=raw.device)
        page, row = rows // PAGE, rows % PAGE
        return torch.stack([torch.stack((k[page, row], v[page, row]), dim=1)
                            for k, v in zip(view.k4, view.v4)], dim=1)

    def write_rows(rows, values, chunk=SNAPSHOT_ROWS):
        rows = torch.as_tensor(rows, dtype=torch.int64, device=raw.device)
        for start in range(0, rows.numel(), chunk):
            part = rows[start:start + chunk]
            page, row = part // PAGE, part % PAGE
            block = values[start:start + chunk].to(raw.device)
            for i, (k, v) in enumerate(zip(view.k4, view.v4)):
                k[page, row] = block[:, i, 0]
                v[page, row] = block[:, i, 1]

    def refuse(*args, **kwargs):
        raise RuntimeError("inkling_kv12s: the page-major window band has no token-row buffers")

    view.set_kv_buffer = set_kv_buffer
    view.set_kv_buffer_prefix_valid = set_kv_buffer_prefix_valid
    view.read_rows = read_rows
    view.write_rows = write_rows
    for name in ("get_kv_buffer", "get_key_buffer", "get_value_buffer", "get_cpu_copy", "load_cpu_copy",
                 "get_device_pool_infos", "host_layout", "get_contiguous_buf_infos", "move_kv_cache"):
        setattr(view, name, refuse)
    view.k_buffer = _NoRowView([None] * spec.layer_num)
    view.v_buffer = _NoRowView([None] * spec.layer_num)
    return view
