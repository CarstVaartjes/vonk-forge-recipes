"""File-backed Engram table store for DeepSeek-V4.1-Flash on GB10 clusters.

The two Engram n-gram tables (layers 1 and 14) are 189.1 GiB of FP8 rows.
None of the stock placements fit a 4x GB10 (TP=4) node: device-sharded needs
+47.3 GiB/rank of unified memory, per-rank host shards the same, and a full
shared host copy 189 GiB — all against ~40 GiB of headroom.

This module keeps the checkpoint shard itself as the store: a read-only
file-backed mmap over the two contiguous tensor ranges, with the upstream
``engram_gather`` kernel reading through the page cache (demand-paged,
reclaimable; GB10 exposes host memory to the GPU via hardware coherency, the
mode upstream logs as "unpinned (ATS)"). Every rank maps the full table and
gathers its own lookups — the shared-store semantics — so the Engram lookup
all-reduce is eliminated.

Written from scratch for the r0b0tlab dsv4.1-flash package. The upstream
SGLang interface it plugs into (EngramEmbedding layout contract,
engram_gather pointer semantics) is credited in the repository README.
"""

from __future__ import annotations

import json
import logging
import mmap
import os
import struct

import numpy as np
import torch
from torch import nn

logger = logging.getLogger(__name__)

FP8_BLOCK_SIZE = 32

_SHARD_INDEX: dict | None = None


def read_safetensors_header(path: str) -> tuple[dict, int]:
    """Return (header dict, data_base) where data_base is the absolute file
    offset where tensor data begins — safetensors data_offsets are relative
    to this point, NOT to the start of the file."""
    with open(path, "rb") as f:
        (n,) = struct.unpack("<Q", f.read(8))
        header = json.loads(f.read(n))
    return {k: v for k, v in header.items() if k != "__metadata__"}, 8 + n


class FileEngramStore:
    """Read-only view of one Engram table inside its checkpoint shard.

    Exposes the raw weight/scale pointers the upstream gather kernel takes,
    plus a CPU reference gather used by the unit tests and the load-time
    fidelity probe.
    """

    def __init__(self, shard_path: str, prefix: str, rows: int, dim: int):
        self.shard_path = shard_path
        self.prefix = prefix
        self.rows = rows
        self.dim = dim
        header, data_base = read_safetensors_header(shard_path)
        wkey, skey = f"{prefix}.embed.weight", f"{prefix}.embed.scale"
        if wkey not in header or skey not in header:
            raise KeyError(f"{wkey}/{skey} not in {shard_path}")
        wv, sv = header[wkey], header[skey]
        wlo, whi = wv["data_offsets"]
        slo, shi = sv["data_offsets"]
        if whi - wlo != rows * dim:
            raise ValueError(
                f"weight range {whi - wlo} != rows*dim {rows * dim}")
        if shi - slo != rows * (dim // FP8_BLOCK_SIZE):
            raise ValueError("scale range mismatch")
        if sv["shape"][0] != rows or wv["shape"][0] != rows:
            raise ValueError("header shape disagrees with requested rows")
        # Contiguity requirement: scales must directly follow weights so both
        # live inside one mapping (kernel takes two pointers, so this also
        # works non-contiguous — but contiguous is what the checkpoint ships
        # and what we assert for).
        if slo != whi:
            raise ValueError(
                f"non-contiguous weight/scale in {shard_path} "
                f"(weight ends {whi}, scale starts {slo}); repack required")

        fd = os.open(shard_path, os.O_RDONLY)
        try:
            self._mm = mmap.mmap(fd, 0, prot=mmap.PROT_READ, flags=mmap.MAP_SHARED)
            self._mm.madvise(mmap.MADV_RANDOM)
        finally:
            os.close(fd)  # mapping holds its own reference
        raw = torch.frombuffer(self._mm, dtype=torch.uint8)
        # data_offsets are relative to the end of the safetensors header.
        self._weight_view = (
            raw[data_base + wlo : data_base + whi]
            .view(torch.float8_e4m3fn).view(rows, dim)
        )
        self._scale_view = (
            raw[data_base + slo : data_base + shi]
            .view(torch.float8_e8m0fnu).view(rows, dim // FP8_BLOCK_SIZE)
        )
        self.weight_ptr = self._weight_view.data_ptr()
        self.scale_ptr = self._scale_view.data_ptr()
        logger.info(
            "engram file store: %s rows=%d dim=%d shard=%s w_off=%d s_off=%d",
            prefix, rows, dim, shard_path, wlo, slo)

    # ---- reference gather (CPU, tests + fidelity probe) -----------------

    def gather(self, ids: torch.Tensor) -> torch.Tensor:
        """Dequant rows `ids` to bf16 — the exact contract of the upstream
        engram_gather kernel: full-table ids, out-of-range → zero rows."""
        ids = ids.to(torch.int64).reshape(-1)
        out = torch.zeros(ids.numel(), self.dim, dtype=torch.bfloat16)
        ok = (ids >= 0) & (ids < self.rows)
        local = ids[ok]
        w = self._weight_view[local].float()
        s = self._scale_view[local].float().repeat_interleave(FP8_BLOCK_SIZE, dim=1)
        out[ok] = (w * s).to(torch.bfloat16)
        return out

    def close(self) -> None:
        self._mm.close()


class _FileHostTableShim:
    """Duck-types the parts of upstream's _HostTable that EngramEmbedding
    touches after construction, with layout="shared" so the forward path is
    the direct full-table gather (no all-reduce) and the layer-14 prefetch
    stream activates. No memfd, no group, nothing to barrier or pin."""

    layout = "shared"
    nbytes = 0
    dirty = False
    registered = False

    def finish_load(self, label: str) -> None:  # loader hook: nothing to write
        logger.info("engram file table %s: ready (file-backed, read-only)", label)


class _InstallState:
    installed = False


def install(model_path: str) -> None:
    """Patch EngramEmbedding to serve each table from the local checkpoint
    shard via FileEngramStore. Idempotent. Must run before the model is
    constructed (sitecustomize does this at interpreter start in the
    serving image)."""
    from sglang.srt.layers import engram as up

    if getattr(up.EngramEmbedding, "_dsv41_file_store", False):
        return

    # SM12x: force _is_sm100_or_newer False so decode uses Triton; prefill uses K-sliced Torch indexer.
    import sglang.srt.layers.attention.deepseek_v4_backend as _dbe
    _cap = torch.cuda.get_device_capability()
    if _cap[0] >= 12:  # SM12x: consumer/GB10 Blackwell without tcgen05
        _dbe._is_sm100_or_newer = lambda: False
        logger.info(
            "dsv41 sm12x gate: DeepGEMM fp4 indexer disabled (cap=%s); "
            "decode/verify route to Triton fallback", _cap)

    # OPTIMAL HYBRID sparse-MLA (SM12x): flashinfer's decode kernel (<=64
    # tokens, incl. all DSPARK verify batches at bs8x6=48) accepts the DSV4.1
    # extra-KV config and is fast; only its PREFILL path rejects it. Route
    # >64-token calls through upstream's Triton kernel (handles the 256-token
    # page layout natively) and keep flashinfer for decode.
    import sglang.kernels.ops.attention.flash_mla_sm120 as _fm
    from sglang.kernels.ops.attention.flash_mla_sm120_triton import (
        flash_mla_sparse_decode_triton as _triton_sparse_mla,
    )

    def _prefill_via_triton(q, k_cache, indices, topk_length, attn_sink,
                            head_dim_v, softmax_scale, extra_k_cache,
                            extra_indices, extra_topk_length):
        return _triton_sparse_mla(
            q, k_cache, indices, topk_length, attn_sink, head_dim_v,
            softmax_scale, extra_k_cache, extra_indices, extra_topk_length,
        )

    _fm._flash_mla_sm120_prefill = _prefill_via_triton
    logger.info("dsv41 sm12x hybrid sparse-MLA: flashinfer decode + triton prefill")

    def patched_init(self, num_embeddings: int, dim: int, layer_id: int):
        # NOTE: deliberately NOT calling the stock __init__ — it would
        # allocate the sharded parameters (~55 GiB unified across the two
        # layers) only for us to drop them.
        nn.Module.__init__(self)
        self.dim = dim
        self.tp_size = 1  # full table on every rank; sharding never applies
        shard = _shard_for_layer(model_path, layer_id)
        st = FileEngramStore(shard, f"layers.{layer_id}.engram",
                             num_embeddings, dim)
        self._file_store = st
        self.weight = nn.Parameter(st._weight_view, requires_grad=False)
        self.scale = nn.Parameter(st._scale_view, requires_grad=False)
        self.weight.weight_loader = _load_rows_noop
        self.scale.weight_loader = _load_rows_noop
        self.host_table = _FileHostTableShim()
        self.row_start = 0
        self.rows = num_embeddings
        logger.info(
            "engram file store armed: layer %d rows=%d dim=%d shard=%s "
            "(shared file-backed layout — no lookup all-reduce)",
            layer_id, num_embeddings, dim, shard)

    def _load_rows_noop(param, loaded_weight, *args, **kwargs):
        # The mmap views already hold the authoritative bytes; safetensors
        # get_tensor() yields lazy mmap views, so returning here reads none.
        return None

    up.EngramEmbedding.__init__ = patched_init
    up.EngramEmbedding._dsv41_file_store = True
    _InstallState.installed = True
    logger.info("dsv41 engram file store installed (model_path=%s)", model_path)


def _shard_for_layer(model_path: str, layer_id: int) -> str:
    """Resolve the shard file holding this layer's engram tensors from the
    checkpoint's model.safetensors.index.json (cached)."""
    global _SHARD_INDEX
    if _SHARD_INDEX is None:
        with open(os.path.join(model_path, "model.safetensors.index.json")) as f:
            _SHARD_INDEX = json.load(f)["weight_map"]
    key = f"layers.{layer_id}.engram.embed.weight"
    if key not in _SHARD_INDEX:
        raise KeyError(f"{key} missing from index")
    return os.path.join(model_path, _SHARD_INDEX[key])
