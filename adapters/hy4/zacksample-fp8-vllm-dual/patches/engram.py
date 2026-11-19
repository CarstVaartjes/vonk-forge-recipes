# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Flash frozen n-gram embeddings with TP head sharding and V2 lookback."""

import os

import numpy as np
import torch
from torch import nn

from vllm.compilation.breakable_cudagraph import eager_break_during_capture
from vllm.distributed import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_reduce,
)
from vllm.forward_context import get_forward_context
from vllm.logger import init_logger
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.utils import set_weight_attrs
from vllm.models.deepseek_v41.common.engram import NgramHashState
from vllm.models.qwen4_exp.common.ngram_embedding import (
    _lookup_ple_embedding_from_pinned_kernel,
)
from vllm.triton_utils import triton
from vllm.utils.platform_utils import is_uva_available
from vllm.utils.torch_utils import get_accelerator_view_from_cpu_tensor

logger = init_logger(__name__)

# --- hy4-flash-sparks patch: Engram tables read from the checkpoint files -------------
# HY4_ENGRAM_DISK=1 keeps the four 31.7 GB tables out of memory. The loader hands this
# module a file-backed (mmap) view of each table; we keep the view and gather each
# step's rows on the host. Touched rows live in the page cache, which the kernel can
# drop, so a 128 GB unified-memory box (DGX Spark) only pays for what it uses.
ENGRAM_ON_DISK = os.environ.get("HY4_ENGRAM_DISK", "0") == "1"


def _file_backing(tensor):
    """(path, byte offset) of the file a loader-provided mmap view starts at."""
    ptr = tensor.data_ptr()
    with open("/proc/self/maps") as maps:
        for line in maps:
            parts = line.split(None, 5)
            lo, hi = (int(x, 16) for x in parts[0].split("-"))
            if lo <= ptr < hi and len(parts) == 6 and parts[5].strip().startswith("/"):
                return parts[5].strip(), int(parts[2], 16) + (ptr - lo)
    raise ValueError("Engram table is not a file-backed view")


class _RowFile:
    """Random reads of fixed-size rows: read-ahead off, every row's page requested first (the kernel
    fetches them together), then copied out. A page fault per row through the mmap was ~10x slower."""

    def __init__(self, path, base, row_bytes):
        self.path, self.base, self.row_bytes = path, base, row_bytes
        self.fd = os.open(path, os.O_RDONLY)
        os.posix_fadvise(self.fd, 0, 0, os.POSIX_FADV_RANDOM)

    def read(self, rows):
        """rows: sorted int64 tensor of row numbers -> uint8 tensor [len(rows), row_bytes]."""
        fd, rb = self.fd, self.row_bytes
        offsets = (rows * rb + self.base).tolist()
        buf = np.empty((len(offsets), rb), dtype=np.uint8)
        for o in offsets:
            os.posix_fadvise(fd, o, rb, os.POSIX_FADV_WILLNEED)
        for i, o in enumerate(offsets):
            if os.preadv(fd, [buf[i]], o) != rb:
                raise ValueError(f"{self.path} is truncated at {o}")
        return torch.from_numpy(buf)


def _local_copy(view, name):
    """With HY4_ENGRAM_LOCAL_DIR set (the worker, whose checkpoint is on NFS): keep this rank's rows in a
    file on local disk, written once."""
    folder = os.environ.get("HY4_ENGRAM_LOCAL_DIR")
    if not folder:
        return None
    path = os.path.join(folder, f"{name}.bf16")
    need = view.numel() * view.element_size()
    if not (os.path.exists(path) and os.path.getsize(path) == need):
        logger.info("Flash Engram on disk: copying %.1f GiB to %s (once)", need / 2**30, path)
        step = 2_000_000
        with open(path + ".part", "wb") as f:
            for i in range(0, view.shape[0], step):
                f.write(view[i : i + step].contiguous().view(torch.uint8).numpy().data)
        os.replace(path + ".part", path)
    return path


def _disk_gather(table, vocab_start, indices, out):
    """Write table[indices - vocab_start] into `out` (GPU). Masked ids (< 0) read row 0;
    the caller zeroes them."""
    from vllm.utils.gpu_sync_debug import gpu_sync_allowed

    with gpu_sync_allowed():
        idx = indices.reshape(-1).cpu()
    local = (idx - vocab_start).clamp_(0, table.shape[0] - 1)
    uniq, inverse = local.unique(return_inverse=True)  # sorted: file order
    host = table.index_select(0, uniq).index_select(0, inverse)
    out.copy_(host.view(out.shape))


class FlashEngramLayout:
    def __init__(self, config):
        self.layer_ids = tuple(config.engram_layer_ids or ())
        self.max_ngram_size = config.engram_max_ngram_size
        self.n_heads = config.engram_n_head_per_ngram
        self.head_dim = config.engram_n_embed_per_ngram // self.n_heads
        self.pad_token_id = config.engram_pad_id
        self.vocab_size = config.vocab_size
        self.seed = config.engram_seed
        self.lookup_table = None
        if not 0 <= self.pad_token_id < self.vocab_size:
            raise ValueError("engram_pad_id is outside the vocabulary")
        groups = config.engram_layer_head_vocab_sizes
        self.primes = tuple(
            groups.get(str(layer), groups.get(layer)) for layer in self.layer_ids
        )
        flat = [[p for order in group for p in order] for group in self.primes]
        self.offsets = np.array(
            [np.cumsum([0, *sizes[:-1]]) for sizes in flat], dtype=np.int64
        )
        if any(sum(sizes) > np.iinfo(np.int32).max for sizes in flat):
            raise ValueError("Engram flattened bucket offsets must fit int32")

    def load_lookup(self, value):
        if (
            value.dtype not in (torch.int32, torch.int64)
            or value.ndim != 1
            or not 0 < value.numel() <= self.vocab_size
        ):
            raise ValueError("Invalid frozen Engram token map")
        lookup = value.detach().cpu().numpy()
        if np.any(lookup < 0) or np.any(lookup > np.iinfo(np.int32).max):
            raise ValueError("Invalid frozen Engram token map")
        if self.lookup_table is not None:
            if not np.array_equal(self.lookup_table, lookup):
                raise ValueError("Engram layers must share the same token map")
            return False
        self.compressed_vocab_size = int(lookup.max()) + 1
        self.lookup_table = lookup.copy()
        self.token_map = lookup.tolist()
        self.token_map.extend(range(len(self.token_map), self.vocab_size))
        bound = max(1, (np.iinfo(np.int64).max // self.compressed_vocab_size) // 2)
        self.multipliers = np.stack(
            [
                np.random.default_rng(self.seed + 10007 * layer).integers(
                    0, bound, size=self.max_ngram_size, dtype=np.int64
                )
                * 2
                + 1
                for layer in self.layer_ids
            ]
        )

        return True


def head_range(sizes, tp, rank):
    if not 1 <= tp <= len(sizes) or not 0 <= rank < tp:
        raise ValueError("Engram requires at least one complete hash head per TP rank")
    prefix = np.cumsum([0, *sizes]).tolist()
    boundaries = [0]
    for r in range(1, tp):
        target = prefix[-1] * r / tp
        candidates = range(boundaries[-1] + 1, len(sizes) - (tp - r) + 1)
        boundaries.append(min(candidates, key=lambda i: abs(prefix[i] - target)))
    boundaries.append(len(sizes))
    return boundaries[rank], boundaries[rank + 1]


class FlashNgramHashState(NgramHashState):
    def __init__(self, config, *, dummy=False):
        nn.Module.__init__(self)
        layout = FlashEngramLayout(config)
        self.layout: FlashEngramLayout = layout  # type: ignore[assignment]
        self.lookback_depth = self.layout.max_ngram_size - 1
        self.block_size = 1
        self.use_slot_cache = False
        self._cache = None
        self.pad_id = 0
        self.visual_token_ids: tuple[int, ...] = ()
        self.register_buffer(
            "token_map",
            torch.zeros(config.vocab_size, dtype=torch.int32),
            persistent=False,
        )
        self.register_buffer(
            "multipliers",
            torch.zeros(
                len(self.layout.layer_ids),
                self.layout.max_ngram_size,
                dtype=torch.int64,
            ),
            persistent=False,
        )
        for name, values in (
            ("primes", self.layout.primes),
            ("offsets", self.layout.offsets),
        ):
            self.register_buffer(
                name, torch.tensor(values, dtype=torch.int64), persistent=False
            )
        if dummy:
            self.load_lookup(torch.arange(config.vocab_size, device="cpu"))

    def load_lookup(self, value):
        if self.layout.load_lookup(value):
            self.token_map.copy_(
                torch.as_tensor(
                    self.layout.token_map,
                    dtype=torch.int32,
                    device=self.token_map.device,
                )
            )
            self.multipliers.copy_(
                torch.as_tensor(self.layout.multipliers, device=self.multipliers.device)
            )
            self.pad_id = self.layout.token_map[self.layout.pad_token_id]

    @eager_break_during_capture
    def prepare_hashes_with_output(self, input_ids, lookback, layer_name, output):
        metadata = get_forward_context().attn_metadata
        output.copy_(
            self.prepare_hashes(
                input_ids,
                lookback,
                metadata[layer_name] if isinstance(metadata, dict) else None,
            )
        )

    def prepare_hashes(self, input_ids, lookback, metadata):
        tokens = input_ids.numel()
        if metadata is None or tokens == 0:
            return input_ids.new_full(
                (len(self.layout.layer_ids), tokens, self.offsets.shape[1]),
                -1,
                dtype=torch.int32,
            )
        if self.layout.lookup_table is None:
            raise ValueError("Missing checkpoint Engram token map")
        if lookback is None or lookback.shape[1] != self.lookback_depth:
            raise ValueError("Flash Engram requires V2 lookback_token_ids")
        starts = metadata.query_start_loc
        rows = torch.arange(tokens, device=input_ids.device)
        requests = torch.searchsorted(starts[1:], rows, right=True)
        requests = requests.clamp_max(starts.numel() - 2).long()
        logical = (
            metadata.seq_lens[requests]
            - (starts[requests + 1] - starts[requests])
            + rows
            - starts[requests]
        ).long()
        dead = (input_ids < 0) | (input_ids >= self.token_map.numel())
        safe_ids = input_ids.clamp(0, self.token_map.numel() - 1)
        lookback_dead = lookback >= self.token_map.numel()
        safe_lookback = lookback.clamp(-1, self.token_map.numel() - 1)
        hashes = super().forward(
            safe_ids,
            logical,
            starts,
            dead,
            safe_lookback,
            lookback_dead,
        )
        offsets = self.offsets.unsqueeze(0)
        primes = self.primes.flatten(1).unsqueeze(0)
        hashes = torch.where(hashes < offsets, hashes + primes, hashes).int()
        no_hash = dead | (rows >= metadata.num_actual_tokens) | (rows >= starts[-1])
        no_hash |= input_ids == self.layout.pad_token_id
        for token_id in self.visual_token_ids:
            no_hash |= input_ids == token_id
        hashes.masked_fill_(no_hash[:, None, None], -1)
        return hashes.permute(1, 0, 2).contiguous()


class FlashEmbedding(nn.Module):
    def __init__(self, sizes, dim, dtype, cpu_offload):
        super().__init__()
        self.tp_size = get_tensor_model_parallel_world_size()
        self.head_start, self.head_end = head_range(
            sizes, self.tp_size, get_tensor_model_parallel_rank()
        )
        self.vocab_start = sum(sizes[: self.head_start])
        self.vocab_end = sum(sizes[: self.head_end])
        self.num_embeddings = sum(sizes)
        self.n_hash_cols = len(sizes)
        self.dim = dim
        self.disk = ENGRAM_ON_DISK and dtype is None
        self._disk_weight = None
        self._disk_scale = None
        self._rowfile = None
        self._staged = None  # [max step tokens, local heads, dim] on the GPU, filled before each step
        if self.disk:
            cpu_offload = False
        self.cpu_offload = cpu_offload
        if cpu_offload and not is_uva_available():
            raise RuntimeError("Flash Engram CPU offload requires UVA")
        rows = self.vocab_end - self.vocab_start
        storage = {"device": "cpu", "pin_memory": True} if cpu_offload else {}
        self.embedding = nn.Module()
        self.embedding.weight = nn.Parameter(
            torch.empty(
                (0,) if dtype is None else (rows, dim),
                dtype=dtype or torch.bfloat16,
                **storage,
            ),
            requires_grad=False,
        )
        self.fp8 = dtype == torch.float8_e4m3fn
        self.embedding.register_parameter("weight_scale_inv", None)
        if self.fp8:
            self.embedding.weight_scale_inv = nn.Parameter(
                torch.empty(rows, dtype=torch.bfloat16, **storage), requires_grad=False
            )
        for parameter in self.embedding.parameters():
            set_weight_attrs(parameter, {"weight_loader": self._load_shard})
        self._uva_weight = None
        self._uva_scale = None
        logger.info(
            "Flash Engram TP table: %d rows x %d, dtype=%s, cpu_offload=%s",
            rows,
            dim,
            dtype,
            cpu_offload,
        )

    def load_weight(self, leaf, value):
        if leaf == "weight_scale_inv" and self.embedding.weight_scale_inv is None:
            self.embedding.weight_scale_inv = nn.Parameter(
                self.embedding.weight.new_empty(0, dtype=torch.bfloat16),
                requires_grad=False,
            )
            set_weight_attrs(
                self.embedding.weight_scale_inv, {"weight_loader": self._load_shard}
            )
        self._load_shard(getattr(self.embedding, leaf), value)

    def _load_shard(self, parameter, value):
        is_weight = parameter is self.embedding.weight
        if self.disk:
            want = (self.num_embeddings, self.dim) if is_weight else (self.num_embeddings,)
            if tuple(value.shape) != want or value.device.type != "cpu":
                raise ValueError(
                    f"Engram on disk needs the checkpoint's CPU view, got "
                    f"{tuple(value.shape)}/{value.dtype}/{value.device}"
                )
            view = value[self.vocab_start : self.vocab_end]
            if is_weight:
                self._disk_weight = view
                self.fp8 = value.dtype == torch.float8_e4m3fn
                row_bytes = self.dim * view.element_size()
                local = _local_copy(view, f"{getattr(self, '_disk_name', 'table')}_{self.vocab_start}_{self.vocab_end}")
                if local:
                    self._rowfile = _RowFile(local, 0, row_bytes)
                else:
                    self._rowfile = _RowFile(*_file_backing(view), row_bytes)
                logger.info("Flash Engram on disk: rows read from %s", self._rowfile.path)
            else:
                self._disk_scale = view
            logger.info(
                "Flash Engram on disk: %s rows %d..%d (%s, %.1f GiB left in the file)",
                "table" if is_weight else "scales",
                self.vocab_start,
                self.vocab_end,
                value.dtype,
                view.numel() * view.element_size() / 2**30,
            )
            return
        expected = (
            (self.num_embeddings, self.dim) if is_weight else (self.num_embeddings,)
        )
        dtypes = (
            (torch.bfloat16, torch.float8_e4m3fn) if is_weight else (torch.bfloat16,)
        )
        if tuple(value.shape) != expected or value.dtype not in dtypes:
            raise ValueError(
                f"Invalid Engram table: {tuple(value.shape)}/{value.dtype}"
            )
        if parameter.numel() and parameter.dtype != value.dtype:
            raise ValueError("Engram table dtype changed during weight reload")
        if is_weight:
            if (
                value.dtype != torch.float8_e4m3fn
                and self.embedding.weight_scale_inv is not None
            ):
                raise ValueError("BF16 Engram table must not have FP8 row scales")
        elif self.embedding.weight.numel() and not self.fp8:
            raise ValueError("BF16 Engram table must not have FP8 row scales")
        if parameter.numel() == 0:
            parameter.data = torch.empty(
                (self.vocab_end - self.vocab_start, *expected[1:]),
                dtype=value.dtype,
                device=parameter.device,
                pin_memory=self.cpu_offload,
            )
        default_weight_loader(parameter, value[self.vocab_start : self.vocab_end])
        if is_weight:
            self.fp8 = value.dtype == torch.float8_e4m3fn
        if is_weight and self.fp8 and self.embedding.weight_scale_inv is None:
            self.embedding.weight_scale_inv = nn.Parameter(
                parameter.new_empty(0, dtype=torch.bfloat16), requires_grad=False
            )
            set_weight_attrs(
                self.embedding.weight_scale_inv, {"weight_loader": self._load_shard}
            )
        self._uva_weight = None
        self._uva_scale = None

    # A breakable CUDA graph stops at these, runs the host read, and starts a new segment. The table is reached
    # through `self` because the capture can only hold GPU tensors (and plain objects) as arguments.
    @eager_break_during_capture
    def _disk_rows_with_output(self, indices, out):
        _disk_gather(self._disk_weight, self.vocab_start, indices, out)

    @eager_break_during_capture
    def _disk_scale_with_output(self, indices, out):
        _disk_gather(self._disk_scale, self.vocab_start, indices, out)

    def _lookup(self, parameter, indices, out, *, scale=False):
        if self.disk:
            table = self._disk_scale if scale else self._disk_weight
            if table is None:
                raise RuntimeError("Engram table was not loaded from the checkpoint")
            if scale:
                self._disk_scale_with_output(indices, out)
            else:
                self._disk_rows_with_output(indices, out)
            return
        view_name = "_uva_scale" if scale else "_uva_weight"
        view = getattr(self, view_name)
        if view is None:
            view = (
                get_accelerator_view_from_cpu_tensor(parameter)
                if self.cpu_offload
                else parameter
            )
            setattr(self, view_name, view)
        row_bytes = parameter.element_size() * (1 if scale else self.dim)
        _lookup_ple_embedding_from_pinned_kernel[(indices.numel(),)](
            view.view(torch.uint8),
            indices,
            out.view(torch.uint8),
            row_bytes,
            self.vocab_start,
            self.vocab_end,
            BLOCK_D=triton.next_power_of_2(row_bytes),
        )

    def forward(self, hashes):
        indices = hashes[:, self.head_start : self.head_end].long().contiguous()
        shape = (*indices.shape, self.dim)
        rows = torch.empty(
            shape,
            dtype=(
                self._disk_weight.dtype if self.disk else self.embedding.weight.dtype
            ),
            device=hashes.device,
        )
        if indices.numel() == 0:
            return rows.float()
        if self.disk and self._staged is not None:
            # Rows were read from disk before this step (HY4DiskEngramState.prepare_inputs). A fixed
            # buffer keeps this graph-safe: a captured CUDA graph only sees a GPU slice.
            return self._staged[: indices.shape[0]].float()
        self._lookup(self.embedding.weight, indices, rows)
        if self.fp8:
            scales = torch.empty(
                indices.shape, dtype=torch.bfloat16, device=hashes.device
            )
            self._lookup(self.embedding.weight_scale_inv, indices, scales, scale=True)
            return rows.float() * scales.float().unsqueeze(-1)
        return rows.float()


class FlashEngram(nn.Module):
    def __init__(self, vllm_config, layout, layer_id, prefix):
        super().__init__()
        config = vllm_config.model_config.hf_text_config
        index = layout.layer_ids.index(layer_id)
        sizes = [p for order in layout.primes[index] for p in order]
        storage = vllm_config.engram_config
        if (
            storage.disk_offload
            or storage.embedding_across_dp
            or storage.dp_shared_memory
        ):
            raise ValueError(
                "Flash Engram supports TP-local tables, not DP or disk offload"
            )
        dtype = None
        if vllm_config.load_config.load_format == "dummy":
            dtype = (
                torch.float8_e4m3fn if config.engram_fp8_embedding else torch.bfloat16
            )
        self.multi_head_embedding = FlashEmbedding(
            sizes, layout.head_dim, dtype, storage.cpu_offload
        )
        self.multi_head_embedding._disk_name = f"engram_layer{layer_id}"
        self.hidden_size, self.hc_mult = config.hidden_size, config.hc_mult
        local = (
            self.multi_head_embedding.head_end - self.multi_head_embedding.head_start
        )
        self.lookup_per_head_norms = nn.ModuleList(
            [nn.RMSNorm(layout.head_dim) for _ in range(local)]
        )
        self.value_proj = nn.Linear(local * layout.head_dim, self.hidden_size)
        self.key_projs = nn.ModuleList(
            [
                nn.Linear(local * layout.head_dim, self.hidden_size)
                for _ in range(self.hc_mult)
            ]
        )
        self.norm_key = nn.ModuleList(
            [nn.RMSNorm(self.hidden_size) for _ in range(self.hc_mult)]
        )
        self.norm_query = nn.ModuleList(
            [nn.RMSNorm(self.hidden_size) for _ in range(self.hc_mult)]
        )
        self.unified_pre_gamma = nn.Parameter(torch.empty(self.hidden_size))
        self.unified_per_hc_scales = nn.ParameterList(
            [nn.Parameter(torch.empty(self.hidden_size)) for _ in range(self.hc_mult)]
        )
        self.total_width = len(sizes) * layout.head_dim
        self.start = self.multi_head_embedding.head_start * layout.head_dim
        self.stop = self.multi_head_embedding.head_end * layout.head_dim
        self.rank = get_tensor_model_parallel_rank()
        for projection in [self.value_proj, *self.key_projs]:
            set_weight_attrs(
                projection.weight, {"weight_loader": self._load_projection}
            )
            set_weight_attrs(projection.bias, {"weight_loader": self._load_bias})

    def _load_projection(self, parameter, value):
        if value.shape != (self.hidden_size, self.total_width):
            raise ValueError("Flash Engram projection weight shape mismatch")
        default_weight_loader(parameter, value[:, self.start : self.stop])

    def _load_bias(self, parameter, value):
        default_weight_loader(
            parameter, value if self.rank == 0 else torch.zeros_like(value)
        )

    def forward(self, hidden, hashes):
        rows = self.multi_head_embedding(hashes)
        eps = torch.finfo(torch.float32).eps
        gamma = torch.stack(
            [norm.weight for norm in self.lookup_per_head_norms]
        ).float()
        local = self.multi_head_embedding
        local_hashes = hashes[:, local.head_start : local.head_end]
        x = rows.masked_fill(local_hashes.unsqueeze(-1) < 0, 0)
        x = (x * (x.square().mean(-1, keepdim=True) + eps).rsqrt() * gamma).to(
            hidden.dtype
        )
        x = x.flatten(1)
        gates = []
        for channel in range(self.hc_mult):
            key = self.key_projs[channel](x)
            if local.tp_size > 1:
                key = tensor_model_parallel_all_reduce(key)
            gate = (
                self.norm_key[channel](key)
                * self.norm_query[channel](hidden[:, channel])
            ).sum(-1)
            gate = gate * self.hidden_size**-0.5
            gates.append((gate.abs().clamp_min(1e-6).sqrt() * gate.sign()).sigmoid())
        gate = torch.stack(gates, dim=1)
        value = self.value_proj(x)
        if local.tp_size > 1:
            value = tensor_model_parallel_all_reduce(value)
        value = value.float()
        signal = (
            self.unified_pre_gamma.float()
            * value
            * (value.square().mean(-1, keepdim=True) + eps).rsqrt()
        )
        scales = torch.stack(list(self.unified_per_hc_scales)).float()
        result = ((gate.float().unsqueeze(-1) + scales) * signal.unsqueeze(1)).to(
            hidden.dtype
        )
        return hidden + result.masked_fill((hashes < 0).all(-1)[:, None, None], 0)


# --- hy4-flash-sparks patch: read each step's Engram rows before the model runs ----------------------
def stage_disk_rows(flash_model, input_ids, lookback, query_start_loc, seq_lens, num_actual_tokens):
    """Hash this step's tokens exactly as the model will, read those rows from the checkpoint files
    and leave them in each table's staging buffer."""
    from types import SimpleNamespace

    from vllm.utils.gpu_sync_debug import gpu_sync_allowed

    state = flash_model.engram_hash
    meta = SimpleNamespace(
        query_start_loc=query_start_loc, seq_lens=seq_lens, num_actual_tokens=num_actual_tokens
    )
    hashes = state.prepare_hashes(input_ids, lookback, meta)  # [layers, tokens, heads]
    with gpu_sync_allowed():
        host = hashes.cpu()
    for i, layer_id in enumerate(flash_model.config.engram_layer_ids):
        emb = flash_model.layers[layer_id - 1].engram_layer.multi_head_embedding
        if emb._staged is None or emb._disk_weight is None:
            continue
        idx = host[i][:, emb.head_start : emb.head_end].long().reshape(-1)
        table = emb._disk_weight
        local = (idx - emb.vocab_start).clamp_(0, table.shape[0] - 1)
        uniq, inverse = local.unique(return_inverse=True)  # sorted: file order
        rows = emb._rowfile.read(uniq).view(torch.bfloat16).index_select(0, inverse)
        n = host.shape[1]
        emb._staged[:n].copy_(rows.view(n, -1, emb.dim))


def make_disk_engram_state_cls():
    from vllm.models.deepseek_v41.nvidia.model_state import DeepseekV41ModelState

    class HY4DiskEngramState(DeepseekV41ModelState):
        """The stock Hy4 flash runner state, plus the disk read of Engram rows before every step."""

        def __init__(self, vllm_config, model, encoder_cache, device):
            super().__init__(vllm_config, model, encoder_cache, device)
            self._flash = next(
                (m for m in model.modules() if getattr(m, "engram_hash", None) is not None), None
            )
            rows = vllm_config.scheduler_config.max_num_batched_tokens
            if self._flash is not None:
                for layer_id in self._flash.config.engram_layer_ids:
                    emb = self._flash.layers[layer_id - 1].engram_layer.multi_head_embedding
                    if emb.disk:
                        if emb.fp8:
                            raise ValueError("Engram on disk supports BF16 tables only")
                        emb._staged = torch.zeros(
                            (rows, emb.head_end - emb.head_start, emb.dim),
                            dtype=torch.bfloat16,
                            device=device,
                        )
                logger.info("Flash Engram on disk: staging %d rows per step and table", rows)

        def prepare_inputs(self, input_batch, req_states):
            model_inputs = super().prepare_inputs(input_batch, req_states)
            if self._flash is None or self.lookback_token_ids is None:
                return model_inputs
            ids = input_batch.input_ids[: input_batch.num_tokens_after_padding]
            stage_disk_rows(
                self._flash,
                ids,
                self.lookback_token_ids,
                input_batch.query_start_loc,
                input_batch.seq_lens,
                input_batch.num_tokens,
            )
            return model_inputs

    return HY4DiskEngramState


if ENGRAM_ON_DISK:
    # model.py imports this module while building the model, i.e. after its classes exist.
    from . import model as _flash_model_module

    _flash_model_module.HYV4FlashForCausalLM.get_model_state_cls = staticmethod(
        make_disk_engram_state_cls
    )
