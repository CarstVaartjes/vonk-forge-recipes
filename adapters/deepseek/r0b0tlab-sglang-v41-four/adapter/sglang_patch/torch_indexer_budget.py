"""Bound DeepSeek-V4 Torch prefill indexer allocations.

SM12x decode/prefill cannot use DeepGEMM fp8_fp4 MQA. Prefill therefore runs
`_low_ratio_index_topk_torch`, which dequants every visible compressed-K row
before scoring. Slice K, take per-slice topk, merge. Union-of-per-slice-topk
is exact for global topk.

indexer.scores layout is [t, n] (heads already reduced).
"""
from __future__ import annotations

import importlib.abc
import importlib.machinery
import logging
import sys

import torch

LOG = logging.getLogger(__name__)
BACKEND = 'sglang.srt.layers.attention.deepseek_v4_backend'
PAGE = 256


def k_chunk_for(score_budget: int, n_heads: int, n_query: int, k_chunk_max: int = 2048) -> int:
    if type(k_chunk_max) is not int or k_chunk_max < 1:
        raise ValueError('k_chunk_max must be a positive int')
    by_score = max(1, int(score_budget) // max(1, int(n_heads) * 2 * max(1, int(n_query))))
    return min(by_score, k_chunk_max)


def sliced_exact_topk(scores: torch.Tensor, lens: torch.Tensor, k: int, k_chunk: int) -> torch.Tensor:
    """scores: [rows, lc]. Returns [rows, k] indices (k clipped to lc)."""
    if scores.ndim == 3:
        if scores.shape[1] != 1:
            raise ValueError('indexer.scores is [t, n]; pass the reduced layout')
        scores = scores.squeeze(1)
    if scores.ndim != 2:
        raise ValueError('scores must be [rows, lc]')
    if type(k_chunk) is not int or k_chunk < 1:
        raise ValueError('k_chunk must be a positive int')
    rows, lc = scores.shape
    k = min(int(k), lc)
    illegal = torch.arange(lc, device=scores.device)[None, :] >= lens.to(device=scores.device, dtype=torch.int64)[:, None]
    scores = scores.masked_fill(illegal, torch.finfo(scores.dtype).min)
    acc_val = acc_idx = None
    for start in range(0, lc, k_chunk):
        sl = scores[:, start:start + k_chunk]
        kk = min(k, sl.shape[-1])
        val, idx = sl.topk(kk, dim=-1, sorted=False)
        idx = idx + start
        acc_val, acc_idx = _merge_topk(acc_val, acc_idx, val, idx, k)
    return acc_idx


def _merge_topk(acc_val, acc_idx, val, idx, k):
    if acc_val is None:
        return val, idx
    acc_val = torch.cat([acc_val, val], dim=-1)
    acc_idx = torch.cat([acc_idx, idx], dim=-1)
    kk = min(k, acc_val.shape[-1])
    val, sel = acc_val.topk(kk, dim=-1, sorted=False)
    return val, acc_idx.gather(-1, sel)


def _candidate_mask(block_scores, compress_lens, topk_blocks, block_size, width):
    num_blocks = block_scores.shape[-1]
    last = (compress_lens.to(dtype=torch.int64) - 1) // block_size
    scores = block_scores.masked_fill(
        torch.arange(num_blocks, device=block_scores.device) == last, torch.inf
    )
    top = scores.topk(min(int(topk_blocks), num_blocks), dim=-1)
    keep = torch.zeros_like(scores, dtype=torch.bool).scatter_(
        -1, top.indices, top.values > -torch.inf
    )
    return keep.repeat_interleave(block_size, dim=-1)[..., :width]


def make_patched(module):
    def patched(self, layer, x, q_lora, req, pos) -> None:
        pool = self.token_to_kv_pool
        core = self.forward_metadata.core_metadata
        ratio = layer.compress_ratio
        indexer = layer.indexer
        page_indices = core.sparse_page_indices(ratio)
        raw_indices = core.sparse_raw_indices(ratio)
        page_indices.fill_(-1)
        if raw_indices is not None:
            raw_indices.fill_(-1)
        q = indexer.queries(q_lora, layer.freqs_cis[pos])
        weights = indexer.head_weights(x)
        compress_lens = (pos + 1) // ratio
        topk = indexer.index_topk
        publish = [] if indexer.is_candidate_source else None
        consume = self.candidate_masks if indexer.uses_candidates else None
        budget = int(getattr(module, '_TORCH_INDEXER_SCORE_BUDGET_BYTES', 1024 << 20))
        k_max = int(getattr(module, '_TORCH_INDEXER_K_CHUNK_MAX', 2048))
        n_heads = int(q.shape[1])
        for b, r in enumerate(torch.unique_consecutive(req).tolist()):
            tok = (req == r).nonzero().squeeze(1)
            if tok.ndim == 0:
                tok = tok.unsqueeze(0)
            lens = compress_lens[tok]
            lc = int(lens.max().item()) if tok.numel() else 0
            if lc == 0:
                if publish is not None:
                    publish.append(torch.zeros(0, 0, dtype=torch.bool, device=pos.device))
                continue
            j = torch.arange(lc, device=pos.device)
            slots_j = self.req_to_token[r, j * ratio].to(torch.int64) // ratio
            k = min(int(topk), lc)
            k_chunk = k_chunk_for(budget, n_heads, int(tok.numel()), k_chunk_max=k_max)
            if publish is not None:
                bs = int(indexer.candidate_block_size)
                k_chunk = max(bs, k_chunk - (k_chunk % bs))
            acc_val = acc_idx = None
            block_scores = None
            block_size = int(getattr(indexer, 'candidate_block_size', 8) or 8)
            num_blocks = (lc + block_size - 1) // block_size
            for kstart in range(0, lc, k_chunk):
                slen = min(k_chunk, lc - kstart)
                sl_slots = slots_j[kstart:kstart + slen]
                if int(sl_slots.numel()) > k_chunk + PAGE:
                    raise RuntimeError('torch indexer dequant exceeded K slice')
                index_k = pool.get_low_ratio_index_k_dequant(layer.layer_id, sl_slots)
                s = indexer.scores(q[tok], index_k, weights[tok])
                s = s.masked_fill(j[kstart:kstart + slen][None, :] >= lens[:, None], -torch.inf)
                if consume is not None and publish is None:
                    s = s.masked_fill(~consume[b][:, kstart:kstart + slen], -torch.inf)
                if publish is not None:
                    padded = torch.nn.functional.pad(
                        s, (kstart % block_size, (-(kstart + slen) % block_size) % block_size),
                        value=float('-inf'),
                    )
                    sl_blocks = padded.unflatten(-1, (-1, block_size)).amax(dim=-1)
                    first = kstart // block_size
                    if block_scores is None:
                        block_scores = torch.full(
                            (s.shape[0], num_blocks), float('-inf'), device=s.device, dtype=s.dtype
                        )
                    block_scores[:, first:first + sl_blocks.shape[-1]] = torch.maximum(
                        block_scores[:, first:first + sl_blocks.shape[-1]], sl_blocks
                    )
                kk = min(k, slen)
                val, idx = s.topk(kk, dim=-1, sorted=False)
                acc_val, acc_idx = _merge_topk(acc_val, acc_idx, val, idx + kstart, k)
            idx = acc_idx.sort(dim=-1).values
            reach = idx < lens[:, None]
            page_indices[tok, :k] = torch.where(
                reach, slots_j[idx.clamp_max(lc - 1)], -1
            ).to(torch.int32)
            if raw_indices is not None:
                raw_indices[tok, :k] = torch.where(reach, idx, -1).to(torch.int32)
            if publish is not None:
                publish.append(_candidate_mask(
                    block_scores, lens[:, None], indexer.candidate_topk_blocks, block_size, lc
                ))
        if publish is not None:
            self.candidate_masks = publish
    patched._dsv41_kslice = True
    return patched


def apply_to_backend(module):
    cls = module.DeepseekV4AttnBackend
    orig = cls._low_ratio_index_topk_torch
    if getattr(orig, '_dsv41_kslice', False):
        return
    cls._low_ratio_index_topk_torch = make_patched(module)
    LOG.warning('DSV41_TORCH_INDEXER_KSLICE_ACTIVE')


class _AfterImport(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname != BACKEND:
            return None
        spec = importlib.machinery.PathFinder.find_spec(fullname, path)
        if spec is None or spec.loader is None:
            raise ImportError('missing pinned module ' + fullname)
        base = spec.loader

        class Loader(importlib.abc.Loader):
            def create_module(self, spec):
                return base.create_module(spec) if hasattr(base, 'create_module') else None

            def exec_module(self, module):
                base.exec_module(module)
                apply_to_backend(module)
        spec.loader = Loader()
        return spec


def install():
    old = next((f for f in sys.meta_path if isinstance(f, _AfterImport)), None)
    if old is not None:
        return
    if BACKEND in sys.modules:
        apply_to_backend(sys.modules[BACKEND])
    sys.meta_path.insert(0, _AfterImport())
