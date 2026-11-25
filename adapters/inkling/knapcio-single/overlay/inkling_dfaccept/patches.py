"""DFlashWorkerV2._selector_sampling_accept -> core.accept_core_sparse on eligible calls."""

import functools

import torch

from . import STATE, DfacceptFatal, _log

MARK = "_inkling_dfaccept"
PRIMS = {}


def _tp1():
    try:
        from sglang.srt.runtime_context import get_parallel

        return int(get_parallel().tp_size) == 1
    except Exception:  # noqa: BLE001 - unknown topology: stock path
        return False


def _min_bs():
    import os

    raw = os.environ.get("INKLING_DFLASH_FUSED_MIN_BS", "").strip()
    return int(raw) if raw else 16


MIN_BS = _min_bs()  # A/B 2026-11-24 (cap off): +1-2 % sampled c24, c8 inconclusive (bench: -0.28 ms at bs 8, -1.4 at 24)


def eligible(sampling_info, next_token_logits, candidates, cutoff_verify_lens):
    """Plain temperature sampling for every request (a greedy row is top_k = 1, so need_top_k_sampling covers it:
    every temperature is then > 0), no verify cutoff, TP1, CUDA, unit vocabulary stride, not under capture."""
    if sampling_info is None or not isinstance(next_token_logits, torch.Tensor) or not next_token_logits.is_cuda:
        return False
    if candidates.shape[0] < MIN_BS:
        return False
    if cutoff_verify_lens is not None or next_token_logits.dim() != 2 or next_token_logits.stride(-1) != 1:
        return False
    if torch.cuda.is_current_stream_capturing():
        return False
    if (getattr(sampling_info, "need_top_k_sampling", True) or getattr(sampling_info, "need_top_p_sampling", True)
            or getattr(sampling_info, "need_min_p_sampling", True)):
        return False
    temps = getattr(sampling_info, "temperatures", None)
    if not isinstance(temps, torch.Tensor) or temps.numel() != candidates.shape[0]:
        return False
    return _tp1()


def patch_worker(mod):
    cls = getattr(mod, "DFlashWorkerV2", None)
    orig = getattr(cls, "_selector_sampling_accept", None) if cls is not None else None
    if orig is None:
        raise DfacceptFatal("inkling_dfaccept: DFlashWorkerV2._selector_sampling_accept is missing")
    if getattr(orig, MARK, False):
        return False

    @functools.wraps(orig)
    def _selector_sampling_accept(self, *, candidates, next_token_logits, candidate_ids, q_rows, sampling_info,
                                  draft_input, cutoff_verify_lens=None):
        STATE["calls"] += 1
        if not eligible(sampling_info, next_token_logits, candidates, cutoff_verify_lens):
            return orig(self, candidates=candidates, next_token_logits=next_token_logits,
                        candidate_ids=candidate_ids, q_rows=q_rows, sampling_info=sampling_info,
                        draft_input=draft_input, cutoff_verify_lens=cutoff_verify_lens)
        from .core import TritonPrims, accept_core_sparse

        prims = PRIMS.get("triton")
        if prims is None:
            prims = PRIMS["triton"] = TritonPrims()
        bs, block = candidates.shape
        coins = torch.rand((bs, block - 1), dtype=torch.float32, device=candidates.device)
        accept_len, bonus, cap_trim = accept_core_sparse(
            prims, candidates=candidates, target_logits=next_token_logits, cand_ids=candidate_ids[:bs],
            q_rows=q_rows[:bs], t_temps=sampling_info.temperatures.view(-1).float(),
            live_lens=None, block_verification=bool(getattr(self, "_block_verification", False)),
            coins=coins)
        STATE["fused"] += 1
        return accept_len.to(torch.int32), bonus.to(torch.int64), cap_trim

    setattr(_selector_sampling_accept, MARK, True)
    cls._selector_sampling_accept = _selector_sampling_accept
    _log("patched DFlashWorkerV2._selector_sampling_accept")
    return True
