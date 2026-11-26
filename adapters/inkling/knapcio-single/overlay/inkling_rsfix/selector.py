"""Part `selector`: the DFLASH / DFlash2 selector walk.

Stock bug: a sampled slot picks ``index = #{lanes: u >= cumsum(q)}`` clamped to
``top_k - 1``. When the coin reaches or passes the scan's last value (FP32: the softmax
sum is just below 1), the count is top_k and the clamp picks the LAST candidate, which
can have q = 0 (a -inf score, or one that underflows at the temperature). The draft then
proposes a token outside the support of the q it reports, so the proposal no longer
follows q (the verifier rejects it, but the joint law of draft and verify is no longer
the one the accept rule assumes).

Fixed rule (sampled rows only; greedy rows are untouched): the first lane with
``(cumsum(q) > u) & (q > 0)``; no match -> the last lane with q > 0 (lane 0 only for a row
with no positive q, which is not a distribution). In exact arithmetic the first lane above
u always has q > 0, so the law is unchanged.

Three targets, each sha-pinned and rebound after import:
- kernels/ops/speculative/dflash.py: ``_selector_walk_kernel`` (CUDA). The launcher
  ``selector_walk_triton`` looks it up as a module global at call time, so every caller
  that imported the launcher by name (lilicorr.py, srt/models/dflash.py) gets the fix.
- srt/models/dflash.py: ``CandidateSelector.sample_path`` (its CPU torch path; the CUDA
  path calls selector_walk_triton). Pinned per function: the DFlash2 compatibility patch
  rewrites other parts of that file.
- kernels/ops/speculative/lilicorr.py: ``_selector_walk_torch`` (CPU torch walk).
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn.functional as F

try:  # Triton is absent on the CPU test machine; the torch paths below do not need it.
    import triton
    import triton.language as tl
except Exception:  # noqa: BLE001
    triton = None
    tl = None

# Bound from the stock module by inkling_rsfix.apply() (no sglang import here).
_STOCK_DEPS = ("selector_walk_triton", "_follow_maps")


def _first_positive_above(u, probs):
    """Index of the first lane with cumsum(probs) > u and probs > 0 along the last dim; if
    none, the last lane with probs > 0 (0 if there is none). ``u`` broadcasts against
    ``probs[..., :1]``."""
    k = probs.shape[-1]
    lanes = torch.arange(k, device=probs.device)
    positive = probs > 0
    hit = (probs.cumsum(dim=-1) > u) & positive
    first = torch.where(hit, lanes, torch.full_like(lanes, k)).amin(dim=-1)
    last = torch.where(positive, lanes, torch.full_like(lanes, -1)).amax(dim=-1)
    return torch.where(first < k, first, last.clamp_min(0))


if triton is not None:

    @triton.jit
    def _selector_walk_kernel(
        scores_ptr,
        candidate_ptr,
        uniforms_ptr,
        temperatures_ptr,
        greedy_ptr,
        tokens_ptr,
        q_ptr,
        slots: tl.constexpr,
        top_k: tl.constexpr,
    ):
        """One program per request: a slot's K scores stay in registers and the walk is a
        loop, so the slot-to-slot dependency costs nothing instead of one kernel each."""
        row = tl.program_id(0)
        offsets = tl.arange(0, top_k)
        temperature = tl.load(temperatures_ptr + row)
        greedy = tl.load(greedy_ptr + row) != 0
        previous = 0
        for slot in range(slots):
            base = (row * slots + slot) * top_k
            scores = tl.load(scores_ptr + (base + previous) * top_k + offsets).to(
                tl.float32
            )
            if greedy:
                # Same pick as torch.argmax: the first NaN if any, else the first max.
                is_nan = scores != scores
                best = tl.max(tl.where(is_nan, -float("inf"), scores), axis=0)
                has_nan = tl.max(is_nan.to(tl.int32), axis=0) > 0
                hit = tl.where(has_nan, is_nan, scores == best)
                index = tl.min(tl.where(hit, offsets, top_k), axis=0)
                probabilities = tl.where(offsets == index, 1.0, 0.0)
            else:
                scaled = scores / temperature
                exponentials = tl.exp(scaled - tl.max(scaled, axis=0))
                probabilities = exponentials / tl.sum(exponentials, axis=0)
                uniform = tl.load(uniforms_ptr + row * slots + slot)
                # rsfix: the first lane whose cumulative sum passes the coin AND that has
                # positive probability; no match (the coin at or past the scan's end, FP32
                # rounding) takes the last positive lane, never a q = 0 lane (stock: the
                # count of lanes at or below the coin, clamped to top_k - 1).
                positive = probabilities > 0.0
                hit = (tl.cumsum(probabilities, axis=0) > uniform) & positive
                first = tl.min(tl.where(hit, offsets, top_k), axis=0)
                last = tl.max(tl.where(positive, offsets, -1), axis=0)
                index = tl.where(first < top_k, first, tl.maximum(last, 0))
            tl.store(q_ptr + base + offsets, probabilities)
            tl.store(tokens_ptr + row * slots + slot, tl.load(candidate_ptr + base + index))
            previous = index


# CandidateSelector.sample_path with the lane rule; installed as the class attribute.
def sample_path(
    self,
    *,
    candidate_ids: torch.Tensor,
    scores: torch.Tensor,
    uniforms: torch.Tensor,
    temperatures: torch.Tensor,
    greedy_mask: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Walk one path, with q over the K candidates for the verify. greedy_mask
    rows take the argmax, selected rather than branched, so one captured graph
    serves greedy and sampling batches alike."""
    if scores.is_cuda:
        return selector_walk_triton(
            candidate_ids=candidate_ids,
            scores=scores,
            uniforms=uniforms,
            temperatures=temperatures,
            greedy_mask=greedy_mask,
        )
    top_k = self.top_k
    temps = temperatures.view(-1, 1)
    initial_probs = torch.softmax(scores[:, 0, 0].float() / temps, dim=-1)
    initial_indices = _first_positive_above(uniforms[:, :1], initial_probs)
    transition_probs = torch.softmax(
        scores[:, 1:].float() / temps[:, :, None, None], dim=-1
    )
    local_maps = _first_positive_above(uniforms[:, 1:, None, None], transition_probs)
    initial_indices = torch.where(
        greedy_mask, scores[:, 0, 0].argmax(dim=-1), initial_indices
    )
    local_maps = torch.where(
        greedy_mask[:, None, None], scores[:, 1:].argmax(dim=-1), local_maps
    )
    torch._dynamo.mark_static(local_maps, 1)
    torch._dynamo.mark_static(local_maps, 2)
    path_indices = _follow_maps(
        local_maps, initial_indices, int(scores.shape[1]) - 1
    )
    tokens = candidate_ids.gather(-1, path_indices.unsqueeze(-1))[:, :, 0]
    realized_rows = transition_probs.gather(
        2, path_indices[:, :-1, None, None].expand(-1, -1, 1, top_k)
    )[:, :, 0]
    q_rows = torch.cat((initial_probs.unsqueeze(1), realized_rows), dim=1)
    # Greedy rows walk the argmax, so their q is the point mass there, not
    # the temperature-1 softmax above. The triton walk stores the same.
    q_rows = torch.where(
        greedy_mask[:, None, None], F.one_hot(path_indices, top_k).float(), q_rows
    )
    return tokens, q_rows


# lilicorr._selector_walk_torch with the lane rule.
def _selector_walk_torch(
    *,
    candidate_ids: torch.Tensor,
    scores: torch.Tensor,
    uniforms: torch.Tensor,
    temperatures: torch.Tensor,
    greedy_mask: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    _, num_slots, topk = candidate_ids.shape
    temps = temperatures.view(-1, 1).to(torch.float32)
    greedy = greedy_mask.view(-1)
    previous = torch.zeros(greedy.shape[0], dtype=torch.int64, device=scores.device)
    tokens, q_rows = [], []
    for slot in range(num_slots):
        node = torch.gather(
            scores[:, slot].float(), 1, previous.view(-1, 1, 1).expand(-1, 1, topk)
        ).squeeze(1)
        probs = torch.softmax(node / temps, dim=-1)
        sampled = _first_positive_above(uniforms[:, slot : slot + 1], probs)
        previous = torch.where(greedy, node.argmax(dim=-1), sampled)
        q_rows.append(
            torch.where(
                greedy.unsqueeze(-1),
                F.one_hot(previous, topk).to(torch.float32),
                probs,
            )
        )
        tokens.append(
            torch.gather(candidate_ids[:, slot], 1, previous.view(-1, 1)).squeeze(1)
        )
    return torch.stack(tokens, dim=-1).to(torch.int64), torch.stack(q_rows, dim=1)
