"""Device-side logic of the verify cap: pure torch, no host syncs, no sglang imports.

Verify block layout (static DSpark verify, request-major): request ``b`` owns rows
``b*S .. b*S+S-1`` where S = gamma + 1. Row 0 is the anchor (last committed token);
row r >= 1 holds draft token d_{r-1}. The target's logits at row r predict the token
after row r, so the greedy accept compares argmax(row r) with d_r for r = 0..S-2.

A live length L (1 <= L <= S) keeps rows 0..L-1 exact and lets the accept check at
most L-1 drafts; the bonus comes from row min(correct, L-1). Rows >= L are "dead".
"""

from __future__ import annotations

from typing import Optional

import torch


def draft_token_probs(
    step_logits: torch.Tensor,
    draft_tokens: torch.Tensor,
    *,
    temperatures: Optional[torch.Tensor] = None,
    greedy_mask: Optional[torch.Tensor] = None,
    use_max: bool = False,
    chunk_rows: int = 64,
) -> torch.Tensor:
    """[bs, gamma] probability the draft gave its proposed token (``use_max``: the
    max probability) at each position, from the draft's kept step logits
    [bs, gamma, V]. Sampled rows use their temperature, greedy rows temperature 1.
    Out-of-range token ids are clamped (only affects the cut, never exactness)."""
    bs, gamma, width = step_logits.shape
    rows = bs * gamma
    flat = step_logits.reshape(rows, width)
    tokens = draft_tokens.reshape(rows).to(torch.long).clamp(0, width - 1)
    temp_rows = None
    if temperatures is not None:
        temps = temperatures.reshape(-1)[:bs].to(torch.float32).clamp_min(1e-5)
        if greedy_mask is not None:
            temps = torch.where(
                greedy_mask.reshape(-1)[:bs].to(torch.bool),
                torch.ones_like(temps),
                temps,
            )
        temp_rows = temps.repeat_interleave(gamma)
    out = torch.empty(rows, dtype=torch.float32, device=step_logits.device)
    step = max(1, int(chunk_rows))
    for start in range(0, rows, step):
        end = min(rows, start + step)
        x = flat[start:end].to(torch.float32)
        if temp_rows is not None:
            x = x / temp_rows[start:end, None]
        lse = torch.logsumexp(x, dim=-1)
        if use_max:
            picked = x.amax(dim=-1)
        else:
            picked = x.gather(1, tokens[start:end, None]).squeeze(1)
        torch.exp(picked - lse, out=out[start:end])
    out = torch.nan_to_num(out, nan=0.0).clamp_(0.0, 1.0)
    return out.view(bs, gamma)


def live_lens_from_probs(
    probs: torch.Tensor, *, threshold: float, min_len: int, max_len: int
) -> torch.Tensor:
    """[bs] int64 live lengths. surv_j = prod_{i<=j} p_i estimates P(d_0..d_j all
    accepted); draft j is worth verifying (row j+1 live) while surv_j >= threshold.
    cumprod of values in [0, 1] is non-increasing, so the count of qualifying
    positions is the leading run: L = 1 + count, clamped to [min_len, max_len]."""
    surv = torch.cumprod(probs.to(torch.float32), dim=1)
    count = (surv >= float(threshold)).sum(dim=1)
    return (count + 1).clamp_(int(min_len), int(max_len)).to(torch.int64)


class RowLayout:
    """Precomputed per-row tensors for ``capacity`` requests of ``stride`` rows."""

    def __init__(self, capacity: int, stride: int, device) -> None:
        self.capacity = int(capacity)
        self.stride = int(stride)
        n = self.capacity * self.stride
        self.rows = torch.arange(n, dtype=torch.int64, device=device)
        self.req = torch.div(self.rows, self.stride, rounding_mode="floor")
        self.pos = self.rows - self.req * self.stride
        self.anchor = self.req * self.stride


def build_src_index(
    live_lens: torch.Tensor,
    layout: RowLayout,
    *,
    num_reqs: Optional[int] = None,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Row -> source row for the expert-id remap. Live rows map to themselves, dead
    rows of a real request (L >= 1) to that request's anchor row, rows of padding
    requests (L == 0) to row 0. With ``num_reqs`` only the first num_reqs*stride rows
    are rebuilt (the rest must already be valid, e.g. built once with zero L)."""
    n = layout.capacity * layout.stride if num_reqs is None else num_reqs * layout.stride
    if out is None:
        out = torch.empty(layout.capacity * layout.stride, dtype=torch.int64,
                          device=layout.rows.device)
    req = layout.req[:n]
    lens_rows = live_lens.to(torch.int64).index_select(0, req)
    dead_target = torch.where(lens_rows > 0, layout.anchor[:n], torch.zeros_like(req))
    torch.where(layout.pos[:n] < lens_rows, layout.rows[:n], dead_target, out=out[:n])
    return out


def remap_rows(
    weights: torch.Tensor, ids: torch.Tensor, src_index: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Overwrite every row's routed (weights, ids) with its source row's."""
    m = ids.shape[0]
    idx = src_index[:m]
    return weights.index_select(0, idx), ids.index_select(0, idx)


def cap_correct_len(
    correct_len: torch.Tensor, live_lens: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reference of the engine's CapCorrectLen: (capped, trimmed)."""
    ell = live_lens.to(correct_len.dtype) - 1
    capped = torch.clamp(torch.minimum(correct_len, ell), min=0)
    return capped, correct_len - capped


def greedy_accept(
    candidates: torch.Tensor, target_predict: torch.Tensor,
    live_lens: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reference greedy accept over [bs, S] candidates and [bs, S] target argmax ids:
    (correct_len, bonus), capped at live_lens - 1 when given."""
    bs, stride = candidates.shape
    match = (candidates[:, 1:] == target_predict[:, : stride - 1]).to(torch.int64)
    correct = match.cumprod(dim=1).sum(dim=1)
    if live_lens is not None:
        correct, _ = cap_correct_len(correct, live_lens)
    bonus = target_predict[torch.arange(bs), correct]
    return correct, bonus


def mask_dead_draft_rows(
    draft_probs: torch.Tensor, draft_tokens: torch.Tensor, live_lens: torch.Tensor
) -> torch.Tensor:
    """Block verification under the cap: make the stock block-verification kernel see
    a draft that stops after its live drafts. In place on ``draft_probs`` [bs, G, V]
    (contiguous FP32 probabilities; row j is the law of d_j), ``draft_tokens``
    [bs, G], ``live_lens`` [>= bs] (L: rows 0..L-1 live, so g = L-1 live drafts).

    For every request with g < G:
      - row g (the law of the first dead draft d_g) becomes all zeros;
      - q_j(d_j) becomes 0 for every j >= g.
    This is exactly the draft of the stopped chain written over the real vocabulary:
    after the stop the draft emits a token outside the vocabulary with probability 1
    (README, "Block verification under the cap"). The kernel then gets prefix
    probability 0 for every step past g (q is not a probability: q <= 0), h_g = p_g
    (its residual mass is p_g x sum(p_next) - 0) and the bonus at tau = g is drawn
    from the target row g alone. Rows g+1.. are never read in full. Requests with
    g >= G are left as they are. Fixed shapes, no host sync (graph-safe)."""
    bs, gamma, vocab = draft_probs.shape
    if not draft_probs.is_contiguous():
        raise RuntimeError("mask_dead_draft_rows needs contiguous draft probabilities")
    if bs == 0:
        return draft_probs
    device = draft_probs.device
    g = (live_lens[:bs].to(torch.int64) - 1).clamp_(min=0)
    flat = draft_probs.view(bs * gamma, vocab)
    # 1) the cut row: zeroed when g < G (a request with g >= G rewrites row G-1 as is)
    idx = torch.arange(bs, device=device, dtype=torch.int64) * gamma + g.clamp(max=gamma - 1)
    keep = (g >= gamma).to(draft_probs.dtype)[:, None]
    flat.index_copy_(0, idx, flat.index_select(0, idx) * keep)
    # 2) the proposed tokens' probabilities past the cut
    tok = draft_tokens[:bs].to(device=device, dtype=torch.int64).clamp(0, vocab - 1)
    dead = torch.arange(gamma, device=device)[None, :] >= g[:, None]
    q = draft_probs.gather(2, tok[..., None])
    draft_probs.scatter_(2, tok[..., None], q.masked_fill(dead[..., None], 0.0))
    return draft_probs


def distinct_experts(ids: torch.Tensor, num_experts: int) -> torch.Tensor:
    """0-d float tensor: number of distinct experts in ``ids``. Graph-safe (no sync)."""
    present = torch.zeros(num_experts, dtype=torch.float32, device=ids.device)
    flat = ids.reshape(-1).to(torch.int64).clamp(0, num_experts - 1)
    present.index_fill_(0, flat, 1.0)
    return present.sum()


def calibrated_live_lens(
    m: torch.Tensor,
    sampled: torch.Tensor,
    *,
    table: torch.Tensor,
    edges: torch.Tensor,
    row_cost: torch.Tensor,
    theta: float,
    min_len: int,
    max_len: int,
    u1: Optional[torch.Tensor] = None,
    pool: float = 0.0,
    rounds: int = 0,
) -> torch.Tensor:
    """[bs] int64 live lengths of the calibrated cost rule (``policy:`` mode).

    m [bs, gamma]: the draft's per-position max probabilities (``draft_token_probs``
    with use_max). sampled [bs] bool picks the calibration table: table [2, gamma, nb]
    (0 = greedy rows, 1 = sampled rows), a_j = table[mode, j, bin(m_j)], bin(m) =
    #{edges <= m}. S_j = a_0 * ... * a_j estimates P(d_0..d_j all accepted).
    (Leading dims are allowed: m [..., bs, gamma], sampled [..., bs]; the batch is
    dim -2. The tests use this to run many independent batches at once.)

    Draft d_j (verify row j+1) is live while S_j >= theta * P_r * row_cost[j]:
    row_cost[j] is the c1 marginal expert union of row j+1 (u1[j+1] - u1[j]) and P_r
    the probability that a new expert of request r is not already read for another
    request. rounds = 0 (or bs = 1, or pool <= 0): P_r = 1, a per-request rule. With
    rounds K > 0 and an effective expert pool E (> u1[-1]):

        L^0 = 1;  P_r^k = prod_{s != r} (1 - u1(L_s^{k-1}) / E);  L^k = rule(P^k)

    P_r^k only falls when other requests' widths grow and L^1 >= L^0, so the rounds
    are monotone (L^k >= L^{k-1}). Each round is a left-to-right rule on request r's
    own S_0..S_j (functions of the drafts before d_j) with a threshold from the other
    requests' widths of the previous round, which in turn only read r's survival up to
    r's final width. So {L_r = l} is decided by d_0..d_{l-2}: a stopping rule, exact
    for sampled rows (README, "policy: mode"). No host sync; fixed launch count."""
    bs, gamma = m.shape[-2], m.shape[-1]
    mode = sampled.to(torch.int64)
    bins = torch.bucketize(m.to(torch.float32).contiguous(), edges, right=True)
    pos = torch.arange(gamma, device=m.device)
    a = table[mode[..., None], pos, bins]  # [..., bs, gamma]
    surv = torch.cumprod(a, dim=-1)
    base = float(theta) * row_cost

    def rule(thr):
        ok = (surv >= thr).to(torch.int64)
        return 1 + torch.cumprod(ok, dim=-1).sum(dim=-1)

    if rounds <= 0 or bs == 1 or pool <= 0.0 or u1 is None:
        lens = rule(base)
    else:
        lens = torch.ones(m.shape[:-1], dtype=torch.int64, device=m.device)
        for _ in range(int(rounds)):
            free = (1.0 - u1[lens - 1] / float(pool)).clamp_min(1e-6)
            logf = torch.log(free)
            p_other = torch.exp(logf.sum(dim=-1, keepdim=True) - logf)  # prod s != r
            lens = rule(base * p_other[..., None])
    return lens.clamp_(int(min_len), int(max_len))
