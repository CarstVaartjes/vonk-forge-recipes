"""The proposal rule as pure torch: the matcher's reference and the override of one draft block.

Matcher (per row b, history h = hist[slot_b, :L_b], L = the anchor position + 1):
  e*  = the largest e with KEY <= e < L and h[e-KEY:e] == h[L-KEY:L]   (all full KEY-gram hits; none -> no copy)
  m   = KEY + the number of further matching tokens going backwards from e*-KEY / L-KEY, capped at MAXLEN
  c   = h[e* : min(e* + G, L)]
  the row is forced when m >= FORCE and len(c) >= 1: slots j < len(c) take c_j.
The measured rule: KEY 12, FORCE 16, MAXLEN 64 (chosen on replayed step logs).

Override of one block (all rows, masked, no host branch):
  draft[b, j]           = c_j where forced (the verify input aliases this buffer)
  sampled rows: owned copies of the selector sample (candidate_ids [bs, G, K], q_rows [bs, G, K]); a forced slot is a
  point mass on c_j: at its existing lane when c_j is a candidate, else in lane 0 after a full-row miss (lane 0's id is
  replaced, so ids stay unique); every other slot keeps its row.
"""

import torch

KEY = 12
FORCE = 16
MAXLEN = 64


def match_ref(hist, slots, lens, valid, gamma, key=KEY, maxlen=MAXLEN):
    """Reference matcher. hist [R, C] int, slots [bs], lens [bs] (L), valid [bs] bool.
    Returns (m [bs] int64, n [bs] int64 = len(c), cont [bs, gamma] int64 (-1 past n))."""
    bs = int(slots.shape[0])
    m_out = torch.zeros(bs, dtype=torch.int64)
    n_out = torch.zeros(bs, dtype=torch.int64)
    cont = torch.full((bs, gamma), -1, dtype=torch.int64)
    for b in range(bs):
        L = int(lens[b])
        slot = int(slots[b])
        if not bool(valid[b]) or L < key + 1 or not (0 <= slot < hist.shape[0]):
            continue
        h = hist[slot].to(torch.int64)
        if L > h.shape[0]:
            continue
        q = h[L - key:L]
        best = -1
        for e in range(L - 1, key - 1, -1):  # e < L: the most recent earlier occurrence
            if torch.equal(h[e - key:e], q):
                best = e
                break
        if best < 0:
            continue
        m = key
        while m < maxlen and best - m - 1 >= 0 and int(h[best - m - 1]) == int(h[L - m - 1]):
            m += 1
        n = min(gamma, L - best)
        m_out[b], n_out[b] = m, n
        cont[b, :n] = h[best:best + n]
    return m_out, n_out, cont


def force_mask(m, n, gamma, force=FORCE):
    """[bs, gamma] bool: slot j of row b is forced."""
    j = torch.arange(gamma, device=m.device)
    return (m >= force)[:, None] & (j[None, :] < n[:, None])


def override_tokens(draft, cont, forced):
    """draft [bs, gamma] (the block's drafted slots, in place) <- cont where forced."""
    draft.copy_(torch.where(forced, cont.to(draft.dtype), draft))
    return draft


def override_sample(candidate_ids, q_rows, cont, forced):
    """Owned copies of the selector sample with forced slots made point masses (see module docstring)."""
    ids = candidate_ids.clone()
    q = q_rows.clone()
    c = cont.to(ids.dtype)
    hit = ids == c[..., None]                                  # [bs, G, K]
    has = hit.any(dim=-1)
    lane = torch.where(has, hit.to(torch.int64).argmax(dim=-1), torch.zeros_like(c, dtype=torch.int64))
    replace0 = forced & ~has
    ids[..., 0] = torch.where(replace0, c, ids[..., 0])
    one_hot = torch.nn.functional.one_hot(lane, num_classes=q.shape[-1]).to(q.dtype)
    q.copy_(torch.where(forced[..., None], one_hot, q))
    return ids, q
