"""Verify-width policy on the DFLASH worker (launch.sh DFCAP = dfpolicy:... | fixed:K | conf:T with SPEC=dflash2;
planned per step in DFlashWorkerV2._plan_verify, see dflash.py).

Width rules over the walked-row features (dflash.DflashFeatures: m, gap per slot, written in the draft graph):
  fixed:K          every request verifies rows 0..K-1 (the step-time-vs-width sweep)
  conf:T           live while cumprod(m_0..m_j) >= T (uncalibrated; for A/B only)
  policy (df1)     calibrated survival a(mode, band, j, mbin, gbin) from tools/dfreplay.py --export-policy; draft j
                   is live while S_j = a_0 * ... * a_j >= theta_band * rowcost_band[j] (leading run)
All three are stopping rules: row j's features are functions of d_0..d_{j-1} (dflash.py docstring), and the batch band
is known before drafting. The result L (1 <= L <= 8) per request goes to STATE.live_lens[:bs] (padding requests 0)
and STATE.src_index (logic.build_src_index: dead rows read their request's anchor row; NOEXP's live mask is
src_index[r] == r); the same live_lens[:bs] becomes the cutoff_verify_lens of _accept_block.

Statistics under the cap: DFLASH's AcceptGreedy counts matches over the full width before
clamping to L - 1, so cap_trim_lens and the result's block_accept_lens include dead-row predictions (remapped rows).
Tokens are unaffected, but every statistic or online calibration fed from a capped step must use only
min(accept_len, L - 1) (accept_len as returned is already clamped; never accept_len + cap_trim_lens). Calibration data
comes only from full-width captures (dfsteplog refuses a capped step). The fused sparse-q sampled accept
(DFLASH_FUSED, inkling_dfaccept) serves only calls without a cutoff; capped calls take the stock accept, which
returns the stock cap_trim_lens. Nothing here reads cap_trim_lens.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Optional, Tuple

import torch

FORMAT = "inkling-vcap-policy/df1+features"   # written by tools/dfreplay.py --export-policy
RUNTIME_FEATURES = ("slot", "m", "gap")         # computed on device from DflashFeatures (tail / mt / prev: not yet)


@dataclass(frozen=True)
class DfPolicy:
    bands: Tuple[int, ...]
    features: Tuple[str, ...]
    band_calib: bool
    edges: dict           # feature -> bin edges (bin = #edges <= x, as np.digitize / bucketize(right=True))
    calib: list           # flat table: ((stratum * nb1 + b1) * nb2 + b2) * nb3 + b3
    theta: Tuple[Tuple[int, float], ...]
    rowcost: dict         # band bs -> [G] ms
    bsmax: int
    gamma: int = 7
    theta_x: Tuple[Tuple[int, int, int, float], ...] = ()   # (band bs, mode 0 greedy / 1 sampled, ctx bucket, theta)
    ctx_edges: Tuple[int, ...] = ()                          # context buckets: bucket = #edges <= committed length

    def theta_req(self, band_bs: int, mode: int, cb: int) -> float:
        """Price of one request (tools/dfreplay.theta_of): (band, mode, bucket), then (band, mode, 0), then the band."""
        tx = {(b, m, c): t for b, m, c, t in self.theta_x}
        for key in ((band_bs, mode, cb), (band_bs, mode, 0)):
            if key in tx:
                return tx[key]
        return dict(self.theta).get(band_bs, 0.0)

    def band_index(self, bs: int) -> int:
        for i, b in enumerate(self.bands):
            if bs <= b:
                return i
        return len(self.bands) - 1

    def theta_for(self, bs: int) -> float:
        """The band's largest price over modes / context buckets (0 = the band never caps: uncapped step)."""
        if bs > self.bsmax:
            return 0.0
        band = self.bands[self.band_index(bs)]
        ncb = len(self.ctx_edges) + 1
        return max(self.theta_req(band, m, c) for m in (0, 1) for c in range(ncb))

    def n_bins(self, feature: str) -> int:
        return self.gamma if feature == "slot" else len(self.edges[feature]) + 1


def load_df_policy(path: str) -> DfPolicy:
    with open(path) as fh:
        obj = json.load(fh)
    if obj.get("format") != FORMAT:
        raise ValueError(f"{path}: format must be {FORMAT!r}, got {obj.get('format')!r}")
    feats = tuple(obj["features"])
    bad = [f for f in feats if f not in RUNTIME_FEATURES]
    if bad or len(feats) > 3:
        raise ValueError(f"{path}: runtime supports <= 3 of {RUNTIME_FEATURES}, got {feats}")
    pol = DfPolicy(bands=tuple(int(b) for b in obj["bands"]), features=feats, band_calib=bool(obj["band_calib"]),
                   edges={k: tuple(float(e) for e in v) for k, v in obj["edges"].items()},
                   calib=[float(x) for x in obj["calib"]],
                   theta=tuple((int(k), float(v)) for k, v in obj["theta"]),
                   rowcost={int(k): [float(c) for c in v] for k, v in obj["rowcost"].items()},
                   bsmax=int(obj["bsmax"]), gamma=int(obj["stride"]) - 1,
                   theta_x=tuple((int(b), int(m), int(c), float(t)) for b, m, c, t in obj.get("theta_x", [])),
                   ctx_edges=tuple(int(e) for e in obj.get("ctx_edges", [])) if obj.get("theta_x") else ())
    if any(m not in (0, 1) or c < 0 or c > len(pol.ctx_edges) or t < 0 for _, m, c, t in pol.theta_x):
        raise ValueError(f"{path}: theta_x entries must be [band, mode 0/1, bucket 0..{len(pol.ctx_edges)}, theta >= 0]")
    strata = 2 * (len(pol.bands) if pol.band_calib else 1)
    size = strata
    for f in feats:
        size *= pol.n_bins(f)
    if len(pol.calib) != size or any(not (0.0 <= v <= 1.0) for v in pol.calib):
        raise ValueError(f"{path}: calib must hold {size} values in [0, 1], got {len(pol.calib)}")
    return pol


class DfPolicyDevice:
    """Device tensors of a df1+features policy: flat table, feature edges, per-band thresholds theta * rowcost."""

    def __init__(self, pol: DfPolicy, device):
        self.pol = pol
        self.table = torch.tensor(pol.calib, dtype=torch.float32, device=device)
        self.edges = {f: torch.tensor(pol.edges[f], dtype=torch.float32, device=device)
                      for f in pol.features if f != "slot"}
        ncb = len(pol.ctx_edges) + 1
        thr = torch.zeros((len(pol.bands), 2, ncb, pol.gamma), dtype=torch.float32)  # [band, mode, bucket, slot]
        for i, b in enumerate(pol.bands):
            cost = pol.rowcost.get(b)
            if cost is None:
                continue
            for mode in (0, 1):
                for cb in range(ncb):
                    th = pol.theta_req(b, mode, cb)
                    if th > 0:
                        thr[i, mode, cb] = th * torch.tensor(cost, dtype=torch.float32)
        self.thr = thr.to(device)
        self.ctx_edges = (torch.tensor(pol.ctx_edges, dtype=torch.int64, device=device) if pol.ctx_edges
                          else None)


def lens_fixed(bs: int, k: int, device) -> torch.Tensor:
    return torch.full((bs,), int(k), dtype=torch.int64, device=device)


def lens_conf(m: torch.Tensor, threshold: float) -> torch.Tensor:
    surv = torch.cumprod(m.to(torch.float32), dim=1)
    ok = (surv >= float(threshold)).to(torch.int64)
    return 1 + torch.cumprod(ok, dim=1).sum(dim=1)


def lens_policy(m: torch.Tensor, gap: torch.Tensor, sampled: torch.Tensor, band: int,
                dev: DfPolicyDevice, context: Optional[torch.Tensor] = None) -> torch.Tensor:
    """[bs] int64 live lengths of the calibrated rule for one decode batch (band known on the host): S_j = prod a,
    draft j live while S_j >= theta_band * rowcost_band[j] (leading run). Same cells as dfreplay.Calib."""
    pol = dev.pol
    bs, gamma = m.shape
    cell = sampled.to(torch.int64)[:, None].expand(bs, gamma)
    if pol.band_calib:
        cell = cell * len(pol.bands) + band
    vals = {"m": m, "gap": gap}
    for f in pol.features:
        nb = pol.n_bins(f)
        if f == "slot":
            b = torch.arange(gamma, device=m.device)[None, :].expand(bs, gamma)
        else:
            b = torch.bucketize(vals[f].to(torch.float32).contiguous(), dev.edges[f], right=True)
        cell = cell * nb + b
    a = dev.table[cell]
    surv = torch.cumprod(a, dim=1)
    mode = sampled.to(torch.int64)
    if dev.ctx_edges is not None and context is not None:
        cb = torch.bucketize(context.reshape(-1)[:bs].to(torch.int64).contiguous(), dev.ctx_edges, right=True)
    else:
        cb = torch.zeros_like(mode)
    thr = dev.thr[band][mode, cb]  # [bs, G]: a request with theta 0 verifies every row
    ok = (surv >= thr).to(torch.int64)
    return 1 + torch.cumprod(ok, dim=1).sum(dim=1)


class Cutoff:
    """The layout handed back from _plan_verify: _accept_block reads verify_lens (cutoff_verify_lens); the result's
    cap_lens and the SPS dumper read verify_lens / graph_num_tokens."""

    __slots__ = ("verify_lens", "graph_num_tokens")

    def __init__(self, verify_lens: torch.Tensor, stride: int):
        self.verify_lens = verify_lens
        self.graph_num_tokens = int(verify_lens.shape[0]) * int(stride)


def plan(state, *, bs: int, mode: str, arg, sampled: Optional[torch.Tensor] = None, lo: int = 1,
         hi: int = 0, context: Optional[torch.Tensor] = None) -> Optional[Cutoff]:
    """Write this step's live lengths and remap index (no host sync) and return the cutoff, or None when the step is
    not capped (then neither remap nor cutoff applies: src_index stays the identity for this step)."""
    from inkling_vcap import logic

    def uncapped():
        # Exactness: a remap must never outlive its capped accept. The verify graphs read src_index at every
        # replay, so an uncapped step needs the identity (one copy when the previous step was capped).
        if not getattr(state, "src_identity", True):
            state.src_index.copy_(state.layout.rows)
            state.src_identity = True
        return None

    if bs <= 0 or bs > state.capacity:
        return uncapped()
    feats = state.df
    stride = state.stride
    if mode == "fixed":
        lens = lens_fixed(bs, min(int(arg), stride), state.live_lens.device)
    elif mode == "conf":
        lens = lens_conf(feats.m[:bs], float(arg))
    elif mode == "policy":
        th = arg.pol.theta_for(bs)
        if th <= 0:
            return uncapped()
        lens = lens_policy(feats.m[:bs], feats.gap[:bs], ~feats.greedy[:bs] if sampled is None else sampled,
                           arg.pol.band_index(bs), arg, context=context)
    else:
        raise ValueError(mode)
    min_live = getattr(state, "df_min_live", None)
    if min_live is not None:
        # inkling_pcopy: a row with k forced copy drafts keeps k + 1 live rows (then clamped like every row; a configured
        # maximum below k + 1 cuts the run like any stopping rule: the copies past it are simply not verified)
        lens = torch.maximum(lens, min_live[:bs].to(lens.dtype))
    state.live_lens[:bs].copy_(lens.clamp_(max(1, lo), hi if hi > 0 else stride))
    state.live_lens[bs:].zero_()  # graph-padding requests: dead, remapped to row 0, capped to 0
    logic.build_src_index(state.live_lens, state.layout, out=state.src_index)
    state.src_identity = False
    return Cutoff(state.live_lens[:bs], stride)
