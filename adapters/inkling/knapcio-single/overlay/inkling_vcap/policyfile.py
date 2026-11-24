"""The ``policy:`` mode's JSON (pure Python; parsed at interpreter start).

    {"format": "inkling-vcap-policy/1",
     "stride": 8,
     "bin_edges": [e_1, ..., e_{nb-1}],         # bin(m) = #{i : e_i <= m}, increasing
     "calib": {"greedy": [[a]*nb]*gamma,          # P(d_j accepted | d_<j accepted,
               "sampled": [[a]*nb]*gamma},        #   bin(m_j)); a in [0, 1]
     "u1": [u1(1), ..., u1(stride)],             # c1 prefix expert union, increasing
     "theta": [[bs_key, theta], ...],             # bs uses the smallest key >= bs (above
                                                  #   the largest: the largest); 0 = full
                                                  #   width (no draft-logit work)
     "pool": [[bs_key, E], ...],                  # optional, same lookup; E > u1[-1]
     "rounds": K}                                 # optional, 0 = per-request rule

A missing "sampled" table falls back to "greedy" (and vice versa). Rule: logic.
calibrated_live_lens. Made by bsreplay.py --export-policy."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Tuple

FORMAT = "inkling-vcap-policy/1"


class PolicyFileError(ValueError):
    pass


@dataclass(frozen=True)
class PolicyFile:
    path: str
    stride: int
    edges: Tuple[float, ...]
    calib: Tuple[Tuple[Tuple[float, ...], ...], ...]  # [2][gamma][nb]: greedy, sampled
    u1: Tuple[float, ...]
    theta: Tuple[Tuple[int, float], ...]
    pool: Tuple[Tuple[int, float], ...]
    rounds: int

    @staticmethod
    def _lookup(table, bs, default):
        if not table:
            return default
        for key, value in table:
            if bs <= key:
                return value
        return table[-1][1]

    def theta_for(self, bs: int) -> float:
        return self._lookup(self.theta, bs, 0.0)

    def pool_for(self, bs: int) -> float:
        return self._lookup(self.pool, bs, 0.0)

    @property
    def row_cost(self) -> Tuple[float, ...]:
        return tuple(self.u1[j + 1] - self.u1[j] for j in range(self.stride - 1))

    def describe(self) -> str:
        th = ",".join(f"{k}={v:g}" for k, v in self.theta)
        pl = ",".join(f"{k}={v:g}" for k, v in self.pool) or "-"
        return f"policy:{self.path} theta[{th}] pool[{pl}] rounds={self.rounds}"


def _table(raw, name, path):
    if not isinstance(raw, list) or not raw:
        raise PolicyFileError(f"{path}: {name} must be a non-empty list of [bs, value]")
    out = {}
    for item in raw:
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            raise PolicyFileError(f"{path}: {name} entries must be [bs, value]")
        key, value = int(item[0]), float(item[1])
        if key < 1 or key in out:
            raise PolicyFileError(f"{path}: {name} has a bad or repeated key {key}")
        out[key] = value
    return tuple(sorted(out.items()))


def parse_policy(obj: dict, path: str = "") -> PolicyFile:
    if not isinstance(obj, dict) or obj.get("format") != FORMAT:
        raise PolicyFileError(f"{path}: format must be {FORMAT!r}")
    try:
        stride = int(obj["stride"])
        edges = tuple(float(e) for e in obj["bin_edges"])
        u1 = tuple(float(u) for u in obj["u1"])
        calib_raw = obj["calib"]
    except (KeyError, TypeError, ValueError) as exc:
        raise PolicyFileError(f"{path}: missing or malformed field ({exc})") from exc
    if stride < 2:
        raise PolicyFileError(f"{path}: stride {stride} < 2")
    gamma, nb = stride - 1, len(edges) + 1
    if any(b <= a for a, b in zip(edges, edges[1:])):
        raise PolicyFileError(f"{path}: bin_edges must be strictly increasing")
    if len(u1) != stride or u1[0] <= 0 or any(b < a for a, b in zip(u1, u1[1:])):
        raise PolicyFileError(f"{path}: u1 needs {stride} positive non-decreasing values")
    if not isinstance(calib_raw, dict) or not set(calib_raw) or set(calib_raw) - {"greedy", "sampled"}:
        raise PolicyFileError(f"{path}: calib must have 'greedy' and/or 'sampled'")
    tables = {}
    for mode, tab in calib_raw.items():
        if (not isinstance(tab, list) or len(tab) != gamma
                or any(not isinstance(r, list) or len(r) != nb for r in tab)):
            raise PolicyFileError(f"{path}: calib[{mode}] must be [{gamma}][{nb}]")
        vals = tuple(tuple(float(v) for v in r) for r in tab)
        if any(not (0.0 <= v <= 1.0) for r in vals for v in r):
            raise PolicyFileError(f"{path}: calib[{mode}] values must be in [0, 1]")
        tables[mode] = vals
    greedy = tables.get("greedy", tables.get("sampled"))
    sampled = tables.get("sampled", tables.get("greedy"))
    theta = _table(obj.get("theta"), "theta", path)
    if any(v < 0 for _, v in theta):
        raise PolicyFileError(f"{path}: theta must be >= 0")
    pool = _table(obj["pool"], "pool", path) if obj.get("pool") else ()
    if any(v <= u1[-1] for _, v in pool):
        raise PolicyFileError(f"{path}: every pool E must exceed u1[-1] = {u1[-1]}")
    rounds = int(obj.get("rounds", 0))
    if rounds < 0 or rounds > 8:
        raise PolicyFileError(f"{path}: rounds must be within 0..8")
    if rounds and not pool:
        raise PolicyFileError(f"{path}: rounds > 0 needs a pool table")
    return PolicyFile(path=path, stride=stride, edges=edges, calib=(greedy, sampled),
                      u1=u1, theta=theta, pool=pool, rounds=rounds)


def load_policy(path: str) -> PolicyFile:
    try:
        with open(path) as f:
            obj = json.load(f)
    except (OSError, ValueError) as exc:
        raise PolicyFileError(f"cannot read policy {path}: {exc}") from exc
    return parse_policy(obj, path)
