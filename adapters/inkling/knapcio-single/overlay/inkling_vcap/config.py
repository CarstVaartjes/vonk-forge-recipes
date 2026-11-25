"""Parsing of the INKLING_VERIFY_CAP* environment variables (pure Python).

INKLING_VERIFY_CAP grammar (case-insensitive, whitespace ignored):

    off | "" | 0                 -> disabled (the overlay patches nothing)
    fixed:K                      -> every request verifies the first K rows
    conf:T  (alias confmax:T)    -> cut when the cumulative product of the draft's
                                    per-position MAX probability drops below T.
                                    (The probability of the proposed token itself is
                                    deliberately not offered: for sampled requests a cut
                                    that depends on the token being verified biases the
                                    output law. For greedy rows the two are identical.)
    cost:B1=T1,B2=T2,...         -> conf with a survival threshold per decode batch size:
                                    bs uses the smallest key >= bs, bs above the largest
                                    key uses the largest key's threshold. T = 0 means
                                    "verify every row" (no draft-logit work that step).
    policy:/path/policy.json     -> calibrated cost rule: per-position acceptance tables
                                    (greedy / sampled) on bins of the draft max
                                    probability, a per-batch-size price theta and the
                                    marginal expert union of each row (optionally
                                    batch-aware). Format: policyfile.py. The path keeps
                                    its case; it must not contain commas.

Options, appended with commas to any mode:
    min=A, max=B                 -> clamp of the live length (default 1 .. verify width)
    bsmax=N                      -> the cap applies only when the decode batch size is
                                    <= N; larger batches verify every row (default: no
                                    limit)

Other variables:
    INKLING_VERIFY_CAP_REMAP=0|1      dead-row expert remap (default 1). 0 keeps the
                                      accept cap but routes dead rows normally; used for
                                      the exactness A/B (same schedule, no remap).
    INKLING_VERIFY_CAP_FUSED=0|1      remap as one Triton launch per MoE visit (default
                                      1 on CUDA, self-checked at startup); 0 = two
                                      torch index_select launches
    INKLING_VERIFY_CAP_LOG_EVERY=N    stats line every N decode steps (default 500, 0 off)
    INKLING_VERIFY_CAP_UNION=0|1      count distinct experts per MoE visit in target
                                      verify (graph-safe device counters; default 0; when
                                      0 no counter kernel is recorded at all)
    INKLING_VERIFY_CAP_FASTPROBS=0|1  1: the draft max probabilities (conf/cost/policy) and
                                      the sampled accept's draft probabilities from one /
                                      two reads of the BF16 step logits with the image's
                                      split softmax kernels (fastprobs.py; self-checked at
                                      startup, falls back on mismatch). Default 0.
    INKLING_VERIFY_CAP_FUSED_ACCEPT=0|1  1: plain-temperature sampled accepts are computed
                                      from the logits (fused_accept.py: lse pass, [bs, 7]
                                      walk, residual masses / exponential-race bonus)
                                      instead of materialised probabilities; chain and
                                      block verification; self-checked at startup. Default 0.
    INKLING_VERIFY_CAP_FUSED_ACCEPT_MIN_BS=N  fused accept only for decode batches >= N
                                      (default 3; GB10 crossover between bs 2 and 3)
    INKLING_VERIFY_CAP_BV=mask|stock  only with --speculative-use-block-verification
                                      (README, "Block verification under the cap"); both
                                      exact. mask (default): the dead drafts are handed to
                                      the stock kernel as a stopped draft (zero cut row,
                                      q(d_j) = 0 past it), so the kernel never scans a dead
                                      level. stock: the stock kernel over every row, then
                                      the usual cap.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Mapping, Optional, Tuple

from inkling_vcap.policyfile import PolicyFile, PolicyFileError, load_policy

ENV_MODE = "INKLING_VERIFY_CAP"
ENV_REMAP = "INKLING_VERIFY_CAP_REMAP"
ENV_FUSED = "INKLING_VERIFY_CAP_FUSED"
ENV_LOG_EVERY = "INKLING_VERIFY_CAP_LOG_EVERY"
ENV_UNION = "INKLING_VERIFY_CAP_UNION"
ENV_STEPLOG = "INKLING_VERIFY_CAP_STEPLOG"
ENV_BV = "INKLING_VERIFY_CAP_BV"
ENV_FASTPROBS = "INKLING_VERIFY_CAP_FASTPROBS"
ENV_FUSED_ACCEPT = "INKLING_VERIFY_CAP_FUSED_ACCEPT"
ENV_FUSED_ACCEPT_MIN_BS = "INKLING_VERIFY_CAP_FUSED_ACCEPT_MIN_BS"

_OFF_VALUES = ("", "off", "0", "none", "false", "no")


class VcapConfigError(ValueError):
    pass


@dataclass(frozen=True)
class Policy:
    """This step's rule: a constant live length, or a survival threshold."""

    const_len: int = 0  # > 0: every request gets this L
    threshold: float = 0.0  # used when const_len == 0 (policy: mode: theta)
    pool: float = 0.0  # policy: mode, batch-aware rounds (0: per-request rule)


@dataclass(frozen=True)
class VcapConfig:
    kind: str  # "fixed" | "conf" | "cost" | "policy"
    fixed_len: int = 0
    threshold: float = 0.0
    cost_table: Tuple[Tuple[int, float], ...] = ()
    min_len: int = 1
    max_len: int = 0  # 0: the verify width
    bsmax: int = 0  # 0: no batch-size limit
    remap: bool = True
    fused: bool = True
    log_every: int = 500
    union: bool = False
    bv_mask: bool = True  # block verification: mask dead drafts (False: stock kernel + cap)
    fastprobs: bool = False  # fastprobs.py for m_j and the accept's draft probabilities
    fused_accept: bool = False  # fused_accept.py for plain sampled accepts
    fused_accept_min_bs: int = 3  # bs 1-2 keep the stock accept (GB10 crossover between 2 and 3)
    raw: str = ""
    policy_file: Optional[PolicyFile] = None
    steplog: str = ""
    dfpolicy_path: str = ""  # dfpolicy: mode (DFLASH worker; dfcap.py loads it at attach)

    @property
    def uses_draft_probs(self) -> bool:
        if self.steplog:
            return True
        if self.kind == "conf":
            return True
        if self.kind == "cost":
            return any(t > 0 for _, t in self.cost_table)
        if self.kind == "policy":
            return any(t > 0 for _, t in self.policy_file.theta)
        if self.kind == "dfpolicy":
            return True
        return False

    def resolved_max(self, stride: int) -> int:
        return stride if self.max_len <= 0 else min(self.max_len, stride)

    def threshold_for(self, bs: int) -> float:
        if self.kind != "cost":
            return self.threshold
        for key, value in self.cost_table:
            if bs <= key:
                return value
        return self.cost_table[-1][1]

    def policy(self, bs: int, stride: int) -> Policy:
        """The rule for a decode step of ``bs`` requests (host-side, no sync)."""
        if self.steplog:
            return Policy(const_len=stride)
        max_len = self.resolved_max(stride)
        if self.bsmax and bs > self.bsmax:
            return Policy(const_len=stride)
        if self.kind == "fixed":
            return Policy(const_len=max(self.min_len, min(self.fixed_len, max_len)))
        if self.kind == "policy":
            theta = self.policy_file.theta_for(bs)
            if theta <= 0.0:
                return Policy(const_len=max_len)
            pool = self.policy_file.pool_for(bs) if self.policy_file.rounds else 0.0
            return Policy(threshold=theta, pool=pool)
        threshold = self.threshold_for(bs)
        if threshold <= 0.0:
            # surv >= 0 always holds: every row up to max_len is live.
            return Policy(const_len=max_len)
        return Policy(threshold=threshold)

    def never_caps(self, stride: int) -> bool:
        """True when every batch size gets the full verify width."""
        if self.steplog:
            return True
        if self.resolved_max(stride) < stride:
            return False
        if self.kind == "fixed":
            return max(self.min_len, min(self.fixed_len, stride)) >= stride
        if self.kind == "conf":
            return self.threshold <= 0.0
        if self.kind == "policy":
            return all(t <= 0.0 for _, t in self.policy_file.theta)
        if self.kind == "dfpolicy":
            return False  # dfcap decides per batch (theta 0 bands verify full width, src_index identity)
        return all(t <= 0.0 for _, t in self.cost_table)

    def describe(self) -> str:
        if self.steplog:
            return f"steplog:{self.steplog} theta=0"
        if self.kind == "fixed":
            body = f"fixed:{self.fixed_len}"
        elif self.kind == "conf":
            body = f"conf:{self.threshold:g}"
        elif self.kind == "policy":
            body = self.policy_file.describe()
        elif self.kind == "dfpolicy":
            body = f"dfpolicy:{self.dfpolicy_path}"
        else:
            body = "cost:" + ",".join(f"{k}={v:g}" for k, v in self.cost_table)
        return (
            f"{body} min={self.min_len} max={self.max_len or 'width'} "
            f"bsmax={self.bsmax or 'none'} remap={'on' if self.remap else 'off'} "
            f"fused={'on' if self.fused else 'off'} union={'on' if self.union else 'off'} "
            f"log_every={self.log_every}"
        )

    def validate_for_stride(self, stride: int) -> None:
        if stride < 2:
            raise VcapConfigError(f"verify width {stride} < 2: nothing to cap")
        if self.kind == "policy" and self.policy_file.stride != stride:
            raise VcapConfigError(
                f"{ENV_MODE}: policy file stride {self.policy_file.stride} != verify "
                f"width {stride}"
            )
        if self.kind == "fixed" and not (1 <= self.fixed_len <= stride):
            raise VcapConfigError(
                f"{ENV_MODE}=fixed:{self.fixed_len} must be within 1..{stride}"
            )
        if self.min_len > self.resolved_max(stride):
            raise VcapConfigError(
                f"{ENV_MODE}: min={self.min_len} exceeds max={self.resolved_max(stride)}"
            )


def _parse_bool(name: str, value: Optional[str], default: bool) -> bool:
    if value is None or value.strip() == "":
        return default
    v = value.strip().lower()
    if v in ("1", "true", "on", "yes"):
        return True
    if v in ("0", "false", "off", "no"):
        return False
    raise VcapConfigError(f"{name}={value!r}: expected 0/1")


def _parse_int(name: str, value: Optional[str], default: int) -> int:
    if value is None or value.strip() == "":
        return default
    try:
        return int(value.strip())
    except ValueError as exc:
        raise VcapConfigError(f"{name}={value!r}: expected an integer") from exc


def _parse_threshold(raw: str, text: str) -> float:
    try:
        value = float(text)
    except ValueError as exc:
        raise VcapConfigError(f"{ENV_MODE}={raw!r}: threshold {text!r} is not a number") from exc
    if not (0.0 <= value <= 1.0):
        raise VcapConfigError(f"{ENV_MODE}={raw!r}: threshold {value} not within [0, 1]")
    return value


def parse_config(env: Mapping[str, str] = os.environ) -> Optional[VcapConfig]:
    """None when the overlay is off. Raises VcapConfigError on a malformed value."""
    steplog = (env.get(ENV_STEPLOG) or "").strip()
    raw = "conf:0" if steplog else env.get(ENV_MODE)
    if raw is None or raw.strip().lower() in _OFF_VALUES:
        return None
    text = "".join(raw.split())
    parts = text.split(",")
    head, opts = parts[0], [o.lower() for o in parts[1:]]
    policy_file = None
    dfpolicy_path = ""
    if head.lower().startswith("dfpolicy:"):
        dfpolicy_path = head[len("dfpolicy:"):]
        if not dfpolicy_path:
            raise VcapConfigError(f"{ENV_MODE}={raw!r}: dfpolicy: needs a JSON path")
        head = "dfpolicy:"
    elif head.lower().startswith("policy:"):
        path = head[len("policy:"):]
        if not path:
            raise VcapConfigError(f"{ENV_MODE}={raw!r}: policy: needs a JSON path")
        try:
            policy_file = load_policy(path)
        except PolicyFileError as exc:
            raise VcapConfigError(f"{ENV_MODE}={raw!r}: {exc}") from exc
        head = "policy:"
    head = head.lower()
    if ":" not in head:
        raise VcapConfigError(
            f"{ENV_MODE}={raw!r}: expected off | fixed:K | conf:T | cost:B=T,... | "
            "policy:/path.json"
        )
    kind, arg = head.split(":", 1)
    fixed_len, threshold = 0, 0.0
    table: dict[int, float] = {}
    if kind == "fixed":
        try:
            fixed_len = int(arg)
        except ValueError as exc:
            raise VcapConfigError(f"{ENV_MODE}={raw!r}: K must be an integer") from exc
        if fixed_len < 1:
            raise VcapConfigError(f"{ENV_MODE}={raw!r}: K must be >= 1")
    elif kind in ("conf", "confmax"):
        kind = "conf"
        threshold = _parse_threshold(raw, arg)
    elif kind == "cost":
        opts = [arg] + opts
    elif kind in ("policy", "dfpolicy"):
        pass
    else:
        raise VcapConfigError(
            f"{ENV_MODE}={raw!r}: unknown mode {kind!r} (fixed | conf | cost | policy)"
        )
    min_len, max_len, bsmax = 1, 0, 0
    for opt in opts:
        if not opt:
            continue
        if "=" not in opt:
            raise VcapConfigError(f"{ENV_MODE}={raw!r}: bad option {opt!r}")
        key, value = opt.split("=", 1)
        if key.isdigit():
            if kind != "cost":
                raise VcapConfigError(
                    f"{ENV_MODE}={raw!r}: batch-size entries need the cost: mode"
                )
            bs_key = int(key)
            if bs_key < 1 or bs_key in table:
                raise VcapConfigError(f"{ENV_MODE}={raw!r}: bad or repeated key {key}")
            table[bs_key] = _parse_threshold(raw, value)
        elif key == "min":
            min_len = _parse_int(ENV_MODE + ":min", value, 1)
        elif key == "max":
            max_len = _parse_int(ENV_MODE + ":max", value, 0)
        elif key == "bsmax":
            bsmax = _parse_int(ENV_MODE + ":bsmax", value, 0)
            if bsmax < 1:
                raise VcapConfigError(f"{ENV_MODE}={raw!r}: bsmax must be >= 1")
        else:
            raise VcapConfigError(f"{ENV_MODE}={raw!r}: unknown option {key!r}")
    if kind == "cost" and not table:
        raise VcapConfigError(f"{ENV_MODE}={raw!r}: cost: needs at least one B=T entry")
    if min_len < 1:
        raise VcapConfigError(f"{ENV_MODE}={raw!r}: min must be >= 1 (the anchor row)")
    if max_len < 0 or (max_len and max_len < min_len):
        raise VcapConfigError(f"{ENV_MODE}={raw!r}: need 1 <= min <= max")
    bv = (env.get(ENV_BV) or "mask").strip().lower()
    if bv not in ("mask", "stock"):
        raise VcapConfigError(f"{ENV_BV} must be mask or stock, got {bv!r}")
    log_every = _parse_int(ENV_LOG_EVERY, env.get(ENV_LOG_EVERY), 500)
    if log_every < 0:
        raise VcapConfigError(f"{ENV_LOG_EVERY} must be >= 0")
    return VcapConfig(
        kind=kind,
        fixed_len=fixed_len,
        threshold=threshold,
        cost_table=tuple(sorted(table.items())),
        min_len=min_len,
        max_len=max_len,
        bsmax=bsmax,
        remap=_parse_bool(ENV_REMAP, env.get(ENV_REMAP), True),
        fused=_parse_bool(ENV_FUSED, env.get(ENV_FUSED), True),
        log_every=0 if steplog else log_every,
        union=False if steplog else _parse_bool(ENV_UNION, env.get(ENV_UNION), False),
        bv_mask=bv == "mask",
        fastprobs=_parse_bool(ENV_FASTPROBS, env.get(ENV_FASTPROBS), False),
        fused_accept=_parse_bool(ENV_FUSED_ACCEPT, env.get(ENV_FUSED_ACCEPT), False),
        fused_accept_min_bs=_parse_int(ENV_FUSED_ACCEPT_MIN_BS, env.get(ENV_FUSED_ACCEPT_MIN_BS), 3),
        raw=raw,
        policy_file=policy_file,
        steplog=steplog,
        dfpolicy_path=dfpolicy_path,
    )
