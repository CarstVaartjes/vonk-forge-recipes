"""INKLING_DRAFT_BYTES parsing.

Comma list, any order, default empty (= the overlay does nothing):

  gemv           draft base logits through the image's sm12x Triton GEMV (<= 16 rows)
  foldk          Markov fold through this package's kernel on the BF16 fold matrix
  fold8[:MODE]   Markov fold from an FP8 e4m3 copy (MODE row | gN | mx; default row)
  head8[:MODE]   draft base logits from an FP8 copy of the LM head (default row)
  draftmx8[:MODE] FP8 copies of the three draft layers' linears (default mx)
  hotvocab:N     draft proposes only from the first N ids of INKLING_DRAFT_HOTVOCAB_FILE
  w8tile         with draftmx8: the draft linears' Triton launches (rows not taken by w8gemv, up to w8_max_m) use the
                 narrow tile BN 16, BK 256 (K <= 2304) / 128 (mx; row/gN keep their BK) (GB10 sweep 2026-11-22: 195 vs 155 GB/s at
                 8 rows); bit-identical to the default tile (the per-output chain does not depend on the tile)
  w8gemv         with draftmx8 (mx): the draft linears at 1-16 rows through a CUDA weight-streaming GEMV (w8gemv.py,
                 csrc/w8gemv.cu) instead of the Triton kernel; same per-output arithmetic (gated bit for bit on
                 the GPU). Rows 17..w8_max_m keep the Triton kernel.
  wide8          this package's FP8 / BF16 kernels run one CTA per output tile for all 64-row blocks (mode 2:
                 one W load per K tile per CTA up to 384 rows; mode 1 / beyond 384: one per row block, L2-hit
                 after the first) instead of a CTA per 64-row block (which re-reads W from DRAM per block);
                 same per-element math (gated bit for bit on the GPU). Applies to whatever head8 / fold8 /
                 foldk / hotvocab / draftmx8 / gemv route through them.
  rnnhoist       Markov walk: the step-invariant 90 % of rnn_proj (the hidden_s and f_rnn inputs, 4608 of 5120
                 columns) runs as one FP32-output GEMM over all 7 steps when the walk starts; each step then reads
                 only the 768 x 512 state/prev-token slice and adds the hoisted partial before the stock BF16
                 rounding (same math up to FP32 summation order). Up to INKLING_DRAFT_RNNHOIST_MAX_BS rows (16).
  tau:X          the draft samples at X times the request temperature (sampled rows only; the accept reads
                 q at the same X x T, so the output law is unchanged). Also INKLING_DRAFT_TAU=X (alone it
                 needs sitecustomize to call install(); with any other option it rides along). X in
                 [0.05, 4]; 1 = off. See tau.py.
  dwin:W[:S]     the drafter's global attention layer (layer 1; local_layer_ids [0, 2]) reads only the first S
                 tokens (sink, default 128 = one page; 0 = none) plus every page holding one of the last W tokens
                 of its K/V (W <= kept tail < W + page) instead of the whole context; the identity below about
                 S + W tokens. W, S: integers, optional k suffix (x1024). INKLING_DRAFT_DWIN_MAXK=0 keeps FA4's
                 stock SplitKV planning bound. Draft-only (dwin.py).

MODE: row = one fp32 scale per output row; gN = one fp32 scale per N inputs of a
row (N a power of two >= 16); mx = MXFP8, one power-of-two (e8m0) scale per 32.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import Optional

ENV = "INKLING_DRAFT_BYTES"
HOT_FILE_ENV = "INKLING_DRAFT_HOTVOCAB_FILE"
TAU_ENV = "INKLING_DRAFT_TAU"


class OptionError(ValueError):
    pass


@dataclass(frozen=True)
class QuantMode:
    kind: str  # "row" | "group" | "mx"
    group: int = 0  # elements per scale for "group" (and 32 for "mx")

    @property
    def label(self) -> str:
        if self.kind == "row":
            return "row"
        if self.kind == "mx":
            return "mx"
        return f"g{self.group}"


def parse_mode(text: Optional[str], default: str) -> QuantMode:
    text = (text or default).strip().lower()
    if text == "row":
        return QuantMode("row")
    if text == "mx":
        return QuantMode("mx", 32)
    m = re.fullmatch(r"g(\d+)", text)
    if m:
        g = int(m.group(1))
        if g < 16 or g & (g - 1):
            raise OptionError(f"group size must be a power of two >= 16, got {g}")
        return QuantMode("group", g)
    raise OptionError(f"unknown quant mode {text!r} (use row, gN or mx)")


@dataclass(frozen=True)
class Options:
    gemv: bool = False
    foldk: bool = False
    fold8: Optional[QuantMode] = None
    head8: Optional[QuantMode] = None
    draftmx8: Optional[QuantMode] = None
    hotvocab: int = 0
    hotvocab_file: Optional[str] = None
    wide8: bool = False
    w8gemv: bool = False
    w8tile: bool = False
    w8gemv_cfg: int = 0  # 0: the per-shape table in w8gemv.py
    rnnhoist: bool = False
    rnnhoist_max_bs: int = 16
    tau: float = 1.0  # draft temperature factor (tau.py); 1.0 = stock
    dwin: int = 0  # tokens kept at the tail of the drafter's global layer (0 = off)
    dwin_sink: int = 128  # tokens kept at the start (sink)
    dwin_maxk: bool = True  # window's static max_seqlen_k for FA4 planning (INKLING_DRAFT_DWIN_MAXK)
    # wide8 launches (more than one 64-row block), from the GB10 sweep of 2026-11-21 (tools/gpu_check_wide8.py
    # --tile-sweep; every configuration bit-identical to the default kernel): mode 1, BN 64, 8 warps, 3 stages
    wide8_mode: int = 1  # 1: row blocks outer (W tile re-read from L2); 2: K outer, <= 6 blocks (spills; slower)
    wide8_stages: int = 3  # Triton num_stages (0: Triton's default)
    wide8_bn: int = 64
    wide8_warps: int = 8
    raw: str = ""
    # Tunables (env, read once).
    gemv_max_m: int = 16
    w8_max_m: int = 64
    fold_bn: int = 64
    fold_bk: int = 128
    linear_bn: int = 32
    linear_bk: int = 256
    num_warps: int = 4
    extra: dict = field(default_factory=dict)

    @property
    def active(self) -> bool:
        return bool(
            self.gemv
            or self.foldk
            or self.fold8
            or self.head8
            or self.draftmx8
            or self.hotvocab
            or self.tau != 1.0
            or self.dwin
            or self.rnnhoist
        )

    @property
    def fold_kernel(self) -> bool:
        return bool(self.fold8 or self.foldk)

    def describe(self) -> str:
        parts = []
        if self.gemv:
            parts.append("gemv")
        if self.foldk and not self.fold8:
            parts.append("foldk")
        if self.fold8:
            parts.append(f"fold8:{self.fold8.label}")
        if self.head8:
            parts.append(f"head8:{self.head8.label}")
        if self.draftmx8:
            parts.append(f"draftmx8:{self.draftmx8.label}")
        if self.hotvocab:
            parts.append(f"hotvocab:{self.hotvocab}")
        if self.wide8:
            parts.append(f"wide8(mode {self.wide8_mode})")
        if self.w8tile:
            parts.append("w8tile")
        if self.w8gemv:
            parts.append("w8gemv" + (f"(cfg {self.w8gemv_cfg})" if self.w8gemv_cfg else ""))
        if self.rnnhoist:
            parts.append(f"rnnhoist(<= {self.rnnhoist_max_bs} rows)")
        if self.tau != 1.0:
            parts.append(f"tau:{self.tau:g}")
        if self.dwin:
            parts.append(f"dwin:{self.dwin}:{self.dwin_sink}")
        return ",".join(parts) or "off"


def _int_env(name: str, default: int) -> int:
    value = os.environ.get(name, "").strip()
    if not value:
        return default
    try:
        return int(value)
    except ValueError as e:
        raise OptionError(f"{name} must be an integer, got {value!r}") from e


def _parse_tau(text: Optional[str], where: str) -> float:
    from .tau import check_tau

    try:
        return check_tau(float(text or ""))
    except ValueError as e:
        raise OptionError(f"{where}: {e} (e.g. tau:0.7)") from e


def _tokens(text: str, what: str) -> int:
    m = re.fullmatch(r"(\d+)([kK]?)", text.strip())
    if not m:
        raise OptionError(f"{what} must be an integer token count (optional k suffix), got {text!r}")
    return int(m.group(1)) * (1024 if m.group(2) else 1)


def parse(text: Optional[str] = None, environ=None) -> Options:
    environ = os.environ if environ is None else environ
    raw = (environ.get(ENV, "") if text is None else text).strip()
    kw: dict = {"raw": raw}
    for item in filter(None, (t.strip() for t in raw.split(","))):
        name, _, arg = item.partition(":")
        name = name.strip().lower()
        arg = arg.strip() or None
        if name in ("off", "none", "0"):
            continue
        if name == "gemv":
            kw["gemv"] = True
        elif name == "foldk":
            kw["foldk"] = True
        elif name == "fold8":
            kw["fold8"] = parse_mode(arg, "row")
        elif name == "head8":
            kw["head8"] = parse_mode(arg, "row")
        elif name == "draftmx8":
            kw["draftmx8"] = parse_mode(arg, "mx")
        elif name == "wide8":
            kw["wide8"] = True
        elif name == "w8gemv":
            kw["w8gemv"] = True
        elif name == "w8tile":
            kw["w8tile"] = True
        elif name == "rnnhoist":
            kw["rnnhoist"] = True
        elif name == "tau":
            kw["tau"] = _parse_tau(arg, "tau")
        elif name == "dwin":
            if arg is None:
                raise OptionError("dwin needs a window size in tokens, e.g. dwin:16384 or dwin:16k:128")
            w, _, sink = arg.partition(":")
            kw["dwin"] = _tokens(w, "dwin window")
            if kw["dwin"] <= 0:
                raise OptionError("dwin window must be positive")
            if sink.strip():
                kw["dwin_sink"] = _tokens(sink, "dwin sink")
        elif name == "hotvocab":
            if arg is None or not arg.isdigit() or int(arg) <= 0:
                raise OptionError("hotvocab needs a positive size, e.g. hotvocab:32768")
            kw["hotvocab"] = int(arg)
        else:
            raise OptionError(f"unknown {ENV} option {item!r}")
    env_tau = environ.get(TAU_ENV, "").strip()
    if env_tau:
        tau = _parse_tau(env_tau, TAU_ENV)
        if "tau" in kw and kw["tau"] != tau:
            raise OptionError(f"tau:{kw['tau']:g} in {ENV} disagrees with {TAU_ENV}={env_tau}")
        kw["tau"] = tau
    if kw.get("w8tile") and kw.get("draftmx8") is None:
        raise OptionError("w8tile needs draftmx8")
    if kw.get("w8gemv") and (kw.get("draftmx8") is None or kw["draftmx8"].kind != "mx"):
        raise OptionError("w8gemv needs draftmx8 in mx mode (draftmx8 or draftmx8:mx)")
    if kw.get("hotvocab"):
        path = environ.get(HOT_FILE_ENV, "").strip()
        if not path:
            raise OptionError(f"hotvocab needs {HOT_FILE_ENV} (a JSON list of token ids)")
        kw["hotvocab_file"] = path
    kw["rnnhoist_max_bs"] = _int_env("INKLING_DRAFT_RNNHOIST_MAX_BS", 16)
    if not 1 <= kw["rnnhoist_max_bs"] <= 64:
        raise OptionError("INKLING_DRAFT_RNNHOIST_MAX_BS must be in [1, 64]")
    kw["dwin_maxk"] = _int_env("INKLING_DRAFT_DWIN_MAXK", 1) != 0
    kw["gemv_max_m"] = _int_env("INKLING_DRAFT_GEMV_MAX_M", 16)
    kw["w8_max_m"] = _int_env("INKLING_DRAFT_W8_MAX_M", 64)
    kw["fold_bn"] = _int_env("INKLING_DRAFT_FOLD_BN", 64)
    kw["fold_bk"] = _int_env("INKLING_DRAFT_FOLD_BK", 128)
    kw["linear_bn"] = _int_env("INKLING_DRAFT_LINEAR_BN", 32)
    kw["linear_bk"] = _int_env("INKLING_DRAFT_LINEAR_BK", 256)
    kw["num_warps"] = _int_env("INKLING_DRAFT_NUM_WARPS", 4)
    kw["w8gemv_cfg"] = _int_env("INKLING_DRAFT_W8GEMV_CFG", 0)
    if kw["w8gemv_cfg"] not in (0, 1142, 1144, 1182, 1184, 1242, 1244, 1282, 1284, 1442, 1444, 1482, 1484):
        raise OptionError(f"INKLING_DRAFT_W8GEMV_CFG={kw['w8gemv_cfg']} is not a compiled variant (see w8gemv.py)")
    kw["wide8_mode"] = _int_env("INKLING_DRAFT_WIDE8_MODE", 1)
    kw["wide8_stages"] = _int_env("INKLING_DRAFT_WIDE8_STAGES", 3)
    kw["wide8_bn"] = _int_env("INKLING_DRAFT_WIDE8_BN", 64)
    kw["wide8_warps"] = _int_env("INKLING_DRAFT_WIDE8_WARPS", 8)
    if kw["wide8_bn"] not in (16, 32, 64, 128) or kw["wide8_warps"] not in (1, 2, 4, 8, 16):
        raise OptionError("INKLING_DRAFT_WIDE8_BN must be 16..128 (power of two), _WARPS 1..16 (power of two)")
    if kw["wide8_mode"] not in (1, 2):
        raise OptionError(f"INKLING_DRAFT_WIDE8_MODE must be 1 or 2, got {kw['wide8_mode']}")
    if not 0 <= kw["wide8_stages"] <= 6:
        raise OptionError(f"INKLING_DRAFT_WIDE8_STAGES must be in [0, 6], got {kw['wide8_stages']}")
    for name in ("fold_bn", "fold_bk", "linear_bn", "linear_bk"):
        v = kw[name]
        if v < 16 or v & (v - 1):
            raise OptionError(f"{name} must be a power of two >= 16, got {v}")
    return Options(**kw)
