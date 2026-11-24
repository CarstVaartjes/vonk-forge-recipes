"""Per-token-count tile tables for Inkling2's SM120 expert grouped GEMMs (sglang.kernels.ops.gemm.inkling2_grouped_sm120).

NVFP4 (layers 0-19): `_tiles(tokens, width)` is a module global that `nvfp4_expert_matmul` looks up on every call.
The wrapper installed here answers from its own table for the token ranges it covers and delegates everything
else to the function it wrapped (inkling_kernels.moe_tiles' wrapper when that lever is on, else the stock table),
so it extends moe_tiles instead of replacing it.

MXFP8 (layers 20-23): `mxfp8_expert_matmul` calls FlashInfer's `moe_gemm_mxfp8_nt_groupwise`, whose C++ heuristic
picks the CuTe kernel's (tile_m, tile_n) from the routed row count. For the row ranges in the MXFP8 table the
wrapper calls the same module's `moe_gemm_mxfp8_nt_groupwise_tuned` entry with an explicit tactic (the tactics
FlashInfer's own autotuner offers); other calls go to the stock function.

Both choices are host-side functions of static shapes (tokens = ids.shape[0], rows = tokens * top_k), so a CUDA
graph keeps the kernel it was captured with. Only configurations measured bit-identical to the shipped choice on
GB10 are in the default tables (results/ in the track directory, bench_moe2.py).

Grammar (whitespace ignored), ranges inclusive:
  INKLING_MOE2_NVFP4 = rule (";" rule)*   rule = LO "-" HI ["@" WIDTH] ":" kv ("," kv)*
                                          kv = swap_ab=0|1 | tile_n=32|64|128 | tile_k=128|256
  INKLING_MOE2_MXFP8 = rule (";" rule)*   rule = LO "-" HI ["@" N "x" K] ":" tile_m=..,tile_n=..
LO/HI are tokens (NVFP4, ids.shape[0]) or routed rows (MXFP8, values.shape[0]); WIDTH is the unpacked K (2304
for w13, 1152 for w2); N x K is the weight's [N, K] (2304x2304 for w13, 2304x1152 for w2). The first matching
rule wins; a width-specific rule is tried before a generic one.
"""

TARGET_MODULE = "sglang.kernels.ops.gemm.inkling2_grouped_sm120"
SOURCE_NEEDLES = (
    "def _tiles(",
    "_tiles(tokens=tokens, width=packed_width * 2)",
    "group_gemm_nvfp4_nt_groupwise(",
    "def mxfp8_expert_matmul(values, scales, weight, weight_scales, offsets):",
    "moe_gemm_mxfp8_nt_groupwise(",
    "scale_granularity_mnk=(1, 1, 32),",
)

NV_TILE_N = (32, 64, 128)
NV_TILE_K = (128, 256)
MX_TACTICS = ((32, 128), (64, 64), (64, 128), (128, 128), (128, 8))  # FlashInfer 0.7.4 plain MoE tactics
MARK = "inkling_moe2_rules"


class PatchError(BaseException):
    """Raised from the import hook when a target changed: a BaseException, because SGLang's model registry
    swallows Exception during model imports (overlay/README.md, "Writing a new overlay")."""


def _ints(head):
    lo, hi = head.split("-", 1)
    lo, hi = int(lo), int(hi)
    if not 1 <= lo <= hi:
        raise ValueError(f"bad range {head!r}")
    return lo, hi


def parse_nvfp4(spec):
    rules = []
    for raw in spec.replace(" ", "").split(";"):
        if not raw:
            continue
        head, _, body = raw.partition(":")
        if not body:
            raise ValueError(f"INKLING_MOE2_NVFP4 rule {raw!r} lacks ':'")
        rng, _, width = head.partition("@")
        lo, hi = _ints(rng)
        width = int(width) if width else None
        vals = {}
        for kv in body.split(","):
            key, _, val = kv.partition("=")
            if key not in ("swap_ab", "tile_n", "tile_k") or not val:
                raise ValueError(f"INKLING_MOE2_NVFP4 item {kv!r}")
            val = int(val)
            if key == "swap_ab":
                if val not in (0, 1):
                    raise ValueError("swap_ab must be 0 or 1")
                val = bool(val)
            elif val not in (NV_TILE_N if key == "tile_n" else NV_TILE_K):
                raise ValueError(f"{key}={val} is not a FlashInfer 0.7.4 NVFP4 tile")
            vals[key] = val
        if set(vals) != {"swap_ab", "tile_n", "tile_k"}:
            raise ValueError(f"INKLING_MOE2_NVFP4 rule {raw!r} must set swap_ab, tile_n and tile_k")
        rules.append((lo, hi, width, (vals["swap_ab"], vals["tile_n"], vals["tile_k"])))
    rules.sort(key=lambda r: r[2] is None)  # width-specific first, stable otherwise
    return rules


def parse_mxfp8(spec):
    rules = []
    for raw in spec.replace(" ", "").split(";"):
        if not raw:
            continue
        head, _, body = raw.partition(":")
        if not body:
            raise ValueError(f"INKLING_MOE2_MXFP8 rule {raw!r} lacks ':'")
        rng, _, shape = head.partition("@")
        lo, hi = _ints(rng)
        nk = None
        if shape:
            n, _, k = shape.partition("x")
            nk = (int(n), int(k))
        vals = dict(kv.split("=", 1) for kv in body.split(","))
        try:
            tac = (int(vals.pop("tile_m")), int(vals.pop("tile_n")))
        except KeyError as exc:
            raise ValueError(f"INKLING_MOE2_MXFP8 rule {raw!r} needs tile_m and tile_n") from exc
        if vals or tac not in MX_TACTICS:
            raise ValueError(f"INKLING_MOE2_MXFP8 rule {raw!r}: tactic {tac} not in {MX_TACTICS}")
        rules.append((lo, hi, nk, tac))
    rules.sort(key=lambda r: r[2] is None)
    return rules


def lookup(rules, count, key):
    for lo, hi, k, val in rules:
        if lo <= count <= hi and (k is None or k == key):
            return val
    return None


def describe(nv, mx):
    parts = [f"{lo}-{hi}" + (f"@{w}" if w else "") + f":{int(v[0])}/{v[1]}/{v[2]}" for lo, hi, w, v in nv]
    parts += [f"mx{lo}-{hi}" + (f"@{k[0]}x{k[1]}" if k else "") + f":{v[0]}x{v[1]}" for lo, hi, k, v in mx]
    return ";".join(parts) or "none"


def make_tiles(inner, rules):
    def _tiles(tokens, width):
        hit = lookup(rules, tokens, width)
        return hit if hit is not None else inner(tokens, width)

    _tiles.__wrapped__ = inner
    setattr(_tiles, MARK, rules)
    return _tiles


def make_mxfp8(stock, rules):
    state = {"checked": set()}

    def _module():
        mod = state.get("mod")
        if mod is None:
            from flashinfer.grouped_mm.cute_sm120_mxfp8_groupwise.core import get_gemm_sm120_module_cute_mxfp8

            mod = get_gemm_sm120_module_cute_mxfp8()
            if not hasattr(mod, "moe_gemm_mxfp8_nt_groupwise_tuned"):
                raise PatchError("inkling_moe2: FlashInfer lacks moe_gemm_mxfp8_nt_groupwise_tuned")
            state["mod"] = mod
        return mod

    def mxfp8_expert_matmul(values, scales, weight, weight_scales, offsets):
        import torch

        rows = values.shape[0]
        tac = lookup(rules, rows, (weight.shape[1], weight.shape[2])) if rows else None
        if (
            tac is None
            or values.dtype != torch.float8_e4m3fn
            or weight.dtype != torch.float8_e4m3fn
            or offsets.dtype != torch.int32
            or offsets.dim() != 1
            or offsets.numel() != weight.shape[0] + 1
        ):
            return stock(values, scales, weight, weight_scales, offsets)
        key = (rows, tuple(weight.shape))
        capturing = torch.cuda.is_current_stream_capturing()
        if capturing and key not in state["checked"]:
            # Never record an unchecked tactic into a graph (replay cannot run the comparison below).
            return stock(values, scales, weight, weight_scales, offsets)
        out = torch.empty((rows, weight.shape[1]), dtype=torch.bfloat16, device=values.device)
        # Same entry and arguments as FlashInfer's own tuned launch (cute_sm120_mxfp8_groupwise/core.py,
        # _CuteSm120Mxfp8MoeRunner._launch), with SGLang's scale granularity (1, 1, 32).
        _module().moe_gemm_mxfp8_nt_groupwise_tuned(
            values, weight, scales, weight_scales, offsets, out, "MN", 1, 1, 32, False, tac[0], tac[1]
        )
        if key not in state["checked"]:
            # Once per (rows, weight shape), on an eager call (warm-up): the tactic must give the stock bits.
            ref = stock(values, scales, weight, weight_scales, offsets)
            if not torch.equal(out.view(torch.int16), ref.view(torch.int16)):
                import sys

                sys.stderr.write(f"inkling_moe2.tiles: MXFP8 tactic {tac} differs from stock at {key}\n")
                raise SystemExit(78)
            state["checked"].add(key)
        return out

    mxfp8_expert_matmul.__wrapped__ = stock
    setattr(mxfp8_expert_matmul, MARK, rules)
    return mxfp8_expert_matmul


def patch(module, nv_rules, mx_rules):
    """Wrap module._tiles (NVFP4) and module.mxfp8_expert_matmul (MXFP8). Returns what was applied."""
    applied = []
    if nv_rules:
        inner = getattr(module, "_tiles", None)
        if not callable(inner):
            raise PatchError(f"inkling_moe2: {TARGET_MODULE}._tiles is missing")
        if getattr(inner, MARK, None) is None:
            probe = inner(tokens=128, width=2304)
            if not (isinstance(probe, tuple) and len(probe) == 3):
                raise PatchError(f"inkling_moe2: {TARGET_MODULE}._tiles returned {probe!r}")
            module._tiles = make_tiles(inner, nv_rules)
            under = "moe_tiles" if getattr(inner, "inkling_rules", None) is not None else "stock"
            applied.append(f"nvfp4 over {under}")
    if mx_rules:
        stock = getattr(module, "mxfp8_expert_matmul", None)
        if not callable(stock):
            raise PatchError(f"inkling_moe2: {TARGET_MODULE}.mxfp8_expert_matmul is missing")
        if getattr(stock, MARK, None) is None:
            module.mxfp8_expert_matmul = make_mxfp8(stock, mx_rules)
            applied.append("mxfp8")
    return applied
