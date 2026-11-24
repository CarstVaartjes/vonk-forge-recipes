"""Lever B: override the NVFP4 grouped-GEMM tile table of Inkling2's SM120 experts.

Target: sglang.kernels.ops.gemm.inkling2_grouped_sm120._tiles(tokens, width), which
nvfp4_expert_matmul looks up as a module global on every call, so replacing the
module attribute is enough. The choice is host-side Python evaluated while CUDA
graphs are captured; each captured graph keeps the tiles it was captured with.

INKLING_MOE_TILES grammar (whitespace ignored):

    rule  := MAXTOK ["@" WIDTH] ":" kv ("," kv)*
    kv    := ("swap_ab" | "tile_n" | "tile_k") "=" INT
    spec  := rule (";" rule)*

A call with `tokens` MoE tokens (ids.shape[0], i.e. 8 for c1 verify of 8 rows) and
unpacked K `width` (2304 for w13, 1152 for w2) takes the rule with the smallest
MAXTOK >= tokens among the rules whose WIDTH is absent or equal to width; at equal
MAXTOK a rule with WIDTH wins. Keys left out keep the stock value for that call.
Calls no rule covers keep the stock table. Example:

    INKLING_MOE_TILES='8:swap_ab=1,tile_n=64,tile_k=256;64@1152:tile_n=128'
"""

TARGET_MODULE = "sglang.kernels.ops.gemm.inkling2_grouped_sm120"
TARGET_ATTR = "_tiles"
SOURCE_NEEDLES = ("def _tiles(", "_tiles(tokens=", "group_gemm_nvfp4_nt_groupwise")

_KEYS = ("swap_ab", "tile_n", "tile_k")


def parse(spec):
    """List of (max_tokens, width or None, {key: value}) sorted for lookup."""
    rules = []
    for raw in spec.replace(" ", "").split(";"):
        if not raw:
            continue
        if ":" not in raw:
            raise ValueError(f"INKLING_MOE_TILES rule {raw!r} lacks ':'")
        head, body = raw.split(":", 1)
        if "@" in head:
            tok, width = head.split("@", 1)
            width = int(width)
        else:
            tok, width = head, None
        tok = int(tok)
        if tok <= 0 or (width is not None and width <= 0):
            raise ValueError(f"INKLING_MOE_TILES rule {raw!r}: bad bound")
        values = {}
        for kv in body.split(","):
            if not kv:
                continue
            if "=" not in kv:
                raise ValueError(f"INKLING_MOE_TILES item {kv!r} lacks '='")
            key, val = kv.split("=", 1)
            if key not in _KEYS:
                raise ValueError(f"INKLING_MOE_TILES key {key!r} not in {_KEYS}")
            val = int(val)
            if key == "swap_ab":
                if val not in (0, 1):
                    raise ValueError("INKLING_MOE_TILES swap_ab must be 0 or 1")
                val = bool(val)
            elif val <= 0 or val % 8:
                raise ValueError(f"INKLING_MOE_TILES {key}={val} is not a tile size")
            values[key] = val
        if not values:
            raise ValueError(f"INKLING_MOE_TILES rule {raw!r} sets nothing")
        rules.append((tok, width, values))
    if not rules:
        raise ValueError("INKLING_MOE_TILES is empty")
    # Smallest bound first; at equal bound the width-specific rule first.
    rules.sort(key=lambda r: (r[0], r[1] is None))
    return rules


def lookup(rules, tokens, width):
    for tok, w, values in rules:
        if tokens <= tok and (w is None or w == width):
            return values
    return None


def make_tiles(stock, rules):
    def _tiles(tokens, width):
        swap_ab, tile_n, tile_k = stock(tokens, width)
        values = lookup(rules, tokens, width)
        if values is None:
            return swap_ab, tile_n, tile_k
        return (
            values.get("swap_ab", swap_ab),
            values.get("tile_n", tile_n),
            values.get("tile_k", tile_k),
        )

    _tiles.__wrapped__ = stock
    _tiles.inkling_rules = rules
    return _tiles


def describe(rules):
    parts = []
    for tok, w, values in rules:
        head = f"<={tok}" + (f"@{w}" if w is not None else "")
        parts.append(head + ":" + ",".join(f"{k}={int(v)}" for k, v in values.items()))
    return ";".join(parts)


def patch(module, rules):
    stock = getattr(module, TARGET_ATTR, None)
    if not callable(stock):
        raise RuntimeError(
            f"inkling_kernels.moe_tiles: {TARGET_MODULE}.{TARGET_ATTR} is missing"
        )
    if getattr(stock, "inkling_rules", None) is not None:
        return  # already patched in this process
    # The stock table must accept (tokens=, width=) and return a 3-tuple.
    probe = stock(tokens=8, width=2304)
    if not (isinstance(probe, tuple) and len(probe) == 3):
        raise RuntimeError(
            f"inkling_kernels.moe_tiles: {TARGET_MODULE}._tiles returned {probe!r}"
        )
    module._tiles = make_tiles(stock, rules)
