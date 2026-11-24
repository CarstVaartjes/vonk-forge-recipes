"""Inkling-2-Mini NVFP4 expert scale codes on one GB10 (SGLang overlay). Exact.

    INKLING_SCODES=1          (default: empty -> install() does nothing)

The NVFP4 routed experts (layers 0-19) store one UE4M3 scale byte per 16 weights (11 % of their bytes). At load,
every weight row's scales are re-encoded as a row base byte + a 5-bit code per scale (encode.py; 0.39 of the scale
bytes); the SM120 grouped GEMM reads the codes and expands them in shared memory into exactly the bytes the stock
kernel would have loaded (csrc/scodes_sm120_mma_array.hpp), so the MMAs see bit-identical operands. -4.1 % of the
NVFP4 expert bytes read per decode step.

Hooks (post-import, no engine file edited):
  sglang.srt.models.inkling_common.inkling2_moe.Inkling2Experts.swizzle_nvfp4_scales  (post_load_weights) ->
      afterwards encode w13/w2 scales of NVFP4 SM120 experts; the codes ride on the scale Parameter
      (attribute _inkling_scodes). Tensors the 5-bit code cannot hold stay stock.
  Inkling2Experts._load_shard -> a reload of a coded scale re-encodes it in place (captured graphs keep their
      addresses); scales that no longer fit the code end the process (exit 78).
  sglang.kernels.ops.gemm.inkling2_grouped_sm120.nvfp4_expert_matmul -> coded weights with a swap_ab tile the
      scodes kernel has (tile_n 32/64/128 x tile_k 128/256, not 128x256) run it; a non-swap choice at <= 3072
      tokens (shipped moe_tiles rule for w13 at 9-64 tokens) runs swap 32x256 coded (bit-identical, checked);
      everything else runs stock. Tiles come from the module's _tiles at call time, so inkling_kernels.moe_tiles
      and inkling_moe2 keep working.
Self-check: the first 3 eager calls per (n, k, stock tactic, coded tile) also run the stock kernel and compare the BF16 outputs bit
for bit (one host sync each, warm-up only); a difference exits the process with code 78. During CUDA-graph capture
a configuration with no passed check runs stock, so a graph never records an unchecked kernel. INKLING_SCODES_CHECK=0 skips
the self-check (not recommended).
Patch targets are checked as source strings at install() (no sglang/torch import); a changed target raises
RuntimeError (the combined loader exits 78); one found changed at import time raises PatchError (BaseException).
"""

import importlib.util
import os
import sys

STATE = {"installed": False, "encoded": [], "checked": {}, "calls": 0, "stock_calls": 0}
SWAP_OVERRIDE_MAX_TOKENS = 3072
SWAP_OVERRIDE_TILE = (32, 256)
CHECKS_PER_CONFIG = 3  # eager self-checks per (n, k, tile) before the config counts as checked
MARK = "_inkling_scodes_wrapped"
ATTR = "_inkling_scodes"
MOE_MODULE = "sglang.srt.models.inkling_common.inkling2_moe"
GEMM_MODULE = "sglang.kernels.ops.gemm.inkling2_grouped_sm120"

_SOURCE_CHECKS = (
    ("kernels/ops/gemm/inkling2_grouped_sm120.py", (
        "def nvfp4_expert_matmul(values, scales, weight, weight_scales, offsets, tokens):",
        "swap_ab, tile_n, tile_k = _tiles(tokens=tokens, width=packed_width * 2)",
        "weight_scales.view(torch.uint8).view(experts, n, packed_width // 8),",
        "group_gemm_nvfp4_nt_groupwise(",
    )),
    ("srt/models/inkling_common/inkling2_moe.py", (
        "def swizzle_nvfp4_scales(self):",
        "def _prepare_sm120_scales(self):",
        "scale.data = swizzle_blockscale(scale.data).flatten(1)",
        "def _load_shard(self, param, loaded, dim, expert_axis=True, checkpoint_shape=None):",
        "weight_scales=self.w13_scale,",
        "weight_scales=self.w2_scale,",
        "grouped_sm120.nvfp4_expert_matmul(",
        "from sglang.kernels.ops.gemm import inkling2_grouped_sm120 as grouped_sm120",
    )),
)


class PatchError(BaseException):
    """A patch target changed (BaseException: SGLang's model registry swallows Exception during imports)."""


def _log(msg):
    sys.stderr.write(f"[inkling_scodes pid={os.getpid()}] {msg}\n")
    sys.stderr.flush()


def _on(value):
    return value.strip().lower() not in ("", "0", "off", "none", "false", "no")


def _sglang_root():
    spec = importlib.util.find_spec("sglang")
    if spec is None or not spec.submodule_search_locations:
        raise RuntimeError("inkling_scodes: the sglang package is not importable")
    return list(spec.submodule_search_locations)[0]


def check_sources(root=None):
    root = root or _sglang_root()
    for rel, needles in _SOURCE_CHECKS:
        path = os.path.join(root, rel)
        try:
            with open(path, encoding="utf-8") as f:
                text = f.read()
        except OSError as exc:
            raise RuntimeError(f"inkling_scodes: patch target {path} is missing") from exc
        missing = [n for n in needles if n not in text]
        if missing:
            raise RuntimeError(f"inkling_scodes: patch target {path} lacks {missing}")


def encode_experts(experts):
    """Attach scale codes to the NVFP4 SM120 w13/w2 scales of one Inkling2Experts module (after swizzling)."""
    import torch

    from . import encode

    if not getattr(experts, "native_sm120", False) or getattr(experts, "_routed_kernels", None) != torch.uint8:
        return
    shapes = getattr(experts, "_nvfp4_scale_shapes", {})
    for name in ("w13", "w2"):
        weight = getattr(experts, name + "_weight", None)
        scale = getattr(experts, name + "_scale", None)
        if weight is None or scale is None or name not in shapes or not weight.is_cuda or not weight.numel():
            continue
        e, n, half_k = weight.shape
        k = half_k * 2
        if tuple(scale.shape) == shapes[name]:
            continue  # not swizzled (yet): leave stock
        if getattr(scale, ATTR, None) is not None:
            reencode(scale)  # post_load_weights after a reload: keep the buffers captured graphs hold
            continue
        try:
            codes, bases, stride = encode.encode(torch, scale.data, n, k)
        except encode.ScodesError as exc:
            _log(f"{name} {tuple(weight.shape)} stays stock: {exc}")
            setattr(scale, ATTR, None)
            continue
        setattr(scale, ATTR, (codes, bases, stride, n, k))
        STATE["encoded"].append((name, e, n, k, scale.numel(), codes.numel() + bases.numel()))
    if any(getattr(getattr(experts, nm + "_scale", None), ATTR, None) is not None for nm in ("w13", "w2")):
        # SGLang's direct tensor updates skip _load_shard/post_load_weights and would leave stale codes.
        experts._direct_weight_update_error = (
            "inkling_scodes: NVFP4 expert scales are mirrored as scale codes; direct weight updates would "
            "leave them stale (reload through load_weights or run without INKLING_SCODES)")
        from . import kernel

        kernel.module()  # build/load the JIT module now, never during graph capture
        kernel.prepare(torch, scale.device)


def reencode(scale):
    """After an in-place reload of a coded scale: re-encode into the SAME code/base buffers (graph addresses)."""
    import torch

    from . import encode

    codes, bases, stride, n, k = getattr(scale, ATTR)
    try:
        new_codes, new_bases, new_stride = encode.encode(torch, scale.data, n, k)
    except encode.ScodesError as exc:
        # Graphs may already embed the old codes; running on would silently use stale scales.
        _log(f"FATAL reloaded scales cannot be coded ({exc}); restart without INKLING_SCODES")
        raise SystemExit(78)
    codes.copy_(new_codes)
    bases.copy_(new_bases)


def _wrap_moe(module):
    cls = getattr(module, "Inkling2Experts", None)
    if cls is None:
        raise PatchError("inkling_scodes: Inkling2Experts is missing")
    swz = getattr(cls, "swizzle_nvfp4_scales", None)
    load = getattr(cls, "_load_shard", None)
    if not callable(swz) or not callable(load):
        raise PatchError("inkling_scodes: Inkling2Experts.swizzle_nvfp4_scales/_load_shard missing")
    if getattr(swz, MARK, False):
        return

    def swizzle_nvfp4_scales(self, *args, **kwargs):
        out = swz(self, *args, **kwargs)
        encode_experts(self)
        return out

    def _load_shard(self, param, *args, **kwargs):
        out = load(self, param, *args, **kwargs)
        if getattr(param, ATTR, None) is not None:
            reencode(param)  # captured graphs hold the code addresses: rewrite them in place
        return out

    for f, inner in ((swizzle_nvfp4_scales, swz), (_load_shard, load)):
        f.__wrapped__ = inner
        setattr(f, MARK, True)
    cls.swizzle_nvfp4_scales = swizzle_nvfp4_scales
    cls._load_shard = _load_shard


def _wrap_gemm(module):
    stock = getattr(module, "nvfp4_expert_matmul", None)
    if not callable(stock):
        raise PatchError("inkling_scodes: nvfp4_expert_matmul is missing")
    if getattr(stock, MARK, False):
        return
    if not callable(getattr(module, "_tiles", None)):
        raise PatchError("inkling_scodes: _tiles is missing")
    check = _on(os.environ.get("INKLING_SCODES_CHECK", "1"))

    from . import kernel

    def nvfp4_expert_matmul(values, scales, weight, weight_scales, offsets, tokens):
        import torch

        coded = getattr(weight_scales, ATTR, None)
        if coded is None or not values.shape[0]:
            STATE["stock_calls"] += 1
            return stock(values, scales, weight, weight_scales, offsets, tokens)
        codes, bases, stride, n, k = coded
        experts, wn, packed_width = weight.shape
        stock_tactic = module._tiles(tokens=tokens, width=packed_width * 2)
        swap_ab, tile_n, tile_k = stock_tactic
        if not swap_ab and tokens <= SWAP_OVERRIDE_MAX_TOKENS:
            # The coded kernel exists only with swap_ab; at decode sizes swap 32x256 is the bit-identical, faster
            # choice (inkling_moe2 measurements). The self-check compares against the stock non-swap call.
            tile_n, tile_k = SWAP_OVERRIDE_TILE
        elif not swap_ab:
            STATE["stock_calls"] += 1
            return stock(values, scales, weight, weight_scales, offsets, tokens)
        if (tile_n, tile_k) not in kernel.TILES or (wn, packed_width * 2) != (n, k):
            STATE["stock_calls"] += 1
            return stock(values, scales, weight, weight_scales, offsets, tokens)
        # Checked per (shape, stock tactic, coded tile): an override of a non-swap tactic is qualified separately.
        key = (n, k, tuple(stock_tactic), (tile_n, tile_k))
        done = STATE["checked"].get(key, 0)
        ref = None
        if check and done < CHECKS_PER_CONFIG:
            if torch.cuda.is_current_stream_capturing():
                if done == 0:
                    STATE["stock_calls"] += 1
                    return stock(values, scales, weight, weight_scales, offsets, tokens)
            else:
                ref = stock(values, scales, weight, weight_scales, offsets, tokens)
        out = torch.empty((values.shape[0], n), dtype=torch.bfloat16, device=values.device)
        kernel.grouped_gemm(torch, values, scales, weight, weight_scales, offsets, codes, bases, stride,
                            tile_n, tile_k, out)
        STATE["calls"] += 1
        if ref is not None:
            if not torch.equal(ref.view(torch.int16), out.view(torch.int16)):
                bad = int((ref.view(torch.int16) != out.view(torch.int16)).sum())
                _log(f"FATAL self-check: {bad} of {out.numel()} outputs differ from stock at rows "
                     f"{values.shape[0]}, n/k/tile {key}")
                raise SystemExit(78)
            STATE["checked"][key] = done + 1
        return out

    nvfp4_expert_matmul.__wrapped__ = stock
    setattr(nvfp4_expert_matmul, MARK, True)
    module.nvfp4_expert_matmul = nvfp4_expert_matmul


def install(root=None):
    """Activate when INKLING_SCODES is on. Idempotent; raises RuntimeError on a changed patch target."""
    if STATE["installed"]:
        return
    STATE["installed"] = True
    if not _on(os.environ.get("INKLING_SCODES", "")):
        return
    check_sources(root)
    from . import _hook

    _hook.when_imported(MOE_MODULE, _wrap_moe)
    _hook.when_imported(GEMM_MODULE, _wrap_gemm)
    _log("enabled: NVFP4 expert scale codes (5-bit, row base), self-check "
         + ("on" if _on(os.environ.get("INKLING_SCODES_CHECK", "1")) else "OFF"))
