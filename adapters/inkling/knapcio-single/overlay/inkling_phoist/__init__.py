"""inkling_phoist: prefill loop-bank hoist over inkling_pfuse's loop_mix (exact up to FP32 summation order).

    INKLING_PFUSE_HOIST=1     on, Option A (one bank-down launch per pass side; FP32-order class vs pfuse under the
                              serving ptxas). =exact: the bit-identical variant (groups 3-4 through the per-visit down
                              kernel, slower). Unset / 0: nothing is imported. Needs INKLING_PFUSE with loopmix (or all).

In loop passes 2 and 3 the message bank (two BF16 sources) and the 48 message queries are fixed for the whole pass,
but pfuse's loop_mix recomputes the bank attention-residual weights, the message RMS statistics, the normed message
planes and the controller down projection over the message columns at each of the 24 visits. Here the first
loop_mix call of a pass runs phoist.prepare() once (bank weights + message rstd for all 12 layers x attention /
mlp side, and the down-projection K groups 3-4, which read message columns only); every visit then runs
phoist.loop_mix_hoisted (stats without the bank, down groups 0-2 with group 2's message columns rebuilt from the
bank, mix1 adding the prepared groups 3-4 in group order). Same expressions, same order, same BF16 rounding:
outputs bit-identical to pfuse (tools/phoist_probe.py: 512-8192 rows, eager and graph replay, per-group partials).
Measured: one loop pass 110.6 -> 97.1 ms at 8,192 rows (-26.9 ms per 8K chunk), 29.0 -> 23.4 ms at 2,048.

Patches (post-import hooks, signature / source checked at install):
  inkling_pfuse.mix.loop_mix                       wrapped; anything it cannot map falls through to pfuse
  sglang...inkling2_loop.LoopMessageBank.__init__  marks a new pass (its block list identifies the bank)
  sglang...inkling2.Inkling2CausalLLM.prepare_loop also stacks the 12 core layers' controller down weights
                                                   [12, 4, 64, 4608] (28 MB; refreshed in place on reload)
CUDA graphs: prepare() is issued inside the first loop_mix call of the pass, so a captured prefill graph contains
it at that position and replays it; nothing syncs with the host. Eager runs do the same.
"""

import importlib.util
import os
import sys

from . import _hook

STATE = {"installed": False, "applied": [], "hoisted": 0, "fallthrough": 0, "prepared": 0,
         "checked": {}, "cfg_ok": {}, "trusted": False, "failed": None, "variant": None, "worst": 0.0}
# Option A self-check tolerance vs pfuse (FP32-order class): max |hoist - pfuse| <= TOL_MAX * max |pfuse| and
# mean |hoist - pfuse| <= TOL_MEAN * mean |pfuse|. A wrong layer / side mapping or a stale bank gives O(1) errors.
TOL_MAX = 2e-2
TOL_MEAN = 2e-3
ABS_FLOOR = 1e-6
# Self-check (toolchain guard): bit identity depends on the code the assembler emits (runs/w18-s06.log: a kernel
# variant matched pfuse under Triton's bundled ptxas but not under the CUDA 13 ptxas SGLang selects). The first eager
# pass at each row count runs pfuse next to the hoist on all 24 calls and compares bit for bit (pfuse's result is
# returned meanwhile); a captured graph hoists only row counts that passed, or row counts whose launch
# configuration (split / summed down groups, BLOCK_K) passed at TRUST_AFTER sizes; any mismatch disables the hoist for
# the process (logged once).
CHECK_CALLS = 24
TRUST_AFTER = 2
# model-owned packs, keyed by the data_ptr of the model's loop_scaled_queries (the tensor is kept in the pack, so
# the address cannot be reused while the pack lives): {"sq": tensor, "down_all": [12, 4, 64, 4608],
# "ptrs": {(layer, side): data_ptr of the model's controller down slice pfuse passes}}
PACKS = {}


class BankList(list):
    """LoopMessageBank.blocks as a list subclass carrying the pass's state: `sq` (the bank's scaled queries) and
    `prep` (phoist.prepare output, built on the pass's first hoisted call; lives and dies with the bank)."""

    __slots__ = ("sq", "prep")

    def __init__(self, *a):
        super().__init__(*a)
        self.sq = None
        self.prep = None
D = 2304
_SOURCES = (
    ("srt/models/inkling_common/inkling2_loop.py", (
        "class LoopMessageBank:",
        "def __init__(self, blocks: tuple[torch.Tensor, ...], scaled_queries: torch.Tensor):",
        "self.blocks = [x.contiguous() for x in blocks]",
        "self.scaled_queries = scaled_queries",
    )),
    ("srt/models/inkling2.py", (
        "def prepare_loop(self) -> None:",
        "layer.controller_weights = stack_controller_weights(layer.controllers)",
        "messages = LoopMessageBank(bank, self.loop_scaled_queries)",
    )),
)


class PhoistFatal(BaseException):
    """BaseException: SGLang's model registry swallows Exception during model imports."""


def _log(msg):
    print(f"[inkling_phoist] {msg}", file=sys.stderr, flush=True)


def enabled(env=None):
    env = os.environ if env is None else env
    on = env.get("INKLING_PFUSE_HOIST", "").strip().lower()
    if on in ("", "0", "off", "false", "no"):
        return False
    if on not in ("1", "on", "true", "yes", "a", "exact"):
        raise RuntimeError(f"inkling_phoist: INKLING_PFUSE_HOIST={on!r} (use 1, exact or 0)")
    pf = {x.strip().lower() for x in env.get("INKLING_PFUSE", "").split(",")}
    if not ({"all", "loopmix"} & pf):
        raise RuntimeError("inkling_phoist: needs INKLING_PFUSE with loopmix (or all)")
    return "exact" if on == "exact" else "a"


def check_sources(root=None):
    if root is None:
        spec = importlib.util.find_spec("sglang")
        if spec is None or not spec.submodule_search_locations:
            raise RuntimeError("inkling_phoist: the sglang package is not importable")
        root = list(spec.submodule_search_locations)[0]
    for rel, needles in _SOURCES:
        path = os.path.join(root, rel)
        try:
            text = open(path, encoding="utf-8").read()
        except OSError as exc:
            raise RuntimeError(f"inkling_phoist: patch target {path} is missing") from exc
        missing = [n for n in needles if n not in text]
        if missing:
            raise RuntimeError(f"inkling_phoist: patch target {path} lacks {missing}")


def locate(sq, bank_queries, nm):
    """(core layer index, side) of a loop_mix call from its bank-query view into the pass's scaled queries
    [12, 4, D] (rows attn, k, v = 0:3, mlp = 3:4), or None."""
    if sq is None or bank_queries.dim() != 2 or bank_queries.shape != (nm, D) or not bank_queries.is_contiguous():
        return None
    if sq.shape != (12, 4, D) or not sq.is_contiguous() or sq.dtype != bank_queries.dtype:
        return None
    if sq.device != bank_queries.device:
        return None
    off = bank_queries.data_ptr() - sq.data_ptr()
    if off < 0 or off % (D * sq.element_size()) or off >= sq.numel() * sq.element_size():
        return None
    row = off // (D * sq.element_size())
    layer, r0 = divmod(row, 4)
    if r0 == 0 and nm == 3:
        return layer, "attn"
    if r0 == 3 and nm == 1:
        return layer, "mlp"
    return None


def build_pack(model):
    """Stack the 12 core layers' controller down weights of `model` into its pack (in place on reload)."""
    import torch

    core = model.config.loop_core_layers
    layers = [model.layers[i] for i in range(core.start, core.stop)]
    downs = [getattr(l_, "controller_weights", None) for l_ in layers]
    if len(downs) != 12 or any(d is None or d[0].shape != (4, 64, 4608) or d[0].dtype != torch.bfloat16
                               for d in downs):
        _log(f"unexpected loop layout ({len(downs)} core layers); hoist disabled for this model")
        return None
    sq = model.loop_scaled_queries
    stacked = torch.stack([d[0] for d in downs])
    key = sq.data_ptr()
    old = PACKS.get(key)
    if (old is not None and old["sq"] is sq and old["down_all"].shape == stacked.shape
            and old["down_all"].device == stacked.device and old["down_all"].dtype == stacked.dtype):
        old["down_all"].copy_(stacked)  # reload of the same model: same buffer (captured graphs keep the pointer)
        pack = old
        pack["startup_ok"] = False  # new weights: prepare_loop re-runs the startup self-test
    else:
        pack = {"sq": sq, "down_all": stacked.contiguous()}
        PACKS[key] = pack
    pack["ptrs"] = {(i, side): d[0][rows].data_ptr() for i, d in enumerate(downs)
                    for side, rows in (("attn", slice(0, 3)), ("mlp", slice(3, 4)))}
    return pack


def tolerance_ok(y, ref):
    """Option A guard: y and ref finite, and per leading slice (controller) max |y - ref| <= TOL_MAX * max |ref| +
    ABS_FLOOR and mean |y - ref| <= TOL_MEAN * mean |ref| + ABS_FLOOR. Returns (ok, worst relative max error)."""
    import torch

    if y.shape != ref.shape or y.dtype != ref.dtype:
        return False, float("inf")
    yf, rf = y.float().reshape(y.shape[0], -1), ref.float().reshape(ref.shape[0], -1)
    fin = torch.isfinite(yf).all() & torch.isfinite(rf).all()
    d = (yf - rf).abs()
    r = rf.abs()
    stats = torch.stack([d.amax(1), d.mean(1), r.amax(1), r.mean(1)]).cpu()  # one transfer
    if not bool(fin):
        return False, float("inf")
    emax, emean, rmax, rmean = stats
    ok = bool(((emax <= TOL_MAX * rmax + ABS_FLOOR) & (emean <= TOL_MEAN * rmean + ABS_FLOOR)).all())
    return ok, float((emax / rmax.clamp_min(1e-30)).max())


def config_key(tokens, device):
    """The launch / specialization regime of a hoisted loop pass at `tokens` rows: pfuse's down tiling (split K
    groups, BLOCK_K) and Triton's 16-divisibility specialization of the row count."""
    from . import phoist

    split, _, _, bk = phoist._down_tiles(tokens, device)
    return bool(split), int(bk), tokens % 16 == 0


STARTUP_ROWS = (600, 1024, 1025, 4096, 4097, 8192)  # odd and 16-aligned rows in each down-tiling regime


def mixed_chunk_on():
    """True with --enable-mixed-chunk (MIXED batches replay through EXTEND prefill graphs: decode rows would get the
    hoisted prefill math); None when the server args are not published yet."""
    try:
        from sglang.srt.runtime_context import get_server_args

        return bool(getattr(get_server_args(), "enable_mixed_chunk", False))
    except Exception:  # noqa: BLE001
        return None


def startup_check(model, pack):
    """Option A self-test at model load (prepare_loop, before KV sizing and CUDA-graph capture): one loop pass of
    all 24 loop_mix calls on synthetic sources / bank with this model's controller weights and loop queries, pfuse
    vs hoist, at STARTUP_ROWS (odd and 16-aligned row counts in every down-tiling regime, up to the 8K chunk),
    attention side with a RMSNorm (NORM mode) and the mlp side in FP32 mode. Sets pack["startup_ok"] and
    pack["armed"] (the config_key regimes that passed; other regimes run pfuse)."""
    import torch

    from inkling_pfuse import mix

    from . import phoist

    orig = getattr(mix.loop_mix, "__wrapped__", mix.loop_mix)
    core = model.config.loop_core_layers
    layers = [model.layers[i] for i in range(core.start, core.stop)]
    sq = pack["sq"]
    dev = sq.device
    worst = 0.0
    g = torch.Generator(device=dev).manual_seed(1234)
    try:
        with torch.no_grad():
            armed = set()
            for T in STARTUP_ROWS:
                # explicit dtypes: model loading runs under SGLang's default dtype (BF16)
                bf = lambda sc: (torch.randn(T, D, device=dev, generator=g, dtype=torch.float32) * sc).to(  # noqa: E731
                    torch.bfloat16)
                bank = [bf(1.5), bf(0.8)]
                srcs = [bf(0.6), bf(2.0), bf(1.0), bf(1.2), bf(0.9)]
                ones = torch.ones(D, device=dev, dtype=torch.bfloat16)
                q3 = torch.randn(3, D, device=dev, generator=g, dtype=torch.float32) * D ** -0.5
                q1 = torch.randn(1, D, device=dev, generator=g, dtype=torch.float32) * D ** -0.5
                fn = phoist.loop_mix_hoisted
                prep = phoist.prepare(bank, sq, pack["down_all"], T, "a")
                for li, lay in enumerate(layers):
                    cw = lay.controller_weights
                    n = 2 + li % 3  # 2..4 state sources
                    mlp_src, mlp_add = (srcs[:n] + [srcs[4]], None) if li % 4 == 3 else (srcs[:n], srcs[4])
                    for side, rows, q, src, add, nw in (
                            ("attn", slice(0, 3), q3 if li % 2 else q1, srcs[:n], None, [ones] * 3),
                            ("mlp", slice(3, 4), q1, mlp_src, mlp_add, None)):
                        d_, gn, fg = (w[rows] for w in cw)
                        qc = q.contiguous()
                        ref = orig(src, add, qc, True, bank, sq[li, rows], d_, gn, fg, norm_weights=nw, eps_norm=1e-6)
                        # loop_mix_hoisted takes no bank_queries (the prep holds the bank weights)
                        y = fn(prep, li, side, src, add, qc, True, bank, d_, gn, fg, norm_weights=nw, eps_norm=1e-6)
                        ok, w = tolerance_ok(y, ref)
                        worst = max(worst, w)
                        if not ok:
                            STATE["failed"] = (T, li, side)
                            _log(f"startup self-test FAILED at {T} rows, core layer {li} {side} (rel max err {w:.2e}): "
                                 f"hoist disabled (pfuse runs)")
                            return False
                del prep, bank, srcs
                torch.cuda.synchronize()
                armed.add(config_key(T, dev))
                torch.cuda.empty_cache()
    except Exception as exc:  # noqa: BLE001  (compile / resource errors leave the model on pfuse)
        STATE["failed"] = ("startup", repr(exc)[:200])
        _log(f"startup self-test ERROR ({exc!r:.200}): hoist disabled (pfuse runs)")
        return False
    finally:
        torch.cuda.empty_cache()
    STATE["worst"] = worst
    pack["armed"] = armed
    pack["startup_ok"] = True
    _log(f"startup self-test passed ({STATE['variant']}: {24 * len(STARTUP_ROWS)} calls at "
         f"{' / '.join(map(str, STARTUP_ROWS))} rows, worst rel max err vs pfuse {worst:.2e}; armed {sorted(armed)})")
    return True


def make_loop_mix(orig):
    def loop_mix(sources, addend, queries, normalize_values, bank, bank_queries, down, gain, forget,
                 norm_weights=None, eps_norm=1e-6, fuse_norm=True):
        args = (sources, addend, queries, normalize_values, bank, bank_queries, down, gain, forget)
        kw = dict(norm_weights=norm_weights, eps_norm=eps_norm, fuse_norm=fuse_norm)
        pack = PACKS.get(bank.sq.data_ptr()) if isinstance(bank, BankList) and bank.sq is not None else None
        nm = bank_queries.shape[0] if bank_queries.dim() == 2 else 0
        where = None
        if pack is not None and pack["sq"] is bank.sq and (norm_weights is None or fuse_norm) and normalize_values:
            where = locate(bank.sq, bank_queries, nm)
        tokens = sources[0].shape[0]
        if (where is None or len(bank) != 2 or bank[0].shape[0] != tokens
                or down.data_ptr() != pack["ptrs"].get(where)):  # pfuse's down must be this model's layer slice
            STATE["fallthrough"] += 1
            return orig(*args, **kw)
        import torch

        from . import phoist

        if isinstance(getattr(phoist, "VARIANT", None), dict):
            phoist.VARIANT["name"] = STATE["variant"]
        if STATE["failed"] is not None:
            STATE["fallthrough"] += 1
            return orig(*args, **kw)
        capturing = torch.cuda.is_current_stream_capturing()
        if STATE["variant"] != "exact":
            # Option A: no per-call check (FP32-order class, so a bitwise check is meaningless and a tolerance check
            # would need host syncs); the hoist runs only if the startup self-test on this model passed
            # (startup_check, before any CUDA-graph capture). Validated offline by tools/gpu_check_phoist.py.
            if not pack.get("startup_ok") or config_key(tokens, bank[0].device) not in pack.get("armed", ()):
                STATE["fallthrough"] += 1
                return orig(*args, **kw)
            verified = True
        else:
            checked = STATE["checked"].get(tokens, 0)
            cfg = phoist._down_tiles(tokens, bank[0].device)
            verified = checked >= CHECK_CALLS or STATE["cfg_ok"].get(cfg, 0) >= TRUST_AFTER
        if capturing and not verified:
            STATE["fallthrough"] += 1
            return orig(*args, **kw)
        layer, side = where
        prep = bank.prep
        if prep is None or prep["T"] != tokens:
            prep = bank.prep = phoist.prepare(list(bank), bank.sq, pack["down_all"], tokens, STATE["variant"])
            STATE["prepared"] += 1
        y = phoist.loop_mix_hoisted(prep, layer, side, sources, addend, queries, normalize_values, list(bank), down,
                                    gain, forget, norm_weights=norm_weights, eps_norm=eps_norm)
        if verified or capturing:
            STATE["hoisted"] += 1
            return y
        ref = orig(*args, **kw)
        bits = torch.int32 if ref.dtype == torch.float32 else torch.int16
        bad = ref.shape != y.shape or ref.dtype != y.dtype or not torch.equal(ref.view(bits), y.view(bits))
        if bad:
            STATE["failed"] = (tokens, layer, side)
            _log(f"self-check MISMATCH at {tokens} rows, core layer {layer} {side}: hoist disabled (pfuse runs)")
            return ref
        STATE["checked"][tokens] = checked + 1
        if checked + 1 == CHECK_CALLS:
            n = STATE["cfg_ok"][cfg] = STATE["cfg_ok"].get(cfg, 0) + 1
            if n == TRUST_AFTER:
                _log(f"self-check passed at {n} row counts with down config {cfg}: trusted for that config")
        return ref

    loop_mix.__wrapped__ = orig
    loop_mix._inkling_phoist = True
    return loop_mix


def _patch_mix(mod):
    orig = getattr(mod, "loop_mix", None)
    if orig is None:
        raise PhoistFatal("inkling_phoist: inkling_pfuse.mix.loop_mix is missing")
    if getattr(orig, "_inkling_phoist", False):
        return
    mix = getattr(mod, "MIX", {})
    if not (mix.get("split_down") and mix.get("per_ctrl")):
        raise PhoistFatal("inkling_phoist: pfuse MIX must use split_down and per_ctrl")
    mod.loop_mix = make_loop_mix(orig)
    STATE["applied"].append("inkling_pfuse.mix.loop_mix")


def _patch_bank(mod):
    cls = getattr(mod, "LoopMessageBank", None)
    if cls is None:
        raise PhoistFatal("inkling_phoist: LoopMessageBank is missing")
    init = cls.__init__
    if getattr(init, "_inkling_phoist", False):
        return

    def __init__(self, blocks, scaled_queries):
        init(self, blocks, scaled_queries)
        bl = BankList(self.blocks)           # the list object pfuse passes as `bank`; state is pass-local
        bl.sq = self.scaled_queries
        bl.prep = None
        self.blocks = bl

    __init__._inkling_phoist = True
    __init__.__wrapped__ = init
    cls.__init__ = __init__
    STATE["applied"].append("LoopMessageBank.__init__")


def _patch_model(mod):
    cls = getattr(mod, "Inkling2CausalLLM", None)
    orig = getattr(cls, "prepare_loop", None) if cls is not None else None
    if orig is None:
        raise PhoistFatal("inkling_phoist: Inkling2CausalLLM.prepare_loop is missing")
    if getattr(orig, "_inkling_phoist", False):
        return

    def prepare_loop(self):
        orig(self)
        if self.config.loop_config is None or getattr(self, "loop_scaled_queries", None) is None:
            return
        if mixed_chunk_on():
            raise PhoistFatal("inkling_phoist: --enable-mixed-chunk is on; refusing (decode rows in MIXED batches "
                              "replayed through EXTEND graphs would get the hoisted prefill math)")
        try:
            pack = build_pack(self)
        except Exception as exc:  # noqa: BLE001  (e.g. an OOM stacking the weights: pfuse runs)
            STATE["failed"] = ("pack", repr(exc)[:200])
            PACKS.pop(getattr(self.loop_scaled_queries, "data_ptr", lambda: None)(), None)
            _log(f"weight pack failed ({exc!r:.200}): hoist disabled (pfuse runs)")
            return
        if pack is not None and STATE["variant"] != "exact" and not pack.get("startup_ok"):
            startup_check(self, pack)

    prepare_loop._inkling_phoist = True
    prepare_loop.__wrapped__ = orig
    cls.prepare_loop = prepare_loop
    STATE["applied"].append("Inkling2CausalLLM.prepare_loop")


def install(env=None, root=None):
    if STATE["installed"]:
        return bool(STATE["applied"]) or None
    STATE["installed"] = True
    variant = enabled(env)
    if not variant:
        return False
    check_sources(root)
    STATE["variant"] = variant
    _hook.when_imported("inkling_pfuse.mix", _patch_mix)
    _hook.when_imported("sglang.srt.models.inkling_common.inkling2_loop", _patch_bank)
    _hook.when_imported("sglang.srt.models.inkling2", _patch_model)
    _log(f"enabled (variant {variant}): loop-bank work of pfuse loop_mix prepared once per pass "
         f"({'bit-identical' if variant == 'exact' else 'exact up to FP32 summation order'})")
    return True
