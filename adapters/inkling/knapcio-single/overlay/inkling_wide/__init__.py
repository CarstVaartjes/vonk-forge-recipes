"""Inkling-2-Mini on DGX Spark (GB10, sm_121a): weight-streaming BF16 GEMM for 17..256 rows (DSpark verify).

    INKLING_WIDE=dense,lmhead        (default: empty -> install() does nothing)

dense   Inkling2Attention.qkvr (plain calls and stacked q/k/v calls, M rows per source) and .wo_ud
lmhead  the target's FP32-logit LM head (unquant.sm12x_fp32_lm_head)
all     both

The weights are read from DRAM once per call: inkling_pico12's exactly coded copy (pico's bf16x format, ~0.775 of
the bytes) when inkling_pico12 coded the layer, else the layer's own BF16 weight. Install AFTER inkling_pico12
(and, to take stacked calls of 128..256 rows before inkling_prefill's qkvr_split, after inkling_prefill too).

Other settings (env):
  INKLING_WIDE_MIN_M     rows of a call taken from (per source for stacked calls; 1..256, default 17)
  INKLING_WIDE_MAX_M     rows of a call taken up to (MIN_M..256, default 256)
  INKLING_WIDE_FORMAT    auto (default: pico's coded copy when there is one, else raw) | raw (always the BF16 weight)
  INKLING_WIDE_SPLITK    force split-K slices per weight kind, e.g. "wo=4,qkvr=1,head=1" (default: by shape)
  INKLING_WIDE_MT        tiles per warp above 128 rows: 0 (default: 1, 16 warps) | 1 | 2 (8 warps)
  INKLING_WIDE_MAX_M_KIND  per-kind row caps (default "qkvr=96,wo=32,head=256": where wide beats cuBLAS on GB10)
  INKLING_WIDE_MG        warp groups along M at 65..128 rows: 2 (default; 16 warps, half the rows each) | 1
  INKLING_WIDE_PFD       L2 prefetch distance in 64-value batches (default 1; 0 = off)
  INKLING_WIDE_SCOPE     verify (default: only target decode / target-verify forwards; prefill and extend keep the
                         stock path) | all
  INKLING_WIDE_PDL       1 (default): launch with programmatic dependent launch; 0: plain launches (same bits)
  INKLING_WIDE_CACHE     extension build cache (default ~/.cache/inkling-wide)
  INKLING_WIDE_VERBOSE   1: print the armed line in every process and the nvcc build output

install() parses the env, checks the patch targets in the installed sglang sources (without importing sglang or
torch) and attaches the patches through post-import hooks; the extension is built (or reused) and the weights
registered when the model's weights finish loading. Any problem raises RuntimeError here, or ends the process with
exit code 78 later.
"""

import importlib.util
import os
import sys

LEVERS = ("dense", "lmhead")
STATE = {"installed": False, "levers": (), "cfg": None}

_SOURCE_CHECKS = {
    "common": (
        (
            "srt/layers/quantization/unquant.py",
            ("class UnquantizedLinearMethod(", "def apply_into(", "is_batch_invariant_mode_enabled"),
        ),
        (
            "srt/models/inkling2.py",
            ("class Inkling2ForConditionalGeneration(", "def post_load_weights(", "forward_batch: ForwardBatch,"),
        ),
        ("srt/model_executor/forward_mode.py", ("def is_decode(self):", "def is_target_verify(self):")),
    ),
    "dense": (
        (
            "srt/models/inkling_common/inkling2_attn.py",
            (
                "class Inkling2Attention(",
                "def _project_sources(",
                "class AttentionSources(",
                "self.qkvr = MergedColumnParallelLinear(",
                "self.wo_ud = RowParallelLinear(",
                "self.split_sizes = [",
                "q, k_proj[1], v_proj[2], r, gate, sinks",
            ),
        ),
    ),
    "lmhead": (
        ("srt/layers/quantization/unquant.py", ("def sm12x_fp32_lm_head(", "def sm12x_fp32_lm_head_applies(")),
        (
            "srt/layers/logits_processor.py",
("sm12x_fp32_lm_head_applies(hidden_states, lm_head.weight, out)",),
        ),
    ),
}


def _log(msg):
    sys.stderr.write(f"[inkling_wide pid={os.getpid()}] {msg}\n")
    sys.stderr.flush()


def _int_env(env, name, default, lo, hi):
    raw = env.get(name, "").strip()
    try:
        value = int(raw) if raw else default
    except ValueError as exc:
        raise ValueError(f"{name}={raw!r} is not an integer") from exc
    if not lo <= value <= hi:
        raise ValueError(f"{name}={value} not in [{lo}, {hi}]")
    return value


def parse_kinds(raw, name, lo, hi):
    """'qkvr=1,wo=4' -> {'qkvr': 1, 'wo': 4} (kinds qkvr, wo, head; values in [lo, hi])."""
    out = {}
    for item in (v.strip() for v in raw.split(",") if v.strip()):
        kind, sep, val = item.partition("=")
        kind = kind.strip().lower()
        if not sep or kind not in ("qkvr", "wo", "head"):
            raise ValueError(f"{name} entry {item!r}: expected qkvr|wo|head=<value>")
        try:
            n = int(val)
        except ValueError as exc:
            raise ValueError(f"{name} entry {item!r}: {val!r} is not an integer") from exc
        if not lo <= n <= hi:
            raise ValueError(f"{name} entry {item!r}: not in [{lo}, {hi}]")
        out[kind] = n
    return out


def parse_splitk(raw):
    return parse_kinds(raw, "INKLING_WIDE_SPLITK", 1, 16)


def config_from_env(env=None):
    env = os.environ if env is None else env
    min_m = _int_env(env, "INKLING_WIDE_MIN_M", 17, 1, 256)
    max_m = _int_env(env, "INKLING_WIDE_MAX_M", 256, min_m, 256)
    max_m_kind = {"qkvr": 96, "wo": 32, "head": 256}
    max_m_kind.update(parse_kinds(env.get("INKLING_WIDE_MAX_M_KIND", ""), "INKLING_WIDE_MAX_M_KIND", 1, 256))
    pdl = _int_env(env, "INKLING_WIDE_PDL", 1, 0, 1)
    mt = _int_env(env, "INKLING_WIDE_MT", 0, 0, 2)
    mg = _int_env(env, "INKLING_WIDE_MG", 2, 0, 2)
    pfd = _int_env(env, "INKLING_WIDE_PFD", 1, 0, 64)
    scope = env.get("INKLING_WIDE_SCOPE", "verify").strip().lower() or "verify"
    if scope not in ("verify", "all"):
        raise ValueError(f"INKLING_WIDE_SCOPE={scope!r}: verify or all")
    fmt = env.get("INKLING_WIDE_FORMAT", "auto").strip().lower() or "auto"
    if fmt not in ("auto", "raw"):
        raise ValueError(f"INKLING_WIDE_FORMAT={fmt!r}: auto or raw")
    splitk = parse_splitk(env.get("INKLING_WIDE_SPLITK", ""))
    cache = env.get("INKLING_WIDE_CACHE", "").strip() or os.path.join(os.path.expanduser("~"), ".cache", "inkling-wide")
    verbose = env.get("INKLING_WIDE_VERBOSE", "").strip().lower() not in ("", "0", "no", "off", "false")
    return {
        "min_m": min_m,
        "max_m": max_m,
        "max_m_kind": max_m_kind,
        "pdl": bool(pdl),
        "mt": mt,
        "mg": mg,
        "pfd": pfd,
        "scope": scope,
        "fmt": fmt,
        "splitk": splitk,
        "cache_dir": cache,
        "verbose": verbose,
    }


def parse_levers(value):
    items = [v.strip().lower() for v in value.split(",") if v.strip()]
    if any(v in ("0", "off", "none", "false", "no") for v in items):
        if len(items) > 1:
            raise RuntimeError(f"inkling_wide: INKLING_WIDE={value!r} mixes 'off' with levers")
        return ()
    out = []
    for v in items:
        if v == "all":
            out.extend(LEVERS)
        elif v in LEVERS:
            out.append(v)
        else:
            raise RuntimeError(f"inkling_wide: unknown INKLING_WIDE entry {v!r}; known: {', '.join(LEVERS)}, all")
    return tuple(dict.fromkeys(out))


def describe(levers, cfg):
    return (
        f"levers={','.join(levers)} min_m={cfg['min_m']} max_m={cfg['max_m']} fmt={cfg['fmt']} "
        f"splitk={cfg['splitk'] or 'auto'} pdl={int(cfg['pdl'])} cache={cfg['cache_dir']}"
    )


def _sglang_root():
    spec = importlib.util.find_spec("sglang")
    if spec is None or not spec.submodule_search_locations:
        raise RuntimeError("inkling_wide: the sglang package is not importable")
    return list(spec.submodule_search_locations)[0]


def check_sources(levers, root=None):
    root = root or _sglang_root()
    for lever in ("common", *levers):
        for rel, needles in _SOURCE_CHECKS[lever]:
            path = os.path.join(root, rel)
            try:
                with open(path, encoding="utf-8") as f:
                    text = f.read()
            except OSError as exc:
                raise RuntimeError(f"inkling_wide: {lever} patch target {path} is missing") from exc
            missing = [n for n in needles if n not in text]
            if missing:
                raise RuntimeError(f"inkling_wide: {lever} patch target {path} lacks {missing}")


UNQUANT = "sglang.srt.layers.quantization.unquant"
ATTN = "sglang.srt.models.inkling_common.inkling2_attn"
MODEL = "sglang.srt.models.inkling2"


def hook_plan(levers):
    out = [(UNQUANT, "patch_unquant")]
    if "dense" in levers:
        out.append((ATTN, "patch_attn"))
    out.append((MODEL, "patch_model"))
    return out


def _configure():
    from . import ops, patches

    if patches.STATE["cfg"] is None:
        cfg = STATE["cfg"]
        ops.SET.min_m, ops.SET.max_m = cfg["min_m"], cfg["max_m"]
        ops.SET.max_m_kind = dict(cfg["max_m_kind"])
        ops.SET.pdl, ops.SET.fmt, ops.SET.splitk = cfg["pdl"], cfg["fmt"], dict(cfg["splitk"])
        ops.SET.mt, ops.SET.mg, ops.SET.pfd = cfg["mt"], cfg["mg"], cfg["pfd"]
        ops.SET.scope = cfg["scope"]
        ops.LOG = _log
        patches.STATE.update(levers=STATE["levers"], cfg=cfg, log=_log)
    return patches


def _callback(name):
    def run(module):
        try:
            patches = _configure()
        except BaseException as exc:  # noqa: BLE001
            _log(f"FATAL importing the patches: {exc!r}")
            os._exit(78)
        patches._guarded(getattr(patches, name))(module)

    run.__name__ = f"inkling_wide.{name}"
    return run


def install(levers=None, env=None, sglang_root=None):
    """Arm the levers named in INKLING_WIDE (or `levers`). Idempotent; returns the armed levers."""
    if STATE["installed"]:
        return STATE["levers"]
    env = os.environ if env is None else env
    levers = parse_levers(env.get("INKLING_WIDE", "") if levers is None else ",".join(levers))
    if not levers:
        STATE["installed"] = True
        return ()
    try:
        cfg = config_from_env(env)
    except ValueError as exc:
        raise RuntimeError(f"inkling_wide: {exc}") from exc
    check_sources(levers, sglang_root)

    from . import _hook

    STATE.update(levers=levers, cfg=cfg)
    for module, callback in hook_plan(levers):
        _hook.when_imported(module, _callback(callback))
    STATE.update(installed=True, levers=levers, cfg=cfg)
    if cfg["verbose"]:
        _log("armed " + describe(levers, cfg))
    return levers
