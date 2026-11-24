"""Inkling-2-Mini on DGX Spark (GB10, sm_121a): small-M BF16 GEMVs on pico's exactly coded 12-bit weights.

    INKLING_PICO12=dense,lmhead        (default: empty -> install() does nothing)

dense      Inkling2Attention.qkvr (plain and stacked q/k/v calls) and .wo_ud at M <= 16 rows
lmhead     the target's FP32-logit LM head (unquant.sm12x_fp32_lm_head) at M <= 16 rows
drafthead  the DSpark draft's BF16 projection through the target head (inkling2_mtp.project_through_lm_head)
draft      the draft's own BF16 linears (draft model .layers)
all        all four
stack1     (with dense; not part of `all`) stacked q/k/v calls of more than 16 rows read the one coded qkvr once: the
           q rows over the columns outside the k/v rows and the k/v rows over those only, by a range launch of
           pico's kernel (a second small extension); no kv pack (~175 MB less), no copy; the same bits

Other settings (env):
  INKLING_PICO12_MAX_M          rows of a plain call pico takes (1..16, default 16)
  INKLING_PICO12_STACKED_MAX_M  rows per source of a stacked qkvr call pico takes (0..16, default 16; 0 = leave
                                stacked calls to the stock / qkvr_gemv path; <= 5 needs no kv pack)
  INKLING_PICO12_STACK1_KV_MT   n-tiles per warp of the stack1 k/v range call (1 or 2, default 1; same bits)
  INKLING_PICO12_STACK1_FUSE    0 (default): stack1 runs q and k/v as two launches; 1: one launch at <= 8 rows a
                                source (same bits; GB10 2026-11-22: within +-2 us of two launches)
  INKLING_PICO12_WK             warps along K per CTA (1..16, default 16 = the probe's setting)
  INKLING_PICO12_PDL            1 (default): launch with programmatic dependent launch, as the probe; 0: plain
  INKLING_PICO12_CACHE          extension build cache (default ~/.cache/inkling-pico12)
  INKLING_PICO12_STD            nvcc -std (default c++20, pico's; c++17 is the fallback)
  INKLING_PICO12_VERBOSE        1: print the armed line in every process and the nvcc build output

install() parses the env, checks that every patch target exists in the installed sglang sources (without
importing sglang or torch), and attaches each patch to its module through a post-import hook. The extension is
built (or reused from the cache) and the weights are coded when the model's weights finish loading, in the
process that loads them; that process then logs one line with what was patched and the memory used. Any
problem raises RuntimeError here, or ends the process with exit code 78 later (a missing patch target, a failed
build, a weight pico cannot code exactly).
"""

import importlib.util
import os
import sys

LEVERS = ("dense", "lmhead", "drafthead", "draft")
OPT_LEVERS = ("stack1",)  # named explicitly only (not part of `all`)
STATE = {"installed": False, "levers": (), "cfg": None}

# Files (relative to the sglang package) and strings each lever relies on.
_SOURCE_CHECKS = {
    "common": (
        (
            "srt/layers/quantization/unquant.py",
            ("class UnquantizedLinearMethod(", "def apply_into(", "is_batch_invariant_mode_enabled"),
        ),
    ),
    "dense": (
        (
            "srt/models/inkling_common/inkling2_attn.py",
            (
                "class Inkling2Attention(",
                "def _project_sources(",
                "def _stacked(",
                "self.qkvr = MergedColumnParallelLinear(",
                "self.wo_ud = RowParallelLinear(",
                "self.split_sizes = [",
            ),
        ),
        ("srt/models/inkling2.py", ("class Inkling2ForConditionalGeneration(", "def post_load_weights(")),
    ),
    "lmhead": (
        ("srt/layers/quantization/unquant.py", ("def sm12x_fp32_lm_head(", "def sm12x_fp32_lm_head_applies(")),
        (
            "srt/layers/logits_processor.py",
            ("sm12x_fp32_lm_head_applies(hidden_states, lm_head.weight, out)",),
        ),
        ("srt/models/inkling2.py", ("class Inkling2ForConditionalGeneration(", "def post_load_weights(")),
    ),
    "drafthead": (
        (
            "srt/models/inkling2_mtp.py",
            ("class Inkling2ForConditionalGenerationMTP(", "project_through_lm_head(hidden, self.lm_head)"),
        ),
        ("srt/models/dspark.py", ("def project_through_lm_head(",)),
        ("srt/models/inkling2.py", ("class Inkling2ForConditionalGeneration(", "def post_load_weights(")),
    ),
    "draft": (("srt/models/inkling2_mtp.py", ("class Inkling2ForConditionalGenerationMTP(", "def load_weights(")),),
}


def _log(msg):
    sys.stderr.write(f"[inkling_pico12 pid={os.getpid()}] {msg}\n")
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


def config_from_env(env=None):
    env = os.environ if env is None else env
    max_m = _int_env(env, "INKLING_PICO12_MAX_M", 16, 1, 16)
    stacked = _int_env(env, "INKLING_PICO12_STACKED_MAX_M", 16, 0, 16)
    wk = _int_env(env, "INKLING_PICO12_WK", 16, 1, 16)
    pdl = _int_env(env, "INKLING_PICO12_PDL", 1, 0, 1)
    kv_mt = _int_env(env, "INKLING_PICO12_STACK1_KV_MT", 1, 1, 2)
    fuse = _int_env(env, "INKLING_PICO12_STACK1_FUSE", 0, 0, 1)
    std = env.get("INKLING_PICO12_STD", "c++20").strip() or "c++20"
    if std not in ("c++17", "c++20"):
        raise ValueError(f"INKLING_PICO12_STD={std!r}: c++17 or c++20")
    cache = env.get("INKLING_PICO12_CACHE", "").strip() or os.path.join(
        os.path.expanduser("~"), ".cache", "inkling-pico12"
    )
    verbose = env.get("INKLING_PICO12_VERBOSE", "").strip().lower() not in ("", "0", "no", "off", "false")
    return {
        "max_m": max_m,
        "stacked_max_m": stacked,
        "wk": wk,
        "pdl": bool(pdl),
        "stack1_kv_mt": kv_mt,
        "stack1_fuse": bool(fuse),
        "std": std,
        "cache_dir": cache,
        "verbose": verbose,
    }


def parse_levers(value):
    items = [v.strip().lower() for v in value.split(",") if v.strip()]
    if any(v in ("0", "off", "none", "false", "no") for v in items):
        if len(items) > 1:
            raise RuntimeError(f"inkling_pico12: INKLING_PICO12={value!r} mixes 'off' with levers")
        return ()
    out = []
    for v in items:
        if v == "all":
            out.extend(LEVERS)
        elif v in LEVERS or v in OPT_LEVERS:
            out.append(v)
        else:
            raise RuntimeError(
                f"inkling_pico12: unknown INKLING_PICO12 entry {v!r}; known: {', '.join(LEVERS + OPT_LEVERS)}, all"
            )
    if "stack1" in out and "dense" not in out:
        raise RuntimeError("inkling_pico12: stack1 needs dense (it changes how dense's stacked qkvr calls read)")
    return tuple(dict.fromkeys(out))


def describe(levers, cfg):
    return (
        f"levers={','.join(levers)} max_m={cfg['max_m']} stacked_max_m={cfg['stacked_max_m']} wk={cfg['wk']} "
        f"pdl={int(cfg['pdl'])} std={cfg['std']} cache={cfg['cache_dir']}"
    )


def _sglang_root():
    spec = importlib.util.find_spec("sglang")  # does not execute sglang
    if spec is None or not spec.submodule_search_locations:
        raise RuntimeError("inkling_pico12: the sglang package is not importable")
    return list(spec.submodule_search_locations)[0]


def check_sources(levers, root=None):
    root = root or _sglang_root()
    for lever in ("common", *(v for v in levers if v not in OPT_LEVERS)):
        for rel, needles in _SOURCE_CHECKS[lever]:
            path = os.path.join(root, rel)
            try:
                with open(path, encoding="utf-8") as f:
                    text = f.read()
            except OSError as exc:
                raise RuntimeError(f"inkling_pico12: {lever} patch target {path} is missing") from exc
            missing = [n for n in needles if n not in text]
            if missing:
                raise RuntimeError(f"inkling_pico12: {lever} patch target {path} lacks {missing}")


UNQUANT = "sglang.srt.layers.quantization.unquant"
ATTN = "sglang.srt.models.inkling_common.inkling2_attn"
MODEL = "sglang.srt.models.inkling2"
MTP = "sglang.srt.models.inkling2_mtp"


def hook_plan(levers):
    """(module, patches.<callback>) pairs the levers need, in registration order."""
    out = [(UNQUANT, "patch_unquant")]
    if "dense" in levers:
        out.append((ATTN, "patch_attn"))
    if {"dense", "lmhead", "drafthead"} & set(levers):
        out.append((MODEL, "patch_model"))
    if {"draft", "drafthead"} & set(levers):
        out.append((MTP, "patch_mtp"))
    return out


def _configure():
    """Import the torch-side modules and hand them the parsed settings (once)."""
    from . import ops, patches

    if patches.STATE["cfg"] is None:
        cfg = STATE["cfg"]
        ops.SET.max_m, ops.SET.stacked_max_m = cfg["max_m"], cfg["stacked_max_m"]
        ops.SET.wk, ops.SET.pdl = cfg["wk"], cfg["pdl"]
        ops.SET.stack1 = "stack1" in STATE["levers"]
        ops.SET.stack1_kv_mt = cfg.get("stack1_kv_mt", 1)
        ops.SET.stack1_fuse = cfg.get("stack1_fuse", False)
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

    run.__name__ = f"inkling_pico12.{name}"
    return run


def install(levers=None, env=None, sglang_root=None):
    """Arm the levers named in INKLING_PICO12 (or `levers`). Idempotent; returns the armed levers."""
    if STATE["installed"]:
        return STATE["levers"]
    env = os.environ if env is None else env
    levers = parse_levers(env.get("INKLING_PICO12", "") if levers is None else ",".join(levers))
    if not levers:
        STATE["installed"] = True
        return ()
    try:
        cfg = config_from_env(env)
    except ValueError as exc:
        raise RuntimeError(f"inkling_pico12: {exc}") from exc
    check_sources(levers, sglang_root)

    from . import _hook  # no torch / sglang import here: the callbacks import them when their module loads

    STATE.update(levers=levers, cfg=cfg)
    for module, callback in hook_plan(levers):
        _hook.when_imported(module, _callback(callback))
    STATE.update(installed=True, levers=levers, cfg=cfg)
    if cfg["verbose"]:
        _log("armed " + describe(levers, cfg))
    return levers
