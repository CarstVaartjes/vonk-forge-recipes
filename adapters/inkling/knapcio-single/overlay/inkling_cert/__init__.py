"""Inkling-2-Mini on DGX Spark: certified greedy argmax of the target LM head (exact; see README.md).

    INKLING_CERT_HEAD=1 | check | 0      (default: unset -> install() does nothing)

1      greedy target-verify steps (the in-graph folded greedy accept, no logprobs / grammar / penalties / logit bias /
       watermark / sampling masks) read a 6-bit copy of the head (358 MB) plus pico's exact head GEMV on the
       candidate tiles instead of pico's whole coded head (715 MB); every other step runs the head unchanged.
check  serve pico's full head; build the certified result beside it and count token mismatches (must be 0).

INKLING_CERT_WIDE=1 with INKLING_CERT_HEAD=1 also certifies 17–192 rows
when the packed, unsplit inkling_wide head would serve the same call.

Needs INKLING_PICO12 with `lmhead` armed first (pico's GEMV is the exact pass and the fallback). Other settings:
  INKLING_CERT_HEAD_LOG_EVERY   counters line every N verify steps (default 2000, 0 = never)
  INKLING_CERT_HEAD_CACHE       extension build cache (default ~/.cache/inkling-cert)
"""

import importlib.util
import os
import sys

STATE = {"installed": False, "mode": "0", "cfg": None}

_SOURCE_CHECKS = (
    ("srt/layers/quantization/unquant.py", ("def sm12x_fp32_lm_head(", "def sm12x_fp32_lm_head_applies(")),
    ("srt/layers/logits_processor.py", ("sm12x_fp32_lm_head_applies(hidden_states, lm_head.weight, out)",
                                        "logits = self._compute_lm_head(")),
    ("srt/models/inkling2.py", ("class Inkling2LogitsProcessor(", "def _get_logits(", "fused_softcap(logits, self.shard_softcap)",
                                "logits[..., self.config.vocab_size :] = float(\"-inf\")", "def post_load_weights(")),
    ("srt/speculative/dspark_components/dspark_verify.py",
     ("class DsparkVerifyEpilogue:", "def begin_static_step(self, bs: int, armed: bool)",
      "def begin_step(self, verify_lens, armed: bool)", "def run_non_compact(",
      "target_predict = sampling_argmax(logits).view(-1)")),
    ("srt/speculative/dspark_components/dspark_worker_v2.py",
     ("self._verify_epilogue.begin_static_step(bs, fold_eligible)", "and (sampling_info is None or sampling_info.is_all_greedy)",
      "and verify_logits_adjustments_are_noop(sampling_info)", "compute_spec_logprobs(")),
)

_WIDE_SOURCE_CHECKS = (
    ("kernels/ops/activation/softcap.py",
     ("x = x / softcapping_value", "x = libdevice.tanh(x)", "x = x * softcapping_value")),
    ("kernels/ops/speculative/dspark/dspark_accept.py",
     ("target_predict: Optional[torch.Tensor] = None", "if target_predict is None:")),
    ("srt/speculative/dspark_components/dspark_verify.py",
     ("AcceptGreedy.execute(", "target_logits=logits", "logits=out.next_token_logits")),
    ("srt/model_executor/runner/decode_cuda_graph_runner.py",
     ("class DecodeCudaGraphRunner(", "def execute(", "output = self.backend.replay(")),
)

UNQUANT = "sglang.srt.layers.quantization.unquant"
MODEL = "sglang.srt.models.inkling2"
VERIFY = "sglang.srt.speculative.dspark_components.dspark_verify"
GRAPH_RUNNER = "sglang.srt.model_executor.runner.decode_cuda_graph_runner"


def _log(msg):
    sys.stderr.write(f"[inkling_cert pid={os.getpid()}] {msg}\n")
    sys.stderr.flush()


def parse_mode(value):
    v = (value or "").strip().lower()
    if v in ("", "0", "off", "none", "false", "no"):
        return "0"
    if v in ("1", "on", "true", "yes"):
        return "1"
    if v == "check":
        return "check"
    raise RuntimeError(f"inkling_cert: INKLING_CERT_HEAD={value!r}: 1, check or 0")


def config_from_env(env):
    raw = env.get("INKLING_CERT_HEAD_LOG_EVERY", "").strip()
    try:
        every = int(raw) if raw else 2000
    except ValueError as exc:
        raise RuntimeError(f"inkling_cert: INKLING_CERT_HEAD_LOG_EVERY={raw!r}") from exc
    cache = env.get("INKLING_CERT_HEAD_CACHE", "").strip() or os.path.join(os.path.expanduser("~"), ".cache",
                                                                          "inkling-cert")
    wide = parse_mode(env.get("INKLING_CERT_WIDE", ""))
    if wide == "check":
        raise RuntimeError("inkling_cert: INKLING_CERT_WIDE accepts 0 or 1")
    return {"log_every": max(0, every), "cache_dir": cache, "wide": wide == "1"}


def check_sources(root=None, wide=False):
    if root is None:
        spec = importlib.util.find_spec("sglang")
        if spec is None or not spec.submodule_search_locations:
            raise RuntimeError("inkling_cert: the sglang package is not importable")
        root = list(spec.submodule_search_locations)[0]
    checks = _SOURCE_CHECKS + (_WIDE_SOURCE_CHECKS if wide else ())
    for rel, needles in checks:
        path = os.path.join(root, rel)
        try:
            with open(path, encoding="utf-8") as f:
                text = f.read()
        except OSError as exc:
            raise RuntimeError(f"inkling_cert: patch target {path} is missing") from exc
        missing = [n for n in needles if n not in text]
        if missing:
            raise RuntimeError(f"inkling_cert: patch target {path} lacks {missing}")


def _callback(name):
    def run(module):
        try:
            from . import patches

            if patches.STATE["cfg"] is None:
                patches.STATE.update(mode=STATE["mode"], cfg=STATE["cfg"], log=_log)
        except BaseException as exc:  # noqa: BLE001
            _log(f"FATAL importing the patches: {exc!r}")
            os._exit(78)
        patches._guarded(getattr(patches, name))(module)

    run.__name__ = f"inkling_cert.{name}"
    return run


def install(env=None, sglang_root=None):
    """Arm INKLING_CERT_HEAD. Idempotent; returns the mode ("0" = nothing armed)."""
    if STATE["installed"]:
        return STATE["mode"]
    env = os.environ if env is None else env
    mode = parse_mode(env.get("INKLING_CERT_HEAD", ""))
    STATE["installed"] = True
    STATE["mode"] = mode
    if mode == "0":
        return mode
    pico = sys.modules.get("inkling_pico12")
    if pico is None or "lmhead" not in tuple(pico.STATE.get("levers", ())):
        raise RuntimeError("inkling_cert: INKLING_CERT_HEAD needs inkling_pico12 installed first with `lmhead`")
    STATE["cfg"] = config_from_env(env)
    if STATE["cfg"]["wide"]:
        wide = sys.modules.get("inkling_wide")
        if mode != "1" or wide is None or "lmhead" not in tuple(wide.STATE.get("levers", ())):
            raise RuntimeError("inkling_cert: INKLING_CERT_WIDE needs INKLING_CERT_HEAD=1 and INKLING_WIDE with lmhead")
    check_sources(sglang_root, wide=STATE["cfg"]["wide"])
    from . import _hook

    for module, cb in ((UNQUANT, "patch_unquant"), (MODEL, "patch_model"), (VERIFY, "patch_verify")):
        _hook.when_imported(module, _callback(cb))
    if STATE["cfg"]["wide"]:
        _hook.when_imported(GRAPH_RUNNER, _callback("patch_graph_runner"))
    return mode
