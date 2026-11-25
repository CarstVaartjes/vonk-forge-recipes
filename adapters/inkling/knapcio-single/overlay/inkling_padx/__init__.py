"""inkling_padx: CUDA-graph padding rows get no experts (exact for live rows).

    INKLING_PADX=1      on (unset / 0: nothing is imported)

A decode / target-verify graph replays at its captured batch size; with fewer live requests the tail rows are
padding. The graph runner zeroes their positions / cache locations / request indices but keeps the previous
replay's input_ids (PaddingPolicy FOREACH_COPY), so every padded row still routes to 8 experts. Measured on GB10
(X1P-s05, v2.8.1 stack, sampled thinking): c9 replays the c10 graph with 8 padded rows and its step is 125.3 ms
against 116.1 ms at c10 (one more real request); the difference is the NVFP4 / MXFP8 grouped GEMMs (+8.1 / +1.9 ms).

Here every padded row's 8 expert ids become the out-of-range id E (= 128) and route_metadata runs over E + 1 global
experts with the E real ones local, exactly as inkling_noexp does for dead verify rows: the SM120 kernels never
touch rows past offsets[E] and the combine gives those rows 0. Live rows are quantized, multiplied and combined by
the same per-row arithmetic (inkling_noexp's GPU check, tools/gpu_check_noexp.py, is the same argument). Padded rows'
outputs are never read.

Mechanism: DecodeCudaGraphRunner._set_dp_real_num_tokens(raw_num_token) runs on every replay (both load_batch
paths); it now also fills a per-runner device int32 [1] with the live row count (a fill kernel, stream-ordered, no
host copy). Inkling2Experts._sm120_route, while that runner is capturing, reads it in-graph: row r is live <=> r <
real. Only graphs that can be replayed with padding are masked (can_pad: bs - 1 is not a captured size, decided at
capture); the others keep the stock route, so the c1..c8 speculative verify graphs carry no extra nodes (the mask's
4 small kernels x 48 MoE visits cost 0.36 ms per c1 step when applied everywhere). Composes with inkling_noexp (dead verify rows under the cap): a row is routed only if live for both. Eager
calls and other runners' captures keep stock routing.
"""

import importlib.util
import os
import sys

from . import _hook

MODEL_MODULE = "sglang.srt.models.inkling_common.inkling2_moe"
RUNNER_MODULE = "sglang.srt.model_executor.runner.decode_cuda_graph_runner"
MARK = "_inkling_padx"
PHANTOM = 128
BIG = 1 << 30
STATE = {"installed": False, "applied": 0, "runner": None, "calls": 0, "passthrough": 0, "fills": 0}
_NEEDLES_MODEL = ("def _sm120_route(self, ids, weights):", "src2dst, dst2src, gammas, _, offsets, *_ = route_metadata(",
                  "if flat.numel() <= ROUTE_MAX_IDS:")
_NEEDLES_RUNNER = ("def _set_dp_real_num_tokens(self, raw_num_token: int) -> None:",
                   "self._set_dp_real_num_tokens(self.raw_num_token)", "self._set_dp_real_num_tokens(raw_num_token)",
                   "def capture(self) -> None:")


class PadxFatal(BaseException):
    """BaseException: SGLang's model registry swallows Exception during model imports."""


def _log(msg):
    print(f"[inkling_padx] {msg}", file=sys.stderr, flush=True)


_OFF = ("", "0", "off", "false", "no", "none")


def enabled(env=None):
    env = os.environ if env is None else env
    on = env.get("INKLING_PADX", "").strip().lower()
    if on in _OFF:
        return False
    if on not in ("1", "on", "true", "yes"):
        raise RuntimeError(f"inkling_padx: INKLING_PADX={on!r} (use 1 or 0)")
    return True


def check_sources(root=None):
    if root is None:
        spec = importlib.util.find_spec("sglang")
        if spec is None or not spec.submodule_search_locations:
            raise RuntimeError("inkling_padx: the sglang package is not importable")
        root = list(spec.submodule_search_locations)[0]
    for rel, needles in (("srt/models/inkling_common/inkling2_moe.py", _NEEDLES_MODEL),
                         ("srt/model_executor/runner/decode_cuda_graph_runner.py", _NEEDLES_RUNNER)):
        path = os.path.join(root, rel)
        try:
            text = open(path, encoding="utf-8").read()
        except OSError as exc:
            raise RuntimeError(f"inkling_padx: patch target {path} is missing") from exc
        missing = [n for n in needles if n not in text]
        if missing:
            raise RuntimeError(f"inkling_padx: patch target {path} lacks {missing}")


def install(env=None, root=None):
    if STATE["installed"]:
        return bool(STATE["applied"])
    STATE["installed"] = True
    if not enabled(env):
        return False
    check_sources(root)
    from . import patches

    _hook.when_imported(MODEL_MODULE, patches.patch_model)
    _hook.when_imported(RUNNER_MODULE, patches.patch_runner)
    _log("enabled: CUDA-graph padding rows route to the phantom expert (live rows exact)")
    return True
