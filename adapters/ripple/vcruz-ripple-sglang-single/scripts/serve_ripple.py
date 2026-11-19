#!/usr/bin/env python3
"""Serve Ripple-65B-Preview EXL3 (5.41bpw / 2.57bpw / 1.95bpw) + EXL3/BF16 DFlash2
draft on SGLang 0.5.19 with vcruz305/sglang-exl3 (https://github.com/vcruz305/sglang-exl3).

Defaults mirror the measured configuration; every knob is overridable via
environment variables or CLI flags:

    MODEL_DIR          target pack directory (default: ./models/target)
    DRAFT_DIR          DFlash2 draft pack directory (default: ./models/draft-4.55bpw)
    DRAFT_QUANT        `exl3` (default) or `unquant` (BF16 draft)
    CTX_LEN            context length (default 131072)
    MEM_FRAC           --mem-fraction-static (default 0.82)
    PORT               server port (default 30000)
    HOST               bind host (default 127.0.0.1)
    SERVED_MODEL_NAME  served model id (default ripple)
    MAX_REQS           --max-running-requests (default 8)
    SGLANG_EXL3_MGEMM  1 (default) / 0
"""
import os
import shlex
import sys
from pathlib import Path

sys.dont_write_bytecode = True

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
PLUGIN_SRC = ROOT / ".deps" / "sglang-exl3" / "src"
PY_BIN_DIR = Path(sys.executable).resolve().parent

for p in (HERE, PLUGIN_SRC):
    if p.is_dir() and str(p) not in sys.path:
        sys.path.insert(0, str(p))

# Ensure PATH includes the active virtualenv's bin/ (where ninja lives for
# FlashInfer JIT compilation during CUDA-graph capture) and /usr/local/cuda/bin.
path_entries = [str(PY_BIN_DIR)]
for cuda_bin in (
    os.environ.get("CUDA_HOME", ""),
    "/usr/local/cuda",
):
    if cuda_bin and (Path(cuda_bin) / "bin").is_dir():
        path_entries.append(str(Path(cuda_bin) / "bin"))
existing_path = os.environ.get("PATH", "")
if existing_path:
    path_entries.append(existing_path)
os.environ["PATH"] = ":".join(path_entries)

# Ensure child sglang::scheduler processes inherit PYTHONPATH for sglang_exl3 and ripple_format
existing_pp = os.environ.get("PYTHONPATH", "")
pp_entries = [str(p) for p in (PLUGIN_SRC, HERE) if p.is_dir()]
if existing_pp:
    pp_entries.append(existing_pp)
os.environ["PYTHONPATH"] = ":".join(pp_entries)

os.environ.setdefault("SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN", "1")
os.environ.setdefault("SGLANG_EXL3_DENSE", "1")
os.environ.setdefault("SGLANG_EXL3_MGEMM", "1")

from sglang_exl3 import register  # noqa: E402

register()

from sglang.srt.models.registry import ModelRegistry  # noqa: E402
from sglang_exl3.models.ricursive import RicursiveForCausalLM  # noqa: E402
import ripple_format  # noqa: E402

ModelRegistry.models["RicursiveForCausalLM"] = RicursiveForCausalLM
ripple_format.register()


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default)


def _resolve_draft_dir(raw_dir: str) -> str:
    if not raw_dir:
        return ""
    p = Path(raw_dir)
    if (p / "config.json").is_file():
        return str(p)
    for sub in ("4.55bpw", "3.60bpw", "3.08bpw"):
        if (p / sub / "config.json").is_file():
            return str(p / sub)
    return str(p)


DEFAULT_ARGS = shlex.split(
    """
    --served-model-name {served_name} --trust-remote-code
    --dtype bfloat16 --kv-cache-dtype bfloat16 --tp-size 1
    --attention-backend triton --page-size 64
    --speculative-algorithm DFLASH --speculative-num-draft-tokens 16
    --speculative-draft-model-quantization {draft_quant}
    --speculative-draft-window-size 16384
    --speculative-draft-attention-backend triton
    --mem-fraction-static {mem_frac}
    --cuda-graph-backend-prefill disabled
    --max-running-requests {max_reqs}
    --tool-call-parser ripple --reasoning-parser ripple
    """.format(
        served_name=_env("SERVED_MODEL_NAME", "ripple"),
        draft_quant=_env("DRAFT_QUANT", "exl3"),
        mem_frac=_env("MEM_FRAC", "0.82"),
        max_reqs=_env("MAX_REQS", "8"),
    )
)

DEFAULT_ARGS += [
    "--model-path",
    _env("MODEL_DIR", str(ROOT / "models" / "target")),
    "--host",
    _env("HOST", "127.0.0.1"),
    "--port",
    _env("PORT", "30000"),
    "--context-length",
    _env("CTX_LEN", "131072"),
]

draft_dir = _resolve_draft_dir(_env("DRAFT_DIR", str(ROOT / "models" / "draft-4.55bpw")))
if draft_dir and "--speculative-draft-model-path" not in sys.argv:
    DEFAULT_ARGS += [
        "--speculative-draft-model-path",
        draft_dir,
    ]

if "--chat-template" not in sys.argv:
    DEFAULT_ARGS += [
        "--chat-template",
        str(HERE / "chat_template.jinja"),
    ]


def main():
    from sglang.srt.entrypoints.http_server import launch_server
    from sglang.srt.server_args import prepare_server_args
    from sglang.srt.utils import kill_process_tree

    args = prepare_server_args(DEFAULT_ARGS + sys.argv[1:])
    try:
        launch_server(args)
    finally:
        kill_process_tree(os.getpid(), include_parent=False)


if __name__ == "__main__":
    main()
