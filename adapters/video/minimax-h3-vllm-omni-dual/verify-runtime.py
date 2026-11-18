"""Verify the MiniMax H3 vLLM-Omni image at build time (no GPU): imports, plugin, wrapper, pins."""

from __future__ import annotations

import importlib
import importlib.metadata
import importlib.util
import os
import sys
from pathlib import Path

PINS = {"vllm": "0.30.0", "nvidia-cutlass-dsl": "4.7.1"}
MODULES = (
    "vllm",
    "vllm_omni",
    "h3_tp2.executor",
    "h3_tp2.agent",
    "h3_tp2.wire",
    "h3_tp2.plugin",
    "av",
)
SPECS = ("cutlass",)  # the CuTe DSL initialises CUDA on import: found, not imported
WRAPPER = Path("/opt/vonk/bin/vllm-omni-job")


def main() -> None:
    for name, version in PINS.items():
        found = importlib.metadata.version(name)
        if found != version:
            raise SystemExit(f"{name} {found} is installed, the creator's stack needs {version}")
    for module in MODULES:
        importlib.import_module(module)
    for spec in SPECS:
        if importlib.util.find_spec(spec) is None:
            raise SystemExit(f"{spec} is not installed")
    plugins = {ep.name: ep.value for ep in importlib.metadata.entry_points(group="vllm.general_plugins")}
    if plugins.get("h3_tp2") != "h3_tp2.plugin:register":
        raise SystemExit(f"the h3_tp2 plugin entry point is missing: {sorted(plugins)}")
    import av

    for codec in ("h264", "aac"):  # the MP4 mux is PyAV's: no ffmpeg binary is needed
        av.codec.Codec(codec, "w")
    for path in (WRAPPER, Path(sys.prefix) / "bin" / "vllm"):
        if not path.is_file() or not os.access(path, os.X_OK):
            raise SystemExit(f"required launcher is missing or not executable: {path}")


if __name__ == "__main__":
    main()
