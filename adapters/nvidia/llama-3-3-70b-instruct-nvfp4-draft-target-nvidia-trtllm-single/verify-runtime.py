"""Verify the Vonk packaging of the NVIDIA TensorRT-LLM image at build time."""

from __future__ import annotations

import argparse
import os
import shutil

import yaml
from pathlib import Path

WRAPPER = Path("/opt/vonk/bin/trtllm-serve")
CONFIG = Path("/opt/vonk/trtllm/llm-api-config.yml")


def main(*, engine_schema: bool = False) -> None:
    if not os.access(WRAPPER, os.X_OK):
        raise SystemExit("Vonk trtllm-serve wrapper is not executable")
    if shutil.which("trtllm-serve") is None:
        raise SystemExit("trtllm-serve is not on PATH in the base image")
    if not CONFIG.is_file() or not CONFIG.read_text(encoding="utf-8").strip():
        raise SystemExit("the playbook LLM API config is missing or empty")

    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    if not isinstance(config, dict) or not isinstance(config.get("speculative_config"), dict):
        raise ValueError("the playbook config must contain a speculative_config mapping")
    if engine_schema:
        # NVIDIA imports its CUDA bindings with the schema. Run this unchanged
        # engine validation on the GPU host, before starting the server.
        from pydantic import TypeAdapter
        from tensorrt_llm.llmapi.llm_args import SpeculativeConfig

        TypeAdapter(SpeculativeConfig).validate_python(config["speculative_config"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine-schema", action="store_true")
    main(engine_schema=parser.parse_args().engine_schema)
