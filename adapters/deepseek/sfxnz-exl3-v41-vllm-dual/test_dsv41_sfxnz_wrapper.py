from __future__ import annotations

import importlib.util
import json
from pathlib import Path

ADAPTER = Path(__file__).parent
RECIPE = (
    ADAPTER.parents[2] / "recipes/deepseek-v4-1-flash-exl3-sfxnz-vllm-dual.json"
)


def _wrapper():
    spec = importlib.util.spec_from_file_location(
        "dsv41_wrapper", ADAPTER / "dsv41-vllm-wrapper.py"
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_entrypoint_and_runtime_metadata_are_declared() -> None:
    dockerfile = (ADAPTER / "Dockerfile").read_text()
    assert 'ai.vonkforge.runtime-interface="v1"' in dockerfile
    recipe = json.loads(RECIPE.read_text())
    assert recipe["runtime"]["entrypoint"] == ["/opt/vonk/bin/dsv41-vllm-serve"]


def test_upstream_patch_directory_is_baked_at_the_bind_mount_paths() -> None:
    dockerfile = (ADAPTER / "Dockerfile").read_text()
    assert "COPY docker/patch /opt/dsv41-patch\n" in dockerfile
    assert (
        "COPY docker/patch/sitecustomize.py /usr/lib/python3.12/sitecustomize.py"
        in dockerfile
    )


def test_switches_rewrite_arguments_like_upstream_run_sh(monkeypatch) -> None:
    wrapper = _wrapper()
    base = [
        "--speculative-config",
        "{}",
        "--compilation-config",
        "{}",
        "--mm-encoder-tp-mode",
        "data",
        "--block-size",
        "64",
    ]
    for name in ("DSV41_SPECULATION", "ENFORCE_EAGER", "LANGUAGE_MODEL_ONLY"):
        monkeypatch.delenv(name, raising=False)
    untouched = list(base)
    wrapper.apply_switches(untouched)
    assert untouched == base

    monkeypatch.setenv("DSV41_SPECULATION", "off")
    monkeypatch.setenv("ENFORCE_EAGER", "1")
    monkeypatch.setenv("LANGUAGE_MODEL_ONLY", "1")
    rewritten = list(base)
    wrapper.apply_switches(rewritten)
    assert rewritten == [
        "--block-size",
        "64",
        "--enforce-eager",
        "--language-model-only",
    ]
