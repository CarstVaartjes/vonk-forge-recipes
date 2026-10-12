"""Every OpenAI model alias must be a name the engine actually serves.

The platform forwards the alias unchanged, so an engine that serves another
name (vLLM and SGLang default to the model path) answers every request with
a 404 while `/v1/models` still passes readiness.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

RECIPES = sorted((Path(__file__).resolve().parents[1] / "recipes").glob("*.json"))

# vLLM and SGLang take the name from the same runtime argument.
CHECKED_ENGINES = {"vllm", "sglang"}
# The other engines are not checked here: llama.cpp, TensorFold, EXL3 and DS4
# launchers do not route on the request's model name.


def _served_names(runtime: dict) -> set[str]:
    names = {
        argument["value"]
        for argument in runtime["arguments"]
        if argument["name"] == "served-model-name"
    }
    # r0b0tlab's profile-driven launcher compiles the whole server argv,
    # `--served-model-name` included, from the named profile.
    if any(argument["name"] == "profile" for argument in runtime["arguments"]):
        return {"*"}
    return names


@pytest.mark.parametrize("path", RECIPES, ids=lambda path: path.stem)
def test_openai_aliases_are_served_names(path: Path) -> None:
    recipe = json.loads(path.read_text(encoding="utf-8"))
    runtime = recipe["runtime"]
    if runtime["engine"] not in CHECKED_ENGINES:
        pytest.skip("engine does not route on the request's model name")
    aliases = {
        alias
        for interface in recipe["interfaces"]
        if interface["adapter"] == "openai"
        for alias in interface.get("model_aliases", [])
    }
    served = _served_names(runtime)
    if "*" in served:
        # Launcher-owned name: the proven validation request must use the alias.
        bodies = [
            check["request"]["body"]
            for check in recipe["validation"]["serving"]["checks"]
            if check["request"].get("body")
        ]
        assert all(body.get("model") in aliases for body in bodies)
        return
    assert aliases <= served, f"aliases {sorted(aliases - served)} are not served"


# Mirrors vonk-forge control/src/vonk_control/litellm.py `_UPSTREAM_MODEL`:
# the gateway rejects any other upstream model name (a path is not a name).
_GATEWAY_MODEL_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:/+-]{0,119}")


@pytest.mark.parametrize("path", RECIPES, ids=lambda path: path.stem)
def test_openai_aliases_are_gateway_model_names(path: Path) -> None:
    recipe = json.loads(path.read_text(encoding="utf-8"))
    for interface in recipe["interfaces"]:
        if interface["adapter"] != "openai":
            continue
        for alias in interface.get("model_aliases", []):
            assert _GATEWAY_MODEL_NAME.fullmatch(alias), f"{alias!r} is not a model name"
