"""Run the real wrapper shell with local stand-ins for the GPU-only programs."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from test_vllm_memory_fit import loads_draft_model
from vonk_forge_contracts import RecipeDefinition

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    "slug,draft",
    [
        ("deepseek-v4-flash-0731-sparkinfer-single", True),
        ("deepseek-v4-flash-0731-mia-sparkinfer-single", True),
        ("deepseek-v4-flash-0731-sparkinfer-target-only-canary-single", False),
    ],
)
def test_wrapper_forwards_authored_memory_budget(
    tmp_path: Path, slug: str, draft: bool
) -> None:
    recipe = RecipeDefinition.model_validate_json(
        (ROOT / "recipes" / f"{slug}.json").read_text()
    )
    assert recipe.settings.kind == "generation"
    assert recipe.settings.context_tokens is not None
    assert recipe.settings.concurrency is not None
    settings = {
        "context_tokens": recipe.settings.context_tokens.value,
        "concurrency": recipe.settings.concurrency.value,
    }
    arguments: list[str] = []
    for item in recipe.runtime.arguments:
        value = settings[item.setting] if item.setting is not None else item.value
        if value is False:
            continue
        arguments.append(f"--{item.name}")
        if value is not True:
            arguments.append(str(value))
    arguments.extend(["--host", "0.0.0.0", "--port", "8000"])
    expected = arguments[arguments.index("--gpu-memory-utilization") + 1]

    model = tmp_path / "model"
    model.mkdir()
    for name in (
        "config.json",
        "quantization_config.json",
        "model.safetensors.index.json",
        "EXL3_MANIFEST.json",
        "REAP_K216_PLAN.json",
    ):
        (model / name).write_text("{}")
    commands = tmp_path / "bin"
    commands.mkdir()
    uname = commands / "uname"
    uname.write_text("#!/bin/sh\necho aarch64\n")
    uname.chmod(0o755)
    python = commands / "runtime-python"
    python.write_text("#!/bin/sh\nexit 0\n")
    python.chmod(0o755)
    engine = commands / "engine"
    engine.write_text(
        f"#!{sys.executable}\n"
        "import json, os, sys\n"
        "print(os.environ['GPU_MEMORY_UTILIZATION'])\n"
        "print(json.dumps(sys.argv[1:]))\n"
        "print('SPEC_MODEL_PATH' in os.environ)\n"
    )
    engine.chmod(0o755)
    # Only relocate absolute runtime paths. All parsing, profile checks,
    # exports and final exec are the actual shipped wrapper's shell code.
    source = (
        ROOT / recipe.execution.build.context.path / "vllm-wrapper.sh"
    ).read_text()
    wrapper = tmp_path / "wrapper.sh"
    wrapper.write_text(
        source.replace("/outputs/", f"{tmp_path}/outputs/")
        .replace("/opt/runtime-venv/bin/python", str(python))
        .replace("/opt/vllm/serve-ds4-flash.sh", str(engine))
    )
    result = subprocess.run(
        ["bash", str(wrapper), "serve", str(model), *arguments],
        env={**os.environ, "PATH": f"{commands}:{os.environ['PATH']}"},
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    memory, forwarded, has_draft = result.stdout.splitlines()
    assert memory == expected
    assert json.loads(forwarded) == arguments
    assert has_draft == str(draft)
    assert loads_draft_model(recipe) == draft
