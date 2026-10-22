"""The rc9 CLI rejects the telemetry switch accepted by newer TRT-LLM images."""

import os
import subprocess
from pathlib import Path

import pytest
from vonk_forge_contracts import RecipeDefinition

ROOT = Path(__file__).resolve().parents[1]
RC9_IMAGE = "3f293455e9450298f548c5c3268186f80703dfc490926f398aec9a746422e840"
RC9_RECIPES = [
    RecipeDefinition.model_validate_json(path.read_text())
    for path in sorted((ROOT / "recipes").glob("*.json"))
    if RC9_IMAGE in path.read_text()
]


@pytest.mark.parametrize("recipe", RC9_RECIPES, ids=lambda r: r.identity.slug)
def test_rc9_launch_uses_environment_for_telemetry(
    recipe: RecipeDefinition, tmp_path: Path
) -> None:
    """Replay rc9's option refusal through the shipped launcher and recipe argv."""
    assert recipe.settings.kind == "generation"
    assert recipe.settings.context_tokens is not None
    assert recipe.settings.concurrency is not None
    settings = {
        "context_tokens": recipe.settings.context_tokens.value,
        "concurrency": recipe.settings.concurrency.value,
    }
    arguments = list(recipe.runtime.entrypoint[1:])
    for argument in recipe.runtime.arguments:
        value = settings[argument.setting] if argument.setting else argument.value
        if value is False:
            continue
        arguments.append(f"--{argument.name}")
        if value is not True:
            arguments.append(str(value))
    # Inject the observed rc9 parser boundary without importing GPU libraries.
    engine = tmp_path / "trtllm-serve"
    engine.write_text(
        "#!/bin/sh\nfor arg do\n"
        '  if [ "$arg" = --no-telemetry ]; then\n'
        '    echo "Error: No such option: --no-telemetry" >&2; exit 2\n'
        "  fi\ndone\n"
        'printf "%s\\n" "$TRTLLM_NO_USAGE_STATS"\n'
        'printf "%s\\n" "$@"\n'
    )
    engine.chmod(0o755)
    result = subprocess.run(
        [
            "/bin/sh",
            str(ROOT / recipe.execution.build.context.path / "trtllm-serve"),
            *arguments,
        ],
        env={
            **os.environ,
            **{item.name: str(item.value) for item in recipe.runtime.environment},
            "PATH": str(tmp_path),
        },
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines() == ["1", *arguments]
