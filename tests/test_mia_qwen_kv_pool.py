"""A fixed maximum KV pool must not consume a smaller Spark startup budget."""

import json
import os
import subprocess
import sys
from pathlib import Path

from vonk_forge_contracts import RecipeDefinition
from vonk_forge_contracts.recipe import RecipeGenerationSettings

ROOT = Path(__file__).resolve().parents[1]


def test_qwen_uses_demand_growth_instead_of_pinning_the_maximum_pool(
    tmp_path: Path,
) -> None:
    recipe = RecipeDefinition.model_validate_json(
        (ROOT / "recipes/qwen3-8-27b-mia-tensorfold-single.json").read_text()
    )
    # Exercise the actual launcher + authored environment at the engine boundary.
    # The upstream pool patch allocates this many GiB before serving any request;
    # a small available pool must still serve short requests at the chosen context.
    assert isinstance(recipe.settings, RecipeGenerationSettings)
    assert recipe.settings.concurrency is not None
    engine = tmp_path / "tensorfold"
    engine.write_text(
        f"#!{sys.executable}\n"
        "import json, os, sys\n"
        "pool = float(os.environ.get('TENSORFOLD_KV_POOL_GIB', '0'))\n"
        "if pool > 16: raise SystemExit('startup pool exceeds available KV memory')\n"
        "print(json.dumps(sys.argv[1:]))\n"
    )
    engine.chmod(0o755)
    env = {**os.environ, "PATH": f"{tmp_path}:{os.environ['PATH']}"}
    env.pop("TENSORFOLD_KV_POOL_GIB", None)
    env.update({item.name: str(item.value) for item in recipe.runtime.environment})
    for name in ("HOME", "TORCH_EXTENSIONS_DIR", "TRITON_CACHE_DIR", "TMPDIR"):
        env[name] = str(tmp_path / name)
    result = subprocess.run(
        [
            sys.executable,
            str(
                ROOT
                / "adapters/qwen/mia-qwen3-8-27b-tensorfold-single/tensorfold-serve"
            ),
            "--context",
            str(recipe.settings.context_tokens.value),
            "--parallel",
            str(recipe.settings.concurrency.value),
        ],
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    assert json.loads(result.stdout)[-4:] == ["--context", "262144", "--parallel", "8"]
