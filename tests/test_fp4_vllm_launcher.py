"""Keep the Controller's serve/model prefix intact at the thin CLI boundary."""

import json
import os
import subprocess
import sys
from pathlib import Path


def test_fp4_launcher_preserves_canonical_serve_command(tmp_path: Path) -> None:
    package = tmp_path / "vllm/entrypoints/cli"
    package.mkdir(parents=True)
    (package / "main.py").write_text(
        "import json, sys\nprint(json.dumps(sys.argv[1:]))\n"
    )
    command = ["serve", "/models", "--served-model-name", "name with spaces"]
    wrapper = (
        Path(__file__).resolve().parents[1]
        / "adapters/qwen/vcruz-fp4-local-vllm-single/vllm-serve"
    )
    result = subprocess.run(
        [sys.executable, str(wrapper), *command],
        env=dict(os.environ, PYTHONPATH=str(tmp_path)),
        capture_output=True,
        text=True,
        check=True,
    )
    assert json.loads(result.stdout) == command
