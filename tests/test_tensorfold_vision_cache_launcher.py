"""The EXL3 launcher must convert into writable cache before starting the engine."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

LAUNCHER = (
    Path(__file__).resolve().parents[1]
    / "adapters/qwen/tournier-exl3-4-05-tensorfold-single/tensorfold-serve"
)


@pytest.mark.parametrize("conversion_fails", [False, True])
def test_conversion_precedes_engine_and_failure_stops_launch(
    tmp_path: Path, conversion_fails: bool
) -> None:
    package = tmp_path / "tensorfold/vision"
    package.mkdir(parents=True)
    (package.parent / "__init__.py").write_text("")
    (package / "__init__.py").write_text("")
    (package / "exl3_convert.py").write_text(
        "import json, os, sys\nfrom pathlib import Path\n"
        "Path(os.environ['CONVERSION']).write_text(json.dumps(sys.argv[1:]))\n"
        "sys.exit(int(os.environ['FAIL_CONVERSION']))\n"
    )
    binaries = tmp_path / "bin"
    binaries.mkdir()
    engine = binaries / "tensorfold"
    engine.write_text(
        f"#!{sys.executable}\nimport os\nfrom pathlib import Path\n"
        "assert Path(os.environ['CONVERSION']).exists()\n"
        "Path(os.environ['ENGINE_STARTED']).touch()\n"
    )
    engine.chmod(0o755)
    converted = tmp_path / "cache/vision/model.safetensors"
    receipt = tmp_path / "conversion.json"
    started = tmp_path / "started"
    env = dict(
        os.environ,
        PATH=f"{binaries}:{os.environ['PATH']}",
        PYTHONPATH=str(tmp_path),
        TENSORFOLD_VISION_WEIGHTS=str(converted),
        CONVERSION=str(receipt),
        ENGINE_STARTED=str(started),
        FAIL_CONVERSION=str(int(conversion_fails)),
    )
    for name in ("HOME", "TORCH_EXTENSIONS_DIR", "TRITON_CACHE_DIR", "TMPDIR"):
        env[name] = str(tmp_path / "cache" / name)
    result = subprocess.run(
        [sys.executable, str(LAUNCHER), "--vision"],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert json.loads(receipt.read_text()) == [
        "/models/vision_k6.safetensors",
        str(converted),
    ]
    assert (result.returncode == 0) is not conversion_fails
    assert started.exists() is not conversion_fails
