from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    "adapter", ["sparkinfer-single", "sparkinfer-target-only-single"]
)
def test_creator_adapters_select_venv_python_via_vllm_python(adapter: str) -> None:
    """Follow the creator: VLLM_PYTHON makes serve-ds4-flash.sh run `python -m`.

    Rewriting /usr/local/bin/vllm without a __main__ guard broke vLLM's spawn
    start method (engine child re-imported the script), so it must stay absent.
    """
    root = ROOT / "adapters/deepseek" / adapter
    assert "/usr/local/bin/vllm" not in (root / "Dockerfile").read_text()
    wrapper = (root / "vllm-wrapper.sh").read_text()
    assert "\nexport VLLM_PYTHON=/opt/runtime-venv/bin/python\n" in wrapper


@pytest.mark.parametrize("adapter", ["mia-sparkinfer-single"])
def test_sparkinfer_console_uses_native_extension_interpreter(
    tmp_path: Path, adapter: str
) -> None:
    """Bypassing the image ENTRYPOINT must still use its Torch ABI environment."""
    dockerfile = (ROOT / "adapters/deepseek" / adapter / "Dockerfile").read_text()
    # Execute the shipped build instruction against a local console script,
    # then invoke that script as the upstream serve-ds4-flash.sh does.
    instruction = next(
        line.removeprefix("RUN ").removesuffix(" \\")
        for line in dockerfile.splitlines()
        if line.startswith("RUN printf '#!/opt/runtime-venv/bin/python")
    )
    interpreter = tmp_path / "runtime-python"
    interpreter.symlink_to(sys.executable)
    package = tmp_path / "vllm/entrypoints/cli"
    package.mkdir(parents=True)
    (package / "main.py").write_text(
        "import json, sys\ndef main():\n    print(json.dumps(sys.argv))\n"
    )
    console = tmp_path / "vllm-console"
    instruction = instruction.replace("/opt/runtime-venv/bin/python", str(interpreter))
    instruction = instruction.replace("/usr/local/bin/vllm", str(console))
    subprocess.run(["sh", "-c", instruction], check=True, timeout=10)
    console.chmod(0o755)
    args = ["serve", "/models", "--gpu-memory-utilization", "0.85"]
    result = subprocess.run(
        [str(console), *args],
        env={**os.environ, "PYTHONPATH": str(tmp_path)},
        capture_output=True,
        text=True,
        check=True,
        timeout=10,
    )
    assert json.loads(result.stdout) == [str(console), *args]
    assert "from vllm.entrypoints.cli.main import main" in console.read_text()


@pytest.mark.parametrize(
    (
        "relative_wrapper",
        "served_model_name",
        "model_len",
        "seqs",
        "batch",
        "graph",
        "utilization",
    ),
    [
        (
            "adapters/deepseek/sparkinfer-single/vllm-wrapper.sh",
            "deepseek-v4-flash-0731-spark",
            "262144",
            "4",
            "8224",
            "6",
            "0.9465",
        ),
        (
            "adapters/deepseek/sparkinfer-target-only-single/vllm-wrapper.sh",
            "deepseek-v4-flash-0731-spark",
            "262144",
            "4",
            "8192",
            "4",
            "0.95",
        ),
        (
            "adapters/deepseek/mia-sparkinfer-single/vllm-wrapper.sh",
            "deepseek-v4-flash-0731",
            "384000",
            "1",
            "8224",
            "24",
            "0.94",
        ),
    ],
)
def test_sparkinfer_wrappers_preserve_authored_engine_argv(
    tmp_path: Path,
    relative_wrapper: str,
    served_model_name: str,
    model_len: str,
    seqs: str,
    batch: str,
    graph: str,
    utilization: str,
) -> None:
    """Unknown, repeated, empty, and structured values reach the launcher unchanged."""

    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    (fake_bin / "uname").write_text(
        "#!/bin/sh\nprintf '%s\\n' aarch64\n", encoding="utf-8"
    )
    (fake_bin / "uname").chmod(0o755)

    fake_python = tmp_path / "runtime-python"
    fake_python.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    fake_python.chmod(0o755)

    captured_launcher = tmp_path / "capture-launcher.sh"
    captured_launcher.write_text(
        "#!/bin/sh\nprintf '%s\\0' \"$@\"\n",
        encoding="utf-8",
    )
    captured_launcher.chmod(0o755)

    model_source = tmp_path / "models" / "weights"
    model_source.mkdir(parents=True)
    for name in (
        "config.json",
        "quantization_config.json",
        "model.safetensors.index.json",
        "EXL3_MANIFEST.json",
        "REAP_K216_PLAN.json",
    ):
        (model_source / name).write_text("{}\n", encoding="utf-8")

    wrapper = (ROOT / relative_wrapper).read_text(encoding="utf-8")
    wrapper = wrapper.replace("/opt/runtime-venv/bin/python", str(fake_python))
    wrapper = wrapper.replace("/opt/recipe/scripts", str(tmp_path / "scripts"))
    wrapper = wrapper.replace("/outputs", str(tmp_path / "outputs"))
    wrapper = wrapper.replace("/opt/vllm/serve-ds4-flash.sh", str(captured_launcher))
    runnable_wrapper = tmp_path / "vllm-wrapper.sh"
    runnable_wrapper.write_text(wrapper, encoding="utf-8")
    runnable_wrapper.chmod(0o755)

    structured = json.dumps(
        {"unicode": "Δ", "punctuation": ";$HOME && {json}", "payload": "x" * 5000},
        ensure_ascii=False,
        separators=(",", ":"),
    )
    authored_args = [
        "--served-model-name",
        served_model_name,
        "--host",
        "0.0.0.0",
        "--port",
        "8000",
        "--max-model-len",
        model_len,
        "--max-num-seqs",
        seqs,
        "--max-num-batched-tokens",
        batch,
        "--max-cudagraph-capture-size",
        graph,
        "--gpu-memory-utilization",
        utilization,
        "--kv-cache-dtype",
        "nvfp4_ds_mla",
        "--enable-prefix-caching",
        "--unfamiliar-option",
        "",
        "--unfamiliar-option",
        "punctuation; $HOME/Δ",
        "--structured-option",
        structured,
    ]
    environment = dict(os.environ)
    environment["PATH"] = f"{fake_bin}{os.pathsep}{environment['PATH']}"

    result = subprocess.run(
        [str(runnable_wrapper), "serve", str(model_source), *authored_args],
        env=environment,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr.decode("utf-8", errors="replace")
    captured = result.stdout.split(b"\0")[:-1]
    assert [token.decode("utf-8") for token in captured] == authored_args
