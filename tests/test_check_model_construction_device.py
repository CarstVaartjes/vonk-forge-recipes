"""New adapter code may not build models eagerly on CPU or pin them to CUDA."""

from __future__ import annotations

import importlib.machinery
import importlib.util
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "tools/check-model-construction-device"
LOADER = importlib.machinery.SourceFileLoader("check_model_construction", str(SCRIPT))
SPEC = importlib.util.spec_from_loader(LOADER.name, LOADER)
assert SPEC is not None
guard = importlib.util.module_from_spec(SPEC)
sys.modules[LOADER.name] = guard
LOADER.exec_module(guard)

EXISTING = "pipe = load(low_cpu_mem_usage=False)\n"
CPU_BUILD = 'with init_weights_on_device(device=torch.device("cpu")):\n'


def _git(root: Path, *args: str) -> None:
    subprocess.run(
        ["git", "-C", str(root), "-c", "user.name=t", "-c", "user.email=t@t", *args],
        check=True,
        capture_output=True,
    )


def _repo(tmp_path: Path) -> Path:
    adapter = tmp_path / "adapters/video/demo"
    (adapter / "vendor").mkdir(parents=True)
    (adapter / "run.py").write_text(EXISTING)
    _git(tmp_path, "init", "-q", "-b", "main")
    _git(tmp_path, "add", "-A")
    _git(tmp_path, "commit", "-q", "-m", "base")
    return tmp_path


def _change(root: Path, files: dict[str, str]) -> list[str]:
    for name, text in files.items():
        (root / "adapters/video/demo" / name).write_text(text)
    _git(root, "add", "-A")
    _git(root, "commit", "-q", "-m", "change")
    return [finding.message() for finding in guard.violations(root, "HEAD~1")]


def test_existing_lines_pass_and_new_eager_construction_fails(tmp_path) -> None:
    root = _repo(tmp_path)
    problems = _change(
        root,
        {
            "run.py": EXISTING + "x = 1\n",
            "load.py": "m = Model.from_pretrained(p, low_cpu_mem_usage=False)\n"
            + CPU_BUILD
            + "    model = Model()\n",
        },
    )
    assert len(problems) == 2
    assert problems[0].startswith("adapters/video/demo/load.py:1: ")
    assert problems[1].startswith("adapters/video/demo/load.py:2: ")


def test_cuda_pinned_only_inside_a_model_constructor(tmp_path) -> None:
    root = _repo(tmp_path)
    problems = _change(
        root,
        {
            "model.py": (
                "class Encoder(nn.Module):\n"
                "    def __init__(self):\n"
                "        self.attn = nn.MultiheadAttention(8, 2, device='cuda')\n"
                "class Runner:\n"
                "    def __init__(self):\n"
                '        self.flag = torch.zeros((1,), device="cuda")\n'
                "def run(model):\n"
                '    return model.to(device="cuda")\n'
            )
        },
    )
    assert len(problems) == 1
    assert problems[0].startswith("adapters/video/demo/model.py:3: ")


def test_patch_scripts_are_judged_by_the_text_they_insert(tmp_path) -> None:
    root = _repo(tmp_path)
    fix = (
        "replace_once(\n"
        "    manager,\n"
        f"    {CPU_BUILD.strip()!r},\n"
        "    'with init_weights_on_device(device=torch.device(\"meta\")):',\n"
        ")\n"
        "replace_once(dit, \"rotary=rotary,\\n    device='cuda',\\n\", 'rotary=rotary,\\n')\n"
    )
    assert _change(root, {"patch-upstream.py": fix}) == []

    pin = (
        "replace_once(dit, 'rotary=rotary,\\n', \"rotary=rotary, device='cuda',\\n\")\n"
    )
    problems = _change(root, {"patch-upstream.py": fix + pin})
    assert len(problems) == 1
    assert problems[0].startswith("adapters/video/demo/patch-upstream.py:7: ")


def test_vendored_sources_are_not_checked(tmp_path) -> None:
    root = _repo(tmp_path)
    assert _change(root, {"vendor/upstream.py": CPU_BUILD}) == []


def test_main_reports_failure(tmp_path, capsys) -> None:
    root = _repo(tmp_path)
    _change(root, {"load.py": CPU_BUILD + "    model = Model()\n"})
    assert guard.main(["--root", str(root), "--base", "HEAD~1"]) == 1
    assert "adapters/video/demo/load.py:1" in capsys.readouterr().err
