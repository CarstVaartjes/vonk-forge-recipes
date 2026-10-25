"""New adapter runtime code may not build models eagerly on CPU or pin them to CUDA."""

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
        path = root / "adapters/video/demo" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    _git(root, "add", "-A")
    _git(root, "commit", "-q", "-m", "change")
    return [finding.message() for finding in guard.violations(root, "HEAD~1")]


def _lines(problems: list[str]) -> list[str]:
    return [problem.split(": ")[0] for problem in problems]


def test_existing_lines_pass_and_new_eager_construction_fails(tmp_path) -> None:
    root = _repo(tmp_path)
    problems = _change(
        root,
        {
            "run.py": EXISTING + "x = 1\n",
            "load.py": "m = Model.from_pretrained(p, low_cpu_mem_usage=False)\n"
            + CPU_BUILD
            + "    model = Model()\n"
            + "with init_weights_on_device('cpu'):\n"
            + "    model = Model()\n"
            + "with init_weights_on_device(device=torch.device('meta')):\n"
            + "    model = Model()\n",
        },
    )
    assert _lines(problems) == [
        "adapters/video/demo/load.py:1",
        "adapters/video/demo/load.py:2",
        "adapters/video/demo/load.py:4",
    ]


def test_cuda_device_only_inside_a_model_constructor(tmp_path) -> None:
    root = _repo(tmp_path)
    problems = _change(
        root,
        {
            "model.py": (
                "class Encoder(nn.Module):\n"
                "    def __init__(self):\n"
                "        self.attn = nn.MultiheadAttention(8, 2, device='cuda')\n"
                "        self.proj = nn.Linear(4, 4, device=torch.device('cuda'))\n"
                '        self.where = torch.device(type="cuda", index=0)\n'
                "        self.cpu = torch.device('cpu')\n"
                "class Runner:\n"
                "    def __init__(self):\n"
                '        self.flag = torch.zeros((1,), device="cuda")\n'
                "def run(model):\n"
                '    return model.to(device="cuda")\n'
            )
        },
    )
    assert _lines(problems) == [
        "adapters/video/demo/model.py:3",
        "adapters/video/demo/model.py:4",
        "adapters/video/demo/model.py:5",
    ]


def test_comment_only_edit_passes(tmp_path) -> None:
    root = _repo(tmp_path)
    edit = "# keep low_cpu_mem_usage=False until TorchAo loads on meta\n" + EXISTING
    assert _change(root, {"run.py": edit}) == []


def test_patch_script_text_is_out_of_scope(tmp_path) -> None:
    # Upstream source a patch inserts is text here; the adapter's own
    # patched-source behaviour test covers it.
    root = _repo(tmp_path)
    pin = "replace_once(dit, 'rotary,', \"rotary, device='cuda',\")\n" + (
        "replace_once(mm, 'meta', 'init_weights_on_device(device=torch.device(\"cpu\"))')\n"
    )
    assert _change(root, {"patch-upstream.py": pin}) == []


def test_vendored_and_verbatim_creator_files_are_not_checked(tmp_path) -> None:
    root = _repo(tmp_path)
    manifest = '{"source": {"repo": "o/r", "commit": "c"}, "ours": ["job.py", "ours/"]}'
    problems = _change(
        root,
        {
            "vendor/upstream.py": CPU_BUILD + "    m()\n",
            "vendored-upstream.json": manifest,
            "creator.py": CPU_BUILD + "    m()\n",
            "job.py": CPU_BUILD + "    m()\n",
            "ours/load.py": CPU_BUILD + "    m()\n",
        },
    )
    assert _lines(problems) == [
        "adapters/video/demo/job.py:1",
        "adapters/video/demo/ours/load.py:1",
    ]


def test_uncommitted_and_untracked_edits_are_checked(tmp_path) -> None:
    # The upstream refresh verifies its edits before it commits them.
    root = _repo(tmp_path)
    (root / "adapters/video/demo/run.py").write_text(EXISTING + CPU_BUILD + "    m()\n")
    (root / "adapters/video/demo/new.py").write_text(
        "x = load(low_cpu_mem_usage=False)\n"
    )
    problems = [finding.message() for finding in guard.violations(root, "HEAD")]
    assert _lines(problems) == [
        "adapters/video/demo/new.py:1",
        "adapters/video/demo/run.py:2",
    ]


def test_main_reports_failure(tmp_path, capsys) -> None:
    root = _repo(tmp_path)
    _change(root, {"load.py": CPU_BUILD + "    model = Model()\n"})
    assert guard.main(["--root", str(root), "--base", "HEAD~1"]) == 1
    assert "adapters/video/demo/load.py:1" in capsys.readouterr().err
