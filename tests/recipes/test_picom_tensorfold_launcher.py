"""The Picom Qwen3.8 TensorFold wrapper makes the two-stage draft head file once and starts without it when that fails."""

import subprocess
from pathlib import Path

from repo_root import ROOT

WRAPPER = ROOT / "adapters/qwen/picom-flash-next-tensorfold-single/tensorfold-serve"
HEAD2 = (
    "TF_FLASHNEXT_Q38_HEAD2",
    "TF_FLASHNEXT_Q38_HEAD2_FAST",
    "TF_FLASHNEXT_Q38_HEAD2_SAMPLED",
)


def run_wrapper(
    tmp_path: Path, python3_body: str
) -> tuple[list[str], dict[str, str], list[str]]:
    """Run the wrapper with a fake python3 (the file's maker) and a fake engine that prints what it was started with."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(exist_ok=True)
    calls = tmp_path / "python3-calls"
    (bin_dir / "python3").write_text(
        f'#!/bin/sh\necho "$*" >> "{calls}"\n{python3_body}\n'
    )
    (bin_dir / "tensorfold-native").write_text(
        '#!/bin/sh\nfor a in "$@"; do echo "A:$a"; done\nenv | sed -n "s/^\\(TF_FLASHNEXT_Q38_HEAD2[A-Z_]*=.*\\)$/E:\\1/p"\n'
    )
    for tool in bin_dir.iterdir():
        tool.chmod(0o755)
    cache = tmp_path / "cache"
    env = {
        "PATH": f"{bin_dir}:/usr/bin:/bin",
        "HOME": str(cache / "home"),
        "CUDA_CACHE_PATH": str(cache / "cuda"),
        "TORCH_EXTENSIONS_DIR": str(cache / "torch_extensions"),
        "TRITON_CACHE_DIR": str(cache / "triton"),
        "TMPDIR": str(cache / "tmp"),
    }
    result = subprocess.run(
        [str(WRAPPER), "--context", "262144"],
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    argv = [line[2:] for line in result.stdout.splitlines() if line.startswith("A:")]
    seen = dict(
        line[2:].split("=", 1)
        for line in result.stdout.splitlines()
        if line.startswith("E:")
    )
    return argv, seen, calls.read_text().splitlines() if calls.exists() else []


def test_first_start_makes_the_head_file_into_the_cache_and_enables_the_lever(
    tmp_path: Path,
) -> None:
    argv, seen, calls = run_wrapper(tmp_path, 'printf head2 > "$6"')
    head2 = tmp_path / "cache/home/.cache/tensorfold/q38/head2_r9.bin"
    assert head2.read_text() == "head2"
    assert argv == ["serve", "/models", "--context", "262144"]
    assert seen == {
        HEAD2[0]: str(head2),
        HEAD2[1]: "3",
        HEAD2[2]: "1",
    }
    assert len(calls) == 1
    assert calls[0].startswith("-I /opt/vonk/head2/mkhead2.py /models ")


def test_a_cached_head_file_is_reused(tmp_path: Path) -> None:
    head2 = tmp_path / "cache/home/.cache/tensorfold/q38/head2_r9.bin"
    head2.parent.mkdir(parents=True)
    head2.write_text("kept")
    _, seen, calls = run_wrapper(tmp_path, "exit 1")
    assert calls == []
    assert seen[HEAD2[0]] == str(head2)
    assert head2.read_text() == "kept"


def test_a_failed_head_file_starts_the_engine_without_the_lever(tmp_path: Path) -> None:
    argv, seen, calls = run_wrapper(tmp_path, 'printf partial > "$6"; exit 1')
    assert argv == ["serve", "/models", "--context", "262144"]
    assert seen == {}
    assert len(calls) == 1
    assert not list((tmp_path / "cache/home/.cache/tensorfold/q38").iterdir())
