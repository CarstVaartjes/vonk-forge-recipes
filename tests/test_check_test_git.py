"""New test code that runs git fails; plain names and old lines pass."""

from __future__ import annotations

import importlib.machinery
import importlib.util
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest
from added_diff import added_diff

ROOT = Path(__file__).resolve().parents[1]
LOADER = importlib.machinery.SourceFileLoader(
    "check_test_git", str(ROOT / "tools/check-test-git")
)
SPEC = importlib.util.spec_from_loader(LOADER.name, LOADER)
assert SPEC is not None
guard = importlib.util.module_from_spec(SPEC)
sys.modules[LOADER.name] = guard
LOADER.exec_module(guard)

HITS = {
    "argv list": 'subprocess.run(["git", "init"], cwd=root)',
    "argv tuple": 'subprocess.check_output(("git", "status"))',
    "starred prefix": 'subprocess.run([*GIT, "log"])',
    "prefix join": 'subprocess.run(GIT + ["diff", base])',
    "argv string head": 'check(["git commit -m x"])',
    "helper name": '_git(root, "init")',
    "attribute helper": 'tool.git("rev-parse", "HEAD")',
    "run helper": 'run_git("status")',
    "shell string": 'subprocess.run("cd x && git add .", shell=True)',
    "bash -c": 'subprocess.run(["bash", "-c", "git rev-parse HEAD"])',
    "os.system": 'os.system("git status")',
    "environment key": 'env = {"GIT_AUTHOR_NAME": "t"}',
}
CLEAN = {
    "file name": 'stub = tmp_path / "git"',
    "patch": 'patch("tool.which", return_value="git")',
    "argv data in a tuple": 'rows = [("push.yml", "git push")]',
    "variable name": 'key = "VONK_GIT_SIGNING_KEY_FILE"',
    "dict of other keys": 'env = {"HOME": "x", "VONK_GIT_SIGNING_KEY_FILE": "y"}',
    "other command": 'subprocess.run(["uv", "run", "pytest"])',
    "word git in a string": 'subprocess.run(["echo", "digit"])',
    "text mentioning git": 'assert "git ls-files" not in text',
    "comment": "# git init",
}


def scan(code: str) -> list[tuple[int, str]]:
    return guard.scan(code + "\n", {1})


@pytest.mark.parametrize("code", HITS.values(), ids=HITS.keys())
def test_added_git_use_fails(code: str) -> None:
    assert [line for line, _ in scan(code)] == [1]


@pytest.mark.parametrize("code", CLEAN.values(), ids=CLEAN.keys())
def test_plain_names_and_data_pass(code: str) -> None:
    assert scan(code) == []


def test_only_added_lines_count() -> None:
    source = 'subprocess.run(["git", "init"])\nx = 1\n'
    assert guard.scan(source, {2}) == []
    assert [line for line, _ in guard.scan(source, {1})] == [1]


def test_check_reads_the_added_lines_of_test_files_only(tmp_path: Path) -> None:
    code = 'subprocess.run(["git", "init"])\n'
    for name in ("tests/test_x.py", "tools/helper.py"):
        (tmp_path / name).parent.mkdir(exist_ok=True)
        (tmp_path / name).write_text(code)
    problems = guard.check(
        tmp_path, added_diff({"tests/test_x.py": code, "tools/helper.py": code})
    )
    assert [problem.split(":")[0] for problem in problems] == ["tests/test_x.py"]
    assert guard.check(tmp_path, "") == []


def test_cli_reads_the_diff_from_stdin_and_exits_one_on_a_hit(tmp_path: Path) -> None:
    code = textwrap.dedent('subprocess.run(["git", "init"])\n')
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests/test_x.py").write_text(code)
    run = subprocess.run(
        [
            sys.executable,
            str(ROOT / "tools/check-test-git"),
            "--diff",
            "-",
            "--root",
            str(tmp_path),
        ],
        input=added_diff({"tests/test_x.py": code}),
        capture_output=True,
        text=True,
        check=False,
    )
    assert run.returncode == 1 and "tests/test_x.py:1" in run.stdout
