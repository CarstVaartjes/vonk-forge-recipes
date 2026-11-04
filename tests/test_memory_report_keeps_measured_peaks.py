"""The catalog report applies the keep rule a move was checked by: its pre-move content."""

import runpy
import subprocess
from pathlib import Path

import pytest

from qualification import documents, memory_need

ROOT = Path(__file__).resolve().parents[1]
SLUG = "llama-3-1-405b-instruct-awq-int4-nvidia-vllm-four"
UNDERSTATES = "peak understates vLLM budget"


def git(root: Path, *args: str) -> None:
    subprocess.run(
        ["git", "-C", str(root), "-c", "user.name=t", "-c", "user.email=t@t", *args],
        check=True,
        capture_output=True,
    )


def write_recipe(root: Path, peak_bytes: int | None = None) -> None:
    """The real recipe, its peaks lowered below the envelope when given (a move)."""
    recipe = documents.recipe(SLUG)
    if peak_bytes is not None:
        for role in recipe.topology.roles:
            role.resources.memory.peak_bytes = peak_bytes
    (root / "recipes" / f"{SLUG}.json").write_text(recipe.model_dump_json())


def catalog(tmp_path: Path, *, history: bool) -> Path:
    (tmp_path / "recipes").mkdir()
    (tmp_path / "models").symlink_to(ROOT / "models", target_is_directory=True)
    if history:
        git(tmp_path, "init", "-q", "-b", "main")
        write_recipe(tmp_path)
        git(tmp_path, "add", "-A")
        git(tmp_path, "commit", "-q", "-m", "measured content")
    write_recipe(tmp_path, peak_bytes=100_000_000_000)
    if history:
        git(tmp_path, "commit", "-q", "-am", "same-budget move")
    return tmp_path


def report(root: Path, monkeypatch: pytest.MonkeyPatch, proven: bool) -> str:
    monkeypatch.setattr(memory_need, "hardware_proven", lambda *_: proven)
    main = runpy.run_path(str(ROOT / "tools/check-recipe-memory-need"))["main"]
    monkeypatch.setitem(main.__globals__, "ROOT", root)
    monkeypatch.delenv("GITHUB_STEP_SUMMARY", raising=False)
    body = root / "report.md"
    monkeypatch.setattr(
        "sys.argv", ["check-recipe-memory-need", "--report", str(body), SLUG]
    )
    assert main() == 0
    return body.read_text()


def test_same_budget_move_of_measured_content_is_not_listed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert UNDERSTATES not in report(
        catalog(tmp_path, history=True), monkeypatch, proven=True
    )


def test_same_budget_move_of_unmeasured_content_is_listed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert UNDERSTATES in report(
        catalog(tmp_path, history=True), monkeypatch, proven=False
    )


def test_without_history_the_move_is_listed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert UNDERSTATES in report(
        catalog(tmp_path, history=False), monkeypatch, proven=True
    )
