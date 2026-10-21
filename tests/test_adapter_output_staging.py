from __future__ import annotations

import ast
import subprocess
import types
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest

ROOT = Path(__file__).resolve().parents[1]
REPAIRED_ADAPTERS = ("video/mova-pytorch/run.py",)


@pytest.fixture(params=REPAIRED_ADAPTERS)
def runner(request: pytest.FixtureRequest) -> types.ModuleType:
    path = ROOT / "adapters" / request.param
    module = types.ModuleType("output_staging_runner")
    exec(compile(path.read_bytes(), str(path), "exec"), module.__dict__)  # noqa: S102
    return module


@pytest.fixture
def outputs(tmp_path: Path) -> Path:
    root = tmp_path / "outputs"
    for relative in ("tmp/vonk-mova-abandoned/partial.mp4", "cache/model.bin"):
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"retained")
    return root


def test_output_preparation_is_fresh_and_preserves_existing_entries(
    runner: types.ModuleType, outputs: Path
) -> None:
    retained = {
        path: path.read_bytes() for path in outputs.rglob("*") if path.is_file()
    }
    with runner._prepare_output_directory(outputs) as first:
        first_path = Path(first)
        assert first_path.parent == outputs / "tmp"
        assert list(first_path.iterdir()) == []
        (first_path / "partial.mp4").write_bytes(b"incomplete")
        with runner._prepare_output_directory(outputs) as second:
            assert Path(second) != first_path
            assert list(Path(second).iterdir()) == []
    assert not first_path.exists()
    assert not Path(second).exists()
    assert {
        path: path.read_bytes() for path in outputs.rglob("*") if path.is_file()
    } == retained


@pytest.mark.parametrize("failure", [None, "inference", "validation", "publication"])
def test_mova_publishes_only_validated_output_and_can_retry(
    runner: types.ModuleType, outputs: Path, failure: str | None
) -> None:
    final = outputs / "mova.mp4"
    final.write_bytes(b"previous complete artifact")
    args = SimpleNamespace(
        output_dir=outputs, seed=0, num_frames=1, num_inference_steps=2
    )
    staged: list[Path] = []
    replace = runner.os.replace

    def infer(command: list[str], *_args: object) -> None:
        path = Path(command[command.index("--output_path") + 1])
        assert path.parent.parent == outputs / "tmp"
        staged.append(path)
        path.write_bytes(b"new complete artifact")
        assert final.read_bytes() == b"previous complete artifact"
        if failure == "inference":
            raise subprocess.TimeoutExpired(command, 10)

    def verify(path: Path, **_kwargs: object) -> None:
        assert path == staged[-1]
        assert final.read_bytes() == b"previous complete artifact"
        if failure == "validation":
            raise ValueError("invalid media")

    def publish(source: Path, target: Path) -> None:
        assert source == staged[-1]
        assert target == final
        if failure == "publication":
            raise OSError("injected rename failure")
        replace(source, target)

    with (
        mock.patch.dict(runner.os.environ, {"VONK_JOB_TIMEOUT_SECONDS": "10"}),
        mock.patch.object(runner, "_arguments", return_value=args),
        mock.patch.object(runner, "_input_manifest", return_value=None),
        mock.patch.object(runner, "_prompt", return_value="A lake"),
        mock.patch.object(runner, "_variant", return_value=(352, 640)),
        mock.patch.object(runner, "_OFFICIAL_RUNNER", final),
        mock.patch.object(
            runner, "_reference_frame", return_value=outputs / "input.png"
        ),
        mock.patch.object(runner, "_generate", side_effect=infer),
        mock.patch.object(runner, "_verify_synchronized_mp4", side_effect=verify),
        mock.patch.object(runner.os, "replace", side_effect=publish),
    ):
        if failure is not None:
            with pytest.raises((subprocess.TimeoutExpired, ValueError, OSError)):
                runner.main()
            assert final.read_bytes() == b"previous complete artifact"
            assert not staged[-1].parent.exists()
            failure = None
        runner.main()

    assert final.read_bytes() == b"new complete artifact"
    assert len(set(staged)) == len(staged)
    assert all(not path.parent.exists() for path in staged)
    assert (outputs / "tmp/vonk-mova-abandoned/partial.mp4").read_bytes() == b"retained"
    assert (outputs / "cache/model.bin").read_bytes() == b"retained"
    assert {path.name for path in outputs.iterdir()} == {"tmp", "cache", "mova.mp4"}


def test_job_adapters_do_not_require_an_empty_output_root() -> None:
    violations: list[str] = []
    for path in sorted((ROOT / "adapters").rglob("*.py")):
        source = path.read_text(encoding="utf-8")
        if "--output-dir" not in source:
            continue
        for branch in ast.walk(ast.parse(source)):
            if not isinstance(branch, ast.If):
                continue
            for node in ast.walk(branch.test):
                if not isinstance(node, ast.Call) or not isinstance(
                    node.func, ast.Attribute
                ):
                    continue
                if node.func.attr not in {"iterdir", "listdir", "scandir"}:
                    continue
                # Input inventories are valid; conditioning a job on output-root
                # contents conflicts with the platform's tmp/cache ownership.
                directory = ast.unparse(node.func.value)
                if "output" in directory.lower() or any(
                    "output" in ast.unparse(argument).lower() for argument in node.args
                ):
                    violations.append(f"{path.relative_to(ROOT)}:{node.lineno}")
    assert not violations, "job adapters check output-root contents: " + ", ".join(
        violations
    )
