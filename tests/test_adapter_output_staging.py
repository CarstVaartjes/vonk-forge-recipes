from __future__ import annotations

import ast
import contextlib
import shutil
import subprocess
import sys
import tempfile
import types
from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest

ROOT = Path(__file__).resolve().parents[1]
REPAIRED_ADAPTERS = ("video/mova-pytorch/run.py",)
# Entries the platform's output collector leaves to the runtime: its scratch
# (TMPDIR=/outputs/tmp) and the cache mount. Everything else must be a file.
RUNTIME_OWNED = frozenset({"tmp", "cache"})
# Positional index of ``dir`` in each tempfile constructor.
TEMPFILE_DIR_ARGUMENT = {
    "TemporaryDirectory": 2,
    "mkdtemp": 2,
    "mkstemp": 2,
    "NamedTemporaryFile": 6,
    "TemporaryFile": 6,
    "SpooledTemporaryFile": 7,
}


def _load(relative: str, name: str) -> types.ModuleType:
    path = ROOT / "adapters" / relative
    module = types.ModuleType(name)
    exec(compile(path.read_bytes(), str(path), "exec"), module.__dict__)  # noqa: S102
    return module


@pytest.fixture(params=REPAIRED_ADAPTERS)
def runner(request: pytest.FixtureRequest) -> types.ModuleType:
    return _load(request.param, "output_staging_runner")


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


def _job_entrypoints() -> Iterator[tuple[Path, ast.Module]]:
    for path in sorted((ROOT / "adapters").rglob("*.py")):
        relative = path.relative_to(ROOT / "adapters")
        if {"vendor", "tests"} & set(relative.parts):
            continue  # upstream copies and creator test suites are not job code
        source = path.read_text(encoding="utf-8")
        if "--output-dir" in source:
            yield path, ast.parse(source)


def _mentions_output(node: ast.expr) -> bool:
    return "output" in ast.unparse(node).lower()


def _output_root_child(node: ast.expr) -> bool:
    """``<output dir> / "<name>"``: a fixed entry directly in the output root."""
    return (
        isinstance(node, ast.BinOp)
        and isinstance(node.op, ast.Div)
        and _mentions_output(node.left)
        and isinstance(node.right, ast.Constant)
        and node.right.value not in RUNTIME_OWNED
    )


def test_job_adapters_keep_scratch_out_of_the_output_root() -> None:
    """A SIGKILL skips every cleanup; scratch left in the output root is not a
    plain output file and fails the signed collection of the next attempt.
    Create scratch with tempfile defaults: the platform sets TMPDIR to its
    collector-exempt ``/outputs/tmp``."""
    violations: list[str] = []
    for path, tree in _job_entrypoints():
        assigned: dict[str, list[ast.expr]] = {}
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        assigned.setdefault(target.id, []).append(node.value)
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            name = func.attr if isinstance(func, ast.Attribute) else ""
            if isinstance(func, ast.Name):
                name = func.id
            if name in TEMPFILE_DIR_ARGUMENT:
                position = TEMPFILE_DIR_ARGUMENT[name]
                directories = [k.value for k in node.keywords if k.arg == "dir"]
                directories += node.args[position : position + 1]
                if any(
                    _mentions_output(directory)
                    and not any(
                        isinstance(part, ast.Constant) and part.value == "tmp"
                        for part in ast.walk(directory)
                    )
                    for directory in directories
                ):
                    violations.append(f"{path.relative_to(ROOT)}:{node.lineno}")
            elif name in {"mkdir", "makedirs"} and isinstance(func, ast.Attribute):
                # Path.mkdir() on the target, or os.mkdir/os.makedirs(target).
                os_call = isinstance(func.value, ast.Name) and func.value.id == "os"
                receiver = node.args[0] if os_call and node.args else func.value
                candidates = [receiver]
                if isinstance(receiver, ast.Name):
                    candidates += assigned.get(receiver.id, [])
                if any(_output_root_child(candidate) for candidate in candidates):
                    violations.append(f"{path.relative_to(ROOT)}:{node.lineno}")
    assert not violations, "job adapters create scratch in the output root: " + (
        ", ".join(violations)
    )


class _Killed(BaseException):
    """The job process is killed (SIGKILL): no ``finally`` or ``__exit__`` runs."""


@contextlib.contextmanager
def _killed_run(outputs: Path) -> Iterator[None]:
    """Run with the platform's TMPDIR and with every cleanup path skipped."""

    @contextlib.contextmanager
    def uncleaned(
        suffix: str | None = None,
        prefix: str | None = None,
        dir: str | Path | None = None,
        **_options: object,
    ) -> Iterator[str]:
        yield tempfile.mkdtemp(suffix, prefix, dir)

    (outputs / "tmp").mkdir(parents=True, exist_ok=True)
    with (
        mock.patch.object(tempfile, "tempdir", str(outputs / "tmp")),
        mock.patch.object(tempfile, "TemporaryDirectory", uncleaned),
        mock.patch.object(shutil, "rmtree"),
        pytest.raises(_Killed),
    ):
        yield


def _exported(outputs: Path) -> dict[str, bool]:
    """What the output collector inspects: name -> is a plain file."""
    return {
        path.name: path.is_file() and not path.is_symlink()
        for path in outputs.iterdir()
        if not (path.name in RUNTIME_OWNED and path.is_dir())
    }


@pytest.mark.parametrize(
    "relative",
    ["video/wan-dancer-native/run.py", "video/wan-dancer-diffsynth-disk/run.py"],
)
def test_killed_wan_dancer_leaves_no_entry_for_the_collector(
    relative: str, tmp_path: Path
) -> None:
    runner = _load(relative, "wan_dancer_killed_runner")
    outputs = tmp_path / "outputs"
    source = tmp_path / "source"
    for global_style, local_style in getattr(runner, "STYLE_PROMPTS", {}).values():
        for name in (global_style, local_style):
            prompt = source / "gen_video/prompt" / name
            prompt.parent.mkdir(parents=True, exist_ok=True)
            prompt.write_text("style", encoding="utf-8")
    args = SimpleNamespace(
        output_mime="video/mp4", seed=0, timeout_seconds=600, output_dir=outputs
    )
    request = {"reference_image": "a.png", "music": "a.wav", "prompt": "Dance"}
    with (
        mock.patch.object(runner, "_arguments", return_value=args),
        mock.patch.object(runner, "REQUIRED_MODEL_FILES", set()),
        mock.patch.object(runner, "SOURCE_ROOT", source, create=True),
        mock.patch.object(runner, "_manifest", return_value=None),
        mock.patch.object(runner, "_request", return_value=request),
        mock.patch.object(runner, "_input_file", return_value=tmp_path / "input"),
        mock.patch.object(runner, "_generation_dimensions", return_value=(512, 512)),
        mock.patch.object(runner, "_probe_duration", return_value=1.0),
        mock.patch.object(runner, "_expected_output", return_value=(512, 512, 24, 0.8)),
        mock.patch.object(runner, "_run", side_effect=_Killed),
        _killed_run(outputs),
    ):
        runner.main()
    assert _exported(outputs) == {}


@pytest.mark.parametrize("killed", [True, False])
def test_foley_stages_in_runtime_scratch_and_survives_a_kill(
    killed: bool, tmp_path: Path
) -> None:
    runner = _load(
        "audio/hunyuan-video-foley-native/pipelines/run.py", "foley_staging_runner"
    )
    outputs = tmp_path / "outputs"
    models = tmp_path / "models"
    for name in (
        "target/hunyuanvideo_foley_xl.pth",
        "target/synchformer_state_dict.pth",
        "target/vae_128d_48k.pth",
        "siglip2/config.json",
        "clap/config.json",
    ):
        (models / name).parent.mkdir(parents=True, exist_ok=True)
        (models / name).write_bytes(b"model")
    upstream = tmp_path / "infer.py"
    upstream.write_text("", encoding="utf-8")

    def rooted(value: str) -> Path:
        path = Path(value)
        return (
            models / path.relative_to("/models")
            if path.is_relative_to("/models")
            else path
        )

    def infer(command: list[str], **_options: object) -> None:
        if killed:
            raise _Killed
        staging = Path(command[command.index("--output_dir") + 1])
        assert staging.parent == outputs / "tmp"
        (staging / "clip_generated.wav").write_bytes(b"wav")

    argv = [
        "run.py",
        "--entrypoint",
        "/opt/vonk/source/pipelines/hunyuan_foley_xl.py",
        "--output-mime",
        "audio/wav",
        "--output-dir",
        str(outputs),
    ]
    with contextlib.ExitStack() as stack:
        if killed:
            stack.enter_context(_killed_run(outputs))
        else:
            (outputs / "tmp").mkdir(parents=True)
            stack.enter_context(
                mock.patch.object(tempfile, "tempdir", str(outputs / "tmp"))
            )
        stack.enter_context(mock.patch.object(sys, "argv", argv))
        stack.enter_context(mock.patch.object(runner, "Path", rooted))
        stack.enter_context(mock.patch.object(runner, "_SOURCE", upstream))
        stack.enter_context(
            mock.patch.object(runner, "_one_video", return_value=tmp_path / "a.mp4")
        )
        stack.enter_context(
            mock.patch.object(runner, "_one_prompt", return_value="rain")
        )
        stack.enter_context(mock.patch.object(runner, "_validate_wav"))
        stack.enter_context(
            mock.patch.object(runner.subprocess, "run", side_effect=infer)
        )
        runner.main()
    assert _exported(outputs) == ({} if killed else {"output.wav": True})
    if not killed:
        assert list((outputs / "tmp").iterdir()) == []
