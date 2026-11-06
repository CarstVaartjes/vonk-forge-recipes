"""Kimodo adapter behaviour that needs no GPU, weights or the upstream stack.

The job-input check, the offline text-encoder tree, the generator command and its
process-group deadline, the motion validation and the deterministic zip all run
here, and the zip is judged by the sweep's own ``zip-entries`` evaluator against
the qualification definition. The recipe, Model documents and adapter must agree
on every mounted path and runtime argument. The model load and the generation are
proven once on real weights and again by the hardware sweep.
"""

from __future__ import annotations

import hashlib
import importlib.util
import io
import json
import os
import sys
import tempfile
import time
import zipfile
from pathlib import Path, PurePosixPath

import pytest

from qualification import documents
from spark_sweep.definitions import Definitions
from spark_sweep.job_assertions import Output, evaluate

ROOT = Path(__file__).resolve().parents[1]
ADAPTER = ROOT / "adapters/three-d/kimodo"
WHEEL = next(ADAPTER.glob("*.whl"))
SLUG = "kimodo-soma-rp-v1-2-pytorch-single"
FIXTURE = ROOT / "qualification/fixtures/kimodo-motion-prompt.txt"
sys.path.insert(0, str(WHEEL))


def _load_run():
    spec = importlib.util.spec_from_file_location("kimodo_run", ADAPTER / "run.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


run = _load_run()


def _stage_inputs(
    root: Path,
    payload: bytes,
    *,
    slot: str = "prompt",
    media_type: str = "text/plain",
    name: str = "prompt.txt",
) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / name).write_bytes(payload)
    manifest = {
        "schema_version": 1,
        "total_bytes": len(payload),
        "files": [
            {
                "slot": slot,
                "name": name,
                "media_type": media_type,
                "size_bytes": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
        ],
    }
    (root / "manifest.json").write_text(json.dumps(manifest))


def _numpy():
    """numpy is a producer-test dependency, not a type-checked one (as in the Hunyuan tests)."""
    return __import__("numpy")


def _bvh(frames: int) -> str:
    return (
        "HIERARCHY\nROOT Root\n{\n  OFFSET 0 0 0\n}\nMOTION\n"
        f"Frames: {frames}\nFrame Time: 0.033333\n" + "0 " * 6 + "\n"
    )


def _write_motion(stem: Path, *, bvh_frames: int = 60, npz_frames: int = 60) -> None:
    numpy = _numpy()

    stem.with_suffix(".bvh").write_text(_bvh(bvh_frames), encoding="utf-8")
    joints = numpy.random.default_rng(0).normal(size=(npz_frames, 77, 3))
    numpy.savez(stem.with_suffix(".npz"), posed_joints=joints)


# --- recipe, models and adapter agree ----------------------------------------


def test_the_recipe_passes_exactly_the_arguments_the_adapter_parses() -> None:
    recipe = documents.recipe(SLUG)
    arguments = {item.name: item for item in recipe.runtime.arguments}
    assert set(arguments) == {"entrypoint", "output-mime", "seed", "duration-seconds"}
    assert arguments["entrypoint"].value == run.ENTRYPOINT
    assert arguments["output-mime"].value == "application/zip"
    assert arguments["duration-seconds"].setting == "duration-seconds"
    default = recipe.settings.knobs["duration-seconds"].value
    assert isinstance(default, float)
    assert run.MIN_DURATION_SECONDS <= default <= run.MAX_DURATION_SECONDS
    argv = [
        f"--{name}={item.value}"
        for name, item in arguments.items()
        if item.setting is None
    ]
    args = run._arguments(
        [*argv, f"--duration-seconds={default}", "--output-dir=/outputs"]
    )
    assert args.duration_seconds == default
    assert args.seed == arguments["seed"].value


def test_the_qualification_selects_a_duration_the_adapter_accepts() -> None:
    definition = Definitions.load().artifact(f"vonk-forge/{SLUG}")
    assert definition is not None
    parameters = definition["parameters"]
    assert run.MIN_DURATION_SECONDS <= parameters["duration-seconds"] <= 2.0
    assert set(parameters) == set(documents.recipe(SLUG).settings.knobs)


def test_every_file_the_adapter_reads_is_a_mounted_model_file() -> None:
    mounted: set[PurePosixPath] = set()
    for selection in documents.recipe(SLUG).models:
        paths = {
            item.id: item.path for item in documents.model(selection.model.slug).files
        }
        for item in selection.files:
            mounted.add(PurePosixPath(item.mount.target) / paths[item.file_id])
    wanted = {
        PurePosixPath(directory) / name
        for directory, names in run.MODEL_FILES.items()
        for name in names
    }
    assert wanted <= mounted


def test_the_models_declare_the_text_encoder_as_dependencies() -> None:
    models = documents.recipe(SLUG).models
    primary = next(s for s in models if s.id == "primary")
    dependencies = {
        item.slug for item in documents.model(primary.model.slug).dependencies
    }
    assert dependencies == {s.model.slug for s in models if s.id != "primary"}


def test_the_fixture_prompt_is_one_sentence_of_valid_input() -> None:
    text = FIXTURE.read_text(encoding="utf-8")
    assert text.strip() and "." not in text
    assert len(text.encode()) <= run.MAX_PROMPT_BYTES


# --- job inputs ----------------------------------------------------------------


def test_the_declared_prompt_is_loaded_and_stripped(tmp_path: Path) -> None:
    _stage_inputs(tmp_path, b"  a person waves.\n")
    assert run._load_prompt(tmp_path) == "a person waves."


@pytest.mark.parametrize(
    "stage",
    [
        {"payload": b""},
        {"payload": b"   \n"},
        {"payload": b"\xff\xfe"},
        {"payload": b"wave\x00"},
        {"payload": b"a" * (run.MAX_PROMPT_BYTES + 1)},
        {"payload": b"wave", "slot": "command"},
        {"payload": b"wave", "media_type": "application/json"},
        {"payload": b"wave", "name": "prompt.md"},
    ],
)
def test_an_input_outside_the_declared_slot_is_refused(
    tmp_path: Path, stage: dict
) -> None:
    _stage_inputs(tmp_path, **stage)
    with pytest.raises(SystemExit):
        run._load_prompt(tmp_path)


def test_a_prompt_whose_size_differs_from_the_manifest_is_refused(
    tmp_path: Path,
) -> None:
    _stage_inputs(tmp_path, b"a person waves")
    (tmp_path / "prompt.txt").write_bytes(b"a person waved twice")
    with pytest.raises(SystemExit, match="size"):
        run._load_prompt(tmp_path)


def test_an_undeclared_extra_input_is_refused(tmp_path: Path) -> None:
    _stage_inputs(tmp_path, b"a person waves")
    (tmp_path / "extra.txt").write_text("x")
    with pytest.raises(SystemExit, match="do not match"):
        run._load_prompt(tmp_path)


# --- arguments and deadline ----------------------------------------------------


def _argv(outputs: Path, **overrides: str) -> list[str]:
    values = {
        "entrypoint": run.ENTRYPOINT,
        "output-mime": "application/zip",
        "seed": "7",
        "duration-seconds": "2.0",
        "output-dir": str(outputs),
        **overrides,
    }
    return [f"--{name}={value}" for name, value in values.items()]


@pytest.mark.parametrize(
    "override",
    [
        {"entrypoint": "/other.py"},
        {"output-mime": "model/gltf-binary"},
        {"seed": "-1"},
        {"duration-seconds": "0.1"},
        {"duration-seconds": "10.5"},
        {"duration-seconds": "nan"},
    ],
)
def test_job_arguments_outside_the_contract_are_refused(
    tmp_path: Path, override: dict
) -> None:
    with pytest.raises(SystemExit):
        run._arguments(_argv(tmp_path, **override))


def test_the_deadline_is_the_platform_job_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("VONK_JOB_TIMEOUT_SECONDS", "900")
    now = time.monotonic()
    assert 899 <= run._job_deadline() - now <= 901


@pytest.mark.parametrize("value", [None, "", "abc", "0", "3601"])
def test_a_missing_or_invalid_job_budget_fails_fast(
    monkeypatch: pytest.MonkeyPatch, value: str | None
) -> None:
    monkeypatch.delenv("VONK_JOB_TIMEOUT_SECONDS", raising=False)
    if value is not None:
        monkeypatch.setenv("VONK_JOB_TIMEOUT_SECONDS", value)
    with pytest.raises(SystemExit, match="VONK_JOB_TIMEOUT_SECONDS|between"):
        run._job_deadline()


# --- offline model wiring ------------------------------------------------------


def _fake_mount(root: Path) -> Path:
    for relative, names in {
        run.MNTP: run.MODEL_FILES[run.TEXT_ENCODER_MOUNT / run.MNTP],
        run.SUPERVISED: run.MODEL_FILES[run.TEXT_ENCODER_MOUNT / run.SUPERVISED],
    }.items():
        for name in names:
            path = root / relative / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("{}" if name.endswith(".json") else "weights")
    (root / run.MNTP / "adapter_config.json").write_text(
        json.dumps(
            {"base_model_name_or_path": "meta-llama/Meta-Llama-3-8B-Instruct", "r": 16}
        )
    )
    return root


def test_the_text_encoder_tree_points_the_adapter_at_the_mounted_base(
    tmp_path: Path,
) -> None:
    mount = _fake_mount(tmp_path / "mount")
    base = tmp_path / "base"
    root = run.build_text_encoder_tree(tmp_path / "scratch", mount=mount, base=base)
    config = json.loads((root / run.MNTP / "adapter_config.json").read_text())
    assert config == {"base_model_name_or_path": str(base), "r": 16}
    assert not (root / run.MNTP / "adapter_config.json").is_symlink()
    assert (root / run.MNTP / "adapter_model.safetensors").resolve() == (
        mount / run.MNTP / "adapter_model.safetensors"
    )
    assert (root / run.SUPERVISED).resolve() == mount / run.SUPERVISED
    # the mount itself is never written
    original = json.loads((mount / run.MNTP / "adapter_config.json").read_text())
    assert original["base_model_name_or_path"].startswith("meta-llama/")


def test_a_missing_mounted_file_is_named(tmp_path: Path) -> None:
    files = {tmp_path: ("config.yaml",)}
    with pytest.raises(SystemExit, match="config.yaml"):
        run.check_model_files(files)
    (tmp_path / "config.yaml").write_text("x")
    run.check_model_files(files)


# --- the generator -------------------------------------------------------------


def test_the_generator_keeps_the_creators_sampling_and_asks_for_the_t_pose_bvh() -> (
    None
):
    command = run.generation_command(
        "-a person waves", duration_seconds=2.0, seed=3, output_stem=Path("/s/motion")
    )
    assert command[1:3] == ["-m", "kimodo.scripts.generate"]
    assert command[command.index("--duration") + 1] == "2"
    assert command[command.index("--seed") + 1] == "3"
    for flag in ("--bvh", "--bvh_standard_tpose", "--no-postprocess"):
        assert flag in command
    for creator_default in (
        "--diffusion_steps",
        "--cfg_type",
        "--cfg_weight",
        "--num_samples",
    ):
        assert creator_default not in command
    assert command[-2:] == ["--", "-a person waves"]


def test_the_generator_environment_binds_every_model_to_a_mount() -> None:
    environment = run.generation_environment(Path("/s/text-encoders"), {"HOME": "/h"})
    assert environment["CHECKPOINT_DIR"] == str(run.CHECKPOINT_DIR)
    assert environment["TEXT_ENCODERS_DIR"] == "/s/text-encoders"
    assert environment["TEXT_ENCODER_MODE"] == "local"
    assert environment["PYTHONPATH"] == str(run.SOURCE_ROOT)
    assert environment["HOME"] == "/h"


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


def test_a_generator_past_the_deadline_is_killed_with_its_children(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    marker = tmp_path / "child.pid"
    child = (
        "import os, subprocess, sys, time\n"
        "grandchild = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(600)'])\n"
        f"open({str(marker)!r}, 'w').write(str(grandchild.pid))\n"
        "time.sleep(600)\n"
    )
    monkeypatch.setattr(run, "PUBLISH_RESERVE_SECONDS", 0)
    with pytest.raises(SystemExit, match="exceeded the job budget"):
        run.run_generation(
            [sys.executable, "-c", child],
            dict(os.environ),
            tmp_path,
            time.monotonic() + 3,
        )
    pid = int(marker.read_text())
    for _ in range(100):
        if not _alive(pid):
            break
        time.sleep(0.1)
    assert not _alive(pid)


def test_a_failing_generator_stops_the_job_with_its_status(tmp_path: Path) -> None:
    with pytest.raises(SystemExit, match="status 3"):
        run.run_generation(
            [sys.executable, "-c", "raise SystemExit(3)"],
            dict(os.environ),
            tmp_path,
            time.monotonic() + 60,
        )


def test_an_exhausted_budget_stops_before_generating(tmp_path: Path) -> None:
    with pytest.raises(SystemExit, match="budget exhausted"):
        run.run_generation(
            [sys.executable, "-c", "pass"], dict(os.environ), tmp_path, time.monotonic()
        )


# --- motion validation and publication -----------------------------------------


def test_a_consistent_bvh_and_npz_validate(tmp_path: Path) -> None:
    _write_motion(tmp_path / "motion", bvh_frames=60, npz_frames=60)
    assert run.validate_motion(tmp_path / "motion.bvh", tmp_path / "motion.npz") == 60


@pytest.mark.parametrize(
    "defect", ["no-motion", "no-frames", "zero-frames", "disagree", "no-joints"]
)
def test_an_inconsistent_motion_is_refused(tmp_path: Path, defect: str) -> None:
    numpy = _numpy()

    stem = tmp_path / "motion"
    _write_motion(stem, npz_frames=61 if defect == "disagree" else 60)
    text = _bvh(60)
    if defect == "no-motion":
        stem.with_suffix(".bvh").write_text("HIERARCHY\n")
    elif defect == "no-frames":
        stem.with_suffix(".bvh").write_text(text.replace("Frames: 60", "Frames: many"))
    elif defect == "zero-frames":
        stem.with_suffix(".bvh").write_text(text.replace("Frames: 60", "Frames: 0"))
    elif defect == "no-joints":
        numpy.savez(stem.with_suffix(".npz"), other=numpy.zeros((60, 3)))
    with pytest.raises(SystemExit):
        run.validate_motion(stem.with_suffix(".bvh"), stem.with_suffix(".npz"))


def _members(directory: Path) -> dict[str, Path]:
    _write_motion(directory / "motion")
    return {
        "motion.bvh": directory / "motion.bvh",
        "motion.npz": directory / "motion.npz",
    }


def test_the_zip_has_fixed_members_order_timestamps_and_bytes(tmp_path: Path) -> None:
    first, second = tmp_path / "a", tmp_path / "b"
    for directory in (first, second):
        directory.mkdir()
    members = _members(first)
    time.sleep(1.1)  # file modification times differ; the zip must not
    for path in members.values():
        os.utime(path, (1, 1))
    one = run.publish(members, first, first)
    two = run.publish(_members(second), second, second)
    assert one.read_bytes() == two.read_bytes()
    with zipfile.ZipFile(one) as archive:
        assert archive.namelist() == list(run.MEMBERS)
        assert {info.date_time for info in archive.infolist()} == {run.ZIP_TIMESTAMP}
        assert archive.testzip() is None


def test_the_published_zip_passes_the_qualification_assertions(tmp_path: Path) -> None:
    """The sweep's own evaluator, on the assertions the qualification definition declares."""
    archive = run.publish(_members(tmp_path), tmp_path, tmp_path)
    content = archive.read_bytes()
    definition = Definitions.load().artifact(f"vonk-forge/{SLUG}")
    assert definition is not None
    assertions = definition["assertions"]
    assert [a["kind"] for a in assertions].count("zip-entries") == 1
    assert (
        evaluate(assertions, [Output(archive.name, "application/zip", content)]) == {}
    )


# --- the job ----------------------------------------------------------------------

CALLS: list[list[str]] = []


def _main_with(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, **patches) -> Path:
    inputs = tmp_path / "inputs"
    _stage_inputs(inputs, FIXTURE.read_bytes())
    outputs = tmp_path / "outputs"
    (outputs / "tmp").mkdir(parents=True, exist_ok=True)
    mount = _fake_mount(tmp_path / "mount")
    monkeypatch.setenv("TMPDIR", str(outputs / "tmp"))
    monkeypatch.setenv("VONK_JOB_TIMEOUT_SECONDS", "60")
    tempfile.tempdir = None
    monkeypatch.setattr(run, "INPUT_ROOT", inputs)
    monkeypatch.setattr(run, "TEXT_ENCODER_MOUNT", mount)
    monkeypatch.setattr(run, "check_model_files", lambda *_: None)
    CALLS.clear()

    def generate(command, environment, cwd, deadline):
        CALLS.append(command)
        _write_motion(Path(command[command.index("--output") + 1]))

    monkeypatch.setattr(run, "run_generation", generate)
    for name, value in patches.items():
        monkeypatch.setattr(run.os if name == "replace" else run, name, value)
    return outputs


def test_the_job_publishes_one_plain_file_and_leaves_no_scratch(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    outputs = _main_with(monkeypatch, tmp_path)
    run.main(_argv(outputs))
    assert sorted(path.name for path in outputs.iterdir()) == ["motion.zip", "tmp"]
    assert list((outputs / "tmp").iterdir()) == []
    with zipfile.ZipFile(outputs / "motion.zip") as archive:
        assert archive.namelist() == ["motion.bvh", "motion.npz"]
        assert archive.read("motion.bvh").startswith(b"HIERARCHY")
        with np_load(archive.read("motion.npz")) as arrays:
            assert arrays["posed_joints"].shape[0] == 60
    command = CALLS[0]
    assert command[command.index("--seed") + 1] == "7"
    assert command[-1] == "a person walks forward slowly, then stops and looks around"


def np_load(payload: bytes):
    numpy = _numpy()

    return numpy.load(io.BytesIO(payload), allow_pickle=False)


def test_a_failed_publication_leaves_no_partial_output_and_a_retry_succeeds(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    def broken(source, target):
        raise OSError("injected rename failure")

    outputs = _main_with(monkeypatch, tmp_path, replace=broken)
    with pytest.raises(OSError, match="injected"):
        run.main(_argv(outputs))
    assert [path.name for path in outputs.iterdir()] == ["tmp"]
    assert list((outputs / "tmp").iterdir()) == []
    monkeypatch.undo()
    outputs = _main_with(monkeypatch, tmp_path)
    run.main(_argv(outputs))
    assert (outputs / "motion.zip").is_file()


def test_a_generator_that_wrote_inconsistent_motion_publishes_nothing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    outputs = _main_with(monkeypatch, tmp_path)

    def generate(command, environment, cwd, deadline):
        _write_motion(Path(command[command.index("--output") + 1]), npz_frames=5)

    monkeypatch.setattr(run, "run_generation", generate)
    with pytest.raises(SystemExit, match="disagree"):
        run.main(_argv(outputs))
    assert [path.name for path in outputs.iterdir()] == ["tmp"]


def test_a_generator_that_wrote_no_file_publishes_nothing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    outputs = _main_with(monkeypatch, tmp_path)
    monkeypatch.setattr(run, "run_generation", lambda *_: None)
    with pytest.raises(SystemExit, match="did not write"):
        run.main(_argv(outputs))
    assert [path.name for path in outputs.iterdir()] == ["tmp"]
