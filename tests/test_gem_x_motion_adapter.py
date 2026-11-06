"""GEM-X motion adapter behaviour that needs no GPU, weights or the upstream stack.

The job-input check, the scratch weights tree, the inference and export commands, their
process-group deadline, the deterministic zip and the NPZ writer all run here, and the zip is
judged by the sweep's own ``zip-entries`` evaluator against the qualification definition. The
recipe, Model documents, fixture and adapter must agree on every mounted path and runtime
argument. The model run and the BVH export are proven once on real weights and again by the
hardware sweep.
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
ADAPTER = ROOT / "adapters/three-d/gem-x-motion"
WHEEL = next(ADAPTER.glob("*.whl"))
SLUG = "gem-x-body-motion-pytorch-single"
FIXTURE = ROOT / "qualification/fixtures/gemx-person-2s.mp4"
sys.path.insert(0, str(WHEEL))


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, ADAPTER / f"{name}.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[f"gem_x_{name}"] = module
    spec.loader.exec_module(module)
    return module


run = _load("run")
export = _load("export_motion")


def _stage_inputs(
    root: Path,
    payload: bytes,
    *,
    slot: str = "video",
    media_type: str = "video/mp4",
    name: str = "clip.mp4",
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


# --- recipe, models, fixture and adapter agree -----------------------------------


def test_the_recipe_passes_exactly_the_arguments_the_adapter_parses() -> None:
    recipe = documents.recipe(SLUG)
    arguments = {item.name: item for item in recipe.runtime.arguments}
    assert set(arguments) == {"entrypoint", "output-mime"}
    assert arguments["entrypoint"].value == run.ENTRYPOINT
    assert arguments["output-mime"].value == "application/zip"
    assert recipe.settings.knobs == {}
    argv = [f"--{name}={item.value}" for name, item in arguments.items()]
    run._arguments([*argv, "--output-dir=/outputs"])


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
    assert mounted - wanted == set()


def test_the_primary_model_declares_the_soma_assets_and_the_detector() -> None:
    models = documents.recipe(SLUG).models
    primary = next(s for s in models if s.id == "primary")
    declared = {
        (item.slug, item.content_sha256)
        for item in documents.model(primary.model.slug).dependencies
    }
    assert declared == {
        (s.model.slug, s.model.content_sha256) for s in models if s.id != "primary"
    }


def test_the_release_configuration_names_the_detector_the_job_links() -> None:
    assert run.RELEASE_DETECTOR == "yolox_humanart.onnx"
    assert run.RELEASE_CONFIG.name in run.MODEL_FILES[run.GEM_WEIGHTS]
    assert run.DETECTOR.name in run.MODEL_FILES[run.DETECTOR.parent]


def test_the_disk_declaration_covers_the_mounted_bytes() -> None:
    recipe = documents.recipe(SLUG)
    mounted = 0
    for selection in recipe.models:
        sizes = {
            item.id: item.size_bytes
            for item in documents.model(selection.model.slug).files
        }
        mounted += sum(sizes[item.file_id] for item in selection.files)
    role = recipe.topology.roles[0]
    assert role.resources.disk.artifact_bytes == mounted


def test_the_fixture_is_the_registered_small_mp4() -> None:
    definition = Definitions.load().artifact(f"vonk-forge/{SLUG}")
    assert definition is not None
    shared = json.loads((ROOT / "qualification/shared.json").read_text())["fixtures"]
    fixture = shared[definition["inputs"][0]["fixture"]]
    payload = FIXTURE.read_bytes()
    assert fixture["path"] == f"fixtures/{FIXTURE.name}"
    assert fixture["sha256"] == hashlib.sha256(payload).hexdigest()
    assert fixture["size_bytes"] == len(payload) < 1_000_000
    assert fixture["media_type"] == "video/mp4"
    assert fixture["provenance"]["license_spdx"] == "CC0-1.0"
    assert payload[4:8] == b"ftyp"


# --- job inputs ----------------------------------------------------------------------


def test_the_declared_video_is_returned(tmp_path: Path) -> None:
    _stage_inputs(tmp_path, FIXTURE.read_bytes())
    assert run._load_video(tmp_path) == tmp_path / "clip.mp4"


@pytest.mark.parametrize(
    ("media_type", "name"),
    [
        ("video/mp4", "clip.MP4"),
        ("video/quicktime", "clip.mov"),
        ("video/webm", "clip.webm"),
        ("video/x-matroska", "clip.mkv"),
    ],
)
def test_every_declared_video_container_is_accepted(
    tmp_path: Path, media_type: str, name: str
) -> None:
    _stage_inputs(tmp_path, b"video", media_type=media_type, name=name)
    assert run._load_video(tmp_path).name == name


@pytest.mark.parametrize(
    "stage",
    [
        {"payload": b""},
        {"payload": b"video", "slot": "prompt"},
        {"payload": b"video", "media_type": "audio/wav"},
        {"payload": b"video", "name": "clip.avi"},
    ],
)
def test_an_input_outside_the_declared_slot_is_refused(
    tmp_path: Path, stage: dict
) -> None:
    _stage_inputs(tmp_path, **stage)
    with pytest.raises(SystemExit):
        run._load_video(tmp_path)


def test_a_video_whose_size_differs_from_the_manifest_is_refused(
    tmp_path: Path,
) -> None:
    _stage_inputs(tmp_path, b"video")
    (tmp_path / "clip.mp4").write_bytes(b"longer video")
    with pytest.raises(SystemExit, match="size"):
        run._load_video(tmp_path)


def test_an_undeclared_extra_input_is_refused(tmp_path: Path) -> None:
    _stage_inputs(tmp_path, b"video")
    (tmp_path / "extra.txt").write_text("x")
    with pytest.raises(SystemExit, match="do not match"):
        run._load_video(tmp_path)


def test_a_symlinked_video_is_refused(tmp_path: Path) -> None:
    _stage_inputs(tmp_path, b"video")
    (tmp_path / "clip.mp4").unlink()
    (tmp_path / "clip.mp4").symlink_to(FIXTURE)
    with pytest.raises(SystemExit, match="regular file"):
        run._load_video(tmp_path)


# --- arguments and deadline ------------------------------------------------------------


@pytest.mark.parametrize(
    "override",
    [{"entrypoint": "/other.py"}, {"output-mime": "model/gltf-binary"}],
)
def test_job_arguments_outside_the_contract_are_refused(
    tmp_path: Path, override: dict
) -> None:
    values = {
        "entrypoint": run.ENTRYPOINT,
        "output-mime": "application/zip",
        "output-dir": str(tmp_path),
        **override,
    }
    with pytest.raises(SystemExit):
        run._arguments([f"--{name}={value}" for name, value in values.items()])


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


# --- offline model wiring --------------------------------------------------------------


def test_a_missing_mounted_file_is_named(tmp_path: Path) -> None:
    files = {tmp_path: ("body_s240000.ckpt",)}
    with pytest.raises(SystemExit, match="body_s240000.ckpt"):
        run.check_model_files(files)
    (tmp_path / "body_s240000.ckpt").write_text("x")
    run.check_model_files(files)


def test_the_weights_tree_links_the_mount_and_names_the_detector_the_release_asks_for(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    mount = tmp_path / "mount"
    mount.mkdir(exist_ok=True)
    names = run.MODEL_FILES[run.GEM_WEIGHTS]
    for name in names:
        (mount / name).write_text(f"content of {name}")
    detector = tmp_path / "yolox_x.onnx"
    detector.write_text("detector")
    monkeypatch.setattr(run, "GEM_WEIGHTS", mount)
    monkeypatch.setattr(run, "RELEASE_CONFIG", mount / run.RELEASE_CONFIG.name)
    monkeypatch.setattr(run, "DETECTOR", detector)
    monkeypatch.setattr(run, "MODEL_FILES", {mount: names})

    (tmp_path / "scratch").mkdir()
    config = run.build_weights_tree(tmp_path / "scratch")

    root = config.parent
    assert config.name == "body_video_native77.yaml"
    assert not config.is_symlink()
    assert config.read_bytes() == (mount / config.name).read_bytes()
    for name in ("body_s240000.ckpt", "motion_stats_raw.npz", "vitpose.pth"):
        assert (root / name).is_symlink()
        assert (root / name).resolve() == (mount / name).resolve()
    assert (root / run.RELEASE_DETECTOR).resolve() == detector.resolve()
    # the mount itself is never written
    assert sorted(path.name for path in mount.iterdir()) == sorted(names)


# --- the steps ---------------------------------------------------------------------------


def test_the_inference_is_upstreams_body_video_path_with_the_creators_settings() -> (
    None
):
    command = run.inference_command(
        Path("/inputs/clip.mp4"), Path("/s/weights/c.yaml"), Path("/s/motion.pt")
    )
    assert command[1:5] == ["-m", "gem.inference", "motion-video", "--task"]
    assert command[command.index("--task") + 1] == "body"
    assert command[command.index("--device") + 1] == "cuda"
    assert command[command.index("--soma-assets") + 1] == str(run.SOMA_ASSETS)
    assert "--postprocess" in command
    # the full clip, the release configuration's camera mode
    for flag in ("--start", "--duration", "--camera-mode"):
        assert flag not in command


def test_the_export_receives_the_result_and_an_output_directory() -> None:
    command = run.export_command(Path("/s/motion.pt"), Path("/s/export"))
    assert command[1] == str(run.EXPORT_SCRIPT)
    assert command[command.index("--motion") + 1] == "/s/motion.pt"
    assert command[command.index("--output-dir") + 1] == "/s/export"


def test_the_children_get_the_source_tree_and_keep_the_rest() -> None:
    environment = run.child_environment({"HOME": "/h", "PYTHONPATH": "/platform"})
    assert environment == {"HOME": "/h", "PYTHONPATH": str(run.SOURCE_ROOT)}


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


def test_a_step_past_the_deadline_is_killed_with_its_children(
    tmp_path: Path,
) -> None:
    marker = tmp_path / "child.pid"
    child = (
        "import subprocess, sys, time\n"
        "grandchild = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(600)'])\n"
        f"open({str(marker)!r}, 'w').write(str(grandchild.pid))\n"
        "time.sleep(600)\n"
    )
    with pytest.raises(SystemExit, match="exceeded the job budget"):
        run.run_step(
            "motion inference",
            [sys.executable, "-c", child],
            tmp_path,
            time.monotonic() + 3,
        )
    pid = int(marker.read_text())
    for _ in range(100):
        if not _alive(pid):
            break
        time.sleep(0.1)
    assert not _alive(pid)


def test_a_failing_step_stops_the_job_with_its_status(tmp_path: Path) -> None:
    with pytest.raises(SystemExit, match="export failed with exit status 3"):
        run.run_step(
            "export",
            [sys.executable, "-c", "raise SystemExit(3)"],
            tmp_path,
            time.monotonic() + 60,
        )


def test_an_exhausted_budget_stops_before_the_step(tmp_path: Path) -> None:
    with pytest.raises(SystemExit, match="budget exhausted before export"):
        run.run_step(
            "export", [sys.executable, "-c", "pass"], tmp_path, time.monotonic()
        )


def test_the_publish_reserve_is_kept_back_from_the_model_run(tmp_path: Path) -> None:
    with pytest.raises(SystemExit, match="budget exhausted before motion inference"):
        run.run_step(
            "motion inference",
            [sys.executable, "-c", "pass"],
            tmp_path,
            time.monotonic() + run.PUBLISH_RESERVE_SECONDS,
            reserve=run.PUBLISH_RESERVE_SECONDS,
        )


# --- the NPZ and the zip -----------------------------------------------------------------


def test_the_npz_is_byte_identical_for_identical_arrays_and_numpy_reads_it() -> None:
    numpy = _numpy()
    arrays = {
        "posed_joints": numpy.arange(24, dtype="float32").reshape(2, 4, 3),
        "fps": numpy.array([30.0], dtype="float32"),
    }
    first = export.npz_bytes(arrays)
    time.sleep(1.1)  # the wall clock moves; the bytes must not
    assert export.npz_bytes(arrays) == first
    with zipfile.ZipFile(io.BytesIO(first)) as archive:
        assert {info.date_time for info in archive.infolist()} == {export.ZIP_TIMESTAMP}
    with numpy.load(io.BytesIO(first), allow_pickle=False) as loaded:
        assert loaded.files == ["posed_joints", "fps"]
        assert (loaded["posed_joints"] == arrays["posed_joints"]).all()


def _bvh(frames: int) -> str:
    return (
        "HIERARCHY\nROOT Root\n{\n  OFFSET 0 0 0\n}\nMOTION\n"
        f"Frames: {frames}\nFrame Time: 0.033333\n" + "0 " * 6 + "\n"
    )


def _members(directory: Path) -> dict[str, Path]:
    numpy = _numpy()

    directory.mkdir(parents=True, exist_ok=True)
    (directory / "motion.bvh").write_text(_bvh(60), encoding="utf-8")
    (directory / "motion.npz").write_bytes(
        export.npz_bytes(
            {
                "posed_joints": numpy.random.default_rng(0)
                .normal(size=(60, 77, 3))
                .astype("float32")
            }
        )
    )
    return {
        "motion.bvh": directory / "motion.bvh",
        "motion.npz": directory / "motion.npz",
    }


def test_the_zip_has_fixed_members_order_timestamps_and_bytes(tmp_path: Path) -> None:
    first, second = tmp_path / "a", tmp_path / "b"
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
    definition = Definitions.load().artifact(f"vonk-forge/{SLUG}")
    assert definition is not None
    assertions = definition["assertions"]
    assert [a["kind"] for a in assertions].count("zip-entries") == 1
    assert (
        evaluate(
            assertions, [Output(archive.name, "application/zip", archive.read_bytes())]
        )
        == {}
    )


def test_the_layout_matches_the_kimodo_recipes_zip() -> None:
    """Both recipes declare the same entry names, order and entry rules."""
    definition = Definitions.load().artifact(f"vonk-forge/{SLUG}")
    assert definition is not None
    entries = next(a for a in definition["assertions"] if a["kind"] == "zip-entries")
    assert entries["exact_names"] == list(run.MEMBERS) == ["motion.bvh", "motion.npz"]
    assert entries["allowed_suffixes"] == [".bvh", ".npz"]
    assert entries["nonempty"] is True


# --- the job -------------------------------------------------------------------------------

CALLS: list[tuple[str, list[str]]] = []


def _main_with(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, **patches) -> Path:
    inputs = tmp_path / "inputs"
    _stage_inputs(inputs, FIXTURE.read_bytes())
    outputs = tmp_path / "outputs"
    (outputs / "tmp").mkdir(parents=True, exist_ok=True)
    mount = tmp_path / "mount"
    mount.mkdir(exist_ok=True)
    monkeypatch.setenv("TMPDIR", str(outputs / "tmp"))
    monkeypatch.setenv("VONK_JOB_TIMEOUT_SECONDS", "600")
    tempfile.tempdir = None
    monkeypatch.setattr(run, "INPUT_ROOT", inputs)
    monkeypatch.setattr(run, "check_model_files", lambda *_: None)
    monkeypatch.setattr(run, "build_weights_tree", lambda scratch: scratch / "c.yaml")
    CALLS.clear()

    def step(name, command, cwd, deadline, *, reserve=0):
        CALLS.append((name, command))
        if name == "export":
            _members(Path(command[command.index("--output-dir") + 1]))

    monkeypatch.setattr(run, "run_step", step)
    for name, value in patches.items():
        monkeypatch.setattr(run.os if name == "replace" else run, name, value)
    return outputs


def _argv(outputs: Path) -> list[str]:
    return [
        f"--entrypoint={run.ENTRYPOINT}",
        "--output-mime=application/zip",
        f"--output-dir={outputs}",
    ]


def test_the_job_runs_inference_then_export_and_publishes_one_plain_file(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    outputs = _main_with(monkeypatch, tmp_path)
    run.main(_argv(outputs))
    assert [name for name, _ in CALLS] == ["motion inference", "export"]
    inference = CALLS[0][1]
    assert inference[inference.index("--input") + 1] == str(
        tmp_path / "inputs" / "clip.mp4"
    )
    assert sorted(path.name for path in outputs.iterdir()) == ["motion.zip", "tmp"]
    assert list((outputs / "tmp").iterdir()) == []
    with zipfile.ZipFile(outputs / "motion.zip") as archive:
        assert archive.namelist() == ["motion.bvh", "motion.npz"]
        assert archive.read("motion.bvh").startswith(b"HIERARCHY")


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


def test_an_export_that_wrote_no_file_publishes_nothing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    outputs = _main_with(monkeypatch, tmp_path)
    monkeypatch.setattr(run, "run_step", lambda *_args, **_kwargs: None)
    with pytest.raises(SystemExit, match="did not write"):
        run.main(_argv(outputs))
    assert [path.name for path in outputs.iterdir()] == ["tmp"]
