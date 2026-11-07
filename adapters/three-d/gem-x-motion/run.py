"""NVIDIA GEM-X job adapter: one video to a SOMA motion zip (BVH and NPZ).

The job reads one ``video`` file and runs upstream's own ``gem-infer motion-video``
(``python -m gem.inference``) with the creator's Native77 final775 Body Motion release
(``body_video_native77.yaml``) and its Contact/IK ``--postprocess``. A second step turns the
predicted SOMA parameters into ``motion.bvh`` (somaskel77 skeleton, centimetres, standard
T-pose rest pose, 30 fps, world-space root trajectory) with Kimodo's SOMA exporter and
into ``motion.npz``. The platform's artifact contract has no animation media type, so the
two files travel in one deterministic zip, laid out exactly like the Kimodo recipe's.

Everything the creator's code would fetch is mounted instead: the GEM-X checkpoints and
release configuration (``/models/gem-x``), the SOMA-X v0.3.3 geometry assets
(``/models/soma-x``) and the YOLOX person detector (``/models/yolox``). The release
configuration resolves its checkpoints next to itself and names the creator's HumanArt
detector, ``yolox_humanart.onnx``, which has no catalogued source. The job therefore runs the
unmodified configuration from a scratch directory of links to the mounted files, where that
name is the mounted YOLOX-X.

This module imports only the standard library at import time (the protocol model loads inside
the function that reads the manifest), so the plumbing is testable without a GPU, weights or
the upstream stack.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import zipfile
from argparse import Namespace
from pathlib import Path

ENTRYPOINT = "/opt/vonk/source/run.py"
SOURCE_REPOSITORY = "https://github.com/NVlabs/GEM-X"
SOURCE_COMMIT = "4aef1452e5df171deff4f293c08e643eb450981e"
INPUT_ROOT = Path("/inputs")
SOURCE_ROOT = Path("/opt/vonk/gem-x")
EXPORT_SCRIPT = Path("/opt/vonk/source/export_motion.py")
MODEL_ROOT = Path("/models")
GEM_WEIGHTS = MODEL_ROOT / "gem-x"
SOMA_ASSETS = MODEL_ROOT / "soma-x"
DETECTOR = MODEL_ROOT / "yolox" / "yolox_x.onnx"
RELEASE_CONFIG = GEM_WEIGHTS / "body_video_native77.yaml"
# The detector file name the release configuration asks for.
RELEASE_DETECTOR = "yolox_humanart.onnx"
MODEL_FILES = {
    GEM_WEIGHTS: (
        "body_s240000.ckpt",
        "body_video_native77.yaml",
        "motion_native77_e774_s292175.ckpt",
        "motion_stats_raw.npz",
        "vitpose.pth",
    ),
    SOMA_ASSETS: (
        "SOMA_neutral.npz",
        "SOMA_procedural_transforms.json",
        "SOMA_template_rig.usda",
    ),
    DETECTOR.parent: (DETECTOR.name,),
}
VIDEO_MEDIA_TYPES = frozenset(
    {"video/mp4", "video/quicktime", "video/webm", "video/x-matroska"}
)
VIDEO_SUFFIXES = frozenset({".mkv", ".mov", ".mp4", ".webm"})
MAX_MANIFEST_BYTES = 262144
OUTPUT_NAME = "motion.zip"
MEMBERS = ("motion.bvh", "motion.npz")
# Time kept after the model run for exporting, validating and writing the zip.
PUBLISH_RESERVE_SECONDS = 60
KILL_GRACE_SECONDS = 10
# Fixed so that identical members always give identical zip bytes.
ZIP_TIMESTAMP = (1980, 1, 1, 0, 0, 0)


def _arguments(argv: list[str] | None = None) -> Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--entrypoint", required=True)
    parser.add_argument("--output-mime", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.entrypoint != ENTRYPOINT:
        raise SystemExit("unexpected signed-bundle entrypoint")
    if args.output_mime != "application/zip":
        raise SystemExit("GEM-X emits application/zip only")
    return args


def _load_video(input_root: Path) -> Path:
    """The one declared ``video`` file of the job input manifest."""

    from vonk_agent_protocol.job_inputs import RecipeJobInputManifest

    if not input_root.is_dir() or input_root.is_symlink():
        raise SystemExit("/inputs must be a directory containing the declared inputs")
    manifest_path = input_root / "manifest.json"
    if (
        manifest_path.is_symlink()
        or not manifest_path.is_file()
        or manifest_path.stat().st_size > MAX_MANIFEST_BYTES
    ):
        raise SystemExit("a regular, bounded Vonk job input manifest is required")
    try:
        manifest = RecipeJobInputManifest.model_validate_json(
            manifest_path.read_bytes()
        )
    except (OSError, ValueError) as error:
        raise SystemExit(f"invalid Vonk job input manifest: {error}") from error
    declared = [item for item in manifest.files if item.slot == "video"]
    if len(manifest.files) != 1 or len(declared) != 1:
        raise SystemExit("exactly one declared video file is required")
    item = declared[0]
    if (
        item.media_type not in VIDEO_MEDIA_TYPES
        or Path(item.name).suffix.lower() not in VIDEO_SUFFIXES
    ):
        raise SystemExit("the video must be an mp4, mov, webm or mkv file")
    if {entry.name for entry in input_root.iterdir()} != {item.name, "manifest.json"}:
        raise SystemExit("job input files do not match the manifest")
    path = input_root / item.name
    if path.is_symlink() or not path.is_file():
        raise SystemExit("the declared video must be a regular file")
    if path.stat().st_size != item.size_bytes or item.size_bytes < 1:
        raise SystemExit("video size does not match the job manifest")
    return path


def _job_deadline() -> float:
    """The platform's job budget as one monotonic deadline."""
    try:
        seconds = int(os.environ["VONK_JOB_TIMEOUT_SECONDS"])
    except (KeyError, ValueError) as error:
        raise SystemExit("job must declare VONK_JOB_TIMEOUT_SECONDS") from error
    if not 1 <= seconds <= 3600:
        raise SystemExit("job timeout must be between 1 and 3600 seconds")
    return time.monotonic() + seconds


def check_model_files(files: dict[Path, tuple[str, ...]] = MODEL_FILES) -> None:
    """Fail now, naming the file, instead of deep inside the upstream loader."""
    for directory, names in files.items():
        for name in names:
            if not (directory / name).is_file():
                raise SystemExit(f"model file is not mounted: {directory / name}")


def build_weights_tree(scratch: Path) -> Path:
    """The release configuration and the files it names, side by side in ``scratch/weights``.

    The configuration is copied byte for byte (GEM-X resolves its paths next to the file it
    reads, so a link would send it back to the mount); every checkpoint is a link to the
    mount, and ``yolox_humanart.onnx``, which no catalogued source provides, is a link to the
    mounted YOLOX-X.
    """

    root = scratch / "weights"
    root.mkdir()
    for name in MODEL_FILES[GEM_WEIGHTS]:
        if name == RELEASE_CONFIG.name:
            shutil.copyfile(RELEASE_CONFIG, root / name)
        else:
            (root / name).symlink_to(GEM_WEIGHTS / name)
    (root / RELEASE_DETECTOR).symlink_to(DETECTOR)
    return root / RELEASE_CONFIG.name


def inference_command(video: Path, config: Path, motion: Path) -> list[str]:
    """Upstream's ``gem-infer motion-video`` with the creator's own settings.

    The full clip is processed (GEM-X refuses a source longer than 60 seconds instead of
    shortening it) and the camera mode is the release configuration's default, the
    background estimate. ``--postprocess`` is the creator's Contact/IK correction.
    """

    return [
        sys.executable,
        "-m",
        "gem.inference",
        "motion-video",
        "--task",
        "body",
        "--input",
        str(video),
        "--config",
        str(config),
        "--device",
        "cuda",
        "--postprocess",
        "--soma-assets",
        str(SOMA_ASSETS),
        "--output",
        str(motion),
    ]


def export_command(motion: Path, directory: Path) -> list[str]:
    return [
        sys.executable,
        str(EXPORT_SCRIPT),
        "--motion",
        str(motion),
        "--output-dir",
        str(directory),
    ]


def child_environment(base: dict[str, str] | None = None) -> dict[str, str]:
    environment = dict(os.environ if base is None else base)
    # PYTHONPATH is platform-owned for the image, so only the children get the source tree.
    environment["PYTHONPATH"] = str(SOURCE_ROOT)
    return environment


def _kill_group(process: subprocess.Popen[bytes]) -> None:
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    process.wait(timeout=KILL_GRACE_SECONDS)


def run_step(
    name: str, command: list[str], cwd: Path, deadline: float, *, reserve: float = 0
) -> None:
    """Run one step in its own process group, killed as a group at the deadline."""

    budget = deadline - time.monotonic() - reserve
    if budget <= 0:
        raise SystemExit(f"GEM-X job budget exhausted before {name}")
    process = subprocess.Popen(
        command, env=child_environment(), cwd=cwd, start_new_session=True
    )
    try:
        code = process.wait(timeout=budget)
    except subprocess.TimeoutExpired:
        _kill_group(process)
        raise SystemExit(f"GEM-X {name} exceeded the job budget") from None
    except BaseException:
        _kill_group(process)
        raise
    if code != 0:
        raise SystemExit(f"GEM-X {name} failed with exit status {code}")


def publish(members: dict[str, Path], scratch: Path, output_dir: Path) -> Path:
    """One zip with fixed member order, timestamps and modes, then one atomic rename."""

    staged = scratch / OUTPUT_NAME
    with zipfile.ZipFile(staged, "w") as archive:
        for name in MEMBERS:
            info = zipfile.ZipInfo(name, ZIP_TIMESTAMP)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.create_system = 3
            info.external_attr = 0o644 << 16
            archive.writestr(info, members[name].read_bytes())
    final = output_dir / OUTPUT_NAME
    os.replace(staged, final)
    return final


def main(argv: list[str] | None = None) -> None:
    args = _arguments(argv)
    deadline = _job_deadline()
    video = _load_video(INPUT_ROOT)
    check_model_files()
    with tempfile.TemporaryDirectory(prefix="vonk-gem-x-") as name:
        scratch = Path(name)
        config = build_weights_tree(scratch)
        motion = scratch / "motion.pt"
        run_step(
            "motion inference",
            inference_command(video, config, motion),
            scratch,
            deadline,
            reserve=PUBLISH_RESERVE_SECONDS,
        )
        exported = scratch / "export"
        exported.mkdir()
        run_step(
            "export",
            export_command(motion, exported),
            scratch,
            deadline,
        )
        members = {member: exported / member for member in MEMBERS}
        for path in members.values():
            if not path.is_file():
                raise SystemExit(f"GEM-X export did not write {path.name}")
        publish(members, scratch, args.output_dir)


if __name__ == "__main__":
    main()
