"""NVIDIA MotionBricks job adapter: a JSON motion script to Unitree G1 qpos JSON.

The job reads one ``command`` JSON (a script of motion segments), replays the
upstream headless navigation demo loop for it on the pinned checkpoints, and
publishes one ``motion.json`` holding the generated 30 fps G1 MuJoCo ``qpos``
trajectory. MotionBricks has no text encoder: the style, duration, heading and
optional speed of each segment are the only controls.

This module imports only the standard library and pydantic at import time;
torch, numpy and the vendored ``motionbricks`` package load inside the
functions that need them, so the contract and plumbing are testable without a
GPU, weights or the upstream stack.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import tempfile
import time
from argparse import Namespace
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

ENTRYPOINT = "/opt/vonk/source/run.py"
SOURCE_REPOSITORY = "https://github.com/NVlabs/GR00T-WholeBodyControl"
SOURCE_COMMIT = "45c03facbf848e0c743091b7d9c17952096ff908"
INPUT_ROOT = Path("/inputs")
# Plain git files (configs, skeletons, statistics) are vendored in the image;
# the four Git LFS checkpoints are mounted by the recipe at their repository
# paths under /models.
SOURCE_ROOT = Path("/opt/vonk/motionbricks")
SIDECAR_ROOT = SOURCE_ROOT / "out"
SKELETON_XML = SOURCE_ROOT / "assets/skeletons/g1/g1.xml"
MODEL_ROOT = Path("/models/motionbricks/out")
CHECKPOINTS = (
    "G1-clip.ckpt",
    "motionbricks_pose/version_1/checkpoints/model-step=2000000.ckpt",
    "motionbricks_root/version_1/checkpoints/model-step=2000000.ckpt",
    "motionbricks_vqvae/version_1/checkpoints/model-step=2000000.ckpt",
)
MAX_COMMAND_BYTES = 65536
MAX_MANIFEST_BYTES = 262144
OUTPUT_NAME = "motion.json"
FPS = 30
QPOS_DIM = 36
# The demo regenerates every 8 frames (controller_dt) and replans every
# generate_dt (2.0) controller periods.
GENERATE_DT = (8 / FPS) * 2.0
MAX_TOTAL_SECONDS = 120.0
QUATERNION_TOLERANCE = 1e-3
IDLE_HISTORY = 5
# joint order of g1_29dof.xml / g1.xml after floating_base_joint
QPOS_LAYOUT = (
    "root_pos_x",
    "root_pos_y",
    "root_pos_z",
    "root_quat_w",
    "root_quat_x",
    "root_quat_y",
    "root_quat_z",
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
)
# clip_holder_G1.CLIPS minus walk_left/walk_right, which the agent selects
# itself for sideways walking.
STYLES = (
    "idle",
    "slow_walk",
    "walk",
    "hand_crawling",
    "walk_boxing",
    "elbow_crawling",
    "stealth_walk",
    "injured_walk",
    "walk_stealth",
    "walk_happy_dance",
    "walk_zombie",
    "walk_gun",
    "walk_scared",
)
BLENDSPACE_STYLES = ("walk_left", "walk_right")
Style = Literal[
    "idle",
    "slow_walk",
    "walk",
    "hand_crawling",
    "walk_boxing",
    "elbow_crawling",
    "stealth_walk",
    "injured_walk",
    "walk_stealth",
    "walk_happy_dance",
    "walk_zombie",
    "walk_gun",
    "walk_scared",
]


class Segment(BaseModel):
    """One stretch of motion. Headings are MuJoCo world degrees: z up, 0 is +x,
    counter-clockwise positive. ``idle`` ignores both headings and keeps the
    current momentum and facing."""

    model_config = ConfigDict(extra="forbid", strict=True)

    style: Style
    duration_seconds: float = Field(ge=1 / FPS, le=60, allow_inf_nan=False)
    movement_heading_degrees: float = Field(
        default=0.0, ge=-360, le=360, allow_inf_nan=False
    )
    facing_heading_degrees: float | None = Field(
        default=None, ge=-360, le=360, allow_inf_nan=False
    )
    # Maps to upstream's internal ``target_vel``. The upstream demo never uses
    # it, so its scale is unverified; treat it as a hint, not metres per second.
    speed_mps: float | None = Field(default=None, gt=0, le=4, allow_inf_nan=False)

    def frames(self) -> int:
        return round(self.duration_seconds * FPS)


class Command(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    schema_version: Literal[1]
    segments: list[Segment] = Field(min_length=1, max_length=64)

    @model_validator(mode="after")
    def _bounded_total(self) -> Command:
        total = sum(segment.duration_seconds for segment in self.segments)
        if total > MAX_TOTAL_SECONDS:
            raise ValueError(f"segments total {total:g}s; at most {MAX_TOTAL_SECONDS:g}s")
        return self


def _arguments(argv: list[str] | None = None) -> Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--entrypoint", required=True)
    parser.add_argument("--output-mime", required=True)
    parser.add_argument("--timeout-seconds", type=int, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.entrypoint != ENTRYPOINT:
        raise SystemExit("unexpected signed-bundle entrypoint")
    if args.output_mime != "application/json":
        raise SystemExit("MotionBricks emits application/json only")
    if not 1 <= args.timeout_seconds <= 3600:
        raise SystemExit("timeout is outside the harness contract")
    if not 0 <= args.seed < 2**32:
        raise SystemExit("seed is outside the numpy/torch seed range")
    return args


def _load_command(input_root: Path) -> Command:
    """The one declared ``command`` JSON, verified against the job manifest."""

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
    except (OSError, ValidationError) as error:
        raise SystemExit(f"invalid Vonk job input manifest: {error}") from error
    declared = [item for item in manifest.files if item.slot == "command"]
    if len(manifest.files) != 1 or len(declared) != 1:
        raise SystemExit("exactly one declared command file is required")
    item = declared[0]
    if item.media_type != "application/json" or Path(item.name).suffix != ".json":
        raise SystemExit("the command must be a .json file of media type application/json")
    if {entry.name for entry in input_root.iterdir()} != {item.name, "manifest.json"}:
        raise SystemExit("job input files do not match the manifest")
    path = input_root / item.name
    if path.is_symlink() or not path.is_file():
        raise SystemExit("the declared command must be a regular file")
    if not 1 <= item.size_bytes <= MAX_COMMAND_BYTES:
        raise SystemExit(f"command.json must hold 1..{MAX_COMMAND_BYTES} bytes")
    payload = path.read_bytes()
    if len(payload) != item.size_bytes:
        raise SystemExit("command size does not match the job manifest")
    if hashlib.sha256(payload).hexdigest() != item.sha256:
        raise SystemExit("command digest does not match the job manifest")
    try:
        return Command.model_validate_json(payload)
    except ValidationError as error:
        raise SystemExit(f"invalid motion command: {error}") from error


def _deadline(timeout_seconds: int) -> float:
    return time.monotonic() + timeout_seconds


def _check_deadline(deadline: float) -> None:
    if time.monotonic() >= deadline:
        raise SystemExit("MotionBricks job budget exhausted before publication")


def build_out_tree(scratch: Path, *, sidecars: Path, models: Path) -> Path:
    """``scratch/out`` merged from the vendored sidecars and the mounted checkpoints.

    The upstream hparams name ``out/...`` relative to the working directory and
    ``experiment.test`` reads configs and checkpoints from one result_dir, so
    both file sets must appear in one tree. Every entry is a symlink; a missing
    checkpoint fails now, with its name, instead of deep inside the loader.
    """

    out = scratch / "out"
    for source in sorted(sidecars.rglob("*")):
        if source.is_file():
            link = out / source.relative_to(sidecars)
            link.parent.mkdir(parents=True, exist_ok=True)
            link.symlink_to(source)
    for relative in CHECKPOINTS:
        source = models / relative
        if not source.is_file():
            raise SystemExit(f"model file is not mounted: {source}")
        link = out / relative
        link.parent.mkdir(parents=True, exist_ok=True)
        link.symlink_to(source)
    return out


def _idle_directions(
    positions: list[tuple[float, float]], root_quaternion: tuple[float, float, float, float]
) -> tuple[tuple[float, float], tuple[float, float]]:
    """Movement and facing for ``idle``: keep the recent momentum and facing.

    Mirrors the idle branch of upstream ``WASD_controller``: the mean root xy
    step of the last frames, and the root +x axis projected onto the ground,
    each normalised with the same ``+1e-5``.
    """

    steps = len(positions) - 1
    velocity = (
        sum(b[0] - a[0] for a, b in zip(positions, positions[1:])) / steps * FPS,
        sum(b[1] - a[1] for a, b in zip(positions, positions[1:])) / steps * FPS,
    )
    speed = math.hypot(*velocity) + 1e-5
    movement = (velocity[0] / speed, velocity[1] / speed)
    w, x, y, z = root_quaternion
    forward = (1 - 2 * (y * y + z * z), 2 * (x * y + w * z))
    length = math.hypot(*forward) + 1e-5
    return movement, (forward[0] / length, forward[1] / length)


def _heading_vector(degrees: float) -> tuple[float, float]:
    radians = math.radians(degrees)
    return math.cos(radians), math.sin(radians)


def _control_signals(
    segment: Segment,
    history: list[list[float]],
    torch: Any,
    agent: Any,
    clips: Any,
) -> dict[str, Any]:
    if segment.style == "idle":
        movement, facing = _idle_directions(
            [(row[0], row[1]) for row in history],
            (history[-1][3], history[-1][4], history[-1][5], history[-1][6]),
        )
    else:
        movement = _heading_vector(segment.movement_heading_degrees)
        facing = (
            movement
            if segment.facing_heading_degrees is None
            else _heading_vector(segment.facing_heading_degrees)
        )
    signals = {
        "movement_direction": torch.tensor([[movement[0], movement[1], 0.0]]),
        "facing_direction": torch.tensor([[facing[0], facing[1], 0.0]]),
        "mode": torch.tensor([[list(clips.CLIPS).index(segment.style)]]),
        "allowed_pred_num_tokens": torch.tensor(
            clips.CLIPS[segment.style]["allowed_pred_num_tokens"]
        ).view(1, -1),
        "context_mujoco_qpos": agent.get_context_mujoco_qpos(),
    }
    if segment.speed_mps is not None:
        signals["target_vel"] = torch.tensor([[segment.speed_mps]])
    return signals


def generate_qpos(
    agent: Any, command: Command, deadline: float
) -> list[list[float]]:
    """Step the headless demo loop; one qpos row (36 floats) per output frame."""

    import torch
    from motionbricks.motion_backbone.demo.clips import clip_holder_G1

    rows: list[list[float]] = []
    history: list[list[float]] = []
    for segment in command.segments:
        for _ in range(segment.frames()):
            _check_deadline(deadline)
            row = [float(value) for value in agent.get_next_frame()]
            rows.append(row)
            if not history:
                history = [row] * IDLE_HISTORY
            signals = _control_signals(segment, history, torch, agent, clip_holder_G1)
            # the controller reads the frames before the current one
            history = [*history[1:], row]
            with torch.no_grad():
                agent.generate_new_frames(signals, GENERATE_DT)
    return rows


def validate_qpos(rows: list[list[float]], frames: int) -> None:
    """Refuse to publish anything that is not a finite (frames, 36) trajectory."""

    if len(rows) != frames:
        raise SystemExit(f"generated {len(rows)} frames, expected {frames}")
    for index, row in enumerate(rows):
        if len(row) != QPOS_DIM or not all(math.isfinite(value) for value in row):
            raise SystemExit(f"frame {index} is not {QPOS_DIM} finite values")
        norm = math.sqrt(sum(value * value for value in row[3:7]))
        if abs(norm - 1.0) > QUATERNION_TOLERANCE:
            raise SystemExit(f"frame {index} root quaternion norm is {norm:.6f}")


def motion_document(
    command: Command, rows: list[list[float]], seed: int
) -> dict[str, Any]:
    segments = []
    start = 0
    for segment in command.segments:
        movement = segment.movement_heading_degrees
        segments.append(
            {
                "style": segment.style,
                "start_frame": start,
                "frame_count": segment.frames(),
                "movement_heading_degrees": movement,
                "facing_heading_degrees": (
                    movement
                    if segment.facing_heading_degrees is None
                    else segment.facing_heading_degrees
                ),
                "speed_mps": segment.speed_mps,
            }
        )
        start += segment.frames()
    return {
        "format": "motionbricks-g1-qpos",
        "format_version": 1,
        "robot": "unitree-g1-29dof",
        "fps": FPS,
        "frame_count": len(rows),
        "qpos_dim": QPOS_DIM,
        "seed": seed,
        "coordinate_system": "mujoco-world-z-up",
        "units": {"position": "m", "angle": "rad"},
        "qpos_layout": list(QPOS_LAYOUT),
        "segments": segments,
        "qpos": rows,
        "source": {
            "repository": SOURCE_REPOSITORY,
            "commit": SOURCE_COMMIT,
        },
    }


def publish(document: dict[str, Any], scratch: Path, output_dir: Path) -> Path:
    """Write in scratch, then one atomic rename: a failure leaves no partial file."""

    staged = scratch / OUTPUT_NAME
    staged.write_text(json.dumps(document, allow_nan=False), encoding="utf-8")
    final = output_dir / OUTPUT_NAME
    os.replace(staged, final)
    return final


def load_agent(
    out: Path, *, device: str = "cuda", skeleton_xml: Path = SKELETON_XML
) -> Any:
    """The upstream demo's model stack, with strict checkpoint loading."""

    import torch
    from motionbricks.exp_setup import experiment
    from motionbricks.motion_backbone.demo.clips import clip_holder_G1
    from motionbricks.motion_backbone.demo.full_agent import full_navigation_agent
    from motionbricks.motion_backbone.inference.motion_inference import (
        motion_inference,
    )

    if list(clip_holder_G1.CLIPS) != [*STYLES, *BLENDSPACE_STYLES]:
        raise SystemExit("vendored MotionBricks clips drifted from the command styles")
    models, confs = experiment.test(
        Namespace(
            result_dir=str(out),
            data_root=None,
            explicit_dataset_folder=None,
            EXP="default",
            return_model_configs=True,
            return_dataloader=False,
        )
    )
    for name in ("pose", "root"):
        state = torch.load(confs[name].ckpt_path, map_location="cpu", weights_only=True)
        models[name].load_state_dict(state["state_dict"])
    inferencer = motion_inference(models, models["pose"].args, device)
    return full_navigation_agent(
        inferencer,
        None,
        device=device,
        speed_scale=[1.0, 1.0],
        target_root_realignment=True,
        source_root_realignment=True,
        force_canonicalization=True,
        skip_ending_target_cond=False,
        filter_qpos=True,
        skeleton_xml=str(skeleton_xml),
        clips="G1",
        ckpt_path=str(out / "G1-clip.ckpt"),
        reprocess_clips=False,
        val_dataloader=None,
    ).to(device)


def seed_everything(seed: int) -> None:
    """After model construction: ``experiment.test`` re-seeds with its own 7."""

    import numpy
    import torch

    numpy.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main(argv: list[str] | None = None) -> None:
    args = _arguments(argv)
    deadline = _deadline(args.timeout_seconds)
    command = _load_command(INPUT_ROOT)
    frames = sum(segment.frames() for segment in command.segments)
    sys.path.insert(0, str(SOURCE_ROOT))
    previous_cwd, previous_tempdir = Path.cwd(), tempfile.tempdir
    with tempfile.TemporaryDirectory(prefix="vonk-motionbricks-") as name:
        scratch = Path(name)
        try:
            # upstream's tempfile.mkdtemp scratch and log file land inside ours
            tempfile.tempdir = name
            out = build_out_tree(scratch, sidecars=SIDECAR_ROOT, models=MODEL_ROOT)
            os.chdir(scratch)
            agent = load_agent(out)
            seed_everything(args.seed)
            agent.reset()
            rows = generate_qpos(agent, command, deadline)
            validate_qpos(rows, frames)
            publish(motion_document(command, rows, args.seed), scratch, args.output_dir)
        finally:
            os.chdir(previous_cwd)
            tempfile.tempdir = previous_tempdir


if __name__ == "__main__":
    main()
