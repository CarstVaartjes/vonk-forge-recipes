"""BVH writer for HY-Motion's wooden skeleton, from the creator's own motion arrays.

The creator's only skeleton exporter is the Autodesk FBX SDK (``smplh2woodfbx.py``), which has
no aarch64 build, so this module writes the BVH instead from the same inputs the FBX path
reads: the template skeleton (``j_template.bin`` rest joints in metres, ``kintree.bin``
parents, ``joint_names.json``) and the NPZ's ``poses`` (axis-angle, 52 joints) and ``trans``.

Conventions, each taken from the creator's code:

- Units: the FBX export scales translation by 100 (``scale=100``, metres to centimetres), so
  offsets and root positions are centimetres.
- Root: the FBX export animates the pelvis at ``template translation + trans * 100``
  (``_applyAnimationToSkeleton``), and ``WoodenMesh`` places the pelvis at
  ``j_template[0] + trans``; the root position channels carry exactly that, with the root
  OFFSET at the origin so every reader agrees on where the pelvis is.
- Rotation: the local rotation of joint ``i`` is ``batch_rodrigues(poses[i])`` applied in
  ``WoodenMesh.simple_lbs`` as ``world_i = world_parent @ [R_i | rest_i - rest_parent]``; the FBX
  export writes the same matrices as ``mat2euler(axes="sxyz")``, that is ``R = Rz Ry Rx``,
  so the channels are ``Zrotation Yrotation Xrotation`` in degrees.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

JOINTS = 52
FPS = 30
CENTIMETRES_PER_METRE = 100.0
ROOT_CHANNELS = "Xposition Yposition Zposition Zrotation Yrotation Xrotation"
JOINT_CHANNELS = "Zrotation Yrotation Xrotation"
_GIMBAL = 1.0 - 1e-6


@dataclass(frozen=True)
class Skeleton:
    names: tuple[str, ...]
    parents: tuple[int, ...]
    rest: np.ndarray  # (J, 3) world rest joints, metres


def load_skeleton(asset_dir: Path, joint_names: Path) -> Skeleton:
    """The creator's ``load_model_data`` for the skeleton part: joints, parents and names."""

    rest = np.frombuffer((asset_dir / "j_template.bin").read_bytes(), dtype=np.float32)
    parents = np.frombuffer((asset_dir / "kintree.bin").read_bytes(), dtype=np.int32)
    names = json.loads(joint_names.read_text(encoding="utf-8"))
    if rest.size != JOINTS * 3 or parents.size != JOINTS or len(names) != JOINTS:
        raise SystemExit(
            f"the wooden skeleton must have {JOINTS} joints, found "
            f"{rest.size // 3} rest joints, {parents.size} parents and {len(names)} names"
        )
    ordered = parents[0] == -1 and all(
        0 <= parent < index for index, parent in enumerate(parents[1:], start=1)
    )
    if not ordered or len(set(names)) != JOINTS:
        raise SystemExit("the wooden skeleton must be one tree with parents before children")
    return Skeleton(
        tuple(names),
        tuple(int(parent) for parent in parents),
        rest.reshape(JOINTS, 3).astype(np.float64),
    )


def rotation_matrices(axis_angle: np.ndarray) -> np.ndarray:
    """Rodrigues, as the creator's ``batch_rodrigues``: (..., 3) axis-angle to (..., 3, 3)."""

    angle = np.linalg.norm(axis_angle, axis=-1, keepdims=True)
    safe = np.where(angle < 1e-12, 1.0, angle)
    x, y, z = np.moveaxis(axis_angle / safe, -1, 0)
    zero = np.zeros_like(x)
    cross = np.stack([zero, -z, y, z, zero, -x, -y, x, zero], axis=-1).reshape(
        *x.shape, 3, 3
    )
    sin = np.sin(angle)[..., None]
    cos = np.cos(angle)[..., None]
    return np.eye(3) + sin * cross + (1.0 - cos) * (cross @ cross)


def euler_zyx_degrees(matrix: np.ndarray) -> np.ndarray:
    """(..., 3, 3) to (..., 3) angles about Z, Y and X in degrees, with ``R = Rz(z) Ry(y) Rx(x)``."""

    sine_y = -matrix[..., 2, 0]
    regular = np.abs(sine_y) < _GIMBAL
    y = np.arcsin(np.clip(sine_y, -1.0, 1.0))
    x = np.where(
        regular,
        np.arctan2(matrix[..., 2, 1], matrix[..., 2, 2]),
        np.arctan2(matrix[..., 0, 1] * sine_y, matrix[..., 0, 2] * sine_y),
    )
    z = np.where(regular, np.arctan2(matrix[..., 1, 0], matrix[..., 0, 0]), 0.0)
    return np.degrees(np.stack([z, y, x], axis=-1))


def _children(parents: tuple[int, ...]) -> dict[int, list[int]]:
    children: dict[int, list[int]] = {index: [] for index in range(len(parents))}
    for index, parent in enumerate(parents):
        if parent >= 0:
            children[parent].append(index)
    return children


def _depth_first(children: dict[int, list[int]], index: int = 0) -> list[int]:
    order = [index]
    for child in children[index]:
        order.extend(_depth_first(children, child))
    return order


def bvh_text(
    skeleton: Skeleton, poses: np.ndarray, trans: np.ndarray, fps: int = FPS
) -> str:
    """One BVH clip: ``poses`` (T, 156) axis-angle and ``trans`` (T, 3) metres."""

    frames = poses.shape[0]
    if poses.shape != (frames, JOINTS * 3) or trans.shape != (frames, 3) or frames < 1:
        raise SystemExit(
            f"poses must be (T, {JOINTS * 3}) and trans (T, 3), got {poses.shape} and {trans.shape}"
        )
    if not (np.isfinite(poses).all() and np.isfinite(trans).all()):
        raise SystemExit("the motion holds non-finite values")
    children = _children(skeleton.parents)
    order = _depth_first(children)
    offsets = skeleton.rest.copy()
    offsets[1:] -= skeleton.rest[np.array(skeleton.parents[1:])]
    offsets *= CENTIMETRES_PER_METRE
    offsets[0] = 0.0

    lines = ["HIERARCHY"]

    def emit(index: int, depth: int) -> None:
        pad = "\t" * depth
        keyword, channels = ("ROOT", ROOT_CHANNELS) if index == 0 else ("JOINT", JOINT_CHANNELS)
        x, y, z = offsets[index]
        lines.extend(
            [
                f"{pad}{keyword} {skeleton.names[index]}",
                f"{pad}{{",
                f"{pad}\tOFFSET {x:.6f} {y:.6f} {z:.6f}",
                f"{pad}\tCHANNELS {len(channels.split())} {channels}",
            ]
        )
        for child in children[index]:
            emit(child, depth + 1)
        lines.append(f"{pad}}}")

    emit(0, 0)
    euler = euler_zyx_degrees(rotation_matrices(poses.reshape(frames, JOINTS, 3)))
    position = (skeleton.rest[0] + trans) * CENTIMETRES_PER_METRE
    lines.extend(["MOTION", f"Frames: {frames}", f"Frame Time: {1.0 / fps:.7f}"])
    for frame in range(frames):
        values = [*position[frame], *euler[frame, order].reshape(-1)]
        lines.append(" ".join(f"{value:.6f}" for value in values))
    return "\n".join(lines) + "\n"
