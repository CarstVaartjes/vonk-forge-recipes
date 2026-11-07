"""GEM-X SOMA motion parameters to ``motion.bvh`` and ``motion.npz``.

Reads the ``gem_body_video_predictions_v1`` result that ``gem-infer motion-video
--postprocess`` wrote and exports its world-space SOMA motion with Kimodo's own SOMA
exporter (``kimodo.exports.bvh.motion_to_bvh``, ``standard_tpose=True``).

GEM-X predicts a root orientation plus 76 local joint rotations of the 77 public SOMA
joints, and the world position of the Hips. The public joints are listed in the same order
as Kimodo's ``somaskel77``, and the local rotations are already in Kimodo's standard T-pose
frame: driving Kimodo's forward kinematics with them over the SOMA runtime's own rest
joints reproduces the SOMA runtime's joints to float32 precision. They therefore go to the
exporter unchanged. The BVH hierarchy carries Kimodo's neutral somaskel77 bone lengths, as
every Kimodo BVH does; the person's own SOMA identity and scale parameters are not part of
it.

The NPZ keeps Kimodo's array names and meaning (``posed_joints``, ``local_rot_mats``,
``global_rot_mats``, ``root_positions``, metres, Y up) and adds ``contact_logits``, GEM-X's
six static-contact logits per frame in the order left ankle, left foot, right ankle,
right foot, left wrist, right wrist, and ``fps`` (one value, 30).
"""

from __future__ import annotations

import argparse
import io
import sys
import zipfile
from pathlib import Path

SOURCE_ROOT = Path("/opt/vonk/gem-x")
RESULT_FORMAT = "gem_body_video_predictions_v1"
FPS = 30.0
JOINTS = 77
BODY_JOINTS = 76
# Fixed so that identical arrays always give identical NPZ bytes.
ZIP_TIMESTAMP = (1980, 1, 1, 0, 0, 0)


def npz_bytes(arrays: dict) -> bytes:
    """An uncompressed NPZ whose members carry a fixed timestamp."""

    import numpy

    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_STORED) as archive:
        for name, array in arrays.items():
            member = io.BytesIO()
            numpy.lib.format.write_array(
                member, numpy.ascontiguousarray(array), allow_pickle=False
            )
            archive.writestr(zipfile.ZipInfo(f"{name}.npy", ZIP_TIMESTAMP), member.getvalue())
    return buffer.getvalue()


def motion_tensors(result: dict):
    """Local rotation matrices ``(T, 77, 3, 3)``, Hips positions ``(T, 3)`` and contact logits."""

    import torch
    from gem.common.geometry.rotations import axis_angle_to_matrix

    if result.get("format") != RESULT_FORMAT:
        raise SystemExit(f"not a {RESULT_FORMAT} result")
    if not result.get("postprocessing"):
        raise SystemExit("the result has no Contact/IK postprocessing")
    if abs(float(result["sampling_fps"]) - FPS) > 1e-4:
        raise SystemExit("GEM-X motion must be sampled at 30 fps")
    parameters = result["pred_soma_params_global"]
    root = parameters["global_orient"]
    body = parameters["body_pose"]
    positions = parameters["transl"]
    frames = int(root.shape[1])
    if root.shape[0] != 1 or frames < 2:
        raise SystemExit("one sequence of at least two frames is required")
    axis_angle = torch.cat(
        (root.reshape(frames, 1, 3), body.reshape(frames, BODY_JOINTS, 3)), dim=1
    )
    rotations = axis_angle_to_matrix(axis_angle)
    positions = positions.reshape(frames, 3)
    logits = result["static_conf_logits"].reshape(frames, -1)
    for tensor in (rotations, positions, logits):
        if not bool(torch.isfinite(tensor).all()):
            raise SystemExit("GEM-X predicted non-finite motion")
    return rotations, positions, logits


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--motion", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    sys.path.insert(0, str(SOURCE_ROOT))

    import torch
    from kimodo.exports.bvh import motion_to_bvh
    from kimodo.skeleton import build_skeleton

    # The file was written moments ago by this job's own pinned gem-infer; it holds numpy
    # arrays (the tracking record), which the weights-only loader refuses.
    result = torch.load(args.motion, map_location="cpu", weights_only=False)
    rotations, positions, logits = motion_tensors(result)
    skeleton = build_skeleton(JOINTS)
    bvh = motion_to_bvh(
        rotations, positions, skeleton=skeleton, fps=FPS, standard_tpose=True
    )
    global_rotations, posed_joints, _ = skeleton.fk(rotations, positions)
    arrays = {
        "posed_joints": posed_joints,
        "local_rot_mats": rotations,
        "global_rot_mats": global_rotations,
        "root_positions": positions,
        "contact_logits": logits,
        "fps": torch.tensor(FPS),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "motion.bvh").write_text(bvh, encoding="utf-8")
    (args.output_dir / "motion.npz").write_bytes(
        npz_bytes({name: tensor.detach().cpu().numpy() for name, tensor in arrays.items()})
    )


if __name__ == "__main__":
    main()
